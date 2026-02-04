#include "mla_bwd.cuh"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <kerutils/kerutils.cuh>
#include <cutlass/arch/arch.h>

using namespace test_operator::mla_bwd;

// Kernel implementation: test mla_bwd with Q, KV, dO inputs
__global__ void test_mla_bwd_kernel(
    const bf16* __restrict__ q,      // [B_H, D_Q] = [128, 576]
    const bf16* __restrict__ kv,     // [B_TOPK, D_K] = [64, 576]
    const bf16* __restrict__ dO,     // [B_H, D_V] = [128, 512]
    bf16* __restrict__ q_out,        // [B_H, D_Q] = [128, 576] (output Q from SMEM)
    bf16* __restrict__ kv_out,       // [B_TOPK, D_K] = [64, 576] (output KV from SMEM)
    bf16* __restrict__ dO_out,       // [B_H, D_V] = [128, 512] (output dO from SMEM)
    float* __restrict__ P,            // [B_H, B_TOPK] = [128, 64] (P = Q @ K^T)
    float* __restrict__ dP            // [B_H, B_TOPK] = [128, 64] (dP = dO @ V^T)
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
    // Use cute namespace inside kernel to avoid conflicts with PyTorch's at::Layout
    using namespace cute;
    
    extern __shared__ char smem_raw[];
    SharedMemoryPlan& plan = *reinterpret_cast<SharedMemoryPlan*>(smem_raw);
    
    const int cta_idx = blockIdx.x % 2;  // 0 or 1
    const int tid = threadIdx.x;
    const int warp_idx = tid / 32;
    
    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 1: Kernel started\n", cta_idx);
    }
    
    // Construct SMEM Tensors
    // Q and K are split into NoPE and RoPE parts
    // Q NoPE: [B_H/2, D_V] = [64, 512], Q RoPE: [B_H/2, D_ROPE] = [64, 64]
    // K NoPE: [B_TOPK/2, D_V] = [32, 512], K RoPE: [B_TOPK/2, D_ROPE] = [32, 64]
    Tensor sQNoPE = make_tensor(make_smem_ptr(plan.u.q_kv.q_nope.data()), SmemLayoutQNoPE{});
    Tensor sQRoPE = make_tensor(make_smem_ptr(plan.u.q_kv.q_rope.data()), SmemLayoutQRoPE{});
    Tensor sKNoPE = make_tensor(make_smem_ptr(plan.u.q_kv.k_nope.data()), SmemLayoutKNoPE{});
    Tensor sKRoPE = make_tensor(make_smem_ptr(plan.u.q_kv.k_rope.data()), SmemLayoutKRoPE{});
    Tensor sdO = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO{});
    
    // Load Q from Global Memory to SMEM (split into NoPE and RoPE)
    // Q NoPE: cta0读取前[B_H/2, D_V] = [64, 512], cta1读取后[B_H/2, D_V] = [64, 512]
    // Q RoPE: cta0读取前[B_H/2, D_ROPE] = [64, 64], cta1读取后[B_H/2, D_ROPE] = [64, 64]
    constexpr int Q_NOPE_ELEMENTS_PER_CTA = (B_H / 2) * D_V;  // 64 * 512 = 32768
    constexpr int Q_NOPE_ELEMENTS_PER_THREAD = (Q_NOPE_ELEMENTS_PER_CTA + NUM_THREADS - 1) / NUM_THREADS;
    constexpr int Q_ROPE_ELEMENTS_PER_CTA = (B_H / 2) * D_ROPE;  // 64 * 64 = 4096
    constexpr int Q_ROPE_ELEMENTS_PER_THREAD = (Q_ROPE_ELEMENTS_PER_CTA + NUM_THREADS - 1) / NUM_THREADS;
    const int q_row_offset = cta_idx * (B_H / 2);
    
    // Load Q NoPE part
    for (int i = 0; i < Q_NOPE_ELEMENTS_PER_THREAD; ++i) {
        int linear_idx = tid * Q_NOPE_ELEMENTS_PER_THREAD + i;
        if (linear_idx < Q_NOPE_ELEMENTS_PER_CTA) {
            int row = linear_idx / D_V;
            int col = linear_idx % D_V;
            sQNoPE(row, col) = q[(q_row_offset + row) * D_Q + col];
        }
    }
    
    // Load Q RoPE part
    for (int i = 0; i < Q_ROPE_ELEMENTS_PER_THREAD; ++i) {
        int linear_idx = tid * Q_ROPE_ELEMENTS_PER_THREAD + i;
        if (linear_idx < Q_ROPE_ELEMENTS_PER_CTA) {
            int row = linear_idx / D_ROPE;
            int col = linear_idx % D_ROPE;
            sQRoPE(row, col) = q[(q_row_offset + row) * D_Q + D_V + col];
        }
    }
    
    __syncthreads();
    cluster_sync();
    
    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 2: Q loaded to SMEM (NoPE + RoPE)\n", cta_idx);
    }
    
    // Load KV from Global Memory to SMEM (split into NoPE and RoPE)
    // K NoPE: cta0读取前[B_TOPK/2, D_V] = [32, 512], cta1读取后[B_TOPK/2, D_V] = [32, 512]
    // K RoPE: cta0读取前[B_TOPK/2, D_ROPE] = [32, 64], cta1读取后[B_TOPK/2, D_ROPE] = [32, 64]
    constexpr int K_NOPE_ELEMENTS_PER_CTA = (B_TOPK / 2) * D_V;  // 32 * 512 = 16384
    constexpr int K_NOPE_ELEMENTS_PER_THREAD = (K_NOPE_ELEMENTS_PER_CTA + NUM_THREADS - 1) / NUM_THREADS;
    constexpr int K_ROPE_ELEMENTS_PER_CTA = (B_TOPK / 2) * D_ROPE;  // 32 * 64 = 2048
    constexpr int K_ROPE_ELEMENTS_PER_THREAD = (K_ROPE_ELEMENTS_PER_CTA + NUM_THREADS - 1) / NUM_THREADS;
    const int kv_row_offset = cta_idx * (B_TOPK / 2);
    
    // Load K NoPE part
    for (int i = 0; i < K_NOPE_ELEMENTS_PER_THREAD; ++i) {
        int linear_idx = tid * K_NOPE_ELEMENTS_PER_THREAD + i;
        if (linear_idx < K_NOPE_ELEMENTS_PER_CTA) {
            int row = linear_idx / D_V;
            int col = linear_idx % D_V;
            sKNoPE(row, col) = kv[(kv_row_offset + row) * D_K + col];
        }
    }
    
    // Load K RoPE part
    for (int i = 0; i < K_ROPE_ELEMENTS_PER_THREAD; ++i) {
        int linear_idx = tid * K_ROPE_ELEMENTS_PER_THREAD + i;
        if (linear_idx < K_ROPE_ELEMENTS_PER_CTA) {
            int row = linear_idx / D_ROPE;
            int col = linear_idx % D_ROPE;
            sKRoPE(row, col) = kv[(kv_row_offset + row) * D_K + D_V + col];
        }
    }
    
    __syncthreads();
    cluster_sync();
    
    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 3: KV loaded to SMEM\n", cta_idx);
    }
    
    // Load dO from Global Memory to SMEM
    // dO: cta0读取前[B_H/2, D_V] = [64, 512], cta1读取后[B_H/2, D_V] = [64, 512]
    constexpr int dO_ELEMENTS_PER_CTA = (B_H / 2) * D_V;  // 64 * 512 = 32768
    constexpr int dO_ELEMENTS_PER_THREAD = (dO_ELEMENTS_PER_CTA + NUM_THREADS - 1) / NUM_THREADS;
    const int dO_row_offset = cta_idx * (B_H / 2);
    
    for (int i = 0; i < dO_ELEMENTS_PER_THREAD; ++i) {
        int linear_idx = tid * dO_ELEMENTS_PER_THREAD + i;
        if (linear_idx < dO_ELEMENTS_PER_CTA) {
            int row = linear_idx / D_V;
            int col = linear_idx % D_V;
            sdO(row, col) = dO[(dO_row_offset + row) * D_V + col];
        }
    }
    
    __syncthreads();
    cluster_sync();
    
    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 4: dO loaded to SMEM\n", cta_idx);
    }
    
    // Write Q, KV, dO from SMEM to global memory
    // Write Q from SMEM to q_out (split into NoPE and RoPE)
    // Write Q NoPE part
    for (int i = 0; i < Q_NOPE_ELEMENTS_PER_THREAD; ++i) {
        int linear_idx = tid * Q_NOPE_ELEMENTS_PER_THREAD + i;
        if (linear_idx < Q_NOPE_ELEMENTS_PER_CTA) {
            int row = linear_idx / D_V;
            int col = linear_idx % D_V;
            q_out[(q_row_offset + row) * D_Q + col] = sQNoPE(row, col);
        }
    }
    
    // Write Q RoPE part
    for (int i = 0; i < Q_ROPE_ELEMENTS_PER_THREAD; ++i) {
        int linear_idx = tid * Q_ROPE_ELEMENTS_PER_THREAD + i;
        if (linear_idx < Q_ROPE_ELEMENTS_PER_CTA) {
            int row = linear_idx / D_ROPE;
            int col = linear_idx % D_ROPE;
            q_out[(q_row_offset + row) * D_Q + D_V + col] = sQRoPE(row, col);
        }
    }
    
    __syncthreads();
    
    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 5: Q written to global memory (NoPE + RoPE)\n", cta_idx);
    }
    
    // Write KV from SMEM to kv_out (split into NoPE and RoPE)
    // Write K NoPE part
    for (int i = 0; i < K_NOPE_ELEMENTS_PER_THREAD; ++i) {
        int linear_idx = tid * K_NOPE_ELEMENTS_PER_THREAD + i;
        if (linear_idx < K_NOPE_ELEMENTS_PER_CTA) {
            int row = linear_idx / D_V;
            int col = linear_idx % D_V;
            kv_out[(kv_row_offset + row) * D_K + col] = sKNoPE(row, col);
        }
    }
    
    // Write K RoPE part
    for (int i = 0; i < K_ROPE_ELEMENTS_PER_THREAD; ++i) {
        int linear_idx = tid * K_ROPE_ELEMENTS_PER_THREAD + i;
        if (linear_idx < K_ROPE_ELEMENTS_PER_CTA) {
            int row = linear_idx / D_ROPE;
            int col = linear_idx % D_ROPE;
            kv_out[(kv_row_offset + row) * D_K + D_V + col] = sKRoPE(row, col);
        }
    }
    
    __syncthreads();
    
    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 6: KV written to global memory\n", cta_idx);
    }
    
    // Write dO from SMEM to dO_out
    for (int i = 0; i < dO_ELEMENTS_PER_THREAD; ++i) {
        int linear_idx = tid * dO_ELEMENTS_PER_THREAD + i;
        if (linear_idx < dO_ELEMENTS_PER_CTA) {
            int row = linear_idx / D_V;
            int col = linear_idx % D_V;
            dO_out[(dO_row_offset + row) * D_V + col] = sdO(row, col);
        }
    }
    
    __syncthreads();
    cluster_sync();
    
    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 7: dO written to global memory\n", cta_idx);
    }
    
    // Allocate TMEM (warp 0 in each CTA, all threads in warp participate)
    if (warp_idx == 0) {
        TMEM::Allocator2Sm().allocate(512, plan.tmem_start_addr.data());
        TMEM::Allocator2Sm().release_allocation_lock();
    }
    __syncthreads();  // Wait for TMEM allocation
    
    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 8: TMEM allocated\n", cta_idx);
    }
    
    // Cluster sync before accessing peer SMEM - all CTAs must participate
    __syncthreads();
    cluster_sync();

    // Create TiledMMA and fragments for P computation
    TiledMMA_P tiled_mma_P{};
    Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H/2>, Int<B_TOPK>>{});
    tP.data().get() = tmem_cols::P;
    
    // Extract K from memory: K is split into NoPE and RoPE parts
    // For P = Q @ K^T, we compute in two parts: Q_NoPE @ K_NoPE^T and Q_RoPE @ K_RoPE^T
    // K shape: [B_TOPK/2, D_K] = [32, 576] per CTA, split into [32, 512] (NoPE) and [32, 64] (RoPE)
    // For 2CTA mode, we need to combine K from both CTAs: [B_TOPK, D_K] = [64, 576]
    // TiledMMA_P expects [B_H, B_TOPK] = [128, 64], and handles 2CTA coordination automatically
    
    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 10: About to compute P = Q @ K^T\n", cta_idx);
    }
    
    // Compute P = Q @ K^T using TiledMMA_P
    // Matrix A: plan.u.q_kv.q_nope/q_rope (Q) [B_H/2, D_Q] = [64, 576] (split into NoPE and RoPE)
    // Matrix B: plan.u.q_kv.k_nope/k_rope (K) [B_TOPK/2, D_K] = [32, 576] (split into NoPE and RoPE)
    // Matrix C: tmem_cols.P [B_H/2, B_TOPK] = [64, 64]
    // Note: TiledMMA_P uses 2x1SM_SS_NOELECT, which automatically handles 2CTA coordination
    
    // For 2CTA mode:
    // - CTA0 processes Q[0:64, :] and K[0:32, :]
    // - CTA1 processes Q[64:128, :] and K[32:64, :]
    // - TiledMMA_P automatically combines results to produce P[128, 64]
    
    // K is stored in k_nope and k_rope, shape [B_TOPK/2, D_K] = [32, 576] per CTA
    // For 2CTA mode, TiledMMA_P handles 2CTA coordination automatically
    // We compute P in two parts: NoPE first (clear accumulator), then RoPE (accumulate)
    
    // Both CTAs participate in the 2CTA computation
    // TiledMMA_P automatically coordinates between CTAs
    // Compute P = Q @ K^T in two parts: NoPE first (clear_accum = true), then RoPE (clear_accum = false)
    if (cta_idx == 0 && warp_idx == 0 && elect_one_sync()) {
        // First compute NoPE part: P += Q_NoPE @ K_NoPE^T (clear accumulator)
        ku::utcmma_ss(tiled_mma_P, sQNoPE, sKNoPE, tP, true);  // clear_accum = true
    }
    ku::tcgen05_after_thread_sync();
    
    __syncthreads();
    cluster_sync();
    
    if (cta_idx == 0 && warp_idx == 0 && elect_one_sync()) {
        // Then compute RoPE part: P += Q_RoPE @ K_RoPE^T (accumulate, don't clear)
        ku::utcmma_ss(tiled_mma_P, sQRoPE, sKRoPE, tP, false);  // clear_accum = false
    }
    ku::tcgen05_after_thread_sync();
    
    __syncthreads();
    cluster_sync();
    
    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 11: P computation completed (NoPE + RoPE)\n", cta_idx);
    }
    
    // Create TiledMMA and fragments for dP computation
    TiledMMA_dP tiled_mma_dP{};
    Tensor tdP = partition_fragment_C(tiled_mma_dP, Shape<Int<B_H/2>, Int<B_TOPK>>{});
    tdP.data().get() = tmem_cols::dP;
    
    // Extract V from memory: V uses the same layout as K_NoPE
    // For dP = dO @ V^T, we need V
    // V shape: [B_TOPK/2, D_V] = [32, 512]
    // V is stored in k_nope, which has the same layout as K_NoPE
    // Create a view of K_NoPE for V: [B_TOPK/2, D_V] = [32, 512]
    Tensor sV = make_tensor(make_smem_ptr(plan.u.q_kv.k_nope.data()), SmemLayoutV{});

    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 12: About to compute dP = dO @ V^T\n", cta_idx);
    }
    
    // Compute dP = dO @ V^T using TiledMMA_dP
    // Matrix A: plan.dO (dO) [B_H/2, D_V] = [64, 512]
    // Matrix B: plan.u.q_kv.k_nope (V) [B_TOPK/2, D_V] = [32, 512] (V uses K_NoPE layout)
    // Matrix C: tmem_cols.dP [B_H/2, B_TOPK] = [64, 64]
    // Note: TiledMMA_dP uses 2x1SM_SS_NOELECT, which automatically handles 2CTA coordination
    
    // V is stored in k_nope, shape [B_TOPK/2, D_V] = [32, 512] per CTA
    // For 2CTA mode, TiledMMA_dP automatically coordinates between CTAs
    
    // Both CTAs participate in the 2CTA computation
    // TiledMMA_dP automatically coordinates between CTAs
    if (cta_idx == 0 && warp_idx == 0 && elect_one_sync()) {
        ku::utcmma_ss(tiled_mma_dP, sdO, sV, tdP, true);  // clear_accum = true
    }
    ku::tcgen05_after_thread_sync();
    
    __syncthreads();
    cluster_sync();
    
    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 13: dP computation completed\n", cta_idx);
    }
    
    // Read P and dP from TMEM and write to global memory
    // P shape: [B_H, B_TOPK] = [128, 64]
    constexpr int P_ROWS = B_H / 2;  // 64
    constexpr int P_COLS = B_TOPK;  // 64
    constexpr int FLOAT2_PER_ROW = P_COLS / 2 / 2;  // 16 float2 per row

    int cta_row_offset = cta_idx * P_ROWS;
    int logical_row = tid % P_ROWS + cta_row_offset;  // Logical row index: 0-63 + 64 = 0-127
    int col_offset = tid / P_ROWS;
    
    // Construct TMEM address: base + (lane << 16) | column
    uint32_t tmem_base = plan.tmem_start_addr.data()[0];
    uint32_t tmem_lane = logical_row;  // TMEM lane = logical row
    uint32_t tmem_col = tmem_cols::P;  // Starting column offset
    
    // Full TMEM address: base + (lane << 16) | column
    uint32_t tmem_addr = tmem_base + (tmem_lane << 16) + tmem_col;
    
    // Load one logical row of P from TMEM (128 float2 = 256 floats)
    // This reads from TMEM lane = logical_row, columns = tmem_cols::p to tmem_cols::p + 127
    float2 p_row[FLOAT2_PER_ROW];
    ku::tmem_ld_32dp32bNx<P_COLS/2>(tmem_addr, p_row);
    cutlass::arch::fence_view_async_tmem_load();
    ku::tcgen05_before_thread_sync();
    
    // Write to global memory
    for (int col = 0; col < FLOAT2_PER_ROW; ++col) {
        P[logical_row * P_COLS + P_COLS / 2 * col_offset + col * 2] = p_row[col].x;
        P[logical_row * P_COLS + P_COLS / 2 * col_offset + col * 2 + 1] = p_row[col].y;
    }
    
    __syncthreads();
    
    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 14: P read from TMEM and written to global memory\n", cta_idx);
    }

    // Read dP from TMEM and write to global memory
    // dP shape: [B_H, B_TOPK] = [128, 64]
    {
        constexpr int dP_ROWS = B_H / 2;  // 64
        constexpr int dP_COLS = B_TOPK;  // 64
        constexpr int dP_FLOAT2_PER_ROW = dP_COLS / 2 / 2;  // 16 float2 per row

        int dP_cta_row_offset = cta_idx * dP_ROWS;
        int dP_logical_row = tid % dP_ROWS + dP_cta_row_offset;  // Logical row index: 0-63 + 64 = 0-127
        int dP_col_offset = tid / dP_ROWS;
        
        // Construct TMEM address: base + (lane << 16) | column
        uint32_t dP_tmem_base = plan.tmem_start_addr.data()[0];
        uint32_t dP_tmem_lane = dP_logical_row;  // TMEM lane = logical row
        uint32_t dP_tmem_col = tmem_cols::dP;  // Starting column offset
        
        // Full TMEM address: base + (lane << 16) | column
        uint32_t dP_tmem_addr = dP_tmem_base + (dP_tmem_lane << 16) + dP_tmem_col;
        
        // Load one logical row of dP from TMEM (128 float2 = 256 floats)
        // This reads from TMEM lane = logical_row, columns = tmem_cols::dP to tmem_cols::dP + 127
        float2 d_row[dP_FLOAT2_PER_ROW];
        ku::tmem_ld_32dp32bNx<dP_COLS/2>(dP_tmem_addr, d_row);
        cutlass::arch::fence_view_async_tmem_load();
        ku::tcgen05_before_thread_sync();
        
        // Write to global memory
        for (int col = 0; col < dP_FLOAT2_PER_ROW; ++col) {
            dP[dP_logical_row * dP_COLS + dP_COLS / 2 * dP_col_offset + col * 2] = d_row[col].x;
            dP[dP_logical_row * dP_COLS + dP_COLS / 2 * dP_col_offset + col * 2 + 1] = d_row[col].y;
        }
    }
    
    __syncthreads();
    cluster_sync();
    
    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 15: dP read from TMEM and written to global memory\n", cta_idx);
    }
    
    // Free TMEM
    if (warp_idx == 0 && elect_one_sync()) {
        TMEM::Allocator2Sm().free(plan.tmem_start_addr.data()[0], 512);
    }
    
    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 16: TMEM freed, kernel completed\n", cta_idx);
    }

#endif
}

// C++ wrapper
void launch_test_mla_bwd(
    const bf16* q,
    const bf16* kv,
    const bf16* dO,
    bf16* q_out,
    bf16* kv_out,
    bf16* dO_out,
    float* P,
    float* dP,
    cudaStream_t stream
) {
    dim3 grid(2, 1, 1);  // 2 CTAs
    dim3 block(NUM_THREADS, 1, 1);
    
    // Set dynamic shared memory size
    cudaError_t attr_err = cudaFuncSetAttribute(
        test_mla_bwd_kernel, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        SMEM_SIZE
    );
    if (attr_err != cudaSuccess) {
        fprintf(stderr, "cudaFuncSetAttribute failed with error: %s\n", cudaGetErrorString(attr_err));
        return;
    }
    
    // Cluster configuration
    cudaLaunchConfig_t config;
    memset(&config, 0, sizeof(config));
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = SMEM_SIZE;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    printf("finish 0: Launching kernel...\n");
    cudaError_t err = cudaLaunchKernelEx(&config, test_mla_bwd_kernel, q, kv, dO, q_out, kv_out, dO_out, P, dP);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaLaunchKernelEx failed with error: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("finish 17: Kernel launch completed\n");
}

// Python binding
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> mla_bwd_forward(
    torch::Tensor q, torch::Tensor kv, torch::Tensor dO) {
    // Check inputs
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(kv.is_cuda(), "kv must be a CUDA tensor");
    TORCH_CHECK(dO.is_cuda(), "dO must be a CUDA tensor");
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(kv.dtype() == torch::kBFloat16, "kv must be bfloat16");
    TORCH_CHECK(dO.dtype() == torch::kBFloat16, "dO must be bfloat16");
    TORCH_CHECK(q.dim() == 2 && q.size(0) == B_H && q.size(1) == D_Q, 
                "q shape must be [128, 576]");
    TORCH_CHECK(kv.dim() == 2 && kv.size(0) == B_TOPK && kv.size(1) == D_K,
                "kv shape must be [64, 576]");
    TORCH_CHECK(dO.dim() == 2 && dO.size(0) == B_H && dO.size(1) == D_V,
                "dO shape must be [128, 512]");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(kv.is_contiguous(), "kv must be contiguous");
    TORCH_CHECK(dO.is_contiguous(), "dO must be contiguous");
    
    // Create output tensors
    auto options_f32 = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(q.device());
    auto options_bf16 = torch::TensorOptions()
        .dtype(torch::kBFloat16)
        .device(q.device());
    
    torch::Tensor q_out = torch::empty({B_H, D_Q}, options_bf16);
    torch::Tensor kv_out = torch::empty({B_TOPK, D_K}, options_bf16);
    torch::Tensor dO_out = torch::empty({B_H, D_V}, options_bf16);
    torch::Tensor P = torch::empty({B_H, B_TOPK}, options_f32);
    torch::Tensor dP = torch::empty({B_H, B_TOPK}, options_f32);
    
    // Get data pointers
    const bf16* q_ptr = reinterpret_cast<const bf16*>(q.data_ptr<at::BFloat16>());
    const bf16* kv_ptr = reinterpret_cast<const bf16*>(kv.data_ptr<at::BFloat16>());
    const bf16* dO_ptr = reinterpret_cast<const bf16*>(dO.data_ptr<at::BFloat16>());
    bf16* q_out_ptr = reinterpret_cast<bf16*>(q_out.data_ptr<at::BFloat16>());
    bf16* kv_out_ptr = reinterpret_cast<bf16*>(kv_out.data_ptr<at::BFloat16>());
    bf16* dO_out_ptr = reinterpret_cast<bf16*>(dO_out.data_ptr<at::BFloat16>());
    float* P_ptr = P.data_ptr<float>();
    float* dP_ptr = dP.data_ptr<float>();
    
    // Get CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Call kernel
    printf("finish -1: Starting kernel launch from Python binding\n");
    launch_test_mla_bwd(q_ptr, kv_ptr, dO_ptr, q_out_ptr, kv_out_ptr, dO_out_ptr, P_ptr, dP_ptr, stream);
    
    // Synchronize and wait for kernel completion
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
    printf("finish 18: Kernel execution completed and synchronized\n");
    
    return std::make_tuple(q_out, kv_out, dO_out, P, dP);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mla_bwd", &mla_bwd_forward, 
          "Test mla_bwd kernel (CUDA). Returns (q_out, kv_out, dO_out, P, dP)");
}

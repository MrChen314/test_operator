#include "test_2sm_mma.cuh"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <kerutils/kerutils.cuh>
#include <cutlass/arch/arch.h>

// Kernel 实现：测试 utcmma_ss 矩阵乘
__global__ void test_utcmma_ss_kernel(
    const bf16* __restrict__ Q,
    const bf16* __restrict__ K,
    float* __restrict__ P_out
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
    extern __shared__ char smem_raw[];
    SharedMemory& smem = *reinterpret_cast<SharedMemory*>(smem_raw);
    
    const int cta_idx = blockIdx.x % 2;  // 0 or 1
    const int tid = threadIdx.x;
    const int warp_idx = tid / 32;
    const int lane_idx = tid % 32;
    
    // 构造 SMEM Tensor
    auto sQ = cute::make_tensor(cute::make_smem_ptr(smem.q), SmemLayoutQ{});
    auto sK = cute::make_tensor(cute::make_smem_ptr(smem.k), SmemLayoutK{});
    if (blockIdx.x == 0 && cta_idx == 0 && tid == 0) {
        printf("==================finish 1==================\n");
    }
    
    // 从 Global Memory 加载 Q, K 到 SMEM
    // 每个 CTA 加载各自的一半：CTA0 加载前64行，CTA1 加载后64行
    const int q_row_offset = cta_idx * (M/2);
    const int k_row_offset = cta_idx * (N/2);
    
    // 加载 Q: [64, 256]
    constexpr int Q_ELEMENTS = (M/2) * K_DIM;
    constexpr int Q_ELEMENTS_PER_THREAD = (Q_ELEMENTS + NUM_THREADS - 1) / NUM_THREADS;
    for (int i = 0; i < Q_ELEMENTS_PER_THREAD; ++i) {
        int linear_idx = tid * Q_ELEMENTS_PER_THREAD + i;
        if (linear_idx < Q_ELEMENTS) {
            int row = linear_idx / K_DIM;
            int col = linear_idx % K_DIM;
            int global_row = q_row_offset + row;
            sQ(row, col) = Q[global_row * K_DIM + col];
        }
    }
    if (blockIdx.x == 0 && cta_idx == 0 && tid == 0) {
        printf("==================finish 2==================\n");
    }
    
    // 加载 K: [64, 256]
    constexpr int K_ELEMENTS = (N/2) * K_DIM;
    constexpr int K_ELEMENTS_PER_THREAD = (K_ELEMENTS + NUM_THREADS - 1) / NUM_THREADS;
    for (int i = 0; i < K_ELEMENTS_PER_THREAD; ++i) {
        int linear_idx = tid * K_ELEMENTS_PER_THREAD + i;
        if (linear_idx < K_ELEMENTS) {
            int row = linear_idx / K_DIM;
            int col = linear_idx % K_DIM;
            int global_row = k_row_offset + row;
            sK(row, col) = K[global_row * K_DIM + col];
        }
    }
    if (blockIdx.x == 0 && cta_idx == 0 && tid == 0) {
        printf("==================finish 3==================\n");
    }
    
    __syncthreads();
    cute::cluster_sync();
    
    // Allocate TMEM (warp 0 in each CTA, all threads in warp participate)
    // According to phase1.cuh: TMEM allocation is in warp_idx == 0, but NOT in elect_one_sync()
    // The entire warp 0 participates in the allocation
    if (warp_idx == 0) {
        cute::TMEM::Allocator2Sm().allocate(512, &smem.tmem_start_addr);
        cute::TMEM::Allocator2Sm().release_allocation_lock();
    }
    __syncthreads();  // Wait for TMEM allocation
    if (blockIdx.x == 0 && cta_idx == 0 && tid == 0) {
        printf("==================finish 4==================\n");
    }
    
    // Create TiledMMA and fragments
    TiledMMA_P_sQ tiled_mma_P_sQ{};
    
    // Create tP fragment for output: [M/2, N] = [64, 128]
    auto tP = cute::partition_fragment_C(tiled_mma_P_sQ, cute::Shape<cute::Int<M/2>, cute::Int<N>>{});
    // Set TMEM address: base + column offset
    // Note: tP.data().get() expects column offset relative to base, but we need to set the full address
    // In phase1.cuh, it's set as: tP.data().get() = tmem_cols::p;
    // This works because the fragment automatically adds the base address
    tP.data().get() = tmem_cols::p;
    
    // Create sQl and sKl tensors (same as sQ and sK for this simple test)
    auto sQl = sQ;
    auto sKl = sK;
    
    // Execute matrix multiplication: P = Q @ K^T
    // For 2x1SM MMA, we need only one thread to launch (like phase1.cuh: cta_idx == 0 && warp_idx == 12 && elect_one_sync())
    // Since we only have 4 warps (128 threads), we use warp 0 instead of warp 12
    if (cta_idx == 0 && warp_idx == 0 && cute::elect_one_sync()) {
        ku::utcmma_ss(tiled_mma_P_sQ, sQl, sKl, tP, true);
    }
    if (blockIdx.x == 0 && cta_idx == 0 && tid == 0) {
        printf("==================finish 5==================\n");
    }
    
    // Synchronize after MMA
    // For 2x1SM MMA, we need to wait for both CTAs to complete
    ku::tcgen05_after_thread_sync();
    if (blockIdx.x == 0 && cta_idx == 0 && tid == 0) {
        printf("==================finish 6==================\n");
    }
    __syncthreads();
    if (blockIdx.x == 0 && cta_idx == 0 && tid == 0) {
        printf("==================finish 7==================\n");
    }
    cute::cluster_sync();
    if (blockIdx.x == 0 && cta_idx == 0 && tid == 0) {
        printf("==================finish 8==================\n");
    }
    
    // Read results from TMEM and write to global memory
    // P matrix logical shape: [M/2, N] = [64, 128] (64 rows, 128 columns)
    // P matrix physical shape in TMEM: 128 lane (rows) × 64 column (columns)
    // Mapping: logical row i → TMEM lane i, logical column j → TMEM column (tmem_cols::p + j/2)
    // Each logical row (128 floats) is stored as 64 float2 values in 64 TMEM columns
    
    constexpr int P_ROWS_PER_CTA = M / 2;  // 64 logical rows
    constexpr int P_COLS = N;  // 128 logical columns
    
    // Each thread reads one logical row of P
    // Thread tid reads logical row tid (if tid < 64)
    // TMEM address format: (lane << 16) | column
    // Where lane = logical row index (0-63), column = tmem_cols::p (starting column offset)
    if (tid < P_ROWS_PER_CTA) {
        int logical_row = tid;  // Logical row index: 0-63
        
        // Construct TMEM address: base + (lane << 16) | column
        // lane = logical_row (TMEM lane index, 0-63)
        // column = tmem_cols::p (starting column offset, 256)
        uint32_t tmem_base = smem.tmem_start_addr;
        uint32_t tmem_lane = logical_row;  // TMEM lane = logical row
        uint32_t tmem_col = tmem_cols::p;  // Starting column offset
        
        // Full TMEM address: base + (lane << 16) | column
        // This ensures we read from the correct TMEM lane (row)
        uint32_t tmem_addr = tmem_base + (tmem_lane << 16) + tmem_col;
        
        // Load one logical row of P from TMEM (64 float2 = 128 floats)
        // This reads from TMEM lane = logical_row, columns = tmem_cols::p to tmem_cols::p + 63
        float2 p_row[32];
        ku::tmem_ld_32dp32bNx<64>(tmem_addr, p_row);
        cutlass::arch::fence_view_async_tmem_load();
        ku::tcgen05_before_thread_sync();
        
        // Write to global memory
        int global_row = q_row_offset + logical_row;
        for (int col = 0; col < 32; ++col) {
            P_out[global_row * P_COLS + col * 2] = p_row[col].x;
            P_out[global_row * P_COLS + col * 2 + 1] = p_row[col].y;
        }
    }
    else {
        int logical_row = tid % 64;  // Logical row index: 0-63
        
        // Construct TMEM address: base + (lane << 16) | column
        // lane = logical_row (TMEM lane index, 0-63)
        // column = tmem_cols::p (starting column offset, 256)
        uint32_t tmem_base = smem.tmem_start_addr;
        uint32_t tmem_lane = logical_row;  // TMEM lane = logical row
        uint32_t tmem_col = tmem_cols::p;  // Starting column offset
        
        // Full TMEM address: base + (lane << 16) | column
        // This ensures we read from the correct TMEM lane (row)
        uint32_t tmem_addr = tmem_base + (tmem_lane << 16) + tmem_col;
        
        // Load one logical row of P from TMEM (64 float2 = 128 floats)
        // This reads from TMEM lane = logical_row, columns = tmem_cols::p to tmem_cols::p + 63
        float2 p_row[32];
        ku::tmem_ld_32dp32bNx<64>(tmem_addr, p_row);
        cutlass::arch::fence_view_async_tmem_load();
        ku::tcgen05_before_thread_sync();
        
        // Write to global memory
        int global_row = q_row_offset + logical_row;
        for (int col = 0; col < 32; ++col) {
            P_out[global_row * P_COLS + 64 + col * 2] = p_row[col].x;
            P_out[global_row * P_COLS + 64 + col * 2 + 1] = p_row[col].y;
        }
    }
    if (blockIdx.x == 0 && cta_idx == 0 && tid == 0) {
        printf("==================finish 9==================\n");
    }
    
    __syncthreads();
    cute::cluster_sync();
    
    // Free TMEM
    if (warp_idx == 0 && cute::elect_one_sync()) {
        cute::TMEM::Allocator2Sm().free(smem.tmem_start_addr, 512);
    }

#endif
}

// C++ wrapper
void launch_test_utcmma_ss(
    const bf16* Q,
    const bf16* K,
    float* P_out,
    cudaStream_t stream
) {
    dim3 grid(2, 1, 1);  // 2 CTAs
    dim3 block(NUM_THREADS, 1, 1);
    
    // 设置动态共享内存大小
    cudaError_t attr_err = cudaFuncSetAttribute(
        test_utcmma_ss_kernel, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        SMEM_SIZE
    );
    if (attr_err != cudaSuccess) {
        fprintf(stderr, "cudaFuncSetAttribute failed with error: %s\n", cudaGetErrorString(attr_err));
        return;
    }
    
    // Cluster 配置
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
    
    cudaError_t err = cudaLaunchKernelEx(&config, test_utcmma_ss_kernel, Q, K, P_out);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaLaunchKernelEx failed with error: %s\n", cudaGetErrorString(err));
        return;
    }
}

// Python binding: 返回 P 矩阵 (Q @ K^T)
torch::Tensor utcmma_ss_forward_with_debug(
    torch::Tensor Q, torch::Tensor K) {
    // 检查输入
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bfloat16");
    TORCH_CHECK(K.dtype() == torch::kBFloat16, "K must be bfloat16");
    TORCH_CHECK(Q.dim() == 2 && Q.size(0) == M && Q.size(1) == K_DIM, 
                "Q shape must be [128, 256]");
    TORCH_CHECK(K.dim() == 2 && K.size(0) == N && K.size(1) == K_DIM,
                "K shape must be [128, 256]");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    
    // 创建输出 tensor: P = Q @ K^T, shape [M, N] = [128, 128]
    auto options_f32 = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(Q.device());
    torch::Tensor P_out = torch::empty({M, N}, options_f32);
    
    // 获取数据指针
    const bf16* Q_ptr = reinterpret_cast<const bf16*>(Q.data_ptr<at::BFloat16>());
    const bf16* K_ptr = reinterpret_cast<const bf16*>(K.data_ptr<at::BFloat16>());
    float* P_out_ptr = P_out.data_ptr<float>();
    
    // 获取 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // 调用 kernel
    launch_test_utcmma_ss(Q_ptr, K_ptr, P_out_ptr, stream);
    
    // 同步等待 kernel 完成
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
    
    return P_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("utcmma_ss_debug", &utcmma_ss_forward_with_debug, "Test utcmma_ss matrix multiplication (CUDA)");
}
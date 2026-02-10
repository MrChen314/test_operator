#include "mla_bwd.cuh"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <kerutils/kerutils.cuh>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/barrier.h>
#include <math_constants.h>

using namespace test_operator::mla_bwd;
using cutlass::arch::fence_barrier_init;
using cutlass::arch::fence_view_async_shared;

// Helper function: float2 subtraction using float2_add and float2_neg
CUTE_DEVICE
float2 float2_sub(const float2 &a, const float2 &b) {
    return ku::float2_add(a, ku::float2_neg(b));
}

// bf16x8 structure for vectorized operations
struct bf16x8 {
    __nv_bfloat162 a01;
    __nv_bfloat162 a23;
    __nv_bfloat162 a45;
    __nv_bfloat162 a67;
};

struct int32x8_t {
    int a0, a1, a2, a3, a4, a5, a6, a7;
};

CUTE_DEVICE int32x8_t ldg_256_indices(void* src_ptr) {
    int32x8_t val;
    asm volatile("ld.global.nc.L1::evict_normal.L2::evict_normal.L2::256B.v8.s32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=r"(val.a0), "=r"(val.a1), "=r"(val.a2), "=r"(val.a3),
          "=r"(val.a4), "=r"(val.a5), "=r"(val.a6), "=r"(val.a7)
        : "l"(src_ptr)
    );
    return val;
}

// Preprocess delta kernel: compute delta = sum(O * dO, dim=-1)
__global__ void preprocess_delta_kernel(
    const bf16* __restrict__ O,      // [B_H, D_V] = [128, 512]
    const bf16* __restrict__ dO,     // [B_H, D_V] = [128, 512]
    float* __restrict__ delta        // [B_H] = [128]
) {
    constexpr int D_V = 512;
    constexpr int B_H = 128;
    
    const int head_idx = threadIdx.x;  // Each thread processes one head
    
    if (head_idx >= B_H) return;
    
    // Compute delta: delta[i] = sum_j(O[i,j] * dO[i,j])
    float delta_val = 0.0f;
    
    // Accumulate O * dO using vectorized loads
    CUTE_UNROLL
    for (int col = 0; col < D_V; col += 8) {
        // Vectorized load of O (8 bf16 values = 128 bits)
        uint4 o_raw = __ldg((const uint4*)(O + head_idx * D_V + col));
        bf16x8 o_vec;
        *(uint4*)&o_vec = o_raw;
        
        // Vectorized load of dO (8 bf16 values = 128 bits)
        uint4 do_raw = __ldg((const uint4*)(dO + head_idx * D_V + col));
        bf16x8 do_vec;
        *(uint4*)&do_vec = do_raw;
        
        // Accumulate dot product
        delta_val += __bfloat162float(o_vec.a01.x) * __bfloat162float(do_vec.a01.x);
        delta_val += __bfloat162float(o_vec.a01.y) * __bfloat162float(do_vec.a01.y);
        delta_val += __bfloat162float(o_vec.a23.x) * __bfloat162float(do_vec.a23.x);
        delta_val += __bfloat162float(o_vec.a23.y) * __bfloat162float(do_vec.a23.y);
        delta_val += __bfloat162float(o_vec.a45.x) * __bfloat162float(do_vec.a45.x);
        delta_val += __bfloat162float(o_vec.a45.y) * __bfloat162float(do_vec.a45.y);
        delta_val += __bfloat162float(o_vec.a67.x) * __bfloat162float(do_vec.a67.x);
        delta_val += __bfloat162float(o_vec.a67.y) * __bfloat162float(do_vec.a67.y);
    }
    
    // Write delta to global memory
    delta[head_idx] = delta_val;
}

// Kernel implementation: test mla_bwd with Q, KV, dO inputs
__global__ void test_mla_bwd_kernel(
    const bf16* __restrict__ q,      // [B_H, D_Q] = [128, 576]
    const bf16* __restrict__ kv,     // [B_TOPK, D_K] = [64, 576]
    const bf16* __restrict__ dO,     // [B_H, D_V] = [128, 512]
    const float* __restrict__ lse,   // [B_H] = [128] (log-sum-exp for softmax)
    const bf16* __restrict__ O,     // [B_H, D_V] = [128, 512] (forward output O)
    const int32_t* __restrict__ gIndices,  // [B_TOPK] = [64] (indices for sparse attention)
    int s_kv,                        // KV sequence length
    int topk_length,                 // TopK length
    bf16* __restrict__ q_out,        // [B_H, D_Q] = [128, 576] (output Q from SMEM)
    bf16* __restrict__ kv_out,       // [B_TOPK, D_K] = [64, 576] (output KV from SMEM)
    bf16* __restrict__ dO_out,       // [B_H, D_V] = [128, 512] (output dO from SMEM)
    float* __restrict__ P,            // [B_H, B_TOPK] = [128, 64] (P = Q @ K^T)
    float* __restrict__ dP,           // [B_H, B_TOPK] = [128, 64] (dP = dO @ V^T)
    bf16* __restrict__ s,            // [B_H, B_TOPK] = [128, 64] (softmax values)
    bf16* __restrict__ ds,           // [B_H, B_TOPK] = [128, 64] (dS gradients)
    const float* __restrict__ delta,  // [B_H] = [128] (delta = sum(O * dO))
    float* __restrict__ dKV,          // [B_TOPK, D_K] = [64, 576] (dKV gradient, float32 for atomic add)
    bf16* __restrict__ sdO_t_out,    // [D_V, B_H] = [512, 128] (dO transposed output)
    float* __restrict__ dQ           // [B_H, D_Q] = [128, 576] (dQ gradient, float32)
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
    // Use cute namespace inside kernel to avoid conflicts with PyTorch's at::Layout
    using namespace cute;
    
    extern __shared__ char smem_raw[];
    SharedMemoryPlan& plan = *reinterpret_cast<SharedMemoryPlan*>(smem_raw);
    
    const int cta_idx = blockIdx.x % 2;  // 0 or 1
    const int tid = threadIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();  // Global warp index
    const int lane_idx = threadIdx.x % 32;
    
    // Determine warpgroup: 4 warpgroups, 128 threads each
    // WG0: threads 0-127 (warpgroup_idx = 0, warp_idx = 0-3)
    // WG1: threads 128-255 (warpgroup_idx = 1, warp_idx = 4-7)
    // WG2: threads 256-383 (warpgroup_idx = 2, warp_idx = 8-11)
    // WG3: threads 384-511 (warpgroup_idx = 3, warp_idx = 12-15)
    const int warpgroup_idx = __shfl_sync(0xffffffff, tid / 128, 0);
    const int idx_in_warpgroup = tid % 128;
    const bool is_wg0 = (warpgroup_idx == 0);
    const bool is_wg1 = (warpgroup_idx == 1);
    const bool is_wg2 = (warpgroup_idx == 2);
    const bool is_wg3 = (warpgroup_idx == 3);
    
    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 1: Kernel started\n", cta_idx);
    }

    // Allocate TMEM (warp 0 in each CTA, all threads in warp participate)
    if (warp_idx == 0) {
        TMEM::Allocator2Sm().allocate(512, plan.tmem_start_addr.data());
        TMEM::Allocator2Sm().release_allocation_lock();
    }
    __syncthreads();  // Wait for TMEM allocation
    
    if (tid == 0 && cta_idx == 0 && blockIdx.x == 0) {
        printf("[CTA %d] finish 2: TMEM allocated\n", cta_idx);
    }
    
    // Initialize barriers (warp 0 in CTA0)
    if (warp_idx == 0 && elect_one_sync()) {
        plan.bar_qkv_loaded.init(128*2);     // WG1通知WG3 q/k/dO已加载到SMEM (2CTA sync)
        plan.bar_p_ready.init(1);        // WG3通知WG0 p已准备好 (2CTA sync)
        plan.bar_dp_ready.init(1);       // WG3通知WG0 dp已准备好 (2CTA sync)
        plan.bar_s_ready.init(128);        // WG0通知WG3 s已准备好
        plan.bar_ds_ready.init(128);       // WG0通知WG3 ds已准备好
        plan.bar_k_valid_free.init(128);
        plan.bar_k_valid_ready.init(8);
        // WG3-WG2 barriers for dKV computation
        plan.bar_dkv_part0_ready.init(1);
        plan.bar_dkv_part1_ready.init(1);
        plan.bar_dkv_part2_ready.init(1);
        plan.bar_dkv_part0_done.init(128);
        plan.bar_dkv_part1_done.init(128);
        plan.bar_dkv_part2_done.init(128);
        // WG1-WG3 barriers for kv_peer cp_async
        plan.bar_kv_peer_cp_async.init(1);      // cp_async transaction barrier
        plan.bar_kv_peer_ready.init(1);         // WG1 notifies WG3 kv_peer is ready
        // WG3-WG0 barrier for dQ computation
        plan.bar_dq_ready.init(1);              // WG3 notifies WG0 dQ is ready
        fence_barrier_init();
    }
    
    // Cluster sync before accessing peer SMEM - all CTAs must participate
    __syncthreads();
    cluster_sync();
    
    // Construct SMEM Tensors
    // Q and K are split into NoPE and RoPE parts
    // Q NoPE: [B_H/2, D_V] = [64, 512], Q RoPE: [B_H/2, D_ROPE] = [64, 64]
    // K NoPE: [B_TOPK/2, D_V] = [32, 512], K RoPE: [B_TOPK/2, D_ROPE] = [32, 64]
    Tensor sQNoPE = make_tensor(make_smem_ptr(plan.u.q_kv.q_nope.data()), SmemLayoutQNoPE{});
    Tensor sQRoPE = make_tensor(make_smem_ptr(plan.u.q_kv.q_rope.data()), SmemLayoutQRoPE{});
    Tensor sKNoPE = make_tensor(make_smem_ptr(plan.u.q_kv.k_nope.data()), SmemLayoutKNoPE{});
    Tensor sKRoPE = make_tensor(make_smem_ptr(plan.u.q_kv.k_rope.data()), SmemLayoutKRoPE{});
    Tensor sdO = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO{});
    // kv_peer: [B_TOPK/2, D_K] = [32, 576] for cross-CTA K loading
    Tensor sKV_peer = make_tensor(make_smem_ptr(plan.u.q_kv.kv_peer.data()), SmemLayoutKV{});

    // ========================================
    // Warpgroup 0: Softmax and dS Computation (WG0)
    // Responsibility: Compute softmax(P), load delta, compute ds
    // ========================================
    if (is_wg0) {
        cutlass::arch::warpgroup_reg_alloc<144>();
        
        // Load LSE from global memory (needed for softmax computation)
        float row_lse = 0.0f;
        const int global_row_idx = cta_idx * (B_H/2) + idx_in_warpgroup % (B_H/2);
        row_lse = __ldg(lse + global_row_idx);
        
        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG0 CTA%d] finish 1: LSE loaded from global memory\n", cta_idx);
        }
        
        const float sm_scale = 1.0f / sqrtf(576.0f);  // softmax scale
        const float scale = sm_scale * 1.44269504f;  // sm_scale * log2(e) for exp2
        Tensor sS = make_tensor(make_smem_ptr(plan.s_ds.s.data()), SmemLayoutS{});
        Tensor sDS = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutS{});
        
        // Step 1: Wait for WG3 to compute P
        plan.bar_p_ready.wait(0);
        ku::tcgen05_after_thread_sync();
        
        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG0 CTA%d] P is ready, computing softmax\n", cta_idx);
        }
        {
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

            // Load one logical row of P from TMEM
            float2 p_row[FLOAT2_PER_ROW];
            ku::tmem_ld_32dp32bNx<P_COLS/2>(tmem_addr, p_row);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();

            // Write to global memory
            for (int col = 0; col < FLOAT2_PER_ROW; ++col) {
                P[logical_row * P_COLS + P_COLS / 2 * col_offset + col * 2] = p_row[col].x;
                P[logical_row * P_COLS + P_COLS / 2 * col_offset + col * 2 + 1] = p_row[col].y;
            }
            __threadfence_block();

            if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
                printf("[WG0 CTA%d] finish 2: P read from TMEM and written to global memory\n", cta_idx);
            }   
        }
        
        // Step 2: Load P from TMEM
        float2 p[(B_TOPK/2)/2];
        uint32_t tmem_base = plan.tmem_start_addr.data()[0];
        uint32_t tmem_lane = idx_in_warpgroup % (B_H/2);  // Logical row index
        uint32_t tmem_col = tmem_cols::P;
        uint32_t tmem_addr = tmem_base + (tmem_lane << 16) + tmem_col;
        
        ku::tmem_ld_32dp32bNx<B_TOPK/2>(tmem_addr, p);
        cutlass::arch::fence_view_async_tmem_load();
        ku::tcgen05_before_thread_sync();
        
        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG0 CTA%d] finish 3: P loaded from TMEM\n", cta_idx);
        }

        // plan.bar_k_valid_ready.wait(0);
        // uint32_t is_k_valid_lo = *(uint32_t*)(plan.is_k_valid + (idx_in_warpgroup>=64?B_TOPK/8/2:0));
        // float* p_float = (float*)p;
        // CUTE_UNROLL
        // for (int i = 0; i < (B_TOPK/2)/2; i += 1) {
        //     if (!(is_k_valid_lo >> i & 1))
        //         p_float[i] = -CUDART_INF_F;
        // }

        // if (idx_in_warpgroup == 0 && cta_idx == 0 && blockIdx.x == 0) {
        //     printf("[WG0] finish 4: k valid mask loaded from SMEM\n");
        // }

        // Step 3: Compute softmax(P) = exp2(P*scale - LSE)
        // Compute softmax values: s = exp2(P*scale - LSE)
        float2 s_fp32[(B_TOPK/2)/2];
        float2 neg_lse = make_float2(-row_lse, -row_lse);
        CUTE_UNROLL
        for (int i = 0; i < (B_TOPK/2)/2; ++i) {
            float2 scaled_p = ku::float2_fma(p[i], make_float2(scale, scale), neg_lse);
            scaled_p.x = exp2f(scaled_p.x);
            scaled_p.y = exp2f(scaled_p.y);
            // s_fp32[i] = __float22bfloat162_rn(scaled_p);
            s_fp32[i] = scaled_p;
        }
        
        // Step 4: Store s to SMEM (convert fp32 to bf16)
        // uint128_t* sS_base = (uint128_t*)plan.s_ds.s.data() + idx_in_warpgroup%64 + 64*((idx_in_warpgroup/64)*8);
        // CUTE_UNROLL
        // for (int i = 0; i < B_TOPK/2/8; ++i) {
        //     sS_base[64*i] = *(uint128_t*)(s_fp32 + i*4);
        // }
        // fence_view_async_shared();

        int s_col_offset = (idx_in_warpgroup / 64) * 32;
        CUTE_UNROLL
        for (int i = 0; i < (B_TOPK/2)/2; ++i) {
            // sS(idx_in_warpgroup%64, i*2+s_col_offset) = bf16(s_fp32[i].x);
            // sS(idx_in_warpgroup%64, i*2+1+s_col_offset) = bf16(s_fp32[i].y);
            sS(i*2+s_col_offset, idx_in_warpgroup%64) = bf16(s_fp32[i].x);
            sS(i*2+1+s_col_offset, idx_in_warpgroup%64) = bf16(s_fp32[i].y);
        }
        fence_view_async_shared();

        
        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG0 CTA%d] finish 5: softmax computed and s stored to SMEM\n", cta_idx);
        }
        
        // WG0内同步：使用条件同步，只有WG0的线程参与
        if (is_wg0) {
            __threadfence_block();  // 确保内存操作的可见性
        }
        
        // Step 5: Write s from SMEM to global memory
        if (idx_in_warpgroup < B_H/2) {
            const int global_row_idx = cta_idx * (B_H/2) + idx_in_warpgroup;
            for (int col = 0; col < B_TOPK; ++col) {
                s[global_row_idx * B_TOPK + col] = sS(col, idx_in_warpgroup);
            }
        }
        
        // Notify WG3 that s is ready
        if (cta_idx == 0) {
            plan.bar_s_ready.arrive(0u);
        } else {
            plan.bar_s_ready.arrive(1u);
        }
        
        
        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG0 CTA%d] finish 6: s written to global memory\n", cta_idx);
        }
        
        // Step 6: Load delta from global memory
        float delta_val = 0.0f;
        delta_val = __ldg(delta + global_row_idx);
        
        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG0 CTA%d] finish 7: delta loaded from global memory\n", cta_idx);
        }
        
        // Step 7: Wait for WG3 to compute dp
        plan.bar_dp_ready.wait(0);
        ku::tcgen05_after_thread_sync();
        
        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG0 CTA%d] finish 8: dP is ready, computing ds\n", cta_idx);
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
            
            // Load one logical row of dP from TMEM
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
        
        // Step 8: Load dp from TMEM
        float2 dp[(B_TOPK/2)/2];
        uint32_t dp_tmem_addr = tmem_base + (tmem_lane << 16) + tmem_cols::dP;
        ku::tmem_ld_32dp32bNx<B_TOPK/2>(dp_tmem_addr, dp);
        cutlass::arch::fence_view_async_tmem_load();
        ku::tcgen05_before_thread_sync();
        
        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG0 CTA%d] finish 9: dp loaded from TMEM\n", cta_idx);
        }
        
        // Step 9: Compute ds = s * (dp - delta) * sm_scale
        // Note: Use fp32 format of s (s_fp32), not bf16 format stored in SMEM
        // Note: Use sm_scale (not scale which includes log2(e)) for ds computation
        __nv_bfloat162 ds_fp32[(B_TOPK/2)/2];
        float2 delta_float2 = make_float2(delta_val, delta_val);
        float2 sm_scale_float2 = make_float2(sm_scale, sm_scale);
        CUTE_UNROLL
        for (int i = 0; i < (B_TOPK/2)/2; ++i) {
            // Convert s_fp32 back to float2 for computation
            // float2 s_val = __bfloat1622float2(s_fp32[i]);
            float2 s_val = s_fp32[i];
            float2 dp_val = dp[i];
            float2 dp_minus_delta = float2_sub(dp_val, delta_float2);
            float2 ds_val = ku::float2_mul(ku::float2_mul(s_val, dp_minus_delta), sm_scale_float2);
            ds_fp32[i] = __float22bfloat162_rn(ds_val);
        }
            
        // Step 10: Store ds to SMEM (convert fp32 to bf16)
        // uint128_t* sDS_base = (uint128_t*)plan.s_ds.ds.data() + idx_in_warpgroup%64 + 64*((idx_in_warpgroup/64)*8);
        // CUTE_UNROLL
        // for (int i = 0; i < B_TOPK/2/8; ++i) {
        //     sDS_base[64*i] = *(uint128_t*)(ds_fp32 + i*4);
        // }
        // fence_view_async_shared();
        int ds_col_offset = (idx_in_warpgroup / 64) * 32;
        CUTE_UNROLL
        for (int i = 0; i < (B_TOPK/2)/2; ++i) {
            // sDS(idx_in_warpgroup%64, i*2+ds_col_offset) = bf16(ds_fp32[i].x);
            // sDS(idx_in_warpgroup%64, i*2+1+ds_col_offset) = bf16(ds_fp32[i].y);
            sDS(i*2+ds_col_offset, idx_in_warpgroup%64) = bf16(ds_fp32[i].x);
            sDS(i*2+1+ds_col_offset, idx_in_warpgroup%64) = bf16(ds_fp32[i].y);
        }
        fence_view_async_shared();

        
        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG0 CTA%d] finish 10: ds computed and stored to SMEM\n", cta_idx);
        }
        
        // WG0内同步：使用条件同步，只有WG0的线程参与
        if (is_wg0) {
            __threadfence_block();  // 确保内存操作的可见性
        }
        
        // Step 11: Write ds from SMEM to global memory
        if (idx_in_warpgroup < B_H/2) {
            const int global_row_idx = cta_idx * (B_H/2) + idx_in_warpgroup;
            for (int col = 0; col < B_TOPK; ++col) {
                ds[global_row_idx * B_TOPK + col] = sDS(col, idx_in_warpgroup);
            }
        }
        
        // Notify WG3 that ds is ready
        if (cta_idx == 0) {
            plan.bar_ds_ready.arrive(0u);
        } else {
            plan.bar_ds_ready.arrive(1u);
        }

        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG0 CTA%d] finish 11: ds written to global memory\n", cta_idx);
        }
        
        // ========================================
        // WG0 Step 12-13: Wait for dQ and transfer to global memory
        // ========================================
        
        // Step 12: Wait for WG3 to compute dQ
        plan.bar_dq_ready.wait(0);
        ku::tcgen05_after_thread_sync();
        
        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG0 CTA%d] finish 12: dQ is ready, transferring to global memory\n", cta_idx);
        }
        
        // Step 13: Read dQ from TMEM and write to global memory
        // dQ shape: [B_H, D_Q] = [128, 576], each CTA handles [B_H/2, D_Q] = [64, 576]
        // dQ is stored in two parts: dQ_NoPE (dims 0-511) and dQ_RoPE (dims 512-575)
        // TMEM layout: dQ uses cols dQ (256 cols for [64, 512] in two tiles) and dQ_RoPE (32 cols for [64, 64])
        {
            constexpr int dQ_ROWS = B_H / 2;  // 64
            
            int dQ_cta_row_offset = cta_idx * dQ_ROWS;
            
            // Each thread handles a portion of dQ
            // Thread mapping: 128 threads handle 64 rows x 576 cols
            // We'll have each of the 128 threads handle different rows/cols
            
            // For dQ_NoPE part (dims 0-511): read from TMEM and write to global in chunks
            // Use 8 chunks to reduce per-thread register footprint.
            constexpr int NOPE_FLOATS_PER_HALF = 256 / 2;  // 128 floats per thread per tile half
            constexpr int NOPE_CHUNKS = 8;
            constexpr int NOPE_CHUNK_FLOATS = NOPE_FLOATS_PER_HALF / NOPE_CHUNKS;  // 16 floats per chunk
            constexpr int NOPE_CHUNK_FLOAT2 = NOPE_CHUNK_FLOATS / 2;  // 8 float2 per chunk
            
            int row_in_cta = idx_in_warpgroup % dQ_ROWS;  // 0-63
            int col_half = idx_in_warpgroup / dQ_ROWS;     // 0 or 1
            int global_row = dQ_cta_row_offset + row_in_cta;
            
            uint32_t tmem_base_dq = plan.tmem_start_addr.data()[0];
            uint32_t tmem_addr_dq0 = tmem_base_dq + (row_in_cta << 16) + tmem_cols::dQ;
            uint32_t tmem_addr_dq1 = tmem_base_dq + (row_in_cta << 16) + (tmem_cols::dQ + 128);

            // 8-iteration loop for NoPE output (both 256-col tiles)
            for (int chunk = 0; chunk < NOPE_CHUNKS; ++chunk) {
                const int chunk_col_base = chunk * NOPE_CHUNK_FLOATS;
                float2 dq_chunk[NOPE_CHUNK_FLOAT2];

                // Tile 0: cols [0, 255]
                ku::tmem_ld_32dp32bNx<NOPE_CHUNK_FLOATS>(tmem_addr_dq0 + chunk_col_base, dq_chunk);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();
                for (int i = 0; i < NOPE_CHUNK_FLOAT2; ++i) {
                    int col = col_half * (256 / 2) + chunk_col_base + i * 2;
                    dQ[global_row * D_Q + col] = dq_chunk[i].x;
                    dQ[global_row * D_Q + col + 1] = dq_chunk[i].y;
                }

                // Tile 1: cols [256, 511]
                ku::tmem_ld_32dp32bNx<NOPE_CHUNK_FLOATS>(tmem_addr_dq1 + chunk_col_base, dq_chunk);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();
                for (int i = 0; i < NOPE_CHUNK_FLOAT2; ++i) {
                    int col = 256 + col_half * (256 / 2) + chunk_col_base + i * 2;
                    dQ[global_row * D_Q + col] = dq_chunk[i].x;
                    dQ[global_row * D_Q + col + 1] = dq_chunk[i].y;
                }
            }
            
            // Read dQ_RoPE (cols 512-575)
            constexpr int ROPE_FLOAT2_PER_ROW = D_ROPE / 2 / 2;  // 16 float2 per row
            float2 dq_rope[ROPE_FLOAT2_PER_ROW];
            uint32_t tmem_addr_dq_rope = tmem_base_dq + (row_in_cta << 16) + tmem_cols::dQ_RoPE;
            ku::tmem_ld_32dp32bNx<D_ROPE/2>(tmem_addr_dq_rope, dq_rope);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();
            
            // Write dQ_RoPE to global memory
            for (int i = 0; i < ROPE_FLOAT2_PER_ROW; ++i) {
                int col = D_V + col_half * (D_ROPE / 2) + i * 2;
                dQ[global_row * D_Q + col] = dq_rope[i].x;
                dQ[global_row * D_Q + col + 1] = dq_rope[i].y;
            }
        }
        
        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG0 CTA%d] finish 13: dQ written to global memory\n", cta_idx);
        }
    }
    
    // ========================================
    // Warpgroup 1: KV Loading (WG1)
    // Responsibility: Load Q, KV, dO from global memory and write back
    // ========================================
    if (is_wg1) {
        cutlass::arch::warpgroup_reg_dealloc<96>();
        
        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG1 CTA%d] finish 1: start loading Q from global memory\n", cta_idx);
        }
        
        // Load Q from Global Memory to SMEM (split into NoPE and RoPE)
        // Q NoPE: cta0读取前[B_H/2, D_V] = [64, 512], cta1读取后[B_H/2, D_V] = [64, 512]
        // Q RoPE: cta0读取前[B_H/2, D_ROPE] = [64, 64], cta1读取后[B_H/2, D_ROPE] = [64, 64]
        constexpr int Q_NOPE_ELEMENTS_PER_CTA = (B_H / 2) * D_V;  // 64 * 512 = 32768
        constexpr int Q_NOPE_ELEMENTS_PER_THREAD = (Q_NOPE_ELEMENTS_PER_CTA + 128 - 1) / 128;
        constexpr int Q_ROPE_ELEMENTS_PER_CTA = (B_H / 2) * D_ROPE;  // 64 * 64 = 4096
        constexpr int Q_ROPE_ELEMENTS_PER_THREAD = (Q_ROPE_ELEMENTS_PER_CTA + 128 - 1) / 128;
        const int q_row_offset = cta_idx * (B_H / 2);
        
        // Load Q NoPE part
        for (int i = 0; i < Q_NOPE_ELEMENTS_PER_THREAD; ++i) {
            int linear_idx = idx_in_warpgroup * Q_NOPE_ELEMENTS_PER_THREAD + i;
            if (linear_idx < Q_NOPE_ELEMENTS_PER_CTA) {
                int row = linear_idx / D_V;
                int col = linear_idx % D_V;
                sQNoPE(row, col) = q[(q_row_offset + row) * D_Q + col];
            }
        }
        
        // Load Q RoPE part
        for (int i = 0; i < Q_ROPE_ELEMENTS_PER_THREAD; ++i) {
            int linear_idx = idx_in_warpgroup * Q_ROPE_ELEMENTS_PER_THREAD + i;
            if (linear_idx < Q_ROPE_ELEMENTS_PER_CTA) {
                int row = linear_idx / D_ROPE;
                int col = linear_idx % D_ROPE;
                sQRoPE(row, col) = q[(q_row_offset + row) * D_Q + D_V + col];
            }
        }
        
        // WG1内同步：使用条件同步，只有WG1的线程参与
        // 由于没有专门的warpgroup同步primitive，我们使用条件同步
        // 所有WG1的线程都会执行到这里，其他warpgroup的线程不会执行
        if (is_wg1) {
            // 使用barrier或简单的条件等待来确保WG1内同步
            // 由于所有WG1线程都在执行相同代码，可以使用__syncthreads()但需要确保其他warpgroup不hang
            // 实际上，由于其他warpgroup的线程不会执行这段代码，所以这里不需要额外同步
            // 只需要确保内存操作的顺序性
            __threadfence_block();  // 确保内存操作的可见性
        }
        
        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG1 CTA%d] finish 2: Q loaded to SMEM (NoPE + RoPE)\n", cta_idx);
        }
        
        // Load KV from Global Memory to SMEM (split into NoPE and RoPE)
        // K NoPE: cta0读取前[B_TOPK/2, D_V] = [32, 512], cta1读取后[B_TOPK/2, D_V] = [32, 512]
        // K RoPE: cta0读取前[B_TOPK/2, D_ROPE] = [32, 64], cta1读取后[B_TOPK/2, D_ROPE] = [32, 64]
        constexpr int K_NOPE_ELEMENTS_PER_CTA = (B_TOPK / 2) * D_V;  // 32 * 512 = 16384
        constexpr int K_NOPE_ELEMENTS_PER_THREAD = (K_NOPE_ELEMENTS_PER_CTA + 128 - 1) / 128;
        constexpr int K_ROPE_ELEMENTS_PER_CTA = (B_TOPK / 2) * D_ROPE;  // 32 * 64 = 2048
        constexpr int K_ROPE_ELEMENTS_PER_THREAD = (K_ROPE_ELEMENTS_PER_CTA + 128 - 1) / 128;
        const int kv_row_offset = cta_idx * (B_TOPK / 2);
        
        // Load K NoPE part
        for (int i = 0; i < K_NOPE_ELEMENTS_PER_THREAD; ++i) {
            int linear_idx = idx_in_warpgroup * K_NOPE_ELEMENTS_PER_THREAD + i;
            if (linear_idx < K_NOPE_ELEMENTS_PER_CTA) {
                int row = linear_idx / D_V;
                int col = linear_idx % D_V;
                sKNoPE(row, col) = kv[(kv_row_offset + row) * D_K + col];
            }
        }
        
        // Load K RoPE part
        for (int i = 0; i < K_ROPE_ELEMENTS_PER_THREAD; ++i) {
            int linear_idx = idx_in_warpgroup * K_ROPE_ELEMENTS_PER_THREAD + i;
            if (linear_idx < K_ROPE_ELEMENTS_PER_CTA) {
                int row = linear_idx / D_ROPE;
                int col = linear_idx % D_ROPE;
                sKRoPE(row, col) = kv[(kv_row_offset + row) * D_K + D_V + col];
            }
        }
        
        if (is_wg1) {
            __threadfence_block();
        }
        
        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG1 CTA%d] finish 3: KV loaded to SMEM\n", cta_idx);
        }
        
        // Load dO from Global Memory to SMEM
        // dO: cta0读取前[B_H/2, D_V] = [64, 512], cta1读取后[B_H/2, D_V] = [64, 512]
        constexpr int dO_ELEMENTS_PER_CTA = (B_H / 2) * D_V;  // 64 * 512 = 32768
        constexpr int dO_ELEMENTS_PER_THREAD = (dO_ELEMENTS_PER_CTA + 128 - 1) / 128;
        const int dO_row_offset = cta_idx * (B_H / 2);
        
        for (int i = 0; i < dO_ELEMENTS_PER_THREAD; ++i) {
            int linear_idx = idx_in_warpgroup * dO_ELEMENTS_PER_THREAD + i;
            if (linear_idx < dO_ELEMENTS_PER_CTA) {
                int row = linear_idx / D_V;
                int col = linear_idx % D_V;
                sdO(row, col) = dO[(dO_row_offset + row) * D_V + col];
            }
        }
        
        if (is_wg1) {
            __threadfence_block();
        }
        
        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG1 CTA%d] finish 4: dO loaded to SMEM\n", cta_idx);
        }

        // Barrier: notify WG3 that q, k, dO are all loaded to SMEM
        plan.bar_qkv_loaded.arrive(0u);
        
        // Write Q, KV, dO from SMEM to global memory
        // Write Q from SMEM to q_out (split into NoPE and RoPE)
        // Write Q NoPE part
        for (int i = 0; i < Q_NOPE_ELEMENTS_PER_THREAD; ++i) {
            int linear_idx = idx_in_warpgroup * Q_NOPE_ELEMENTS_PER_THREAD + i;
            if (linear_idx < Q_NOPE_ELEMENTS_PER_CTA) {
                int row = linear_idx / D_V;
                int col = linear_idx % D_V;
                q_out[(q_row_offset + row) * D_Q + col] = sQNoPE(row, col);
            }
        }
        
        // Write Q RoPE part
        for (int i = 0; i < Q_ROPE_ELEMENTS_PER_THREAD; ++i) {
            int linear_idx = idx_in_warpgroup * Q_ROPE_ELEMENTS_PER_THREAD + i;
            if (linear_idx < Q_ROPE_ELEMENTS_PER_CTA) {
                int row = linear_idx / D_ROPE;
                int col = linear_idx % D_ROPE;
                q_out[(q_row_offset + row) * D_Q + D_V + col] = sQRoPE(row, col);
            }
        }
        
        if (is_wg1) {
            __threadfence_block();
        }
        
        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG1 CTA%d] finish 5: Q written to global memory (NoPE + RoPE)\n", cta_idx);
        }
        
        // Write KV from SMEM to kv_out (split into NoPE and RoPE)
        // Write K NoPE part
        for (int i = 0; i < K_NOPE_ELEMENTS_PER_THREAD; ++i) {
            int linear_idx = idx_in_warpgroup * K_NOPE_ELEMENTS_PER_THREAD + i;
            if (linear_idx < K_NOPE_ELEMENTS_PER_CTA) {
                int row = linear_idx / D_V;
                int col = linear_idx % D_V;
                kv_out[(kv_row_offset + row) * D_K + col] = sKNoPE(row, col);
            }
        }
        
        // Write K RoPE part
        for (int i = 0; i < K_ROPE_ELEMENTS_PER_THREAD; ++i) {
            int linear_idx = idx_in_warpgroup * K_ROPE_ELEMENTS_PER_THREAD + i;
            if (linear_idx < K_ROPE_ELEMENTS_PER_CTA) {
                int row = linear_idx / D_ROPE;
                int col = linear_idx % D_ROPE;
                kv_out[(kv_row_offset + row) * D_K + D_V + col] = sKRoPE(row, col);
            }
        }
        
        if (is_wg1) {
            __threadfence_block();
        }
        
        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG1 CTA%d] finish 6: KV written to global memory\n", cta_idx);
        }
        
        // Write dO from SMEM to dO_out
        for (int i = 0; i < dO_ELEMENTS_PER_THREAD; ++i) {
            int linear_idx = idx_in_warpgroup * dO_ELEMENTS_PER_THREAD + i;
            if (linear_idx < dO_ELEMENTS_PER_CTA) {
                int row = linear_idx / D_V;
                int col = linear_idx % D_V;
                dO_out[(dO_row_offset + row) * D_V + col] = sdO(row, col);
            }
        }
        
        if (is_wg1) {
            __threadfence_block();
        }
        
        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG1 CTA%d] finish 7: dO written to global memory\n", cta_idx);
        }
        
        // ========================================
        // WG1 Step 8: Load kv_peer from peer CTA using cp_async
        // Reference: test_smem_cp_async.cu
        // ========================================
        // kv_peer: [B_TOPK/2, D_K] = [32, 576] per CTA
        constexpr int KV_PEER_ELEMENTS = (B_TOPK / 2) * D_K;  // 32 * 576 = 18432
        
        // Prepare for cp_async: expect transaction bytes
        if (idx_in_warpgroup == 0) {
            plan.bar_kv_peer_cp_async.arrive_and_expect_tx(sizeof(bf16) * KV_PEER_ELEMENTS);
        }
        
        // Cluster sync before cp_async
        // cluster_sync();
        
        // Launch cp_async to copy K from peer CTA's sKNoPE + sKRoPE to local sKV_peer
        // CTA0 copies from CTA1's KV, CTA1 copies from CTA0's KV
        if (idx_in_warpgroup == 0) {
            // Get peer address for kv_peer buffer
            bf16* peer_kv_peer_ptr = kerutils::get_peer_addr(plan.u.q_kv.kv_peer.data());
            transac_bar_t* peer_bar_ptr = kerutils::get_peer_addr(&plan.bar_kv_peer_cp_async);
            
            // Source is local K_nope + K_rope (need to be combined in kv_peer layout)
            // For simplicity, we copy the local k_nope and k_rope combined area
            // Note: k_nope and k_rope are contiguous in memory layout
            kerutils::cp_async_bulk_shared_cta_to_shared_cluster(
                peer_kv_peer_ptr,
                plan.u.q_kv.k_nope.data(),  // Source: local k_nope (which continues to k_rope)
                sizeof(bf16) * KV_PEER_ELEMENTS,
                *peer_bar_ptr
            );
        }
        
        fence_view_async_shared();
        
        // Wait for cp_async to complete
        if (idx_in_warpgroup == 0) {
            plan.bar_kv_peer_cp_async.wait(0);
            // __threadfence_block();
            
            // Notify WG3 that kv_peer is ready
            if (cta_idx == 0) {
                plan.bar_kv_peer_ready.arrive(0u);
            } else {
                plan.bar_kv_peer_ready.arrive(1u);
            }
        }
        
        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG1 CTA%d] finish 8: kv_peer loaded via cp_async\n", cta_idx);
        }
    }
    

    // ========================================
    // Warpgroup 2: dKV Transfer (WG2)
    // Responsibility: Read dKV from TMEM and atomicAdd to global memory
    // ========================================
    if (is_wg2) {
        cutlass::arch::warpgroup_reg_alloc<128>();

        const int row = idx_in_warpgroup % B_TOPK;   // 0-63: which KV row
        const int half = idx_in_warpgroup / B_TOPK;   // 0 or 1: which column half
        uint32_t tmem_base_wg2 = plan.tmem_start_addr.data()[0];

        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG2 CTA%d] finish 1: WG2 started\n", cta_idx);
        }

        // ---- Step 3.1: Wait for WG3 to compute dKV_part0 ----
        plan.bar_dkv_part0_ready.wait(0);
        ku::tcgen05_after_thread_sync();

        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG2 CTA%d] finish 2: dKV_part0 ready\n", cta_idx);
        }

        // ---- Step 3.2: Transfer dKV_part0 to global memory (dims 0-255) ----
        {
            constexpr int COLS_PER_HALF = 256 / 2;  // 128 float values per half
            constexpr int CHUNK_SIZE = COLS_PER_HALF / 4;  // 32 float values per chunk
            float* dst = dKV + row * D_K + half * COLS_PER_HALF;

            // Loop 4 times to read in chunks, reducing register usage
            CUTE_UNROLL
            for (int chunk = 0; chunk < 4; ++chunk) {
                float2 dkv_data[CHUNK_SIZE / 2];  // 16 float2 = 32 floats
                ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(tmem_cols::dKV + chunk * CHUNK_SIZE, dkv_data);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                // atom.add.v4: atomically add to global memory
                float* src = (float*)dkv_data;
                CUTE_UNROLL
                for (int i = 0; i < CHUNK_SIZE; i += 4) {
                    atomicAdd(dst + chunk * CHUNK_SIZE + i,     src[i]);
                    atomicAdd(dst + chunk * CHUNK_SIZE + i + 1, src[i + 1]);
                    atomicAdd(dst + chunk * CHUNK_SIZE + i + 2, src[i + 2]);
                    atomicAdd(dst + chunk * CHUNK_SIZE + i + 3, src[i + 3]);
                }
            }
        }

        // ---- Step 3.3: Notify WG3 that dKV_part0 transfer is done ----
        if (cta_idx == 0) {
            plan.bar_dkv_part0_done.arrive(0u);
        } else {
            plan.bar_dkv_part0_done.arrive(1u);
        }

        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG2 CTA%d] finish 4: dKV_part0 transferred\n", cta_idx);
        }

        // ---- Step 3.4: Wait for WG3 to compute dKV_part1 ----
        plan.bar_dkv_part1_ready.wait(0);
        ku::tcgen05_after_thread_sync();

        // ---- Step 3.5: Transfer dKV_part1 to global memory (dims 256-511) ----
        {
            constexpr int COLS_PER_HALF = 256 / 2;  // 128 float values per half
            constexpr int CHUNK_SIZE = COLS_PER_HALF / 4;  // 32 float values per chunk
            float* dst = dKV + row * D_K + 256 + half * COLS_PER_HALF;

            // Loop 4 times to read in chunks, reducing register usage
            CUTE_UNROLL
            for (int chunk = 0; chunk < 4; ++chunk) {
                float2 dkv_data[CHUNK_SIZE / 2];  // 16 float2 = 32 floats
                ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(tmem_cols::dKV + chunk * CHUNK_SIZE, dkv_data);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                // atom.add.v4: atomically add to global memory (offset by 256 for part1)
                float* src = (float*)dkv_data;
                CUTE_UNROLL
                for (int i = 0; i < CHUNK_SIZE; i += 4) {
                    atomicAdd(dst + chunk * CHUNK_SIZE + i,     src[i]);
                    atomicAdd(dst + chunk * CHUNK_SIZE + i + 1, src[i + 1]);
                    atomicAdd(dst + chunk * CHUNK_SIZE + i + 2, src[i + 2]);
                    atomicAdd(dst + chunk * CHUNK_SIZE + i + 3, src[i + 3]);
                }
            }
        }

        // ---- Step 3.6: Notify WG3 that dKV_part1 transfer is done ----
        if (cta_idx == 0) {
            plan.bar_dkv_part1_done.arrive(0u);
        } else {
            plan.bar_dkv_part1_done.arrive(1u);
        }

        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG2 CTA%d] finish 5: dKV_part1 transferred\n", cta_idx);
        }

        // ---- Step 3.7: Wait for WG3 to compute dKV_part2 ----
        plan.bar_dkv_part2_ready.wait(0);
        ku::tcgen05_after_thread_sync();

        // ---- Step 3.8: Transfer dKV_part2 to global memory (dims 512-575, RoPE) ----
        {
            constexpr int ROPE_COLS_PER_HALF = D_ROPE / 2;  // 32 float values per half
            float2 dkv_rope_data[ROPE_COLS_PER_HALF / 2];  // 16 float2 = 32 floats
            ku::tmem_ld_32dp32bNx<ROPE_COLS_PER_HALF>(tmem_cols::dKV_RoPE, dkv_rope_data);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();

            // atom.add.v4: atomically add to global memory (offset by D_V=512 for RoPE part)
            float* dst = dKV + row * D_K + D_V + half * ROPE_COLS_PER_HALF;
            float* src = (float*)dkv_rope_data;
            CUTE_UNROLL
            for (int i = 0; i < ROPE_COLS_PER_HALF; i += 4) {
                atomicAdd(dst + i,     src[i]);
                atomicAdd(dst + i + 1, src[i + 1]);
                atomicAdd(dst + i + 2, src[i + 2]);
                atomicAdd(dst + i + 3, src[i + 3]);
            }
        }

        // ---- Step 3.9: Notify WG3 that dKV_part2 transfer is done ----
        if (cta_idx == 0) {
            plan.bar_dkv_part2_done.arrive(0u);
        } else {
            plan.bar_dkv_part2_done.arrive(1u);
        }

        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG2 CTA%d] finish 6: dKV_part2 (RoPE) transferred, all done\n", cta_idx);
        }
    }

    // ========================================
    // Warpgroup 3: Matrix Multiplication (WG3)
    // Responsibility: Compute P, dP, and dKV
    // ========================================
    if (is_wg3) {
        cutlass::arch::warpgroup_reg_alloc<168>();
        
        // Allocate TMEM tensors for P and dP
        TiledMMA_P tiled_mma_P{};
        TiledMMA_dP tiled_mma_dP{};
        Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H/2>, Int<B_TOPK>>{});
        Tensor tdP = partition_fragment_C(tiled_mma_dP, Shape<Int<B_H/2>, Int<B_TOPK>>{});
        tP.data().get() = tmem_cols::P;
        tdP.data().get() = tmem_cols::dP;

        // Allocate TMEM tensors for dKV
        TiledMMA_dKV tiled_mma_dKV{};
        TiledMMA_dKV_RoPE tiled_mma_dKV_RoPE{};
        Tensor tdKV = partition_fragment_C(tiled_mma_dKV, Shape<Int<B_TOPK>, Int<256>>{});
        tdKV.data().get() = tmem_cols::dKV;
        Tensor tdKV_RoPE = partition_fragment_C(tiled_mma_dKV_RoPE, Shape<Int<B_TOPK>, Int<D_ROPE>>{});
        tdKV_RoPE.data().get() = tmem_cols::dKV_RoPE;

        if (idx_in_warpgroup == 0 && blockIdx.x < 2) {
            printf("[WG3 CTA%d] finish 1: TMEM tensors allocated\n", cta_idx);
        }

        // Extract V from memory: V uses the same layout as K_NoPE
        Tensor sV = make_tensor(make_smem_ptr(plan.u.q_kv.k_nope.data()), SmemLayoutV{});

        // ---- Phase 1: 2x1SM MMA for P and dP (CTA0 only) ----
        if (cta_idx == 0 && warp_idx == 12 && elect_one_sync()) {
            // Wait for WG1 to load q, k, dO to SMEM
            plan.bar_qkv_loaded.wait(0);
            ku::tcgen05_after_thread_sync();
            
            // Compute P = Q @ K^T using TiledMMA_P
            ku::utcmma_ss(tiled_mma_P, sQNoPE, sKNoPE, tP, true);  // clear_accum = true
            ku::utcmma_ss(tiled_mma_P, sQRoPE, sKRoPE, tP, false);  // clear_accum = false
            ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_p_ready, 1|2);
            
            if (idx_in_warpgroup == 0 && cta_idx == 0 && blockIdx.x == 0) {
                printf("[WG3] finish 2: P computed (NoPE + RoPE)\n");
            }

            ku::tcgen05_after_thread_sync();
            
            // Compute dP = dO @ V^T using TiledMMA_dP
            ku::utcmma_ss(tiled_mma_dP, sdO, sV, tdP, true);  // clear_accum = true
            ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_dp_ready, 1|2);
            
            if (idx_in_warpgroup == 0 && cta_idx == 0 && blockIdx.x == 0) {
                printf("[WG3] finish 3: dP computed\n");
            }

            ku::tcgen05_after_thread_sync();
        }

        // ---- Phase 2: WS MMA for dKV (both CTAs) ----
        if (warp_idx == 12 && elect_one_sync()) {
            // Wait for WG0 to compute s
            plan.bar_s_ready.wait(0);

            if (blockIdx.x < 2) {
                printf("[WG3 CTA%d] finish 4: s and ds ready, starting dKV computation\n", cta_idx);
            }

            // S and dS tensors for MMA A operand
            Tensor sS_mma = make_tensor(make_smem_ptr(plan.s_ds.s.data()), SmemLayoutS{});
            Tensor sDS_mma = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutdS{});

            // dO transposed: full [D_V, B_H/2] = [512, 64], then flat_divide into [256, 64] tiles
            Tensor sdO_t_full = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutQTilesTransposed<D_V/64>{});
            auto sdO_t_div = flat_divide(sdO_t_full, Shape<Int<256>, Int<B_H/2>>{});

            // Q NoPE transposed: full [D_V, B_H/2] = [512, 64], then flat_divide into [256, 64] tiles
            Tensor sQ_t_full = make_tensor(make_smem_ptr(plan.u.q_kv.q_nope.data()), SmemLayoutQTilesTransposed<D_V/64>{});
            auto sQ_t_div = flat_divide(sQ_t_full, Shape<Int<256>, Int<B_H/2>>{});

            // Q RoPE transposed: [D_ROPE, B_H/2] = [64, 64]
            Tensor sQ_rope_t = make_tensor(make_smem_ptr(plan.u.q_kv.q_rope.data()), SmemLayoutQRoPETransposed{});

            // ---- Step 2.2: Compute dV[0:256] = s^T @ dO[:, 0:256] (clear) ----
            ku::utcmma_ss(tiled_mma_dKV, sS_mma, sdO_t_div(_, _, _0{}, _0{}), tdKV, true);

            // Wait for WG0 to compute ds
            plan.bar_ds_ready.wait(0);
            // ---- Step 2.3: Compute dK[0:256] = ds^T @ Q[:, 0:256] (accumulate) ----
            ku::utcmma_ss(tiled_mma_dKV, sDS_mma, sQ_t_div(_, _, _0{}, _0{}), tdKV, false);
            ku::umma_arrive_noelect(plan.bar_dkv_part0_ready);

            // ---- Step 2.4: Notify WG2 that dKV_part0 is ready ----
            ku::tcgen05_after_thread_sync();
            // if (cta_idx == 0) {
            //     plan.bar_dkv_part0_ready.arrive(0u);
            // } else {
            //     plan.bar_dkv_part0_ready.arrive(1u);
            // }

            if (blockIdx.x < 2) {
                printf("[WG3 CTA%d] finish 5: dKV_part0 computed, waiting for WG2 transfer\n", cta_idx);
            }

            // ========================================
            // WG3 Step 5.1-5.4: Compute dQ (inserted between dKV_part0 notify and wait)
            // dQ = ds @ K (K_local + K_peer)
            // ========================================
            
            // Allocate TMEM tensors for dQ
            TiledMMA_dQ tiled_mma_dQ{};
            TiledMMA_dQ_RoPE tiled_mma_dQ_RoPE{};
            Tensor tdQ_part0 = partition_fragment_C(tiled_mma_dQ, Shape<Int<B_H/2>, Int<256>>{});
            tdQ_part0.data().get() = tmem_cols::dQ;
            Tensor tdQ_part1 = partition_fragment_C(tiled_mma_dQ, Shape<Int<B_H/2>, Int<256>>{});
            tdQ_part1.data().get() = tmem_cols::dQ + 128;
            Tensor tdQ_RoPE = partition_fragment_C(tiled_mma_dQ_RoPE, Shape<Int<B_H/2>, Int<D_ROPE>>{});
            tdQ_RoPE.data().get() = tmem_cols::dQ_RoPE;
            
            // dS transposed tensor for MMA A operand: [B_H/2, B_TOPK] -> treat as A matrix
            Tensor sDS_t = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutdSTransposed{});
            auto sDS_t_div = flat_divide(sDS_t, Shape<Int<B_H/2>, Int<B_TOPK/2>>{});
            
            // K NoPE transposed: [D_V, B_TOPK/2] per CTA, flat_divide into [256, B_TOPK/2] tiles
            Tensor sK_nope_t_full = make_tensor(make_smem_ptr(plan.u.q_kv.k_nope.data()), SmemLayoutKVTilesTransposed<D_V/64>{});
            auto sK_nope_t_div = flat_divide(sK_nope_t_full, Shape<Int<256>, Int<B_TOPK/2>>{});
            
            // K RoPE transposed: [D_ROPE, B_TOPK/2] = [64, 32]
            Tensor sK_rope_t = make_tensor(make_smem_ptr(plan.u.q_kv.k_rope.data()), SmemLayoutKRoPETransposed{});
            
            // K_peer transposed tensors (peer CTA's K)
            Tensor sK_peer_nope_t_full = make_tensor(make_smem_ptr(plan.u.q_kv.kv_peer.data()), SmemLayoutKVTilesTransposed<D_V/64>{});
            auto sK_peer_nope_t_div = flat_divide(sK_peer_nope_t_full, Shape<Int<256>, Int<B_TOPK/2>>{});
            
            // K_peer RoPE transposed: [D_ROPE, B_TOPK/2] = [64, 32]
            // kv_peer layout: [B_TOPK/2, D_K] = [32, 576], so RoPE part starts at offset D_V
            Tensor sK_peer_rope_t = make_tensor(make_smem_ptr(plan.u.q_kv.kv_peer.data() + (B_TOPK/2) * D_V), SmemLayoutKRoPETransposed{});
            
            // dQ reduction dimension (topk) must match each CTA's K ownership:
            // CTA0: local K is topk[0:32], peer K is topk[32:64]
            // CTA1: local K is topk[32:64], peer K is topk[0:32]
            if (cta_idx == 0) {
                // ---- Step 5.1: Compute dQ_nope[0:256] = ds[:, 0:32] @ K_local_nope[0:256] (clear) ----
                ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _0{}), sK_nope_t_div(_, _, _0{}, _0{}), tdQ_part0, true);
                
                if (blockIdx.x < 2) {
                    printf("[WG3 CTA%d] finish 5.1: dQ_nope[0:256] computed with K_local\n", cta_idx);
                }
                
                // ---- Step 5.2: Wait for WG1 to finish loading kv_peer ----
                plan.bar_kv_peer_ready.wait(0);
                ku::tcgen05_after_thread_sync();
                
                if (blockIdx.x < 2) {
                    printf("[WG3 CTA%d] finish 5.2: kv_peer ready, computing dQ with K_peer\n", cta_idx);
                }
                
                // ---- Step 5.3: Compute dQ_nope[0:256] += ds[:, 32:64] @ K_peer_nope[0:256] (accumulate) ----
                ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _1{}), sK_peer_nope_t_div(_, _, _0{}, _0{}), tdQ_part0, false);
                
                // ---- Step 5.3b: Compute dQ_nope[256:512] = ds[:, 0:32] @ K_local_nope[256:512] + ds[:, 32:64] @ K_peer_nope[256:512] ----
                ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _0{}), sK_nope_t_div(_, _, _1{}, _0{}), tdQ_part1, true);  // clear for part1
                ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _1{}), sK_peer_nope_t_div(_, _, _1{}, _0{}), tdQ_part1, false);
                
                // ---- Step 5.3c: Compute dQ_rope = ds[:, 0:32] @ K_local_rope + ds[:, 32:64] @ K_peer_rope ----
                ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t_div(_, _, _0{}, _0{}), sK_rope_t, tdQ_RoPE, true);   // clear
                ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t_div(_, _, _0{}, _1{}), sK_peer_rope_t, tdQ_RoPE, false);  // accumulate
            } else {
                // ---- Step 5.1: Compute dQ_nope[0:256] = ds[:, 32:64] @ K_local_nope[0:256] (clear) ----
                ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _1{}), sK_nope_t_div(_, _, _0{}, _0{}), tdQ_part0, true);
                
                if (blockIdx.x < 2) {
                    printf("[WG3 CTA%d] finish 5.1: dQ_nope[0:256] computed with K_local\n", cta_idx);
                }
                
                // ---- Step 5.2: Wait for WG1 to finish loading kv_peer ----
                plan.bar_kv_peer_ready.wait(0);
                ku::tcgen05_after_thread_sync();
                
                if (blockIdx.x < 2) {
                    printf("[WG3 CTA%d] finish 5.2: kv_peer ready, computing dQ with K_peer\n", cta_idx);
                }
                
                // ---- Step 5.3: Compute dQ_nope[0:256] += ds[:, 0:32] @ K_peer_nope[0:256] (accumulate) ----
                ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _0{}), sK_peer_nope_t_div(_, _, _0{}, _0{}), tdQ_part0, false);
                
                // ---- Step 5.3b: Compute dQ_nope[256:512] = ds[:, 32:64] @ K_local_nope[256:512] + ds[:, 0:32] @ K_peer_nope[256:512] ----
                ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _1{}), sK_nope_t_div(_, _, _1{}, _0{}), tdQ_part1, true);  // clear for part1
                ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _0{}), sK_peer_nope_t_div(_, _, _1{}, _0{}), tdQ_part1, false);
                
                // ---- Step 5.3c: Compute dQ_rope = ds[:, 32:64] @ K_local_rope + ds[:, 0:32] @ K_peer_rope ----
                ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t_div(_, _, _0{}, _1{}), sK_rope_t, tdQ_RoPE, true);   // clear
                ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t_div(_, _, _0{}, _0{}), sK_peer_rope_t, tdQ_RoPE, false);  // accumulate
            }
            ku::umma_arrive_noelect(plan.bar_dq_ready);

            
            ku::tcgen05_after_thread_sync();
            
            if (blockIdx.x < 2) {
                printf("[WG3 CTA%d] finish 5.3: dQ computed (NoPE + RoPE)\n", cta_idx);
            }
            
            // ---- Step 5.4: Notify WG0 that dQ is ready ----
            // ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_dq_ready, 1|2);
            
            if (blockIdx.x < 2) {
                printf("[WG3 CTA%d] finish 5.4: dQ ready, notified WG0\n", cta_idx);
            }

            // ---- Step 2.5: Wait for WG2 to finish dKV_part0 transfer ----
            plan.bar_dkv_part0_done.wait(0);
            ku::tcgen05_after_thread_sync();

            // ---- Step 2.6: Compute dV[256:512] = s^T @ dO[:, 256:512] (clear) ----
            ku::utcmma_ss(tiled_mma_dKV, sS_mma, sdO_t_div(_, _, _1{}, _0{}), tdKV, true);
            // ---- Step 2.7: Compute dK[256:512] = ds^T @ Q[:, 256:512] (accumulate) ----
            ku::utcmma_ss(tiled_mma_dKV, sDS_mma, sQ_t_div(_, _, _1{}, _0{}), tdKV, false);
            ku::umma_arrive_noelect(plan.bar_dkv_part1_ready);

            // ---- Step 2.8: Notify WG2 that dKV_part1 is ready ----
            ku::tcgen05_after_thread_sync();
            // if (cta_idx == 0) {
            //     plan.bar_dkv_part1_ready.arrive(0u);
            // } else {
            //     plan.bar_dkv_part1_ready.arrive(1u);
            // }

            if (blockIdx.x < 2) {
                printf("[WG3 CTA%d] finish 6: dKV_part1 computed, waiting for WG2 transfer\n", cta_idx);
            }

            // ---- Step 2.9: Wait for WG2 to finish dKV_part1 transfer ----
            plan.bar_dkv_part1_done.wait(0);
            ku::tcgen05_after_thread_sync();

            // ---- Step 2.10: Compute dK_rope = ds^T @ Q_rope (clear) ----
            ku::utcmma_ss(tiled_mma_dKV_RoPE, sDS_mma, sQ_rope_t, tdKV_RoPE, true);
            ku::umma_arrive_noelect(plan.bar_dkv_part2_ready);

            // ---- Step 2.11: Notify WG2 that dKV_part2 is ready ----
            ku::tcgen05_after_thread_sync();
            // if (cta_idx == 0) {
            //     plan.bar_dkv_part2_ready.arrive(0u);
            // } else {
            //     plan.bar_dkv_part2_ready.arrive(1u);
            // }

            if (blockIdx.x < 2) {
                printf("[WG3 CTA%d] finish 7: dKV_part2 (RoPE) computed, waiting for WG2 transfer\n", cta_idx);
            }

            // ---- Step 2.12: Wait for WG2 to finish dKV_part2 transfer ----
            plan.bar_dkv_part2_done.wait(0);

            if (blockIdx.x < 2) {
                printf("[WG3 CTA%d] finish 8: All dKV parts transferred\n", cta_idx);
            }

            // ---- Step 2.13: Output sdO_t_full to global memory ----
            // sdO_t_full shape: [D_V, B_H/2] = [512, 64] per CTA
            // Full output shape: [D_V, B_H] = [512, 128] (both CTAs combined)
            // Only one thread executes (elect_one_sync), so it writes the entire tensor
            constexpr int D_V = 512;
            constexpr int B_H_HALF = 64;
            int row_offset = cta_idx * B_H_HALF;  // CTA0: 0-63, CTA1: 64-127
            
            // Write sdO_t_full to global memory
            // sdO_t_full is stored in SMEM with layout SmemLayoutQTilesTransposed<D_V/64>
            // sdO_t_full(col, row) accesses element at [col, row] in the transposed layout
            // Output to global memory: sdO_t_out[col * B_H + row_offset + row]
            for (int col = 0; col < D_V; ++col) {
                for (int row = 0; row < B_H_HALF; ++row) {
                    sdO_t_out[col * B_H + row_offset + row] = sdO_t_full(col, row);
                }
            }
            
            if (blockIdx.x < 2) {
                printf("[WG3 CTA%d] finish 9: sdO_t_full written to global memory\n", cta_idx);
            }
        }
    }

    // All threads must sync before proceeding
    __syncthreads();
    cluster_sync();
    
    // Free TMEM
    if (warp_idx == 0 && elect_one_sync()) {
        TMEM::Allocator2Sm().free(plan.tmem_start_addr.data()[0], 512);
    }
    
    if (tid == 0 && blockIdx.x < 2) {
        printf("[CTA %d] finish 16: TMEM freed, kernel completed\n", cta_idx);
    }

#endif
}

// Preprocess delta wrapper
void launch_preprocess_delta(
    const bf16* O,
    const bf16* dO,
    float* delta,
    cudaStream_t stream
) {
    constexpr int B_H = 128;
    dim3 grid(1, 1, 1);  // One block
    dim3 block(B_H, 1, 1);  // 128 threads
    
    preprocess_delta_kernel<<<grid, block, 0, stream>>>(O, dO, delta);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "preprocess_delta_kernel failed with error: %s\n", cudaGetErrorString(err));
        return;
    }
}

// C++ wrapper
void launch_test_mla_bwd(
    const bf16* q,
    const bf16* kv,
    const bf16* dO,
    const float* lse,
    const bf16* O,
    const int32_t* gIndices,
    int s_kv,
    int topk_length,
    bf16* q_out,
    bf16* kv_out,
    bf16* dO_out,
    float* P,
    float* dP,
    bf16* s,
    bf16* ds,
    const float* delta,
    float* dKV,
    bf16* sdO_t_out,
    float* dQ,
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
    cudaError_t err = cudaLaunchKernelEx(&config, test_mla_bwd_kernel, 
        q, kv, dO, lse, O, gIndices, s_kv, topk_length, q_out, kv_out, dO_out, P, dP, s, ds, delta, dKV, sdO_t_out, dQ);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaLaunchKernelEx failed with error: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("finish 17: Kernel launch completed\n");
}

// Python binding
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> mla_bwd_forward(
    torch::Tensor q, torch::Tensor kv, torch::Tensor dO, torch::Tensor lse, torch::Tensor O, torch::Tensor indices) {
    // Check inputs
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(kv.is_cuda(), "kv must be a CUDA tensor");
    TORCH_CHECK(dO.is_cuda(), "dO must be a CUDA tensor");
    TORCH_CHECK(lse.is_cuda(), "lse must be a CUDA tensor");
    TORCH_CHECK(O.is_cuda(), "O must be a CUDA tensor");
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(kv.dtype() == torch::kBFloat16, "kv must be bfloat16");
    TORCH_CHECK(dO.dtype() == torch::kBFloat16, "dO must be bfloat16");
    TORCH_CHECK(lse.dtype() == torch::kFloat32, "lse must be float32");
    TORCH_CHECK(O.dtype() == torch::kBFloat16, "O must be bfloat16");
    TORCH_CHECK(indices.dtype() == torch::kInt32, "indices must be int32");
    TORCH_CHECK(q.dim() == 2 && q.size(0) == B_H && q.size(1) == D_Q, 
                "q shape must be [128, 576]");
    TORCH_CHECK(kv.dim() == 2 && kv.size(0) == B_TOPK && kv.size(1) == D_K,
                "kv shape must be [64, 576]");
    TORCH_CHECK(dO.dim() == 2 && dO.size(0) == B_H && dO.size(1) == D_V,
                "dO shape must be [128, 512]");
    TORCH_CHECK(lse.dim() == 1 && lse.size(0) == B_H,
                "lse shape must be [128]");
    TORCH_CHECK(O.dim() == 2 && O.size(0) == B_H && O.size(1) == D_V,
                "O shape must be [128, 512]");
    TORCH_CHECK(indices.dim() == 1 && indices.size(0) == B_TOPK,
                "indices shape must be [64]");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(kv.is_contiguous(), "kv must be contiguous");
    TORCH_CHECK(dO.is_contiguous(), "dO must be contiguous");
    TORCH_CHECK(lse.is_contiguous(), "lse must be contiguous");
    TORCH_CHECK(O.is_contiguous(), "O must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    
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
    torch::Tensor s = torch::empty({B_H, B_TOPK}, options_bf16);
    torch::Tensor ds = torch::empty({B_H, B_TOPK}, options_bf16);
    torch::Tensor delta = torch::empty({B_H}, options_f32);
    // dKV initialized to zeros for atomic add accumulation
    torch::Tensor dKV = torch::zeros({B_TOPK, D_K}, options_f32);
    torch::Tensor sdO_t_out = torch::empty({D_V, B_H}, options_bf16);
    torch::Tensor dQ = torch::empty({B_H, D_Q}, options_f32);
    
    // Get data pointers
    const bf16* q_ptr = reinterpret_cast<const bf16*>(q.data_ptr<at::BFloat16>());
    const bf16* kv_ptr = reinterpret_cast<const bf16*>(kv.data_ptr<at::BFloat16>());
    const bf16* dO_ptr = reinterpret_cast<const bf16*>(dO.data_ptr<at::BFloat16>());
    const float* lse_ptr = lse.data_ptr<float>();
    const bf16* O_ptr = reinterpret_cast<const bf16*>(O.data_ptr<at::BFloat16>());
    const int32_t* indices_ptr = indices.data_ptr<int32_t>();
    bf16* q_out_ptr = reinterpret_cast<bf16*>(q_out.data_ptr<at::BFloat16>());
    bf16* kv_out_ptr = reinterpret_cast<bf16*>(kv_out.data_ptr<at::BFloat16>());
    bf16* dO_out_ptr = reinterpret_cast<bf16*>(dO_out.data_ptr<at::BFloat16>());
    float* P_ptr = P.data_ptr<float>();
    float* dP_ptr = dP.data_ptr<float>();
    bf16* s_ptr = reinterpret_cast<bf16*>(s.data_ptr<at::BFloat16>());
    bf16* ds_ptr = reinterpret_cast<bf16*>(ds.data_ptr<at::BFloat16>());
    float* delta_ptr = delta.data_ptr<float>();
    float* dKV_ptr = dKV.data_ptr<float>();
    bf16* sdO_t_out_ptr = reinterpret_cast<bf16*>(sdO_t_out.data_ptr<at::BFloat16>());
    float* dQ_ptr = dQ.data_ptr<float>();
    
    // Get CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Preprocess delta: compute delta = sum(O * dO, dim=-1)
    printf("finish -2: Computing delta...\n");
    launch_preprocess_delta(O_ptr, dO_ptr, delta_ptr, stream);
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "Delta preprocessing failed: ", cudaGetErrorString(err));
    }
    
    // Determine s_kv and topk_length from indices
    // For test purposes, s_kv is the maximum valid KV index + 1
    // topk_length is the actual number of valid indices (B_TOPK for this test)
    int s_kv = B_TOPK;  // Default to B_TOPK, can be adjusted based on actual usage
    int topk_length = B_TOPK;
    
    // Call kernel
    printf("finish -1: Starting kernel launch from Python binding\n");
    launch_test_mla_bwd(q_ptr, kv_ptr, dO_ptr, lse_ptr, O_ptr, indices_ptr, s_kv, topk_length,
        q_out_ptr, kv_out_ptr, dO_out_ptr, P_ptr, dP_ptr, s_ptr, ds_ptr, delta_ptr, dKV_ptr, sdO_t_out_ptr, dQ_ptr, stream);
    
    // Synchronize and wait for kernel completion
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
    printf("finish 18: Kernel execution completed and synchronized\n");
    
    return std::make_tuple(q_out, kv_out, dO_out, P, dP, s, ds, dKV, sdO_t_out, dQ);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mla_bwd", &mla_bwd_forward, 
          "Test mla_bwd kernel (CUDA). Returns (q_out, kv_out, dO_out, P, dP, s, ds, dKV, sdO_t_out, dQ)");
}

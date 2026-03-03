#include "test_dq_2sm_mma.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cluster_launch.hpp>
#include <kerutils/kerutils.cuh>

namespace test_operator::dq_2sm_mma {

using namespace cute;
using bf16 = cutlass::bfloat16_t;
using cutlass::arch::fence_barrier_init;
using cutlass::arch::fence_view_async_shared;

template <typename Params>
__global__ __launch_bounds__(NUM_THREADS, 1) void dq_2sm_mma_kernel(
    const bf16* __restrict__ ds,      // [B_H, B_TOPK] = [128, 64]
    const bf16* __restrict__ kv,      // [s_kv, D_K]
    const int32_t* __restrict__ indices,  // [B_TOPK] = [64]
    float* __restrict__ dQ_out,       // [B_H, D_Q] = [128, 576]
    bf16* __restrict__ cuda_smem_k_nope,  // [B_TOPK, 512] for verification
    bf16* __restrict__ cuda_smem_k_rope,  // [B_TOPK, D_ROPE] for verification
    bf16* __restrict__ cuda_smem_ds_t,    // [B_TOPK, B_H] for verification
    __grid_constant__ const Params params
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
    using namespace cute;

    extern __shared__ char smem_raw[];
    SharedMemory& smem = *reinterpret_cast<SharedMemory*>(smem_raw);

    const int cta_idx = blockIdx.x % 2;  // 0 or 1
    const int tid = threadIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();

    if (tid == 0) {
        printf("CTA%d: kernel started\n", cta_idx);
    }

    // Initialize barriers (only CTA0 thread 0)
    if (warp_idx == 0 && elect_one_sync()) {
        smem.bar_kv_nope_ready.init(1);
        smem.bar_kv_rope_ready.init(1);
        smem.bar_dq_nope_ready.init(1);
        smem.bar_dq_rope_ready.init(1);
        fence_barrier_init();
        if (cta_idx == 0) {
            printf("CTA0: barriers initialized\n");
        }
    }

    __syncthreads();
    cluster_sync();

    if (tid == 0) {
        printf("CTA%d: after cluster_sync 1\n", cta_idx);
    }

    // Step 1: Load ds to smem.ds_t
    // ds shape: [B_H, B_TOPK] = [128, 64]
    // Each CTA handles [B_H/2, B_TOPK] = [64, 64]
    // Load ds transposed to smem
    const int ds_rows = B_H / 2;  // 64
    const int ds_cols = B_TOPK;   // 64
    const int ds_elements = ds_rows * ds_cols;  // 4096
    const int ds_elements_per_thread = ds_elements / NUM_THREADS;  // 32
    Tensor sDS_t = make_tensor(make_smem_ptr(smem.ds_t.data()), SmemLayoutdSTransposed{});

    for (int i = 0; i < ds_elements_per_thread; ++i) {
        const int flat_idx = tid * ds_elements_per_thread + i;
        const int row = flat_idx / ds_cols;
        const int col = flat_idx % ds_cols;
        const int global_row = cta_idx * ds_rows + row;
        // Use layout-aware tensor accessor so swizzle/interleave mapping is respected.
        sDS_t(row, col) = ds[global_row * ds_cols + col];
    }

    fence_view_async_shared();
    __syncthreads();

    if (tid == 0) {
        printf("CTA%d: ds loaded\n", cta_idx);
    }

    // Step 1.1: Output ds_t from smem to global for verification.
    // Global layout: cuda_smem_ds_t[col, global_row] with shape [B_TOPK, B_H].
    if (cuda_smem_ds_t != nullptr) {
        const int elements_per_cta = ds_rows * ds_cols;  // 64 * 64
        const int elements_per_thread = (elements_per_cta + NUM_THREADS - 1) / NUM_THREADS;
        for (int i = 0; i < elements_per_thread; ++i) {
            const int flat_idx = tid * elements_per_thread + i;
            if (flat_idx < elements_per_cta) {
                const int local_row = flat_idx / ds_cols;  // 0-63
                const int local_col = flat_idx % ds_cols;  // 0-63
                const int global_row = cta_idx * ds_rows + local_row;  // 0-127
                cuda_smem_ds_t[local_col * B_H + global_row] = sDS_t(local_row, local_col);
            }
        }
    }

    __syncthreads();

    if (tid == 0) {
        printf("CTA%d: output ds_t done\n", cta_idx);
    }

    // Step 2: Load KV using TMA gather4
    // Each CTA loads [B_TOPK/2, D_K] = [32, 576]
    if (warp_idx == 0 && elect_one_sync()) {
        if (cta_idx == 0) {
            printf("CTA0: starting TMA load\n");
        }
        const int32_t* cta_indices = indices + cta_idx * (B_TOPK / 2);

        // Load KV NoPE part: [32, 512]
        for (int row = 0; row < B_TOPK / 2; row += 4) {
            int4 indices4;
            indices4.x = __ldg(cta_indices + row + 0);
            indices4.y = __ldg(cta_indices + row + 1);
            indices4.z = __ldg(cta_indices + row + 2);
            indices4.w = __ldg(cta_indices + row + 3);

            // Load 8 tiles of 64 elements each (512 total)
            for (int col_tile = 0; col_tile < 8; ++col_tile) {
                kerutils::tma_gather4_cta_group_2<true>(
                    &(params.tensor_map_kv),
                    smem.bar_kv_nope_ready,
                    // SmemLayoutKNoPE is tile-major on the 64-col axis:
                    // row-group stride = 64, col-tile stride = (B_TOPK/2)*64.
                    smem.k_nope.data() + row * 64 + col_tile * ((B_TOPK / 2) * 64),
                    // smem.k_nope.data() + row * 512 + col_tile * 64,
                    col_tile * 64,
                    indices4,
                    (int64_t)TMA::CacheHintSm90::EVICT_LAST
                );
            }
        }

        // Only CTA0 waits for TMA completion
        if (cta_idx == 0) {
            smem.bar_kv_nope_ready.arrive_and_expect_tx(B_TOPK * 512 * sizeof(bf16));
            smem.bar_kv_nope_ready.wait(0);
            ku::tcgen05_after_thread_sync();
            printf("CTA0: k_nope loaded\n");
        }

        // Load KV RoPE part: [32, 64]
        // Each CTA loads full 64 columns of RoPE data in one TMA call
        for (int row = 0; row < B_TOPK / 2; row += 4) {
            int4 indices4;
            indices4.x = __ldg(cta_indices + row + 0);
            indices4.y = __ldg(cta_indices + row + 1);
            indices4.z = __ldg(cta_indices + row + 2);
            indices4.w = __ldg(cta_indices + row + 3);

            kerutils::tma_gather4_cta_group_2<true>(
                &(params.tensor_map_kv_rope),
                smem.bar_kv_rope_ready,
                smem.k_rope.data() + row * D_ROPE,
                512,  // Start from column 512 (RoPE part)
                indices4,
                (int64_t)TMA::CacheHintSm90::EVICT_LAST
            );
        }

        // Only CTA0 waits for TMA completion
        if (cta_idx == 0) {
            smem.bar_kv_rope_ready.arrive_and_expect_tx(B_TOPK * D_ROPE * sizeof(bf16));
            smem.bar_kv_rope_ready.wait(0);
            ku::tcgen05_after_thread_sync();
            printf("CTA0: k_rope loaded\n");
        }
    }

    __syncthreads();
    cluster_sync();

    if (tid == 0) {
        printf("CTA%d: after TMA load cluster_sync\n", cta_idx);
    }

    // Step 2.2 + 2.5: Direct relayout for dQ MMA (no explicit exchange writeback to k_nope/k_rope).
    // Data is read from local/peer SMEM with the same logical mapping as previous exchange step,
    // then written directly to k_nope_dq/k_rope_dq.
    // k_nope logical transform:
    //   [32,512] -> split A:[32,0:256], B:[32,256:512] -> stack [64,256] -> transpose [256,64]
    // k_rope logical transform:
    //   [32, 64] -> split A:[32,0:32],  B:[32,32:64]  -> stack [64, 32] -> transpose [32,64]
    {
        Tensor sK_nope_local = make_tensor(make_smem_ptr(smem.k_nope.data()), SmemLayoutKNoPE{});
        Tensor sK_nope_dst = make_tensor(make_smem_ptr(smem.k_nope_dq.data()), SmemLayoutKCalcDQNoPE{});
        bf16* peer_k_nope = kerutils::get_peer_addr(smem.k_nope.data());
        Tensor sK_nope_peer = make_tensor(make_smem_ptr(peer_k_nope), SmemLayoutKNoPE{});

        Tensor sK_rope_local = make_tensor(make_smem_ptr(smem.k_rope.data()), SmemLayoutKRoPE{});
        Tensor sK_rope_dst = make_tensor(make_smem_ptr(smem.k_rope_dq.data()), SmemLayoutKCalcDQRoPE{});
        bf16* peer_k_rope = kerutils::get_peer_addr(smem.k_rope.data());
        Tensor sK_rope_peer = make_tensor(make_smem_ptr(peer_k_rope), SmemLayoutKRoPE{});

        constexpr int N_COLS = 256;
        constexpr int K_ROWS = B_TOPK;  // 64
        constexpr int HALF_ROWS = B_TOPK / 2;  // 32
        const int nope_elements = N_COLS * K_ROWS;
        const int nope_elements_per_thread = (nope_elements + NUM_THREADS - 1) / NUM_THREADS;

        for (int i = 0; i < nope_elements_per_thread; ++i) {
            const int flat_idx = tid * nope_elements_per_thread + i;
            if (flat_idx < nope_elements) {
                const int n = flat_idx / K_ROWS;  // 0..255
                const int k = flat_idx % K_ROWS;  // 0..63
                const int src_row = k % HALF_ROWS;
                const int src_col = n + (k / HALF_ROWS) * N_COLS;
                bf16 v;
                if (cta_idx == 0) {
                    v = (src_col < 256)
                        ? sK_nope_local(src_row, src_col)
                        : sK_nope_peer(src_row, src_col - 256);
                } else {
                    v = (src_col < 256)
                        ? sK_nope_peer(src_row, src_col + 256)
                        : sK_nope_local(src_row, src_col);
                }
                sK_nope_dst(n, k) = v;
            }
        }

        constexpr int N_COLS_ROPE = D_ROPE / 2;  // 32
        constexpr int K_ROWS_ROPE = B_TOPK;      // 64
        constexpr int HALF_ROWS_ROPE = B_TOPK / 2;  // 32
        const int rope_elements = N_COLS_ROPE * K_ROWS_ROPE;
        const int rope_elements_per_thread = (rope_elements + NUM_THREADS - 1) / NUM_THREADS;

        for (int i = 0; i < rope_elements_per_thread; ++i) {
            const int flat_idx = tid * rope_elements_per_thread + i;
            if (flat_idx < rope_elements) {
                const int n = flat_idx / K_ROWS_ROPE;  // 0..31
                const int k = flat_idx % K_ROWS_ROPE;  // 0..63
                const int src_row = k % HALF_ROWS_ROPE;
                const int src_col = n + (k / HALF_ROWS_ROPE) * N_COLS_ROPE;
                bf16 v;
                if (cta_idx == 0) {
                    v = (src_col < 32)
                        ? sK_rope_local(src_row, src_col)
                        : sK_rope_peer(src_row, src_col - 32);
                } else {
                    v = (src_col < 32)
                        ? sK_rope_peer(src_row, src_col + 32)
                        : sK_rope_local(src_row, src_col);
                }
                sK_rope_dst(n, k) = v;
            }
        }

        // Step 2.4: Output cuda_kv for verification with the same logical exchange mapping.
        if (cuda_smem_k_nope != nullptr) {
            const int elements_per_cta = (B_TOPK / 2) * 512;
            const int elements_per_thread = (elements_per_cta + NUM_THREADS - 1) / NUM_THREADS;

            for (int i = 0; i < elements_per_thread; ++i) {
                const int flat_idx = tid * elements_per_thread + i;
                if (flat_idx < elements_per_cta) {
                    const int local_row = flat_idx / 512;  // 0-31
                    const int local_col = flat_idx % 512;  // 0-511

                    int global_row, global_col;
                    if (local_col < 256) {
                        global_row = local_row;
                        global_col = cta_idx * 256 + local_col;
                    } else {
                        global_row = (B_TOPK / 2) + local_row;
                        global_col = cta_idx * 256 + (local_col - 256);
                    }

                    bf16 v;
                    if (cta_idx == 0) {
                        v = (local_col < 256)
                            ? sK_nope_local(local_row, local_col)
                            : sK_nope_peer(local_row, local_col - 256);
                    } else {
                        v = (local_col < 256)
                            ? sK_nope_peer(local_row, local_col + 256)
                            : sK_nope_local(local_row, local_col);
                    }
                    cuda_smem_k_nope[global_row * 512 + global_col] = v;
                }
            }
        }

        if (cuda_smem_k_rope != nullptr) {
            const int elements_per_cta = (B_TOPK / 2) * D_ROPE;
            const int elements_per_thread = (elements_per_cta + NUM_THREADS - 1) / NUM_THREADS;

            for (int i = 0; i < elements_per_thread; ++i) {
                const int flat_idx = tid * elements_per_thread + i;
                if (flat_idx < elements_per_cta) {
                    const int local_row = flat_idx / D_ROPE;  // 0-31
                    const int local_col = flat_idx % D_ROPE;  // 0-63

                    int global_row, global_col;
                    if (local_col < 32) {
                        global_row = local_row;
                        global_col = cta_idx * 32 + local_col;
                    } else {
                        global_row = (B_TOPK / 2) + local_row;
                        global_col = cta_idx * 32 + (local_col - 32);
                    }

                    bf16 v;
                    if (cta_idx == 0) {
                        v = (local_col < 32)
                            ? sK_rope_local(local_row, local_col)
                            : sK_rope_peer(local_row, local_col - 32);
                    } else {
                        v = (local_col < 32)
                            ? sK_rope_peer(local_row, local_col + 32)
                            : sK_rope_local(local_row, local_col);
                    }
                    cuda_smem_k_rope[global_row * D_ROPE + global_col] = v;
                }
            }
        }
    }

    fence_view_async_shared();
    __syncthreads();

    if (tid == 0) {
        printf("CTA%d: k_nope/k_rope relayout for dQ done\n", cta_idx);
    }

    // Step 2.3: Compute dQ using MMA
    // Allocate TMEM (warp 0 only)
    if (warp_idx == 0) {
        TMEM::Allocator2Sm().allocate(512, smem.tmem_start_addr.data());
        TMEM::Allocator2Sm().release_allocation_lock();
        if (cta_idx == 0) {
            printf("CTA0: TMEM allocated\n");
        }
    }
    __syncthreads();

    // Compute dQ (2x1SM launch discipline: only CTA0 launches)
    if (cta_idx == 0 && warp_idx == 0 && elect_one_sync()) {
        if (cta_idx == 0) {
            printf("CTA0: starting MMA computation\n");
        }

        // Prepare tensors
        Tensor sDS_t = make_tensor(make_smem_ptr(smem.ds_t.data()), SmemLayoutdSTransposed{});
        Tensor sK_nope_t = make_tensor(make_smem_ptr(smem.k_nope_dq.data()), SmemLayoutKCalcDQNoPE{});
        Tensor sK_rope_t = make_tensor(make_smem_ptr(smem.k_rope_dq.data()), SmemLayoutKCalcDQRoPE{});

        // Allocate TMEM for dQ
        TiledMMA_dQ_2cta tiled_mma_dQ{};
        TiledMMA_dQ_RoPE_2cta tiled_mma_dQ_RoPE{};

        Tensor tdQ = partition_fragment_C(tiled_mma_dQ, Shape<Int<B_H>, Int<256>>{});
        tdQ.data().get() = tmem_cols::dQ;

        Tensor tdQ_RoPE = partition_fragment_C(tiled_mma_dQ_RoPE, Shape<Int<B_H>, Int<D_ROPE>>{});
        tdQ_RoPE.data().get() = tmem_cols::dQ_RoPE;

        // Compute dQ = dS^T @ K
        ku::utcmma_ss(tiled_mma_dQ, sDS_t, sK_nope_t, tdQ, true);
        ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t, sK_rope_t, tdQ_RoPE, true);

        if (cta_idx == 0) {
            printf("CTA0: MMA computation done\n");
        }
    }

    ku::tcgen05_after_thread_sync();
    __syncthreads();
    cluster_sync();

    if (tid == 0) {
        printf("CTA%d: before reading TMEM\n", cta_idx);
    }

    // Read dQ from TMEM and write to global memory (all threads participate)
    {
        constexpr int DQ_ROWS_PER_CTA = B_H / 2;          // 64
        constexpr int NOPE_FLOATS_PER_HALF = 256 / 2;     // 128
        constexpr int NOPE_CHUNKS = 8;
        constexpr int NOPE_CHUNK_FLOATS = NOPE_FLOATS_PER_HALF / NOPE_CHUNKS;  // 16
        constexpr int NOPE_CHUNK_FLOAT2 = NOPE_CHUNK_FLOATS / 2;                // 8
        constexpr int ROPE_FLOAT2_PER_ROW = D_ROPE / 2 / 2;                     // 16

        const uint32_t tmem_base = smem.tmem_start_addr.data()[0];
        const int row_in_cta = tid % DQ_ROWS_PER_CTA;
        const int col_half = tid / DQ_ROWS_PER_CTA;  // 0 or 1
        const int global_row = cta_idx * DQ_ROWS_PER_CTA + row_in_cta;

        const uint32_t tmem_addr_dq0 = tmem_base + (row_in_cta << 16) + tmem_cols::dQ;
        const uint32_t tmem_addr_dq1 = tmem_base + (row_in_cta << 16) + (tmem_cols::dQ + 128);

        // dQ NoPE part0:
        //   col_half==0 -> cols [0,127]
        //   col_half==1 -> cols [256,383]
        for (int chunk = 0; chunk < NOPE_CHUNKS; ++chunk) {
            const int chunk_col_base = chunk * NOPE_CHUNK_FLOATS;
            float2 dq_chunk[NOPE_CHUNK_FLOAT2];
            ku::tmem_ld_32dp32bNx<NOPE_CHUNK_FLOATS>(tmem_addr_dq0 + chunk_col_base, dq_chunk);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();

            for (int i = 0; i < NOPE_CHUNK_FLOAT2; ++i) {
                const int part0_base = (col_half == 0) ? 0 : 256;
                const int col = part0_base + chunk_col_base + i * 2;
                dQ_out[global_row * D_Q + col] = dq_chunk[i].x;
                dQ_out[global_row * D_Q + col + 1] = dq_chunk[i].y;
            }
        }

        // dQ NoPE part1:
        //   col_half==0 -> cols [128,255]
        //   col_half==1 -> cols [384,511]
        for (int chunk = 0; chunk < NOPE_CHUNKS; ++chunk) {
            const int chunk_col_base = chunk * NOPE_CHUNK_FLOATS;
            float2 dq_chunk[NOPE_CHUNK_FLOAT2];
            ku::tmem_ld_32dp32bNx<NOPE_CHUNK_FLOATS>(tmem_addr_dq1 + chunk_col_base, dq_chunk);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();

            for (int i = 0; i < NOPE_CHUNK_FLOAT2; ++i) {
                const int part1_base = (col_half == 0) ? 128 : 384;
                const int col = part1_base + chunk_col_base + i * 2;
                dQ_out[global_row * D_Q + col] = dq_chunk[i].x;
                dQ_out[global_row * D_Q + col + 1] = dq_chunk[i].y;
            }
        }

        // dQ RoPE: cols [512, 575]
        const uint32_t tmem_addr_rope = tmem_base + (row_in_cta << 16) + tmem_cols::dQ_RoPE;
        float2 dq_rope[ROPE_FLOAT2_PER_ROW];
        ku::tmem_ld_32dp32bNx<D_ROPE / 2>(tmem_addr_rope, dq_rope);
        cutlass::arch::fence_view_async_tmem_load();
        ku::tcgen05_before_thread_sync();

        for (int i = 0; i < ROPE_FLOAT2_PER_ROW; ++i) {
            const int col = 512 + col_half * (D_ROPE / 2) + i * 2;
            dQ_out[global_row * D_Q + col] = dq_rope[i].x;
            dQ_out[global_row * D_Q + col + 1] = dq_rope[i].y;
        }
    }

    __syncthreads();

    if (tid == 0) {
        printf("CTA%d: TMEM read done\n", cta_idx);
    }

    // Free TMEM
    if (warp_idx == 0 && elect_one_sync()) {
        TMEM::Allocator2Sm().free(smem.tmem_start_addr.data()[0], 512);
        if (cta_idx == 0) {
            printf("CTA0: TMEM freed\n");
        }
    }

    if (tid == 0) {
        printf("CTA%d: kernel finished\n", cta_idx);
    }
#endif
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> run_dq_2sm_mma(
    torch::Tensor ds,      // [128, 64], bf16
    torch::Tensor kv,      // [s_kv, 576], bf16
    torch::Tensor indices  // [64], int32/int64
) {
    printf("Host: run_dq_2sm_mma started\n");

    TORCH_CHECK(ds.dtype() == torch::kBFloat16);
    TORCH_CHECK(kv.dtype() == torch::kBFloat16);
    TORCH_CHECK(ds.size(0) == B_H && ds.size(1) == B_TOPK);
    TORCH_CHECK(kv.size(1) == D_K);
    TORCH_CHECK(indices.size(0) == B_TOPK);

    printf("Host: input checks passed\n");

    auto dQ = torch::zeros({B_H, D_Q}, torch::dtype(torch::kFloat32).device(ds.device()));
    auto cuda_kv_nope = torch::zeros({B_TOPK, 512}, torch::dtype(torch::kBFloat16).device(ds.device()));
    auto cuda_kv_rope = torch::zeros({B_TOPK, D_ROPE}, torch::dtype(torch::kBFloat16).device(ds.device()));
    auto cuda_ds_t = torch::zeros({B_TOPK, B_H}, torch::dtype(torch::kBFloat16).device(ds.device()));

    // Convert indices to int32 if needed
    auto indices_i32 = indices.to(torch::kInt32);

    // Create tensor maps for TMA
    CUtensorMap tensor_map_kv, tensor_map_kv_rope;

    // Tensor map for k_nope (box_size = 64)
    {
        uint64_t size[2] = {(uint64_t)D_K, (uint64_t)kv.size(0)};
        uint64_t stride[1] = {(uint64_t)kv.stride(0) * sizeof(bf16)};
        uint32_t box_size[2] = {64, 1};
        uint32_t elem_stride[2] = {1, 1};

        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            (bf16*)kv.data_ptr(),
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    // Tensor map for k_rope (box_size = 64, same as k_nope)
    {
        uint64_t size[2] = {(uint64_t)D_K, (uint64_t)kv.size(0)};
        uint64_t stride[1] = {(uint64_t)kv.stride(0) * sizeof(bf16)};
        uint32_t box_size[2] = {64, 1};  // Changed from 32 to 64
        uint32_t elem_stride[2] = {1, 1};

        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv_rope,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            (bf16*)kv.data_ptr(),
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    printf("Host: tensor maps created\n");

    KernelParams params{tensor_map_kv, tensor_map_kv_rope};

    auto kernel = &dq_2sm_mma_kernel<KernelParams>;
    dim3 grid(2, 1, 1);
    dim3 block(NUM_THREADS, 1, 1);

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE);

    // Get pointers first
    bf16* ds_ptr = (bf16*)ds.data_ptr();
    bf16* kv_ptr = (bf16*)kv.data_ptr();
    int32_t* indices_ptr = (int32_t*)indices_i32.data_ptr();
    float* dQ_ptr = (float*)dQ.data_ptr();
    bf16* cuda_kv_nope_ptr = (bf16*)cuda_kv_nope.data_ptr();
    bf16* cuda_kv_rope_ptr = (bf16*)cuda_kv_rope.data_ptr();
    bf16* cuda_ds_t_ptr = (bf16*)cuda_ds_t.data_ptr();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

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

    cudaError_t err = cudaLaunchKernelEx(&config, kernel,
        ds_ptr, kv_ptr, indices_ptr, dQ_ptr, cuda_kv_nope_ptr, cuda_kv_rope_ptr, cuda_ds_t_ptr, params);

    printf("Host: kernel launched\n");

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaLaunchKernelEx failed with error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    printf("Host: kernel finished\n");

    return std::make_tuple(dQ, cuda_kv_nope, cuda_kv_rope, cuda_ds_t);
}

}  // namespace test_operator::dq_2sm_mma

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_dq_2sm_mma", &test_operator::dq_2sm_mma::run_dq_2sm_mma, "DQ 2SM MMA Test");
}

#include "test_dq_2sm_mma.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cluster_launch.hpp>
#include <kerutils/kerutils.cuh>

namespace test_operator::dq_2sm_mma {

using namespace cute;
using bf16 = cutlass::bfloat16_t;

template <typename Params>
__global__ __launch_bounds__(NUM_THREADS, 1) void dq_2sm_mma_kernel(
    const bf16* __restrict__ ds,      // [B_H, B_TOPK] = [128, 64]
    const bf16* __restrict__ kv,      // [s_kv, D_K]
    const int32_t* __restrict__ indices,  // [B_TOPK] = [64]
    float* __restrict__ dQ_out,       // [B_H, D_Q] = [128, 576]
    bf16* __restrict__ cuda_smem_k_nope,  // [B_TOPK, 512] for verification
    bf16* __restrict__ cuda_smem_k_rope,  // [B_TOPK, D_ROPE] for verification
    __grid_constant__ const Params params
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
    using namespace cute;

    extern __shared__ char smem_raw[];
    SharedMemory& smem = *reinterpret_cast<SharedMemory*>(smem_raw);

    const int cta_idx = blockIdx.x % 2;  // 0 or 1
    const int tid = threadIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();

    // Initialize barriers (only CTA0 thread 0)
    if (tid == 0 && cta_idx == 0) {
        smem.bar_kv_nope_ready.init(1);
        smem.bar_kv_rope_ready.init(1);
        smem.bar_dq_nope_ready.init(1);
        smem.bar_dq_rope_ready.init(1);
        fence_barrier_init();
    }

    __syncthreads();
    cluster_sync();

    // Step 1: Load ds to smem.ds_t
    // ds shape: [B_H, B_TOPK] = [128, 64]
    // Each CTA handles [B_H/2, B_TOPK] = [64, 64]
    // Load ds transposed to smem
    const int ds_rows = B_H / 2;  // 64
    const int ds_cols = B_TOPK;   // 64
    const int ds_elements = ds_rows * ds_cols;  // 4096
    const int ds_elements_per_thread = ds_elements / NUM_THREADS;  // 32

    for (int i = 0; i < ds_elements_per_thread; ++i) {
        const int flat_idx = tid * ds_elements_per_thread + i;
        const int row = flat_idx / ds_cols;
        const int col = flat_idx % ds_cols;
        const int global_row = cta_idx * ds_rows + row;
        // Store transposed: smem[col][row]
        smem.ds_t.data()[col * ds_rows + row] = ds[global_row * ds_cols + col];
    }

    fence_view_async_shared();
    __syncthreads();

    // Step 2: Load KV using TMA gather4
    // Each CTA loads [B_TOPK/2, D_K] = [32, 576]
    if (warp_idx == 0 && elect_one_sync()) {
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
                    smem.k_nope.data() + row * 512 + col_tile * 64,
                    col_tile * 64,
                    indices4,
                    (int64_t)TMA::CacheHintSm90::EVICT_LAST
                );
            }
        }

        smem.bar_kv_nope_ready.arrive_and_expect_tx((B_TOPK / 2) * 512 * sizeof(bf16));
        smem.bar_kv_nope_ready.wait(0);
        ku::tcgen05_after_thread_sync();

        // Load KV RoPE part: [32, 64]
        for (int row = 0; row < B_TOPK / 2; row += 4) {
            int4 indices4;
            indices4.x = __ldg(cta_indices + row + 0);
            indices4.y = __ldg(cta_indices + row + 1);
            indices4.z = __ldg(cta_indices + row + 2);
            indices4.w = __ldg(cta_indices + row + 3);

            kerutils::tma_gather4_cta_group_2<true>(
                &(params.tensor_map_kv_rope32),
                smem.bar_kv_rope_ready,
                smem.k_rope.data() + row * 64,
                512,  // offset to RoPE part in global memory
                indices4,
                (int64_t)TMA::CacheHintSm90::EVICT_LAST
            );
        }

        smem.bar_kv_rope_ready.arrive_and_expect_tx((B_TOPK / 2) * D_ROPE * sizeof(bf16));
        smem.bar_kv_rope_ready.wait(0);
        ku::tcgen05_after_thread_sync();
    }

    __syncthreads();
    cluster_sync();

    // Step 2.2: Exchange KV NoPE data between CTA0 and CTA1
    // CTA0 exchanges [32, 256:512] with CTA1's [32, 0:256]
    // Total: 32 * 256 = 8192 bf16 elements
    // Each thread handles: 8192 / 128 = 64 bf16 elements

    constexpr int EXCHANGE_ELEMENTS = 32 * 256;
    constexpr int ELEMENTS_PER_THREAD = EXCHANGE_ELEMENTS / NUM_THREADS;

    bf16 exchange_buffer[ELEMENTS_PER_THREAD];

    // Read from peer CTA's smem
    bf16* peer_k_nope = kerutils::get_peer_addr(smem.k_nope.data());

    if (cta_idx == 0) {
        // CTA0: read peer's [32, 0:256]
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            const int flat_idx = tid * ELEMENTS_PER_THREAD + i;
            const int row = flat_idx / 256;
            const int col = flat_idx % 256;
            exchange_buffer[i] = peer_k_nope[row * 512 + col];
        }
    } else {
        // CTA1: read peer's [32, 256:512]
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            const int flat_idx = tid * ELEMENTS_PER_THREAD + i;
            const int row = flat_idx / 256;
            const int col = flat_idx % 256;
            exchange_buffer[i] = peer_k_nope[row * 512 + 256 + col];
        }
    }

    cluster_sync();

    // Write to local smem
    if (cta_idx == 0) {
        // CTA0: write to [32, 256:512]
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            const int flat_idx = tid * ELEMENTS_PER_THREAD + i;
            const int row = flat_idx / 256;
            const int col = flat_idx % 256;
            smem.k_nope.data()[row * 512 + 256 + col] = exchange_buffer[i];
        }
    } else {
        // CTA1: write to [32, 0:256]
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            const int flat_idx = tid * ELEMENTS_PER_THREAD + i;
            const int row = flat_idx / 256;
            const int col = flat_idx % 256;
            smem.k_nope.data()[row * 512 + col] = exchange_buffer[i];
        }
    }

    fence_view_async_shared();
    __syncthreads();
    cluster_sync();

    // Step 2.4: Output cuda_kv for verification
    // After exchange, each CTA's smem contains the data it needs to output:
    // CTA0: smem[32, 0:256] -> global[0:32, 0:256], smem[32, 256:512] -> global[32:64, 0:256]
    // CTA1: smem[32, 0:256] -> global[0:32, 256:512], smem[32, 256:512] -> global[32:64, 256:512]
    if (cuda_smem_k_nope != nullptr) {
        // Each CTA outputs 32*512 = 16384 elements
        const int elements_per_cta = (B_TOPK / 2) * 512;
        const int elements_per_thread = (elements_per_cta + NUM_THREADS - 1) / NUM_THREADS;

        for (int i = 0; i < elements_per_thread; ++i) {
            const int flat_idx = tid * elements_per_thread + i;
            if (flat_idx < elements_per_cta) {
                const int local_row = flat_idx / 512;  // 0-31
                const int local_col = flat_idx % 512;  // 0-511

                // Determine global position
                int global_row, global_col;
                if (local_col < 256) {
                    // First half columns: output to rows [0:32]
                    global_row = local_row;
                    global_col = cta_idx * 256 + local_col;
                } else {
                    // Second half columns: output to rows [32:64]
                    global_row = (B_TOPK / 2) + local_row;
                    global_col = cta_idx * 256 + (local_col - 256);
                }

                cuda_smem_k_nope[global_row * 512 + global_col] = smem.k_nope.data()[flat_idx];
            }
        }
    }

    if (cuda_smem_k_rope != nullptr) {
        // Each CTA outputs 32*64 = 2048 elements
        const int elements_per_cta = (B_TOPK / 2) * D_ROPE;
        const int elements_per_thread = (elements_per_cta + NUM_THREADS - 1) / NUM_THREADS;

        for (int i = 0; i < elements_per_thread; ++i) {
            const int flat_idx = tid * elements_per_thread + i;
            if (flat_idx < elements_per_cta) {
                const int local_row = flat_idx / D_ROPE;  // 0-31
                const int local_col = flat_idx % D_ROPE;  // 0-63

                // Both CTAs output to the same columns, different rows
                // CTA0: rows [0:32], CTA1: rows [32:64]
                const int global_row = cta_idx * (B_TOPK / 2) + local_row;
                cuda_smem_k_rope[global_row * D_ROPE + local_col] = smem.k_rope.data()[flat_idx];
            }
        }
    }

    __syncthreads();

    // Step 2.3: Compute dQ using MMA
    // Allocate TMEM (warp 0 only)
    if (warp_idx == 0) {
        TMEM::Allocator2Sm().allocate(512, smem.tmem_start_addr.data());
        TMEM::Allocator2Sm().release_allocation_lock();
    }
    __syncthreads();

    // Compute dQ (elected warp only)
    if (warp_idx == 0 && elect_one_sync()) {
        cutlass::arch::warpgroup_reg_alloc<168>();

        // Prepare tensors
        Tensor sDS_t = make_tensor(make_smem_ptr(smem.ds_t.data()), SmemLayoutdSTransposed{});
        Tensor sK_nope_t = make_tensor(make_smem_ptr(smem.k_nope.data()), SmemLayoutKCalcDQNoPE{});
        Tensor sK_rope_t = make_tensor(make_smem_ptr(smem.k_rope.data()), SmemLayoutKCalcDQRoPE{});

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

        ku::tcgen05_after_thread_sync();
    }

    __syncthreads();

    // Read dQ from TMEM and write to global memory (all threads participate)
    {
        const uint32_t tmem_base = smem.tmem_start_addr.data()[0];
        const int row_in_cta = tid % (B_H / 2);
        const int global_row = cta_idx * (B_H / 2) + row_in_cta;

        // Read dQ NoPE part (512 elements)
        for (int col_chunk = 0; col_chunk < 2; ++col_chunk) {
            const uint32_t tmem_addr = tmem_base + (row_in_cta << 16) + tmem_cols::dQ + col_chunk * 128;
            float2 dq_data[16];
            ku::tmem_ld_32dp32bNx<32>(tmem_addr, dq_data);
            cutlass::arch::fence_view_async_tmem_load();

            for (int i = 0; i < 16; ++i) {
                dQ_out[global_row * D_Q + col_chunk * 256 + i * 2] = dq_data[i].x;
                dQ_out[global_row * D_Q + col_chunk * 256 + i * 2 + 1] = dq_data[i].y;
            }
        }

        // Read dQ RoPE part (64 elements)
        const uint32_t tmem_addr_rope = tmem_base + (row_in_cta << 16) + tmem_cols::dQ_RoPE;
        float2 dq_rope_data[16];
        ku::tmem_ld_32dp32bNx<32>(tmem_addr_rope, dq_rope_data);
        cutlass::arch::fence_view_async_tmem_load();

        for (int i = 0; i < 16; ++i) {
            dQ_out[global_row * D_Q + 512 + i * 2] = dq_rope_data[i].x;
            dQ_out[global_row * D_Q + 512 + i * 2 + 1] = dq_rope_data[i].y;
        }
    }

    __syncthreads();

    // Free TMEM
    if (warp_idx == 0 && elect_one_sync()) {
        TMEM::Allocator2Sm().free(smem.tmem_start_addr.data()[0], 512);
    }
#endif
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> run_dq_2sm_mma(
    torch::Tensor ds,      // [128, 64], bf16
    torch::Tensor kv,      // [s_kv, 576], bf16
    torch::Tensor indices  // [64], int32/int64
) {
    TORCH_CHECK(ds.dtype() == torch::kBFloat16);
    TORCH_CHECK(kv.dtype() == torch::kBFloat16);
    TORCH_CHECK(ds.size(0) == B_H && ds.size(1) == B_TOPK);
    TORCH_CHECK(kv.size(1) == D_K);
    TORCH_CHECK(indices.size(0) == B_TOPK);

    auto dQ = torch::zeros({B_H, D_Q}, torch::dtype(torch::kFloat32).device(ds.device()));
    auto cuda_kv_nope = torch::zeros({B_TOPK, 512}, torch::dtype(torch::kBFloat16).device(ds.device()));
    auto cuda_kv_rope = torch::zeros({B_TOPK, D_ROPE}, torch::dtype(torch::kBFloat16).device(ds.device()));

    // Convert indices to int32 if needed
    auto indices_i32 = indices.to(torch::kInt32);

    // Create tensor maps for TMA
    CUtensorMap tensor_map_kv, tensor_map_kv_rope32;

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

    // Tensor map for RoPE part (same as above, just different column offset)
    tensor_map_kv_rope32 = tensor_map_kv;

    KernelParams params{tensor_map_kv, tensor_map_kv_rope32};

    auto kernel = &dq_2sm_mma_kernel<KernelParams>;
    dim3 grid(2, 1, 1);
    dim3 block(NUM_THREADS, 1, 1);

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE);

    cudaLaunchConfig_t config;
    memset(&config, 0, sizeof(config));
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = SMEM_SIZE;
    config.stream = at::cuda::getCurrentCUDAStream();

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    cudaLaunchKernelEx(
        &config,
        (void*)kernel,
        (bf16*)ds.data_ptr(),
        (bf16*)kv.data_ptr(),
        (int32_t*)indices_i32.data_ptr(),
        (float*)dQ.data_ptr(),
        (bf16*)cuda_kv_nope.data_ptr(),
        (bf16*)cuda_kv_rope.data_ptr(),
        params
    );

    return std::make_tuple(dQ, cuda_kv_nope, cuda_kv_rope);
}

}  // namespace test_operator::dq_2sm_mma

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_dq_2sm_mma", &test_operator::dq_2sm_mma::run_dq_2sm_mma, "DQ 2SM MMA Test");
}

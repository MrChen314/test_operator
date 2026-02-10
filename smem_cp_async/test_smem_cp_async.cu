#include "test_smem_cp_async.cuh"

#include <cstring>
#include <cstdio>

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

using namespace test_operator::smem_cp_async;

#ifndef SMEM_CP_ASYNC_DEBUG
#define SMEM_CP_ASYNC_DEBUG 1
#endif

CUTE_DEVICE void store_c_from_tmem(uint32_t tmem_base, int row_offset, float* __restrict__ C) {
    constexpr int FLOAT2_PER_ROW = N_TOTAL / 4;  // 64 float2
    const int tid = threadIdx.x;
    if (tid >= WARP_GROUP_THREADS) {
        return;
    }
    const int logical_row = tid % M_PER_CTA;
    const int col_half = tid / M_PER_CTA;  // 0 or 1

    const uint32_t tmem_addr = tmem_base + (logical_row << 16) + tmem_cols::c;
    float2 row_data[FLOAT2_PER_ROW];
    ku::tmem_ld_32dp32bNx<N_TOTAL / 2>(tmem_addr, row_data);
    cutlass::arch::fence_view_async_tmem_load();
    ku::tcgen05_before_thread_sync();

    #pragma unroll
    for (int i = 0; i < FLOAT2_PER_ROW; ++i) {
        const int col = col_half * (N_TOTAL / 2) + i * 2;
        C[(row_offset + logical_row) * N_TOTAL + col] = row_data[i].x;
        C[(row_offset + logical_row) * N_TOTAL + col + 1] = row_data[i].y;
    }
}

__global__ void test_smem_cp_async_kernel(
    const bf16* __restrict__ A,
    const bf16* __restrict__ B,
    float* __restrict__ C
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
    using namespace cute;

    extern __shared__ char smem_raw[];
    SharedMemory& smem = *reinterpret_cast<SharedMemory*>(smem_raw);

    const int cta_idx = blockIdx.x % 2;
    const int tid = threadIdx.x;
    const int warp_idx = tid / 32;
    const int warpgroup_idx = tid / WARP_GROUP_THREADS;
    const int idx_in_warpgroup = tid % WARP_GROUP_THREADS;
    const bool is_wg0 = (warpgroup_idx == 0);
    const bool is_wg1 = (warpgroup_idx == 1);

    Tensor sA = make_tensor(make_smem_ptr(smem.sA.data()), SmemLayoutA{});
    Tensor sB_local = make_tensor(make_smem_ptr(smem.sB_local.data()), SmemLayoutB{});
    Tensor sB_peer = make_tensor(make_smem_ptr(smem.sB_peer.data()), SmemLayoutB{});

    const int a_row_offset = cta_idx * M_PER_CTA;
    constexpr int A_ELEMENTS = M_PER_CTA * K_TOTAL;
    constexpr int B_ELEMENTS_FULL = N_TOTAL * K_TOTAL;
    constexpr int B_STORAGE_ELEMENTS = cosize_v<SmemLayoutB>;
    constexpr int B_HALF_ELEMENTS = K_PER_CTA * N_TOTAL;
    constexpr int A_ELEMENTS_PER_WG_THREAD = (A_ELEMENTS + WARP_GROUP_THREADS - 1) / WARP_GROUP_THREADS;
    constexpr int B_ELEMENTS_PER_WG_THREAD = (B_ELEMENTS_FULL + WARP_GROUP_THREADS - 1) / WARP_GROUP_THREADS;
    constexpr int B_HALF_ELEMENTS_PER_WG_THREAD = (B_HALF_ELEMENTS + WARP_GROUP_THREADS - 1) / WARP_GROUP_THREADS;

    if (warp_idx == 0 && elect_one_sync()) {
        smem.bar_cp_async.init(1);
        smem.bar_b_local_ready.init(1);
        smem.bar_b_peer_ready.init(1);
        cutlass::arch::fence_barrier_init();
    }

    if (warp_idx == 0) {
        TMEM::Allocator2Sm().allocate(512, &smem.tmem_start_addr);
        TMEM::Allocator2Sm().release_allocation_lock();
    }

    __syncthreads();
    cluster_sync();

    // ========================================
    // Warpgroup 0: Data Load + cp.async Producer (WG0)
    // Responsibility:
    // 1) Load A/B_local from global to smem
    // 2) Launch cp.async to peer CTA
    // 3) Notify WG1 when B_local / B_peer are ready
    // ========================================
    if (is_wg0) {
        #pragma unroll
        for (int i = 0; i < A_ELEMENTS_PER_WG_THREAD; ++i) {
            const int linear_idx = idx_in_warpgroup * A_ELEMENTS_PER_WG_THREAD + i;
            if (linear_idx < A_ELEMENTS) {
                const int row = linear_idx / K_TOTAL;
                const int k = linear_idx % K_TOTAL;
                sA(row, k) = A[(a_row_offset + row) * K_TOTAL + k];
            }
        }

        #pragma unroll
        for (int i = 0; i < B_ELEMENTS_PER_WG_THREAD; ++i) {
            const int linear_idx = idx_in_warpgroup * B_ELEMENTS_PER_WG_THREAD + i;
            if (linear_idx < B_ELEMENTS_FULL) {
                const int n = linear_idx / K_TOTAL;
                const int k = linear_idx % K_TOTAL;
                sB_local(n, k) = bf16(0.0f);
                sB_peer(n, k) = bf16(0.0f);
            }
        }

        const int b_k_offset = cta_idx * K_PER_CTA;
        #pragma unroll
        for (int i = 0; i < B_HALF_ELEMENTS_PER_WG_THREAD; ++i) {
            const int linear_idx = idx_in_warpgroup * B_HALF_ELEMENTS_PER_WG_THREAD + i;
            if (linear_idx < B_HALF_ELEMENTS) {
                const int k_local = linear_idx / N_TOTAL;
                const int n = linear_idx % N_TOTAL;
                const int k_global = b_k_offset + k_local;
                sB_local(n, k_global) = B[k_global * N_TOTAL + n];
            }
        }

        __threadfence_block();

#if SMEM_CP_ASYNC_DEBUG
        if (idx_in_warpgroup == 0) {
            const int logical_b_elems = B_ELEMENTS_FULL;
            const int storage_b_elems = B_STORAGE_ELEMENTS;
            const unsigned long long logical_b_bytes =
                static_cast<unsigned long long>(sizeof(bf16) * static_cast<size_t>(logical_b_elems));
            const unsigned long long storage_b_bytes =
                static_cast<unsigned long long>(sizeof(bf16) * static_cast<size_t>(storage_b_elems));
            const unsigned long long cp_async_tx_bytes =
                static_cast<unsigned long long>(sizeof(bf16) * static_cast<size_t>(B_ELEMENTS_FULL));

            printf(
                "[smem_cp_async debug][cta=%d] logical_B_elems=%d storage_B_elems=%d logical_B_bytes=%llu storage_B_bytes=%llu cp_async_tx_bytes=%llu\n",
                cta_idx,
                logical_b_elems,
                storage_b_elems,
                logical_b_bytes,
                storage_b_bytes,
                cp_async_tx_bytes
            );

            printf("[smem_cp_async debug][cta=%d] sA(row0,k0..7):", cta_idx);
            for (int k = 0; k < 8; ++k) {
                printf(" %7.4f", static_cast<float>(sA(0, k)));
            }
            printf("\n");

            printf("[smem_cp_async debug][cta=%d] sB_local(n0,k0..7):", cta_idx);
            for (int k = 0; k < 8; ++k) {
                printf(" %7.4f", static_cast<float>(sB_local(0, k)));
            }
            printf("\n");

            printf("[smem_cp_async debug][cta=%d] sB_local(n0,k28..35):", cta_idx);
            for (int k = 28; k < 36; ++k) {
                printf(" %7.4f", static_cast<float>(sB_local(0, k)));
            }
            printf("\n");

            printf("[smem_cp_async debug][cta=%d] sB_local(n0,k56..63):", cta_idx);
            for (int k = 56; k < 64; ++k) {
                printf(" %7.4f", static_cast<float>(sB_local(0, k)));
            }
            printf("\n");
        }
#endif

        if (idx_in_warpgroup == 0) {
            smem.bar_cp_async.arrive_and_expect_tx(sizeof(bf16) * B_ELEMENTS_FULL);
        }
    }

    cluster_sync();

    if (is_wg0 && idx_in_warpgroup == 0) {
        if (cta_idx == 0) {
            smem.bar_b_local_ready.arrive(0u);
        } else {
            smem.bar_b_local_ready.arrive(1u);
        }

        bf16* peer_B_peer_ptr = kerutils::get_peer_addr(smem.sB_peer.data());
        transac_bar_t* peer_bar_ptr = kerutils::get_peer_addr(&smem.bar_cp_async);
        kerutils::cp_async_bulk_shared_cta_to_shared_cluster(
            peer_B_peer_ptr,
            smem.sB_local.data(),
            sizeof(bf16) * B_ELEMENTS_FULL,
            *peer_bar_ptr
        );
    }

    cutlass::arch::fence_view_async_shared();

    if (is_wg0 && idx_in_warpgroup == 0) {
        smem.bar_cp_async.wait(0);
        __threadfence_block();
        if (cta_idx == 0) {
            smem.bar_b_peer_ready.arrive(0u);
        } else {
            smem.bar_b_peer_ready.arrive(1u);
        }

#if SMEM_CP_ASYNC_DEBUG
        printf("[smem_cp_async debug][cta=%d] sB_peer(n0,k0..7):", cta_idx);
        for (int k = 0; k < 8; ++k) {
            printf(" %7.4f", static_cast<float>(sB_peer(0, k)));
        }
        printf("\n");

        printf("[smem_cp_async debug][cta=%d] sB_peer(n0,k28..35):", cta_idx);
        for (int k = 28; k < 36; ++k) {
            printf(" %7.4f", static_cast<float>(sB_peer(0, k)));
        }
        printf("\n");

        printf("[smem_cp_async debug][cta=%d] sB_peer(n0,k56..63):", cta_idx);
        for (int k = 56; k < 64; ++k) {
            printf(" %7.4f", static_cast<float>(sB_peer(0, k)));
        }
        printf("\n");

        float dot_local_n0 = 0.0f;
        float dot_peer_n0 = 0.0f;
        float dot_ref_n0 = 0.0f;
        float dot_local_n128 = 0.0f;
        float dot_peer_n128 = 0.0f;
        float dot_ref_n128 = 0.0f;
        for (int k = 0; k < K_TOTAL; ++k) {
            const float a = static_cast<float>(sA(0, k));
            dot_local_n0 += a * static_cast<float>(sB_local(0, k));
            dot_peer_n0 += a * static_cast<float>(sB_peer(0, k));
            dot_ref_n0 += static_cast<float>(A[a_row_offset * K_TOTAL + k]) *
                          static_cast<float>(B[k * N_TOTAL]);

            dot_local_n128 += a * static_cast<float>(sB_local(128, k));
            dot_peer_n128 += a * static_cast<float>(sB_peer(128, k));
            dot_ref_n128 += static_cast<float>(A[a_row_offset * K_TOTAL + k]) *
                            static_cast<float>(B[k * N_TOTAL + 128]);
        }

        float diff_n0 = dot_local_n0 + dot_peer_n0 - dot_ref_n0;
        if (diff_n0 < 0.0f) {
            diff_n0 = -diff_n0;
        }
        float diff_n128 = dot_local_n128 + dot_peer_n128 - dot_ref_n128;
        if (diff_n128 < 0.0f) {
            diff_n128 = -diff_n128;
        }

        printf(
            "[smem_cp_async debug][cta=%d] dot(row=%d,col=0): local=%9.4f peer=%9.4f total=%9.4f ref=%9.4f abs_diff=%9.4f\n",
            cta_idx,
            a_row_offset,
            dot_local_n0,
            dot_peer_n0,
            dot_local_n0 + dot_peer_n0,
            dot_ref_n0,
            diff_n0
        );
        printf(
            "[smem_cp_async debug][cta=%d] dot(row=%d,col=128): local=%9.4f peer=%9.4f total=%9.4f ref=%9.4f abs_diff=%9.4f\n",
            cta_idx,
            a_row_offset,
            dot_local_n128,
            dot_peer_n128,
            dot_local_n128 + dot_peer_n128,
            dot_ref_n128,
            diff_n128
        );
#endif
    }

    // ========================================
    // Warpgroup 1: MMA Consumer (WG1)
    // Responsibility:
    // 1) Wait B_local ready and run first MMA
    // 2) Wait B_peer ready and run second MMA accumulate
    // ========================================
    if (is_wg1) {
        TiledMMA_C tiled_mma{};
        Tensor tC = partition_fragment_C(tiled_mma, Shape<Int<M_PER_CTA>, Int<N_TOTAL>>{});
        tC.data().get() = tmem_cols::c;

        if (warp_idx == 4 && elect_one_sync()) {
            smem.bar_b_local_ready.wait(0);
            ku::utcmma_ss(tiled_mma, sA, sB_local, tC, true);
        }
        ku::tcgen05_after_thread_sync();

        if (warp_idx == 4 && elect_one_sync()) {
            smem.bar_b_peer_ready.wait(0);
            ku::utcmma_ss(tiled_mma, sA, sB_peer, tC, false);
        }
        ku::tcgen05_after_thread_sync();
    }

    __syncthreads();
    cluster_sync();

    // WG0 reads TMEM result and stores to global C.
    if (is_wg0) {
        store_c_from_tmem(smem.tmem_start_addr, a_row_offset, C);
    }
    __syncthreads();
    cluster_sync();

#if SMEM_CP_ASYNC_DEBUG
    if (tid == 0) {
        printf("[smem_cp_async debug][cta=%d] C(row=%d,col0..7):", cta_idx, a_row_offset);
        for (int col = 0; col < 8; ++col) {
            printf(" %9.4f", C[a_row_offset * N_TOTAL + col]);
        }
        printf("\n");
    }

    if (tid == 0 && cta_idx == 0) {
        printf("[smem_cp_async debug][cta=%d] C(row=64,col0..7):", cta_idx);
        for (int col = 0; col < 8; ++col) {
            printf(" %9.4f", C[64 * N_TOTAL + col]);
        }
        printf("\n");
    }
#endif

    if (warp_idx == 0 && elect_one_sync()) {
        TMEM::Allocator2Sm().free(smem.tmem_start_addr, 512);
    }
#else
    (void)A;
    (void)B;
    (void)C;
#endif
}

void launch_test_smem_cp_async(
    const bf16* A,
    const bf16* B,
    float* C,
    cudaStream_t stream
) {
    dim3 grid(2, 1, 1);
    dim3 block(NUM_THREADS, 1, 1);

    cudaError_t attr_err = cudaFuncSetAttribute(
        test_smem_cp_async_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_SIZE
    );
    if (attr_err != cudaSuccess) {
        std::fprintf(stderr, "cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(attr_err));
        return;
    }

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

    cudaError_t err = cudaLaunchKernelEx(&config, test_smem_cp_async_kernel, A, B, C);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaLaunchKernelEx failed: %s\n", cudaGetErrorString(err));
    }
}

torch::Tensor smem_cp_async_forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be bfloat16");
    TORCH_CHECK(A.dim() == 2 && A.size(0) == M_TOTAL && A.size(1) == K_TOTAL,
                "A shape must be [128, 64]");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == K_TOTAL && B.size(1) == N_TOTAL,
                "B shape must be [64, 256]");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
    torch::Tensor C = torch::empty({M_TOTAL, N_TOTAL}, options);

    const bf16* A_ptr = reinterpret_cast<const bf16*>(A.data_ptr<at::BFloat16>());
    const bf16* B_ptr = reinterpret_cast<const bf16*>(B.data_ptr<at::BFloat16>());
    float* C_ptr = C.data_ptr<float>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_test_smem_cp_async(A_ptr, B_ptr, C_ptr, stream);

    cudaError_t err = cudaStreamSynchronize(stream);
    TORCH_CHECK(err == cudaSuccess, "CUDA execution failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smem_cp_async", &smem_cp_async_forward,
          "2CTA cp.async SMEM exchange test (A[128,64] @ B[64,256])");
}

#include "test_dq_2sm_mma.cuh"

#include <c10/cuda/CUDAStream.h>
#include <cutlass/arch/arch.h>
#include <cutlass/cuda_host_adapter.hpp>

#include <cstring>

using cutlass::arch::fence_barrier_init;
using cutlass::arch::fence_view_async_shared;

namespace test_operator::dq_2sm_mma {

template <typename Params>
__global__ __launch_bounds__(NUM_THREADS, 1) void dq_2sm_mma_kernel(
    const bf16* __restrict__ ds,
    const bf16* __restrict__ kv,
    const int32_t* __restrict__ indices,
    float* __restrict__ dQ_out,
    __grid_constant__ const Params params
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
    using namespace cute;

    extern __shared__ char smem_raw[];
    SharedMemory& smem = *reinterpret_cast<SharedMemory*>(smem_raw);

    const int cta_idx = blockIdx.x % 2;
    const int tid = threadIdx.x;
    const int warp_idx = tid / 32;
    const int idx_in_warpgroup = tid % 128;
    const bool dbg_print = (blockIdx.x < 2) && (idx_in_warpgroup == 0);

    if (dbg_print) {
        printf(
            "[DBG][B%d CTA%d WG0] enter idx0..3=(%d,%d,%d,%d)\n",
            blockIdx.x, cta_idx,
            indices[0], indices[1], indices[2], indices[3]
        );
    }

    if (warp_idx == 0 && elect_one_sync()) {
        smem.bar_kv_part0_ready.init(1);
        smem.bar_kv_part1_ready.init(1);
        smem.bar_kv_part2_ready.init(1);
        smem.bar_dq_ready.init(1);
        fence_barrier_init();
    }
    __syncthreads();
    cluster_sync();

    Tensor sDS_t = make_tensor(make_smem_ptr(smem.ds_t.data()), SmemLayoutdSTransposed{});

    constexpr int DS_ROWS_PER_CTA = B_H / 2;
    constexpr int DS_COLS = B_TOPK;
    constexpr int DS_ELEMENTS = DS_ROWS_PER_CTA * DS_COLS;
    for (int linear_idx = tid; linear_idx < DS_ELEMENTS; linear_idx += blockDim.x) {
        const int row = linear_idx / DS_COLS;
        const int col = linear_idx % DS_COLS;
        const int global_row = cta_idx * DS_ROWS_PER_CTA + row;
        sDS_t(row, col) = ds[global_row * DS_COLS + col];
    }
    fence_view_async_shared();
    __syncthreads();
    cluster_sync();

    if (dbg_print) {
        const int row0 = cta_idx * DS_ROWS_PER_CTA;
        printf(
            "[DBG][B%d CTA%d WG0] ds smem(0,0..3)=(%.6f,%.6f,%.6f,%.6f) gmem(row=%d,0..3)=(%.6f,%.6f,%.6f,%.6f)\n",
            blockIdx.x, cta_idx,
            (float)sDS_t(0, 0), (float)sDS_t(0, 1), (float)sDS_t(0, 2), (float)sDS_t(0, 3),
            row0,
            (float)ds[row0 * DS_COLS + 0], (float)ds[row0 * DS_COLS + 1],
            (float)ds[row0 * DS_COLS + 2], (float)ds[row0 * DS_COLS + 3]
        );
    }

    if (warp_idx == 0) {
        TMEM::Allocator2Sm().allocate(512, smem.tmem_start_addr.data());
        TMEM::Allocator2Sm().release_allocation_lock();
    }
    __syncthreads();
    cluster_sync();

    constexpr int NUM_WARPS = 4;
    constexpr int NUM_LOCAL_ROWS_PER_WARP = (B_TOPK / 2) / 4 / NUM_WARPS;
    constexpr int COLS_NOPE_PART = 128;
    constexpr int COLS_ROPE_PART = D_ROPE / 2;

    if (warp_idx < NUM_WARPS && elect_one_sync()) {
        const int local_warp_idx = warp_idx;
        bf16* sKCalc_part0_base = smem.k_calc_dq.data() + local_warp_idx * 4 * 64;
        bf16* sKCalc_part1_base =
            smem.k_calc_dq.data() + cosize_v<SmemLayoutKCalcDQPartNoPE> + local_warp_idx * 4 * 64;
        bf16* sKCalc_part2_base =
            smem.k_calc_dq.data() + cosize_v<SmemLayoutKCalcDQPartNoPE> * 2 + local_warp_idx * 4 * COLS_ROPE_PART;

        int4 indices4[NUM_LOCAL_ROWS_PER_WARP];
        CUTE_UNROLL
        for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
            indices4[local_row] = __ldg(
                (const int4*)(indices + cta_idx * (B_TOPK / 2)) + local_row * NUM_WARPS + local_warp_idx
            );
        }

        const int part0_gmem_col_base = (cta_idx == 0) ? 0 : 128;
        CUTE_UNROLL
        for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
            CUTE_UNROLL
            for (int local_col = 0; local_col < COLS_NOPE_PART / 64; ++local_col) {
                ku::tma_gather4_cta_group_2<true>(
                    &(params.tensor_map_kv),
                    smem.bar_kv_part0_ready,
                    sKCalc_part0_base + local_row * (4 * NUM_WARPS) * 64 + local_col * ((B_TOPK / 2) * 64),
                    part0_gmem_col_base + local_col * 64,
                    indices4[local_row],
                    (int64_t)TMA::CacheHintSm90::EVICT_LAST
                );
            }
        }

        const int part1_gmem_col_base = (cta_idx == 0) ? 256 : 384;
        CUTE_UNROLL
        for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
            CUTE_UNROLL
            for (int local_col = 0; local_col < COLS_NOPE_PART / 64; ++local_col) {
                ku::tma_gather4_cta_group_2<true>(
                    &(params.tensor_map_kv),
                    smem.bar_kv_part1_ready,
                    sKCalc_part1_base + local_row * (4 * NUM_WARPS) * 64 + local_col * ((B_TOPK / 2) * 64),
                    part1_gmem_col_base + local_col * 64,
                    indices4[local_row],
                    (int64_t)TMA::CacheHintSm90::EVICT_LAST
                );
            }
        }

        const int part2_gmem_col_base = (cta_idx == 0) ? 512 : 544;
        CUTE_UNROLL
        for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
            ku::tma_gather4_cta_group_2<true>(
                &(params.tensor_map_kv_rope32),
                smem.bar_kv_part2_ready,
                sKCalc_part2_base + local_row * (4 * NUM_WARPS) * COLS_ROPE_PART,
                part2_gmem_col_base,
                indices4[local_row],
                (int64_t)TMA::CacheHintSm90::EVICT_LAST
            );
        }
    }

    if (cta_idx == 0 && warp_idx == 0 && elect_one_sync()) {
        TiledMMA_dQ_2cta tiled_mma_dQ_2cta{};
        TiledMMA_dQ_RoPE_2cta tiled_mma_dQ_rope_2cta{};

        Tensor tdQ_part0 = partition_fragment_C(tiled_mma_dQ_2cta, Shape<Int<B_H>, Int<256>>{});
        tdQ_part0.data().get() = tmem_cols::dQ;

        Tensor tdQ_part1 = partition_fragment_C(tiled_mma_dQ_2cta, Shape<Int<B_H>, Int<256>>{});
        tdQ_part1.data().get() = tmem_cols::dQ + 128;

        Tensor tdQ_rope = partition_fragment_C(tiled_mma_dQ_rope_2cta, Shape<Int<B_H>, Int<D_ROPE>>{});
        tdQ_rope.data().get() = tmem_cols::dQ_RoPE;

        Tensor sK_calc_part0 = make_tensor(
            make_smem_ptr(smem.k_calc_dq.data()),
            SmemLayoutKCalcDQPartNoPE{}
        );
        Tensor sK_calc_part1 = make_tensor(
            make_smem_ptr(smem.k_calc_dq.data() + cosize_v<SmemLayoutKCalcDQPartNoPE>),
            SmemLayoutKCalcDQPartNoPE{}
        );
        Tensor sK_calc_part2 = make_tensor(
            make_smem_ptr(smem.k_calc_dq.data() + cosize_v<SmemLayoutKCalcDQPartNoPE> * 2),
            SmemLayoutKCalcDQPartRoPE{}
        );

        // kv_part0 expects bytes from both CTAs combined via cta_group::2 + USE_CTA0_MBAR:
        // 2 * (B_TOPK/2 rows) * 128 cols * sizeof(bf16) = B_TOPK * 128 * sizeof(bf16).
        smem.bar_kv_part0_ready.arrive_and_expect_tx(B_TOPK * 128 * sizeof(bf16));
        smem.bar_kv_part0_ready.wait(0);
        ku::tcgen05_after_thread_sync();
        if (dbg_print) {
            const int idx0 = indices[0];
            const int idx1 = indices[1];
            const int idx2 = indices[2];
            const int idx3 = indices[3];
            printf(
                "[DBG][B%d CTA%d WG0] part0 k_smem(k0..3,col0)=(%.6f,%.6f,%.6f,%.6f) kv_ref(idx0..3,col0)=(%.6f,%.6f,%.6f,%.6f)\n",
                blockIdx.x, cta_idx,
                (float)sK_calc_part0(0, 0), (float)sK_calc_part0(0, 1),
                (float)sK_calc_part0(0, 2), (float)sK_calc_part0(0, 3),
                (float)kv[idx0 * D_K + 0], (float)kv[idx1 * D_K + 0],
                (float)kv[idx2 * D_K + 0], (float)kv[idx3 * D_K + 0]
            );
            printf(
                "[DBG][B%d CTA%d WG0] part0 smem4x4 r0=(%.6f,%.6f,%.6f,%.6f) r1=(%.6f,%.6f,%.6f,%.6f)\n",
                blockIdx.x, cta_idx,
                (float)sK_calc_part0(0, 0), (float)sK_calc_part0(0, 1), (float)sK_calc_part0(0, 2), (float)sK_calc_part0(0, 3),
                (float)sK_calc_part0(1, 0), (float)sK_calc_part0(1, 1), (float)sK_calc_part0(1, 2), (float)sK_calc_part0(1, 3)
            );
            printf(
                "[DBG][B%d CTA%d WG0] part0 kv4x4 i0=(%.6f,%.6f,%.6f,%.6f) i1=(%.6f,%.6f,%.6f,%.6f)\n",
                blockIdx.x, cta_idx,
                (float)kv[idx0 * D_K + 0], (float)kv[idx0 * D_K + 1], (float)kv[idx0 * D_K + 2], (float)kv[idx0 * D_K + 3],
                (float)kv[idx1 * D_K + 0], (float)kv[idx1 * D_K + 1], (float)kv[idx1 * D_K + 2], (float)kv[idx1 * D_K + 3]
            );
        }
        ku::utcmma_ss(tiled_mma_dQ_2cta, sDS_t, sK_calc_part0, tdQ_part0, true);

        // kv_part1 has the same footprint as part0 (another 128-column slice).
        smem.bar_kv_part1_ready.arrive_and_expect_tx(B_TOPK * 128 * sizeof(bf16));
        smem.bar_kv_part1_ready.wait(0);
        ku::tcgen05_after_thread_sync();
        if (dbg_print) {
            const int idx0 = indices[0];
            const int idx1 = indices[1];
            const int idx2 = indices[2];
            const int idx3 = indices[3];
            printf(
                "[DBG][B%d CTA%d WG0] part1 k_smem(k0..3,col0)=(%.6f,%.6f,%.6f,%.6f) kv_ref(idx0..3,col256)=(%.6f,%.6f,%.6f,%.6f)\n",
                blockIdx.x, cta_idx,
                (float)sK_calc_part1(0, 0), (float)sK_calc_part1(0, 1),
                (float)sK_calc_part1(0, 2), (float)sK_calc_part1(0, 3),
                (float)kv[idx0 * D_K + 256], (float)kv[idx1 * D_K + 256],
                (float)kv[idx2 * D_K + 256], (float)kv[idx3 * D_K + 256]
            );
            printf(
                "[DBG][B%d CTA%d WG0] part1 smem4x4 r0=(%.6f,%.6f,%.6f,%.6f) r1=(%.6f,%.6f,%.6f,%.6f)\n",
                blockIdx.x, cta_idx,
                (float)sK_calc_part1(0, 0), (float)sK_calc_part1(0, 1), (float)sK_calc_part1(0, 2), (float)sK_calc_part1(0, 3),
                (float)sK_calc_part1(1, 0), (float)sK_calc_part1(1, 1), (float)sK_calc_part1(1, 2), (float)sK_calc_part1(1, 3)
            );
            printf(
                "[DBG][B%d CTA%d WG0] part1 kv4x4 i0=(%.6f,%.6f,%.6f,%.6f) i1=(%.6f,%.6f,%.6f,%.6f)\n",
                blockIdx.x, cta_idx,
                (float)kv[idx0 * D_K + 256], (float)kv[idx0 * D_K + 257], (float)kv[idx0 * D_K + 258], (float)kv[idx0 * D_K + 259],
                (float)kv[idx1 * D_K + 256], (float)kv[idx1 * D_K + 257], (float)kv[idx1 * D_K + 258], (float)kv[idx1 * D_K + 259]
            );
        }
        ku::utcmma_ss(tiled_mma_dQ_2cta, sDS_t, sK_calc_part1, tdQ_part1, true);

        // kv_part2 is the 32-column RoPE slice, again aggregated across both CTAs.
        smem.bar_kv_part2_ready.arrive_and_expect_tx(B_TOPK * (D_ROPE / 2) * sizeof(bf16));
        smem.bar_kv_part2_ready.wait(0);
        ku::tcgen05_after_thread_sync();
        if (dbg_print) {
            const int idx0 = indices[0];
            const int idx1 = indices[1];
            const int idx2 = indices[2];
            const int idx3 = indices[3];
            printf(
                "[DBG][B%d CTA%d WG0] part2 k_smem(k0..3,col0)=(%.6f,%.6f,%.6f,%.6f) kv_ref(idx0..3,col512)=(%.6f,%.6f,%.6f,%.6f)\n",
                blockIdx.x, cta_idx,
                (float)sK_calc_part2(0, 0), (float)sK_calc_part2(0, 1),
                (float)sK_calc_part2(0, 2), (float)sK_calc_part2(0, 3),
                (float)kv[idx0 * D_K + 512], (float)kv[idx1 * D_K + 512],
                (float)kv[idx2 * D_K + 512], (float)kv[idx3 * D_K + 512]
            );
            printf(
                "[DBG][B%d CTA%d WG0] part2 smem4x4 r0=(%.6f,%.6f,%.6f,%.6f) r1=(%.6f,%.6f,%.6f,%.6f)\n",
                blockIdx.x, cta_idx,
                (float)sK_calc_part2(0, 0), (float)sK_calc_part2(0, 1), (float)sK_calc_part2(0, 2), (float)sK_calc_part2(0, 3),
                (float)sK_calc_part2(1, 0), (float)sK_calc_part2(1, 1), (float)sK_calc_part2(1, 2), (float)sK_calc_part2(1, 3)
            );
            printf(
                "[DBG][B%d CTA%d WG0] part2 kv4x4 i0=(%.6f,%.6f,%.6f,%.6f) i1=(%.6f,%.6f,%.6f,%.6f)\n",
                blockIdx.x, cta_idx,
                (float)kv[idx0 * D_K + 512], (float)kv[idx0 * D_K + 513], (float)kv[idx0 * D_K + 514], (float)kv[idx0 * D_K + 515],
                (float)kv[idx1 * D_K + 512], (float)kv[idx1 * D_K + 513], (float)kv[idx1 * D_K + 514], (float)kv[idx1 * D_K + 515]
            );
        }

        if (dbg_print) {
            float smem_dot0 = 0.0f;
            float smem_dot256 = 0.0f;
            float smem_dot512 = 0.0f;
            float gmem_dot0 = 0.0f;
            float gmem_dot256 = 0.0f;
            float gmem_dot512 = 0.0f;
            CUTE_UNROLL
            for (int k = 0; k < B_TOPK; ++k) {
                const float ds_v = (float)sDS_t(0, k);
                smem_dot0 += ds_v * (float)sK_calc_part0(0, k);
                smem_dot256 += ds_v * (float)sK_calc_part1(0, k);
                smem_dot512 += ds_v * (float)sK_calc_part2(0, k);

                const int kv_idx = indices[k];
                const float ds_ref = (float)ds[k];
                gmem_dot0 += ds_ref * (float)kv[kv_idx * D_K + 0];
                gmem_dot256 += ds_ref * (float)kv[kv_idx * D_K + 256];
                gmem_dot512 += ds_ref * (float)kv[kv_idx * D_K + 512];
            }
            printf(
                "[DBG][B%d CTA%d WG0] dot_pre row0 col0/256/512 smem=(%.6f,%.6f,%.6f) gmem=(%.6f,%.6f,%.6f)\n",
                blockIdx.x, cta_idx, smem_dot0, smem_dot256, smem_dot512, gmem_dot0, gmem_dot256, gmem_dot512
            );
        }
        ku::utcmma_ss(tiled_mma_dQ_rope_2cta, sDS_t, sK_calc_part2, tdQ_rope, true);

        ku::umma_arrive_multicast_2x1SM_noelect(smem.bar_dq_ready, 1 | 2);
    }

    if (warp_idx == 0 && elect_one_sync()) {
        smem.bar_dq_ready.wait(0);
        ku::tcgen05_after_thread_sync();
    }
    __syncthreads();
    cluster_sync();

    constexpr int dQ_ROWS_PER_CTA = B_H / 2;
    constexpr int NOPE_FLOATS_PER_HALF = 256 / 2;
    constexpr int NOPE_CHUNKS = 8;
    constexpr int NOPE_CHUNK_FLOATS = NOPE_FLOATS_PER_HALF / NOPE_CHUNKS;
    constexpr int NOPE_CHUNK_FLOAT2 = NOPE_CHUNK_FLOATS / 2;
    constexpr int ROPE_FLOAT2_PER_HALF = (D_ROPE / 2) / 2;

    const int row_in_cta = tid % dQ_ROWS_PER_CTA;
    const int col_half = tid / dQ_ROWS_PER_CTA;
    const int out_row = cta_idx * dQ_ROWS_PER_CTA + row_in_cta;

    const uint32_t tmem_base = smem.tmem_start_addr.data()[0];
    const uint32_t tmem_addr_dq0 = tmem_base + (row_in_cta << 16) + tmem_cols::dQ;
    const uint32_t tmem_addr_dq1 = tmem_base + (row_in_cta << 16) + (tmem_cols::dQ + 128);

    CUTE_UNROLL
    for (int chunk = 0; chunk < NOPE_CHUNKS; ++chunk) {
        const int chunk_col_base = chunk * NOPE_CHUNK_FLOATS;
        float2 dq_chunk[NOPE_CHUNK_FLOAT2];
        ku::tmem_ld_32dp32bNx<NOPE_CHUNK_FLOATS>(tmem_addr_dq0 + chunk_col_base, dq_chunk);
        cutlass::arch::fence_view_async_tmem_load();
        ku::tcgen05_before_thread_sync();

        CUTE_UNROLL
        for (int i = 0; i < NOPE_CHUNK_FLOAT2; ++i) {
            const int out_col = col_half * NOPE_FLOATS_PER_HALF + chunk_col_base + i * 2;
            dQ_out[out_row * D_Q + out_col] = dq_chunk[i].x;
            dQ_out[out_row * D_Q + out_col + 1] = dq_chunk[i].y;
        }
    }

    CUTE_UNROLL
    for (int chunk = 0; chunk < NOPE_CHUNKS; ++chunk) {
        const int chunk_col_base = chunk * NOPE_CHUNK_FLOATS;
        float2 dq_chunk[NOPE_CHUNK_FLOAT2];
        ku::tmem_ld_32dp32bNx<NOPE_CHUNK_FLOATS>(tmem_addr_dq1 + chunk_col_base, dq_chunk);
        cutlass::arch::fence_view_async_tmem_load();
        ku::tcgen05_before_thread_sync();

        CUTE_UNROLL
        for (int i = 0; i < NOPE_CHUNK_FLOAT2; ++i) {
            const int out_col = 256 + col_half * NOPE_FLOATS_PER_HALF + chunk_col_base + i * 2;
            dQ_out[out_row * D_Q + out_col] = dq_chunk[i].x;
            dQ_out[out_row * D_Q + out_col + 1] = dq_chunk[i].y;
        }
    }

    float2 dq_rope[ROPE_FLOAT2_PER_HALF];
    const uint32_t tmem_addr_dq_rope = tmem_base + (row_in_cta << 16) + tmem_cols::dQ_RoPE;
    ku::tmem_ld_32dp32bNx<D_ROPE / 2>(tmem_addr_dq_rope, dq_rope);
    cutlass::arch::fence_view_async_tmem_load();
    ku::tcgen05_before_thread_sync();
    CUTE_UNROLL
    for (int i = 0; i < ROPE_FLOAT2_PER_HALF; ++i) {
        const int out_col = 512 + col_half * (D_ROPE / 2) + i * 2;
        dQ_out[out_row * D_Q + out_col] = dq_rope[i].x;
        dQ_out[out_row * D_Q + out_col + 1] = dq_rope[i].y;
    }

    __syncthreads();
    if (dbg_print) {
        const int row_g = cta_idx * dQ_ROWS_PER_CTA;
        float ref0 = 0.0f;
        float ref256 = 0.0f;
        float ref512 = 0.0f;
        CUTE_UNROLL
        for (int k = 0; k < B_TOPK; ++k) {
            const int kv_idx = indices[k];
            const float ds_v = (float)ds[row_g * B_TOPK + k];
            ref0 += ds_v * (float)kv[kv_idx * D_K + 0];
            ref256 += ds_v * (float)kv[kv_idx * D_K + 256];
            ref512 += ds_v * (float)kv[kv_idx * D_K + 512];
        }
        printf(
            "[DBG][B%d CTA%d WG0] dQ row%d col0/256/512 out=(%.6f,%.6f,%.6f) ref=(%.6f,%.6f,%.6f)\n",
            blockIdx.x, cta_idx, row_g,
            dQ_out[row_g * D_Q + 0], dQ_out[row_g * D_Q + 256], dQ_out[row_g * D_Q + 512],
            ref0, ref256, ref512
        );
    }
    cluster_sync();

    if (warp_idx == 0 && elect_one_sync()) {
        TMEM::Allocator2Sm().free(smem.tmem_start_addr.data()[0], 512);
    }
#else
    (void)ds;
    (void)kv;
    (void)indices;
    (void)dQ_out;
    (void)params;
#endif
}

torch::Tensor run_dq_2sm_mma(
    torch::Tensor ds,
    torch::Tensor kv,
    torch::Tensor indices
) {
    TORCH_CHECK(ds.is_cuda(), "ds must be CUDA tensor");
    TORCH_CHECK(kv.is_cuda(), "kv must be CUDA tensor");
    TORCH_CHECK(indices.is_cuda(), "indices must be CUDA tensor");

    TORCH_CHECK(ds.dtype() == torch::kBFloat16, "ds must be bfloat16");
    TORCH_CHECK(kv.dtype() == torch::kBFloat16, "kv must be bfloat16");
    TORCH_CHECK(indices.dtype() == torch::kInt32 || indices.dtype() == torch::kInt64,
                "indices must be int32 or int64");

    TORCH_CHECK(ds.dim() == 2 && ds.size(0) == B_H && ds.size(1) == B_TOPK,
                "ds shape must be [128, 64]");
    TORCH_CHECK(kv.dim() == 2 && kv.size(1) == D_K,
                "kv shape must be [s_kv, 576]");
    TORCH_CHECK(indices.dim() == 1 && indices.size(0) == B_TOPK,
                "indices shape must be [64]");

    TORCH_CHECK(ds.is_contiguous(), "ds must be contiguous");
    TORCH_CHECK(kv.is_contiguous(), "kv must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");

    const int64_t s_kv = kv.size(0);
    TORCH_CHECK(s_kv > 0, "s_kv must be > 0");

    auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(ds.device());
    torch::Tensor dQ_out = torch::zeros({B_H, D_Q}, options_f32);

    torch::Tensor indices_i32 = (indices.dtype() == torch::kInt32) ? indices : indices.to(torch::kInt32);

    const bf16* ds_ptr = reinterpret_cast<const bf16*>(ds.data_ptr<at::BFloat16>());
    const bf16* kv_ptr = reinterpret_cast<const bf16*>(kv.data_ptr<at::BFloat16>());
    const int32_t* indices_ptr = indices_i32.data_ptr<int32_t>();
    float* dq_ptr = dQ_out.data_ptr<float>();

    CUtensorMap tensor_map_kv;
    CUtensorMap tensor_map_kv_rope32;

    {
        uint64_t size[2] = {(uint64_t)D_K, (uint64_t)s_kv};
        uint64_t stride[1] = {(uint64_t)D_K * sizeof(bf16)};
        uint32_t box_size[2] = {64, 1};
        uint32_t elem_stride[2] = {1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            const_cast<bf16*>(kv_ptr),
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled(kv) failed with code ", int(res));
    }

    {
        uint64_t size[2] = {(uint64_t)D_K, (uint64_t)s_kv};
        uint64_t stride[1] = {(uint64_t)D_K * sizeof(bf16)};
        uint32_t box_size[2] = {32, 1};
        uint32_t elem_stride[2] = {1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv_rope32,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            const_cast<bf16*>(kv_ptr),
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled(kv_rope32) failed with code ", int(res));
    }

    KernelParams params = {tensor_map_kv, tensor_map_kv_rope32};
    auto kernel = &dq_2sm_mma_kernel<KernelParams>;
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    cudaError_t attr_err = cudaFuncSetAttribute(
        (const void*)kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_SIZE
    );
    TORCH_CHECK(attr_err == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(attr_err));

    cudaLaunchConfig_t config;
    memset(&config, 0, sizeof(config));
    config.gridDim = dim3(2, 1, 1);
    config.blockDim = dim3(NUM_THREADS, 1, 1);
    config.dynamicSmemBytes = SMEM_SIZE;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    cudaError_t err = cudaLaunchKernelEx(
        &config,
        kernel,
        ds_ptr,
        kv_ptr,
        indices_ptr,
        dq_ptr,
        params
    );
    TORCH_CHECK(err == cudaSuccess, "dq_2sm_mma_kernel launch failed: ", cudaGetErrorString(err));

    return dQ_out;
}

}  // namespace test_operator::dq_2sm_mma

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run_dq_2sm_mma",
        &test_operator::dq_2sm_mma::run_dq_2sm_mma,
        "Unit test kernel for dQ path with TMA kv_part staging and 2SM MMA"
    );
}

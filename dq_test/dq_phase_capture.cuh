#pragma once

#include "sm100/prefill/sparse/bwd/head128/preprocess_delta.cuh"

#include "dq_config_local.h"

#include <cstring>
#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cuda_host_adapter.hpp>

#include "params.h"
#include "utils.h"
#include "sm100/helpers.h"

namespace sm100::bwd::head128_2kernels::dq {

using namespace cute;

CUTE_DEVICE
int32x8_t ldg_256_indices(void* src_ptr) {
    int32x8_t val;
    asm volatile("ld.global.nc.L1::evict_normal.L2::evict_normal.L2::256B.v8.s32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=r"(val.a0), "=r"(val.a1), "=r"(val.a2), "=r"(val.a3),
          "=r"(val.a4), "=r"(val.a5), "=r"(val.a6), "=r"(val.a7)
        : "l"(src_ptr)
    );
    return val;
}

enum class WarpRole {
    SoftmaxAndDQTransfer = 0,
    MmaAndKLoad = 1,
    KvValidLoad = 2,
};

static constexpr int kNumSoftmaxAndDQTransferWarps = 4;
static constexpr int kNumMmaAndKLoadWarps = 1;
static constexpr int kNumKvValidLoadWarps = 1;
static constexpr int kThreadsPerWarp = 32;
static constexpr int kNumAssignedWarps =
    kNumSoftmaxAndDQTransferWarps + kNumMmaAndKLoadWarps + kNumKvValidLoadWarps;
static constexpr uint32_t kTmemBase = 0;

static_assert(kNumAssignedWarps == 6);
static_assert(NUM_THREADS == kNumAssignedWarps * kThreadsPerWarp);

CUTE_DEVICE
WarpRole warp_idx_to_role(int warp_idx) {
    if (warp_idx < kNumSoftmaxAndDQTransferWarps) {
        return WarpRole::SoftmaxAndDQTransfer;
    }
    if (warp_idx == kNumSoftmaxAndDQTransferWarps) {
        return WarpRole::MmaAndKLoad;
    }
    return WarpRole::KvValidLoad;
}

template<typename TmaParamsType>
__global__ __launch_bounds__(NUM_THREADS, 1) void dq_phase_kernel(
    __grid_constant__ const SparseAttnBwdParams params,
    __grid_constant__ const TmaParamsType tma_params
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
    extern __shared__ char smem_raw[];
    SharedMemoryPlan& plan = *reinterpret_cast<SharedMemoryPlan*>(smem_raw);

    const int cta_idx = blockIdx.x % 2;
    const int s_q_idx = blockIdx.x / 2;
    if (s_q_idx >= params.s_q) {
        return;
    }

    const int max_kv_i = params.q_start_index_s + s_q_idx;
    const int tid = threadIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;
    const WarpRole warp_role = warp_idx_to_role(warp_idx);
    const int topk_length = params.topk_length == nullptr ?
        params.topk :
        min(max(__ldg(params.topk_length + s_q_idx), 0), params.topk);
    const int32_t* gIndices_s = params.indices + (int64_t)s_q_idx * params.stride_indices_s_q;
    const float* lse_s = params.lse + (int64_t)s_q_idx * params.h_q;
    const float* delta_s = params.delta + (int64_t)s_q_idx * params.stride_delta_s_q;
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);

    if (tid == 0) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dQ.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_S.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dS.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv));
    }

    if (warp_idx == 0 && elect_one_sync()) {
        plan.bar_prologue_q.init(1);
        plan.bar_prologue_tQ.init(1);
        plan.bar_prologue_k.init(1);
        plan.bar_prologue_dO.init(1);
        plan.bar_p_ready.init(1);
        plan.bar_dp_ready.init(1);
        plan.bar_s_ready.init(kNumSoftmaxAndDQTransferWarps * kThreadsPerWarp);
        plan.bar_ds_ready.init(kNumSoftmaxAndDQTransferWarps * kThreadsPerWarp);
        plan.bar_k_valid_free.init(kNumSoftmaxAndDQTransferWarps * kThreadsPerWarp);
        plan.bar_k_valid_ready.init(B_TOPK / 8);
        plan.bar_dq_ready.init(1);
        fence_barrier_init();
    }

    cluster_sync();

    Tensor sQFull = make_tensor(make_smem_ptr(plan.u.q_full.data()), SmemLayoutQ{});
    Tensor sdO = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO{});
    Tensor sS_store = make_tensor(make_smem_ptr(plan.s_ds.s.data()), SmemLayoutdSTransposed{});
    Tensor sDS_store = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutdSTransposed{});

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            Tensor gQ = flat_divide(
                tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx),
                Tile<Int<B_H / 2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_Q, gQ, sQFull, plan.bar_prologue_q, TMA::CacheHintSm90::EVICT_FIRST);

            Tensor gdO = flat_divide(
                tma_params.tma_dO.get_tma_tensor(tma_params.shape_dO)(_, _, s_q_idx),
                Tile<Int<B_H / 2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_dO, gdO, sdO, plan.bar_prologue_dO, TMA::CacheHintSm90::EVICT_FIRST);
        }

        TMEM::Allocator2Sm().allocate(tmem_cols::kNumUsedCols, plan.tmem_start_addr.data());
        KU_TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        TMEM::Allocator2Sm().release_allocation_lock();
    }
    __syncthreads();

    if (warp_role == WarpRole::SoftmaxAndDQTransfer) {
        const int idx_in_softmax = warp_idx * kThreadsPerWarp + lane_idx;
        const int global_row_idx = cta_idx * (B_H / 2) + idx_in_softmax % (B_H / 2);
        const float row_lse = __ldg(lse_s + global_row_idx) * 1.44269504f;
        const float neg_delta_val = __ldg(delta_s + global_row_idx);
        const float sm_scale = params.sm_scale;
        const float scale = params.sm_scale_div_log2;

        const float2 neg_lse_f2 = make_float2(-row_lse, -row_lse);
        const float2 scale_f2 = make_float2(scale, scale);
        const float2 neg_delta_f2 = make_float2(neg_delta_val, neg_delta_val);
        const float2 sm_scale_f2 = make_float2(sm_scale, sm_scale);

        const uint32_t tmem_lane = idx_in_softmax % S_DS_ROWS_PER_CTA;
        const uint32_t tmem_p_addr = kTmemBase + (tmem_lane << 16) + tmem_cols::P;
        const uint32_t tmem_dp_addr = kTmemBase + (tmem_lane << 16) + tmem_cols::dP;
        const int row_in_tile = idx_in_softmax % S_DS_ROWS_PER_CTA;
        const int col_half = idx_in_softmax / S_DS_ROWS_PER_CTA;
        bf16* sS_base = plan.s_ds.s.data() +
            row_in_tile * S_DS_VEC_ELEMS + col_half * S_DS_ROWS_PER_CTA * S_DS_COLS_PER_THREAD;
        bf16* sDS_base = plan.s_ds.ds.data() +
            row_in_tile * S_DS_VEC_ELEMS + col_half * S_DS_ROWS_PER_CTA * S_DS_COLS_PER_THREAD;

        constexpr int SMEM_VEC_F2 = S_DS_VEC_ELEMS / 2;
        constexpr int NUM_SMEM_VEC_STORES = S_DS_COLS_PER_THREAD / S_DS_VEC_ELEMS;
        constexpr int SMEM_VEC_STRIDE = S_DS_ROWS_PER_CTA * S_DS_VEC_ELEMS;
        constexpr int DP_CHUNK_F2 = SMEM_VEC_F2;
        constexpr int NUM_DP_CHUNKS = (B_TOPK / 2) / 2 / DP_CHUNK_F2;
        static_assert(SMEM_VEC_F2 == 4);

        CUTE_NO_UNROLL
        for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
            const int phase = k_block & 1;

            plan.bar_p_ready.wait(phase);
            ku::tcgen05_after_thread_sync();

            float2 p[(B_TOPK / 2) / 2];
            ku::tmem_ld_32dp32bNx<B_TOPK / 2>(tmem_p_addr, p);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();

            plan.bar_k_valid_ready.wait(phase);
            const uint32_t is_k_valid_lo =
                *(uint32_t*)(plan.is_k_valid + (idx_in_softmax >= S_DS_ROWS_PER_CTA ? B_TOPK / 8 / 2 : 0));
            const uint32_t is_k_valid_hi =
                *(uint32_t*)(plan.is_k_valid + (idx_in_softmax >= S_DS_ROWS_PER_CTA ? B_TOPK / 8 / 2 : 0) + 4);
            float* p_float = (float*)p;
            CUTE_UNROLL
            for (int i = 0; i < (B_TOPK / 2) / 2; ++i) {
                if (!(is_k_valid_lo >> i & 1)) {
                    p_float[i] = -CUDART_INF_F;
                }
            }
            CUTE_UNROLL
            for (int i = 0; i < (B_TOPK / 2) / 2; ++i) {
                if (!(is_k_valid_hi >> i & 1)) {
                    p_float[i + (B_TOPK / 2) / 2] = -CUDART_INF_F;
                }
            }
            plan.bar_k_valid_free.arrive();

            CUTE_UNROLL
            for (int vec = 0; vec < NUM_SMEM_VEC_STORES; ++vec) {
                const int base_idx = vec * SMEM_VEC_F2;
                bf16x8 s_pack;
                {
                    float2 p_vec = ku::float2_fma(p[base_idx + 0], scale_f2, neg_lse_f2);
                    p_vec = make_float2(exp2f(p_vec.x), exp2f(p_vec.y));
                    p[base_idx + 0] = p_vec;
                    s_pack.a01 = __float22bfloat162_rn(p_vec);
                }
                {
                    float2 p_vec = ku::float2_fma(p[base_idx + 1], scale_f2, neg_lse_f2);
                    p_vec = make_float2(exp2f(p_vec.x), exp2f(p_vec.y));
                    p[base_idx + 1] = p_vec;
                    s_pack.a23 = __float22bfloat162_rn(p_vec);
                }
                {
                    float2 p_vec = ku::float2_fma(p[base_idx + 2], scale_f2, neg_lse_f2);
                    p_vec = make_float2(exp2f(p_vec.x), exp2f(p_vec.y));
                    p[base_idx + 2] = p_vec;
                    s_pack.a45 = __float22bfloat162_rn(p_vec);
                }
                {
                    float2 p_vec = ku::float2_fma(p[base_idx + 3], scale_f2, neg_lse_f2);
                    p_vec = make_float2(exp2f(p_vec.x), exp2f(p_vec.y));
                    p[base_idx + 3] = p_vec;
                    s_pack.a67 = __float22bfloat162_rn(p_vec);
                }
                *reinterpret_cast<bf16x8*>(sS_base + vec * SMEM_VEC_STRIDE) = s_pack;
            }
            fence_view_async_shared();
            __threadfence_block();
            plan.bar_s_ready.arrive(static_cast<uint32_t>(cta_idx));
            NamedBarrier::arrive_and_wait(kNumSoftmaxAndDQTransferWarps * kThreadsPerWarp, 1);
            if (warp_idx == 0 && elect_one_sync()) {
                Tensor gS = flat_divide(
                    tma_params.tma_S.get_tma_tensor(tma_params.shape_S)(_, _, s_q_idx),
                    Tile<Int<B_H / 2>>{}
                )(_, cta_idx, _);
                auto thr_tma_s = tma_params.tma_S.get_slice(_0{});
                cute::copy(
                    tma_params.tma_S,
                    thr_tma_s.partition_S(sS_store),
                    thr_tma_s.partition_D(gS)
                );
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();
            }

            plan.bar_dp_ready.wait(phase);
            ku::tcgen05_after_thread_sync();

            CUTE_UNROLL
            for (int ch = 0; ch < NUM_DP_CHUNKS; ++ch) {
                float2 dp[DP_CHUNK_F2];
                ku::tmem_ld_32dp32bNx<DP_CHUNK_F2 * 2>(tmem_dp_addr + ch * DP_CHUNK_F2 * 2, dp);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                const int base_idx = ch * DP_CHUNK_F2;
                bf16x8 ds_pack;
                {
                    float2 ds_vec = ku::float2_mul(
                        ku::float2_mul(p[base_idx + 0], ku::float2_add(dp[0], neg_delta_f2)),
                        sm_scale_f2
                    );
                    ds_pack.a01 = __float22bfloat162_rn(ds_vec);
                }
                {
                    float2 ds_vec = ku::float2_mul(
                        ku::float2_mul(p[base_idx + 1], ku::float2_add(dp[1], neg_delta_f2)),
                        sm_scale_f2
                    );
                    ds_pack.a23 = __float22bfloat162_rn(ds_vec);
                }
                {
                    float2 ds_vec = ku::float2_mul(
                        ku::float2_mul(p[base_idx + 2], ku::float2_add(dp[2], neg_delta_f2)),
                        sm_scale_f2
                    );
                    ds_pack.a45 = __float22bfloat162_rn(ds_vec);
                }
                {
                    float2 ds_vec = ku::float2_mul(
                        ku::float2_mul(p[base_idx + 3], ku::float2_add(dp[3], neg_delta_f2)),
                        sm_scale_f2
                    );
                    ds_pack.a67 = __float22bfloat162_rn(ds_vec);
                }
                *reinterpret_cast<bf16x8*>(sDS_base + ch * SMEM_VEC_STRIDE) = ds_pack;
            }
            fence_view_async_shared();
            __threadfence_block();
            plan.bar_ds_ready.arrive(static_cast<uint32_t>(cta_idx));
            NamedBarrier::arrive_and_wait(kNumSoftmaxAndDQTransferWarps * kThreadsPerWarp, 2);
            if (warp_idx == 0 && elect_one_sync()) {
                Tensor gdS = flat_divide(
                    tma_params.tma_dS.get_tma_tensor(tma_params.shape_dS)(_, _, s_q_idx),
                    Tile<Int<B_H / 2>>{}
                )(_, cta_idx, _);
                auto thr_tma_ds = tma_params.tma_dS.get_slice(_0{});
                cute::copy(
                    tma_params.tma_dS,
                    thr_tma_ds.partition_S(sDS_store),
                    thr_tma_ds.partition_D(gdS)
                );
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();
            }
        }

        const int final_phase = (num_k_blocks - 1) & 1;
        plan.bar_dq_ready.wait(final_phase);
        ku::tcgen05_after_thread_sync();

        {
            constexpr int dQ_ROWS = B_H / 2;
            constexpr int SQ_FLOATS_PER_HALF = D_sQ / 2;
            constexpr int TQ_FLOATS_PER_HALF = D_tQ / 2;
            constexpr int TMEM_CHUNK_FLOATS = 32;
            constexpr int TMEM_CHUNK_FLOAT2 = TMEM_CHUNK_FLOATS / 2;

            Tensor sdQ = make_tensor(make_smem_ptr(plan.u.dq.data()), SmemLayoutQ{});
            const int row_in_cta = idx_in_softmax % dQ_ROWS;
            const int col_half = idx_in_softmax / dQ_ROWS;

            const uint32_t tmem_addr_dq = kTmemBase + (row_in_cta << 16) + tmem_cols::dQ;

            CUTE_UNROLL
            for (int chunk = 0; chunk < SQ_FLOATS_PER_HALF / TMEM_CHUNK_FLOATS; ++chunk) {
                const int chunk_col_base = chunk * TMEM_CHUNK_FLOATS;
                float2 dq_chunk[TMEM_CHUNK_FLOAT2];
                ku::tmem_ld_32dp32bNx<TMEM_CHUNK_FLOATS>(
                    tmem_addr_dq + col_half * SQ_FLOATS_PER_HALF + chunk_col_base,
                    dq_chunk
                );
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                CUTE_UNROLL
                for (int i = 0; i < TMEM_CHUNK_FLOAT2; ++i) {
                    const int col = col_half * SQ_FLOATS_PER_HALF + chunk_col_base + i * 2;
                    sdQ(row_in_cta, col) = bf16(dq_chunk[i].x);
                    sdQ(row_in_cta, col + 1) = bf16(dq_chunk[i].y);
                }
            }

            CUTE_UNROLL
            for (int chunk = 0; chunk < TQ_FLOATS_PER_HALF / TMEM_CHUNK_FLOATS; ++chunk) {
                const int chunk_col_base = chunk * TMEM_CHUNK_FLOATS;
                float2 dq_chunk[TMEM_CHUNK_FLOAT2];
                ku::tmem_ld_32dp32bNx<TMEM_CHUNK_FLOATS>(
                    tmem_addr_dq + D_sQ / 2 + col_half * TQ_FLOATS_PER_HALF + chunk_col_base,
                    dq_chunk
                );
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                CUTE_UNROLL
                for (int i = 0; i < TMEM_CHUNK_FLOAT2; ++i) {
                    const int col = D_sQ + col_half * TQ_FLOATS_PER_HALF + chunk_col_base + i * 2;
                    sdQ(row_in_cta, col) = bf16(dq_chunk[i].x);
                    sdQ(row_in_cta, col + 1) = bf16(dq_chunk[i].y);
                }
            }

            fence_view_async_shared();
            NamedBarrier::arrive_and_wait(kNumSoftmaxAndDQTransferWarps * kThreadsPerWarp, 0);

            if (warp_idx == 0 && elect_one_sync()) {
                Tensor gdQ = flat_divide(
                    tma_params.tma_dQ.get_tma_tensor(tma_params.shape_dQ)(_, _, s_q_idx),
                    Tile<Int<B_H / 2>>{}
                )(_, cta_idx, _);
                auto thr_tma_dq = tma_params.tma_dQ.get_slice(_0{});
                cute::copy(
                    tma_params.tma_dQ,
                    thr_tma_dq.partition_S(sdQ),
                    thr_tma_dq.partition_D(gdQ)
                );
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();
            }
        }
    }

    if (warp_role == WarpRole::MmaAndKLoad) {
        cutlass::arch::warpgroup_reg_alloc<168>();

        if (elect_one_sync()) {
            auto sanitize_index = [&](int index, int topk_idx) {
                return (index >= 0 && index < params.s_kv && index <= max_kv_i && topk_idx < topk_length) ? index : 0;
            };

            auto sanitize_indices4 = [&](int4 raw, int base_topk_idx) {
                int4 out;
                out.x = sanitize_index(raw.x, base_topk_idx + 0);
                out.y = sanitize_index(raw.y, base_topk_idx + 1);
                out.z = sanitize_index(raw.z, base_topk_idx + 2);
                out.w = sanitize_index(raw.w, base_topk_idx + 3);
                return out;
            };

            auto load_k_half = [&](int k_block, int half_idx, int load_phase) {
                constexpr int ROW_GROUPS_PER_HALF = (B_TOPK / 2) / 4;
                bf16* sK_base = plan.u.q_k.k.data();
                CUTE_UNROLL
                for (int local_row = 0; local_row < ROW_GROUPS_PER_HALF; ++local_row) {
                    const int base_topk_idx = k_block * B_TOPK + half_idx * (B_TOPK / 2) + local_row * 4;
                    int4 raw_indices4 = __ldg((const int4*)(gIndices_s + base_topk_idx));
                    int4 indices4 = sanitize_indices4(raw_indices4, base_topk_idx);
                    CUTE_UNROLL
                    for (int local_col = 0; local_col < D_K / 64; ++local_col) {
                        ku::tma_gather4_cta_group_2<true>(
                            &(tma_params.tensor_map_kv),
                            plan.bar_prologue_k,
                            sK_base + local_row * 4 * 64 + local_col * ((B_TOPK / 2) * 64),
                            local_col * 64,
                            indices4,
                            (int64_t)TMA::CacheHintSm90::EVICT_LAST
                        );
                    }
                }
                plan.bar_prologue_k.arrive_and_expect_tx((B_TOPK / 2) * D_K * sizeof(bf16));
                plan.bar_prologue_k.wait(load_phase & 1);
                ku::tcgen05_after_thread_sync();
            };

            UMMA::SmemDescriptor sQ_desc = UMMA::make_umma_desc<UMMA::Major::K>(
                make_tensor(
                    make_smem_ptr(plan.u.q_full.data() + (B_H / 2) * D_sQ),
                    tile_to_shape(
                        UMMA::Layout_K_SW128_Atom<bf16>{},
                        Shape<Int<B_H / 2>, Int<64>>{}
                    )
                )
            );

            if (cta_idx == 0) {
                plan.bar_prologue_q.arrive_and_expect_tx(B_H * D_Q * sizeof(bf16));
                plan.bar_prologue_q.wait(0);
                ku::tcgen05_after_thread_sync();

                CUTE_UNROLL
                for (int tile_idx = 0; tile_idx < NUM_tQ_TILES; ++tile_idx) {
                    CUTE_UNROLL
                    for (int subtile_idx = 0; subtile_idx < 8; ++subtile_idx) {
                        SM100_UTCCP_2x64dp128bitlw0213_2cta::copy(
                            sQ_desc + tile_idx * ((B_H / 2) * 128 / 16) + subtile_idx,
                            tmem_cols::tQ + tile_idx * 32 + subtile_idx * 4
                        );
                    }
                }
                ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_prologue_tQ, 1 | 2);

                plan.bar_prologue_dO.arrive_and_expect_tx(B_H * D_V * sizeof(bf16));
                plan.bar_prologue_dO.wait(0);
                ku::tcgen05_after_thread_sync();
            }

            plan.bar_prologue_tQ.wait(0);
            ku::tcgen05_after_thread_sync();

            Tensor sQ_sQ = make_tensor(make_smem_ptr(plan.u.q_k.sQ.data()), SmemLayoutSQ{});
            Tensor sK_sQ = make_tensor(make_smem_ptr(plan.u.q_k.k.data()), SmemLayoutKSQ{});
            Tensor sK_tQ = make_tensor(
                make_smem_ptr(plan.u.q_k.k.data() + (B_TOPK / 2) * D_sQ),
                SmemLayoutKTQ{}
            );
            Tensor sV = make_tensor(make_smem_ptr(plan.u.q_k.k.data()), SmemLayoutV{});
            Tensor sS_mma = make_tensor(make_smem_ptr(plan.s_ds.s.data()), SmemLayoutS{});
            Tensor sDS_mma = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutdS{});
            Tensor sDS_t = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutdSTransposed{});
            auto sDS_t_div = flat_divide(sDS_t, Shape<Int<B_H / 2>, Int<B_TOPK / 2>>{});
            Tensor sK_sQ_t = make_tensor(make_smem_ptr(plan.u.q_k.k.data()), SmemLayoutKSQTransposed{});
            Tensor sK_tQ_t = make_tensor(
                make_smem_ptr(plan.u.q_k.k.data() + (B_TOPK / 2) * D_sQ),
                SmemLayoutKTQTransposed{}
            );

            TiledMMA_P_tQ tiled_mma_P_tQ{};
            TiledMMA_P_sQ tiled_mma_P_sQ{};
            TiledMMA_dP tiled_mma_dP{};
            TiledMMA_dQ_sQ tiled_mma_dQ_sQ{};
            TiledMMA_dQ_tQ tiled_mma_dQ_tQ{};

            Tensor tQ_tmem = tiled_mma_P_tQ.get_slice(_0{}).make_fragment_A(
                partition_shape_A(tiled_mma_P_tQ, Shape<Int<B_H / 2>, Int<D_tQ>>{})
            );
            tQ_tmem.data().get() = tmem_cols::tQ;

            Tensor tP0 = partition_fragment_C(tiled_mma_P_tQ, Shape<Int<B_H / 2>, Int<B_TOPK / 2>>{});
            Tensor tP1 = partition_fragment_C(tiled_mma_P_tQ, Shape<Int<B_H / 2>, Int<B_TOPK / 2>>{});
            tP0.data().get() = tmem_cols::P;
            tP1.data().get() = tmem_cols::P + B_TOPK / 4;

            Tensor tdP0 = partition_fragment_C(tiled_mma_dP, Shape<Int<B_H / 2>, Int<B_TOPK / 2>>{});
            Tensor tdP1 = partition_fragment_C(tiled_mma_dP, Shape<Int<B_H / 2>, Int<B_TOPK / 2>>{});
            tdP0.data().get() = tmem_cols::dP;
            tdP1.data().get() = tmem_cols::dP + B_TOPK / 4;

            Tensor tdQ_sQ = partition_fragment_C(tiled_mma_dQ_sQ, Shape<Int<B_H / 2>, Int<D_sQ>>{});
            tdQ_sQ.data().get() = tmem_cols::dQ;
            Tensor tdQ_tQ = partition_fragment_C(tiled_mma_dQ_tQ, Shape<Int<B_H / 2>, Int<D_tQ>>{});
            tdQ_tQ.data().get() = tmem_cols::dQ + D_sQ / 2;

            CUTE_NO_UNROLL
            for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
                const int phase = k_block & 1;
                const bool dq_clear = (k_block == 0);
                const int base_load_phase = k_block * 4;

                load_k_half(k_block, 0, base_load_phase + 0);
                ku::utcmma_ss(tiled_mma_P_sQ, sQ_sQ, sK_sQ, tP0, true);
                ku::utcmma_ts(tiled_mma_P_tQ, tQ_tmem, sK_tQ, tP0, false);
                ku::utcmma_ss(tiled_mma_dP, sdO, sV, tdP0, true);

                load_k_half(k_block, 1, base_load_phase + 1);
                ku::utcmma_ss(tiled_mma_P_sQ, sQ_sQ, sK_sQ, tP1, true);
                ku::utcmma_ts(tiled_mma_P_tQ, tQ_tmem, sK_tQ, tP1, false);
                ku::utcmma_ss(tiled_mma_dP, sdO, sV, tdP1, true);

                ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_p_ready, 1 | 2);
                ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_dp_ready, 1 | 2);
                ku::tcgen05_after_thread_sync();

                plan.bar_s_ready.wait(phase);
                ku::tcgen05_after_thread_sync();
                plan.bar_ds_ready.wait(phase);
                ku::tcgen05_after_thread_sync();

                load_k_half(k_block, 0, base_load_phase + 2);
                ku::utcmma_ss(tiled_mma_dQ_sQ, sDS_t_div(_, _, _0{}), sK_sQ_t, tdQ_sQ, dq_clear);
                ku::utcmma_ss(tiled_mma_dQ_tQ, sDS_t_div(_, _, _0{}), sK_tQ_t, tdQ_tQ, dq_clear);

                load_k_half(k_block, 1, base_load_phase + 3);
                ku::utcmma_ss(tiled_mma_dQ_sQ, sDS_t_div(_, _, _1{}), sK_sQ_t, tdQ_sQ, false);
                ku::utcmma_ss(tiled_mma_dQ_tQ, sDS_t_div(_, _, _1{}), sK_tQ_t, tdQ_tQ, false);

                ku::umma_arrive_noelect(plan.bar_dq_ready);
                ku::tcgen05_after_thread_sync();
            }
        }
    }

    if (warp_role == WarpRole::KvValidLoad) {
        if (lane_idx < B_TOPK / 8) {
            CUTE_NO_UNROLL
            for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
                int32x8_t indices = ldg_256_indices((void*)(gIndices_s + k_block * B_TOPK + lane_idx * 8));
                auto is_valid = [&](int rel_idx, int index) -> char {
                    const int topk_idx = k_block * B_TOPK + lane_idx * 8 + rel_idx;
                    return index >= 0 && index < params.s_kv && index <= max_kv_i && topk_idx < topk_length;
                };
                const char mask =
                    (is_valid(7, indices.a7) << 7) |
                    (is_valid(6, indices.a6) << 6) |
                    (is_valid(5, indices.a5) << 5) |
                    (is_valid(4, indices.a4) << 4) |
                    (is_valid(3, indices.a3) << 3) |
                    (is_valid(2, indices.a2) << 2) |
                    (is_valid(1, indices.a1) << 1) |
                    (is_valid(0, indices.a0) << 0);

                plan.bar_k_valid_free.wait(((k_block & 1) ^ 1));
                plan.is_k_valid[lane_idx] = mask;
                plan.bar_k_valid_ready.arrive();
            }
        }
    }

    cluster_sync();

    if (warp_idx == 0 && elect_one_sync()) {
        TMEM::Allocator2Sm().free(kTmemBase, tmem_cols::kNumUsedCols);
    }
#endif
}

static void launch_dq_phase_capture(
    const SparseAttnBwdParams& params,
    bf16* s_out,
    bf16* ds_out,
    int stride_s_s_q,
    int stride_s_h_q,
    int stride_ds_s_q,
    int stride_ds_h_q
) {
    auto shape_Q = cute::make_shape(B_H, D_Q, params.s_q);
    auto tma_Q = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.q),
            cute::make_layout(
                shape_Q,
                cute::make_stride(params.stride_q_h_q, cute::_1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQ{}
    );

    auto shape_dO = cute::make_shape(B_H, D_V, params.s_q);
    auto tma_dO = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.dO),
            cute::make_layout(
                shape_dO,
                cute::make_stride(params.stride_dO_h_q, cute::_1{}, params.stride_dO_s_q)
            )
        ),
        SmemLayoutdO{}
    );

    auto shape_dQ = cute::make_shape(B_H, D_Q, params.s_q);
    auto tma_dQ = cute::make_tma_copy(
        cute::SM90_TMA_STORE{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.dQ),
            cute::make_layout(
                shape_dQ,
                cute::make_stride(params.stride_dQ_h_q, cute::_1{}, params.stride_dQ_s_q)
            )
        ),
        SmemLayoutQ{}
    );

    auto shape_S = cute::make_shape(B_H, B_TOPK, params.s_q);
    auto tma_S = cute::make_tma_copy(
        cute::SM90_TMA_STORE{},
        cute::make_tensor(
            cute::make_gmem_ptr(s_out),
            cute::make_layout(
                shape_S,
                cute::make_stride(stride_s_h_q, cute::_1{}, stride_s_s_q)
            )
        ),
        SmemLayoutdSTransposed{}
    );

    auto shape_dS = cute::make_shape(B_H, B_TOPK, params.s_q);
    auto tma_dS = cute::make_tma_copy(
        cute::SM90_TMA_STORE{},
        cute::make_tensor(
            cute::make_gmem_ptr(ds_out),
            cute::make_layout(
                shape_dS,
                cute::make_stride(stride_ds_h_q, cute::_1{}, stride_ds_s_q)
            )
        ),
        SmemLayoutdSTransposed{}
    );

    CUtensorMap tensor_map_kv;
    {
        uint64_t size[2] = {(uint64_t)D_K, (unsigned long)params.s_kv};
        uint64_t stride[1] = {(uint64_t)params.stride_kv_s_kv * sizeof(bf16)};
        uint32_t box_size[2] = {64, 1};
        uint32_t elem_stride[2] = {1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            const_cast<bf16*>(params.kv),
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

    using KernelTmaParams = TmaParams<
        decltype(shape_Q), decltype(tma_Q),
        decltype(shape_dO), decltype(tma_dO),
        decltype(shape_dQ), decltype(tma_dQ),
        decltype(shape_S), decltype(tma_S),
        decltype(shape_dS), decltype(tma_dS)
    >;

    KernelTmaParams tma_params = {
        shape_Q, tma_Q,
        shape_dO, tma_dO,
        shape_dQ, tma_dQ,
        shape_S, tma_S,
        shape_dS, tma_dS,
        tensor_map_kv
    };

    auto kernel = &dq_phase_kernel<KernelTmaParams>;
    dim3 grid(2 * params.s_q, 1, 1);
    dim3 block(NUM_THREADS, 1, 1);

    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    cudaLaunchConfig_t config;
    memset(&config, 0, sizeof(config));
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = SMEM_SIZE;
    config.stream = params.stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    KU_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, params, tma_params));
}

template<int DQK>
void run_bwd_dq_phase_capture_kernel(
    const SparseAttnBwdParams& params,
    bf16* s_out,
    bf16* ds_out,
    int stride_s_s_q,
    int stride_s_h_q,
    int stride_ds_s_q,
    int stride_ds_h_q
) {
    static_assert(DQK == D_QK);

    KU_ASSERT(params.d_qk == DQK);
    KU_ASSERT(params.d_v == D_V);
    KU_ASSERT(params.h_q == B_H);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk == B_TOPK);

    sm100::bwd::head128::run_bwd_preprocess_delta_kernel<DQK>(params);
    launch_dq_phase_capture(
        params,
        s_out,
        ds_out,
        stride_s_s_q,
        stride_s_h_q,
        stride_ds_s_q,
        stride_ds_h_q
    );
}

}  // namespace sm100::bwd::head128_2kernels::dq

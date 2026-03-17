#pragma once

#include "sm100/prefill/sparse/bwd/head128/preprocess_delta.cuh"

#include "dq_config_local.h"

#include <cstdio>
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
    SoftmaxAndDQTransfer = 0x1,
    KvTileTransfer = 0x2,
    Mma = 0x4,
    KvValidLoad = 0x5,
};

static constexpr int kNumSoftmaxAndDQTransferWarps = 4;
static constexpr int kNumKvTileTransferWarps = 2;
static constexpr int kNumMmaWarps = 1;
static constexpr int kNumKvValidLoadWarps = 1;
static constexpr int kThreadsPerWarp = 32;

static constexpr int kSoftmaxAndDQTransferFirstWarp = 0;
static constexpr int kKvTileTransferFirstWarp = kSoftmaxAndDQTransferFirstWarp + kNumSoftmaxAndDQTransferWarps;
static constexpr int kMmaFirstWarp = kKvTileTransferFirstWarp + kNumKvTileTransferWarps;
static constexpr int kKvValidLoadFirstWarp = kMmaFirstWarp + kNumMmaWarps;
static constexpr int kNumAssignedWarps =
    kNumSoftmaxAndDQTransferWarps + kNumKvTileTransferWarps +
    kNumMmaWarps + kNumKvValidLoadWarps;

static_assert(kNumAssignedWarps == 8, "Warp assignment must cover exactly 8 warps");
static_assert(kKvValidLoadFirstWarp + kNumKvValidLoadWarps == kNumAssignedWarps, "Warp role ranges must be contiguous");
static_assert(NUM_THREADS == kNumAssignedWarps * kThreadsPerWarp, "NUM_THREADS must match warp assignment");

CUTE_DEVICE
WarpRole warp_idx_to_role(int warp_idx) {
    if (warp_idx < kKvTileTransferFirstWarp) {
        return WarpRole::SoftmaxAndDQTransfer;
    }
    if (warp_idx < kMmaFirstWarp) {
        return WarpRole::KvTileTransfer;
    }
    if (warp_idx < kKvValidLoadFirstWarp) {
        return WarpRole::Mma;
    }
    return WarpRole::KvValidLoad;
}

CUTE_DEVICE
const char* warp_role_name(WarpRole role) {
    switch (role) {
        case WarpRole::SoftmaxAndDQTransfer:
            return "SOFTMAX";
        case WarpRole::KvTileTransfer:
            return "KV_COPY";
        case WarpRole::Mma:
            return "MMA";
        case WarpRole::KvValidLoad:
            return "KVALID";
    }
    return "UNKNOWN";
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
    const int max_kv_i = params.q_start_index_s + s_q_idx;
    const int tid = threadIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;
    const WarpRole warp_role = warp_idx_to_role(warp_idx);
    if (s_q_idx >= params.s_q) {
        return;
    }
    const int topk_length = params.topk_length == nullptr ?
        params.topk :
        min(max(__ldg(params.topk_length + s_q_idx), 0), params.topk);
    const int32_t* gIndices_s = params.indices + (int64_t)s_q_idx * params.stride_indices_s_q;
    const float* lse_s = params.lse + (int64_t)s_q_idx * params.h_q;
    const float* delta_s = params.delta + (int64_t)s_q_idx * params.stride_delta_s_q;
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);
    const bool dbg_block = blockIdx.x < 2;
    const bool dbg_thread0 = dbg_block && tid == 0;
    const bool dbg_role_thread = dbg_block && lane_idx == 0;
    const bool dbg_softmax = dbg_role_thread && warp_idx == kSoftmaxAndDQTransferFirstWarp;
    const bool dbg_kv_transfer = dbg_role_thread && warp_idx == kKvTileTransferFirstWarp;
    const bool dbg_mma = dbg_role_thread && warp_idx == kMmaFirstWarp;
    const bool dbg_kv_valid = dbg_role_thread && warp_idx == kKvValidLoadFirstWarp;

    if (dbg_thread0) {
        printf("[DBG][B%d SQ%d CTA%d] enter topk=%d num_k=%d max_kv=%d\n",
               (int)blockIdx.x, s_q_idx, cta_idx, topk_length, num_k_blocks, max_kv_i);
    }
    if (dbg_softmax || dbg_kv_transfer || dbg_mma || dbg_kv_valid) {
        printf("[DBG][B%d SQ%d CTA%d W%d %s] role_enter\n",
               (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, warp_role_name(warp_role));
    }

    if (tid == 0) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q_nope.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_Q_rope.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dQ.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_S.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dS.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv));
    }

    if (warp_idx == 0 && elect_one_sync()) {
        plan.bar_prologue_q_nope.init(1);
        plan.bar_prologue_q_rope.init(1);
        plan.bar_prologue_kv.init(1);
        plan.bar_prologue_dO.init(1);
        plan.bar_p_ready.init(1);
        plan.bar_dp_ready.init(1);
        plan.bar_s_ready.init(kNumSoftmaxAndDQTransferWarps * kThreadsPerWarp);
        plan.bar_ds_ready.init(kNumSoftmaxAndDQTransferWarps * kThreadsPerWarp);
        plan.bar_k_valid_ready.init(B_TOPK / 8);
        plan.bar_k_valid_free.init(kNumSoftmaxAndDQTransferWarps * kThreadsPerWarp);
        plan.bar_kv_peer_cp_async.init(1);
        plan.bar_kv_peer_ready.init(1);
        plan.bar_dq_ready.init(1);
        fence_barrier_init();
    }

    cluster_sync();

    if (dbg_thread0) {
        printf("[DBG][B%d SQ%d CTA%d] post_init cluster_sync_done\n",
               (int)blockIdx.x, s_q_idx, cta_idx);
    }

    Tensor sQNoPE = make_tensor(make_smem_ptr(plan.u.q_kv.q_nope.data()), SmemLayoutQNoPE{});
    Tensor sQRoPE = make_tensor(make_smem_ptr(plan.u.q_kv.q_rope.data()), SmemLayoutQRoPE{});
    Tensor sKNoPE = make_tensor(make_smem_ptr(plan.u.q_kv.k_nope.data()), SmemLayoutKNoPE{});
    Tensor sKRoPE = make_tensor(make_smem_ptr(plan.u.q_kv.k_rope.data()), SmemLayoutKRoPE{});
    Tensor sdO = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO{});
    Tensor sS_store = make_tensor(make_smem_ptr(plan.s_ds.s.data()), SmemLayoutdSTransposed{});
    Tensor sDS_store = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutdSTransposed{});

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            Tensor gQNoPE = flat_divide(
                tma_params.tma_Q_nope.get_tma_tensor(tma_params.shape_Q_nope)(_, _, s_q_idx),
                Tile<Int<B_H / 2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_Q_nope, gQNoPE, sQNoPE, plan.bar_prologue_q_nope, TMA::CacheHintSm90::EVICT_FIRST);

            Tensor gQRoPE = flat_divide(
                tma_params.tma_Q_rope.get_tma_tensor(tma_params.shape_Q_rope)(_, _, s_q_idx),
                Tile<Int<B_H / 2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_Q_rope, gQRoPE, sQRoPE, plan.bar_prologue_q_rope, TMA::CacheHintSm90::EVICT_FIRST);

            Tensor gdO = flat_divide(
                tma_params.tma_dO.get_tma_tensor(tma_params.shape_dO)(_, _, s_q_idx),
                Tile<Int<B_H / 2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_dO, gdO, sdO, plan.bar_prologue_dO, TMA::CacheHintSm90::EVICT_FIRST);
        }

        TMEM::Allocator2Sm().allocate(tmem_cols::kNumUsedCols, plan.tmem_start_addr.data());
        TMEM::Allocator2Sm().release_allocation_lock();
    }
    __syncthreads();

    const uint32_t tmem_base = plan.tmem_start_addr.data()[0];

    if (dbg_thread0) {
        printf("[DBG][B%d SQ%d CTA%d] prologue_ready tmem_base=%u\n",
               (int)blockIdx.x, s_q_idx, cta_idx, tmem_base);
    }

    if (warp_role == WarpRole::SoftmaxAndDQTransfer) {
        const int idx_in_softmax = (warp_idx - kSoftmaxAndDQTransferFirstWarp) * kThreadsPerWarp + lane_idx;
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
        const uint32_t tmem_p_addr = tmem_base + (tmem_lane << 16) + tmem_cols::P;
        const uint32_t tmem_dp_addr = tmem_base + (tmem_lane << 16) + tmem_cols::dP;
        const int row_in_tile = idx_in_softmax % S_DS_ROWS_PER_CTA;
        const int col_half = idx_in_softmax / S_DS_ROWS_PER_CTA;
        bf16* sS_base = plan.s_ds.s.data() +
            row_in_tile * S_DS_VEC_ELEMS + col_half * S_DS_ROWS_PER_CTA * S_DS_COLS_PER_THREAD;
        bf16* sDS_base = plan.s_ds.ds.data() +
            row_in_tile * S_DS_VEC_ELEMS + col_half * S_DS_ROWS_PER_CTA * S_DS_COLS_PER_THREAD;

        constexpr int SMEM_VEC_F2 = S_DS_VEC_ELEMS / 2;
        constexpr int NUM_SMEM_VEC_STORES = S_DS_COLS_PER_THREAD / S_DS_VEC_ELEMS;
        constexpr int SMEM_VEC_STRIDE = S_DS_ROWS_PER_CTA * S_DS_VEC_ELEMS;
        static_assert(SMEM_VEC_F2 == 4, "Softmax vectorized write path expects 4 float2 per 128-bit store.");

        if (dbg_softmax) {
            printf("[DBG][B%d SQ%d CTA%d W%d SOFTMAX] enter row=%d row_lse=%f neg_delta=%f tmem_p=0x%x tmem_dp=0x%x\n",
                   (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, global_row_idx, row_lse, neg_delta_val,
                   tmem_p_addr, tmem_dp_addr);
        }

        CUTE_NO_UNROLL
        for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
            const int phase = k_block & 1;
            const bool dbg_edge_k = dbg_softmax && (k_block == 0 || k_block + 1 == num_k_blocks);

            if (dbg_edge_k) {
                printf("[DBG][B%d SQ%d CTA%d W%d SOFTMAX] k=%d phase=%d wait_p_ready\n",
                       (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, k_block, phase);
            }

            plan.bar_p_ready.wait(phase);
            ku::tcgen05_after_thread_sync();

            float2 p[(B_TOPK / 2) / 2];
            ku::tmem_ld_32dp32bNx<B_TOPK / 2>(tmem_p_addr, p);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();

            plan.bar_k_valid_ready.wait(phase);
            const uint32_t is_k_valid_lo =
                *(uint32_t*)(plan.is_k_valid + (idx_in_softmax >= S_DS_ROWS_PER_CTA ? B_TOPK / 8 / 2 : 0));
            if (dbg_edge_k) {
                printf("[DBG][B%d SQ%d CTA%d W%d SOFTMAX] k=%d phase=%d k_valid_mask=0x%08x\n",
                       (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, k_block, phase, is_k_valid_lo);
            }
            float* p_float = (float*)p;
            CUTE_UNROLL
            for (int i = 0; i < B_TOPK / 2; ++i) {
                if (!(is_k_valid_lo >> i & 1)) {
                    p_float[i] = -CUDART_INF_F;
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
            NamedBarrier::arrive_and_wait(kNumSoftmaxAndDQTransferWarps * kThreadsPerWarp, 2);
            if (dbg_edge_k) {
                printf("[DBG][B%d SQ%d CTA%d W%d SOFTMAX] k=%d phase=%d s_ready\n",
                       (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, k_block, phase);
            }

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
            if (dbg_edge_k) {
                printf("[DBG][B%d SQ%d CTA%d W%d SOFTMAX] k=%d phase=%d dp_ready\n",
                       (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, k_block, phase);
            }

            constexpr int DP_CHUNK_F2 = SMEM_VEC_F2;
            constexpr int NUM_DP_CHUNKS = (B_TOPK / 2) / 2 / DP_CHUNK_F2;
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
            NamedBarrier::arrive_and_wait(kNumSoftmaxAndDQTransferWarps * kThreadsPerWarp, 3);
            if (dbg_edge_k) {
                printf("[DBG][B%d SQ%d CTA%d W%d SOFTMAX] k=%d phase=%d ds_ready\n",
                       (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, k_block, phase);
            }

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
        if (dbg_softmax) {
            printf("[DBG][B%d SQ%d CTA%d W%d SOFTMAX] wait_dq_ready final_phase=%d\n",
                   (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, final_phase);
        }
        plan.bar_dq_ready.wait(final_phase);
        ku::tcgen05_after_thread_sync();
        if (dbg_softmax) {
            printf("[DBG][B%d SQ%d CTA%d W%d SOFTMAX] dq_ready final_phase=%d\n",
                   (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, final_phase);
        }

        {
            constexpr int dQ_ROWS = B_H / 2;
            constexpr int NOPE_FLOATS_PER_HALF = 256 / 2;
            constexpr int NOPE_CHUNKS = 8;
            constexpr int NOPE_CHUNK_FLOATS = NOPE_FLOATS_PER_HALF / NOPE_CHUNKS;
            constexpr int NOPE_CHUNK_FLOAT2 = NOPE_CHUNK_FLOATS / 2;
            constexpr int ROPE_FLOAT2_PER_ROW = D_ROPE / 2 / 2;

            Tensor sdQ = make_tensor(make_smem_ptr(plan.u.dq.data()), SmemLayoutQ{});

            const int row_in_cta = idx_in_softmax % dQ_ROWS;
            const int col_half = idx_in_softmax / dQ_ROWS;

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
                    int col = col_half * NOPE_FLOATS_PER_HALF + chunk_col_base + i * 2;
                    sdQ(row_in_cta, col) = bf16(dq_chunk[i].x);
                    sdQ(row_in_cta, col + 1) = bf16(dq_chunk[i].y);
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
                    int col = 256 + col_half * NOPE_FLOATS_PER_HALF + chunk_col_base + i * 2;
                    sdQ(row_in_cta, col) = bf16(dq_chunk[i].x);
                    sdQ(row_in_cta, col + 1) = bf16(dq_chunk[i].y);
                }
            }

            constexpr int ROPE_F2_CHUNK = 8;
            constexpr int ROPE_NUM_CHUNKS = ROPE_FLOAT2_PER_ROW / ROPE_F2_CHUNK;
            const uint32_t tmem_addr_dq_rope = tmem_base + (row_in_cta << 16) + tmem_cols::dQ_RoPE;
            CUTE_UNROLL
            for (int rch = 0; rch < ROPE_NUM_CHUNKS; ++rch) {
                float2 dq_rope_chunk[ROPE_F2_CHUNK];
                ku::tmem_ld_32dp32bNx<ROPE_F2_CHUNK * 2>(
                    tmem_addr_dq_rope + rch * ROPE_F2_CHUNK * 2, dq_rope_chunk);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                CUTE_UNROLL
                for (int i = 0; i < ROPE_F2_CHUNK; ++i) {
                    const int fi = rch * ROPE_F2_CHUNK + i;
                    int col = D_V + col_half * (D_ROPE / 2) + fi * 2;
                    sdQ(row_in_cta, col) = bf16(dq_rope_chunk[i].x);
                    sdQ(row_in_cta, col + 1) = bf16(dq_rope_chunk[i].y);
                }
            }

            fence_view_async_shared();
            NamedBarrier::arrive_and_wait(kNumSoftmaxAndDQTransferWarps * kThreadsPerWarp, 0);

            if (warp_idx == 0 && elect_one_sync()) {
                if (dbg_block) {
                    printf("[DBG][B%d SQ%d CTA%d W%d SOFTMAX] dQ_tma_store_begin\n",
                           (int)blockIdx.x, s_q_idx, cta_idx, warp_idx);
                }
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
                if (dbg_block) {
                    printf("[DBG][B%d SQ%d CTA%d W%d SOFTMAX] dQ_tma_store_done\n",
                           (int)blockIdx.x, s_q_idx, cta_idx, warp_idx);
                }
            }
        }
        if (dbg_softmax) {
            printf("[DBG][B%d SQ%d CTA%d W%d SOFTMAX] exit\n",
                   (int)blockIdx.x, s_q_idx, cta_idx, warp_idx);
        }
    }

    if (warp_role == WarpRole::KvTileTransfer) {
        constexpr int NUM_WARPS = kNumKvTileTransferWarps;
        static_assert((B_TOPK / 2) % (4 * NUM_WARPS) == 0);
        constexpr int NUM_LOCAL_ROWS_PER_WARP = (B_TOPK / 2) / 4 / NUM_WARPS;
        constexpr int KV_PEER_ELEMENTS = (B_TOPK / 2) * D_K;
        const int local_warp_idx = warp_idx - kKvTileTransferFirstWarp;

        if (dbg_kv_transfer) {
            printf("[DBG][B%d SQ%d CTA%d W%d KV_COPY] enter local_warp=%d\n",
                   (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, local_warp_idx);
        }

        CUTE_NO_UNROLL
        for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
            const int phase = k_block & 1;
            const bool dbg_edge_k = dbg_kv_transfer && (k_block == 0 || k_block + 1 == num_k_blocks);

            if (k_block > 0) {
                plan.bar_dq_ready.wait((k_block - 1) & 1);
            }

            if (elect_one_sync()) {
                bf16* sKV_base = plan.u.q_kv.k_nope.data() + local_warp_idx * 4 * 64;
                int4 indices4[NUM_LOCAL_ROWS_PER_WARP];
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    indices4[local_row] = __ldg(
                        (const int4*)(gIndices_s + k_block * B_TOPK + cta_idx * (B_TOPK / 2)) +
                        local_row * NUM_WARPS + local_warp_idx
                    );
                }
                if (dbg_block && local_warp_idx == 0 && (k_block == 0 || k_block + 1 == num_k_blocks)) {
                    printf("[DBG][B%d SQ%d CTA%d W%d KV_COPY] k=%d phase=%d gather indices4={%d,%d,%d,%d}\n",
                           (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, k_block, phase,
                           indices4[0].x, indices4[0].y, indices4[0].z, indices4[0].w);
                }

                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    CUTE_UNROLL
                    for (int local_col = 0; local_col < D_K / 64; ++local_col) {
                        ku::tma_gather4_cta_group_2<true>(
                            &(tma_params.tensor_map_kv),
                            plan.bar_prologue_kv,
                            sKV_base + local_row * (4 * NUM_WARPS) * 64 + local_col * ((B_TOPK / 2) * 64),
                            local_col * 64,
                            indices4[local_row],
                            (int64_t)TMA::CacheHintSm90::EVICT_LAST
                        );
                    }
                }
            }

            plan.bar_p_ready.wait(phase);
            NamedBarrier::arrive_and_wait(NUM_WARPS * kThreadsPerWarp, 1);
            if (dbg_edge_k) {
                printf("[DBG][B%d SQ%d CTA%d W%d KV_COPY] k=%d phase=%d p_ready\n",
                       (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, k_block, phase);
            }

            if (local_warp_idx == 0 && elect_one_sync()) {
                plan.bar_kv_peer_cp_async.arrive_and_expect_tx(sizeof(bf16) * KV_PEER_ELEMENTS);
                bf16* peer_kv_peer_ptr = kerutils::get_peer_addr(plan.u.q_kv.kv_peer.data());
                transac_bar_t* peer_bar_ptr = kerutils::get_peer_addr(&plan.bar_kv_peer_cp_async);
                kerutils::cp_async_bulk_shared_cta_to_shared_cluster(
                    peer_kv_peer_ptr,
                    plan.u.q_kv.k_nope.data(),
                    sizeof(bf16) * KV_PEER_ELEMENTS,
                    *peer_bar_ptr
                );
                fence_view_async_shared();
                if (dbg_block && (k_block == 0 || k_block + 1 == num_k_blocks)) {
                    printf("[DBG][B%d SQ%d CTA%d W%d KV_COPY] k=%d phase=%d peer_cp_async_launched bytes=%d\n",
                           (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, k_block, phase,
                           (int)(sizeof(bf16) * KV_PEER_ELEMENTS));
                }
            }

            NamedBarrier::arrive_and_wait(NUM_WARPS * kThreadsPerWarp, 1);
            plan.bar_kv_peer_cp_async.wait(phase);
            if (dbg_edge_k) {
                printf("[DBG][B%d SQ%d CTA%d W%d KV_COPY] k=%d phase=%d peer_cp_async_done\n",
                       (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, k_block, phase);
            }

            if (local_warp_idx == 0 && elect_one_sync()) {
                plan.bar_kv_peer_ready.arrive(static_cast<uint32_t>(cta_idx));
                if (dbg_block && (k_block == 0 || k_block + 1 == num_k_blocks)) {
                    printf("[DBG][B%d SQ%d CTA%d W%d KV_COPY] k=%d phase=%d peer_ready_arrive\n",
                           (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, k_block, phase);
                }
            }
        }
        if (dbg_kv_transfer) {
            printf("[DBG][B%d SQ%d CTA%d W%d KV_COPY] exit\n",
                   (int)blockIdx.x, s_q_idx, cta_idx, warp_idx);
        }
    }

    if (warp_role == WarpRole::Mma) {
        TiledMMA_P tiled_mma_P{};
        TiledMMA_dP tiled_mma_dP{};
        Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H / 2>, Int<B_TOPK>>{});
        Tensor tdP = partition_fragment_C(tiled_mma_dP, Shape<Int<B_H / 2>, Int<B_TOPK>>{});
        tP.data().get() = tmem_cols::P;
        tdP.data().get() = tmem_cols::dP;

        Tensor sV = make_tensor(make_smem_ptr(plan.u.q_kv.k_nope.data()), SmemLayoutV{});

        if (elect_one_sync()) {
            if (dbg_block) {
                printf("[DBG][B%d SQ%d CTA%d W%d MMA] enter tmem_base=%u\n",
                       (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, tmem_base);
            }
            if (cta_idx == 0) {
                plan.bar_prologue_q_nope.arrive_and_expect_tx(B_H * D_V * sizeof(bf16));
                plan.bar_prologue_q_rope.arrive_and_expect_tx(B_H * D_ROPE * sizeof(bf16));
                plan.bar_prologue_dO.arrive_and_expect_tx(B_H * D_V * sizeof(bf16));
                plan.bar_prologue_q_nope.wait(0);
                plan.bar_prologue_q_rope.wait(0);
                plan.bar_prologue_dO.wait(0);
                ku::tcgen05_after_thread_sync();
                if (dbg_block) {
                    printf("[DBG][B%d SQ%d CTA%d W%d MMA] q_dO_prologue_done\n",
                           (int)blockIdx.x, s_q_idx, cta_idx, warp_idx);
                }
            }

            TiledMMA_dQ tiled_mma_dQ{};
            TiledMMA_dQ_RoPE tiled_mma_dQ_RoPE{};
            Tensor tdQ_part0 = partition_fragment_C(tiled_mma_dQ, Shape<Int<B_H / 2>, Int<256>>{});
            tdQ_part0.data().get() = tmem_cols::dQ;
            Tensor tdQ_part1 = partition_fragment_C(tiled_mma_dQ, Shape<Int<B_H / 2>, Int<256>>{});
            tdQ_part1.data().get() = tmem_cols::dQ + 128;
            Tensor tdQ_RoPE = partition_fragment_C(tiled_mma_dQ_RoPE, Shape<Int<B_H / 2>, Int<D_ROPE>>{});
            tdQ_RoPE.data().get() = tmem_cols::dQ_RoPE;

            Tensor sDS_t = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutdSTransposed{});
            auto sDS_t_div = flat_divide(sDS_t, Shape<Int<B_H / 2>, Int<B_TOPK / 2>>{});

            Tensor sK_nope_t_full = make_tensor(make_smem_ptr(plan.u.q_kv.k_nope.data()), SmemLayoutKVTilesTransposed<D_V / 64>{});
            auto sK_nope_t_div = flat_divide(sK_nope_t_full, Shape<Int<256>, Int<B_TOPK / 2>>{});
            Tensor sK_rope_t = make_tensor(make_smem_ptr(plan.u.q_kv.k_rope.data()), SmemLayoutKRoPETransposed{});

            Tensor sK_peer_nope_t_full = make_tensor(make_smem_ptr(plan.u.q_kv.kv_peer.data()), SmemLayoutKVTilesTransposed<D_V / 64>{});
            auto sK_peer_nope_t_div = flat_divide(sK_peer_nope_t_full, Shape<Int<256>, Int<B_TOPK / 2>>{});
            Tensor sK_peer_rope_t = make_tensor(
                make_smem_ptr(plan.u.q_kv.kv_peer.data() + (B_TOPK / 2) * D_V),
                SmemLayoutKRoPETransposed{}
            );

            CUTE_NO_UNROLL
            for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
                const int phase = k_block & 1;
                const bool dq_clear = (k_block == 0);
                const bool dbg_edge_k = dbg_block && (k_block == 0 || k_block + 1 == num_k_blocks);

                if (cta_idx == 0) {
                    if (dbg_edge_k) {
                        printf("[DBG][B%d SQ%d CTA%d W%d MMA] k=%d phase=%d begin_p_dp dq_clear=%d\n",
                               (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, k_block, phase, dq_clear ? 1 : 0);
                    }
                    plan.bar_prologue_kv.arrive_and_expect_tx(B_TOPK * D_K * sizeof(bf16));
                    plan.bar_prologue_kv.wait(phase);
                    ku::tcgen05_after_thread_sync();
                    ku::utcmma_ss(tiled_mma_P, sQNoPE, sKNoPE, tP, true);
                    ku::utcmma_ss(tiled_mma_P, sQRoPE, sKRoPE, tP, false);
                    ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_p_ready, 1 | 2);
                    ku::tcgen05_after_thread_sync();

                    ku::utcmma_ss(tiled_mma_dP, sdO, sV, tdP, true);
                    ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_dp_ready, 1 | 2);
                    ku::tcgen05_after_thread_sync();
                    if (dbg_edge_k) {
                        printf("[DBG][B%d SQ%d CTA%d W%d MMA] k=%d phase=%d p_dp_done\n",
                               (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, k_block, phase);
                    }
                }

                if (dbg_edge_k) {
                    printf("[DBG][B%d SQ%d CTA%d W%d MMA] k=%d phase=%d dq_begin branch=%d\n",
                           (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, k_block, phase, cta_idx);
                }
                if (cta_idx == 0) {
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _0{}), sK_nope_t_div(_, _, _0{}, _0{}), tdQ_part0, dq_clear);
                    if (dbg_edge_k) {
                        printf("[DBG][B%d SQ%d CTA%d W%d MMA] k=%d phase=%d dq_part0_local_done\n",
                               (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, k_block, phase);
                    }
                    plan.bar_kv_peer_ready.wait(phase);
                    ku::tcgen05_after_thread_sync();
                    if (dbg_edge_k) {
                        printf("[DBG][B%d SQ%d CTA%d W%d MMA] k=%d phase=%d kv_peer_ready\n",
                               (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, k_block, phase);
                    }
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _1{}), sK_peer_nope_t_div(_, _, _0{}, _0{}), tdQ_part0, false);

                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _0{}), sK_nope_t_div(_, _, _1{}, _0{}), tdQ_part1, dq_clear);
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _1{}), sK_peer_nope_t_div(_, _, _1{}, _0{}), tdQ_part1, false);

                    ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t_div(_, _, _0{}, _0{}), sK_rope_t, tdQ_RoPE, dq_clear);
                    ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t_div(_, _, _0{}, _1{}), sK_peer_rope_t, tdQ_RoPE, false);
                } else {
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _1{}), sK_nope_t_div(_, _, _0{}, _0{}), tdQ_part0, dq_clear);
                    if (dbg_edge_k) {
                        printf("[DBG][B%d SQ%d CTA%d W%d MMA] k=%d phase=%d dq_part0_local_done\n",
                               (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, k_block, phase);
                    }
                    plan.bar_kv_peer_ready.wait(phase);
                    ku::tcgen05_after_thread_sync();
                    if (dbg_edge_k) {
                        printf("[DBG][B%d SQ%d CTA%d W%d MMA] k=%d phase=%d kv_peer_ready\n",
                               (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, k_block, phase);
                    }
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _0{}), sK_peer_nope_t_div(_, _, _0{}, _0{}), tdQ_part0, false);

                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _1{}), sK_nope_t_div(_, _, _1{}, _0{}), tdQ_part1, dq_clear);
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _0{}), sK_peer_nope_t_div(_, _, _1{}, _0{}), tdQ_part1, false);

                    ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t_div(_, _, _0{}, _1{}), sK_rope_t, tdQ_RoPE, dq_clear);
                    ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t_div(_, _, _0{}, _0{}), sK_peer_rope_t, tdQ_RoPE, false);
                }
                if (dbg_edge_k) {
                    printf("[DBG][B%d SQ%d CTA%d W%d MMA] k=%d phase=%d dq_done\n",
                           (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, k_block, phase);
                }
                ku::umma_arrive_noelect(plan.bar_dq_ready);
                ku::tcgen05_after_thread_sync();
                if (dbg_edge_k) {
                    printf("[DBG][B%d SQ%d CTA%d W%d MMA] k=%d phase=%d dq_ready_arrive\n",
                           (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, k_block, phase);
                }
            }
            if (dbg_block) {
                printf("[DBG][B%d SQ%d CTA%d W%d MMA] exit\n",
                       (int)blockIdx.x, s_q_idx, cta_idx, warp_idx);
            }
        }
    }

    if (warp_role == WarpRole::KvValidLoad) {
        if (lane_idx < B_TOPK / 8) {
            if (dbg_kv_valid) {
                printf("[DBG][B%d SQ%d CTA%d W%d KVALID] enter\n",
                       (int)blockIdx.x, s_q_idx, cta_idx, warp_idx);
            }
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                int32x8_t indices = ldg_256_indices((void*)(gIndices_s + k * B_TOPK + lane_idx * 8));
                auto is_valid = [&](int rel_idx, int index) -> char {
                    const int topk_idx = k * B_TOPK + lane_idx * 8 + rel_idx;
                    return index >= 0 && index < params.s_kv && index <= max_kv_i && topk_idx < topk_length;
                };
                char is_ks_valid_mask =
                    is_valid(7, indices.a7) << 7 |
                    is_valid(6, indices.a6) << 6 |
                    is_valid(5, indices.a5) << 5 |
                    is_valid(4, indices.a4) << 4 |
                    is_valid(3, indices.a3) << 3 |
                    is_valid(2, indices.a2) << 2 |
                    is_valid(1, indices.a1) << 1 |
                    is_valid(0, indices.a0) << 0;
                if (dbg_kv_valid && (k == 0 || k + 1 == num_k_blocks)) {
                    printf("[DBG][B%d SQ%d CTA%d W%d KVALID] k=%d mask=0x%02x idx0=%d idx7=%d\n",
                           (int)blockIdx.x, s_q_idx, cta_idx, warp_idx, k,
                           static_cast<unsigned int>(static_cast<unsigned char>(is_ks_valid_mask)),
                           indices.a0, indices.a7);
                }

                plan.bar_k_valid_free.wait((k & 1) ^ 1);
                plan.is_k_valid[lane_idx] = is_ks_valid_mask;
                plan.bar_k_valid_ready.arrive();
            }
            if (dbg_kv_valid) {
                printf("[DBG][B%d SQ%d CTA%d W%d KVALID] exit\n",
                       (int)blockIdx.x, s_q_idx, cta_idx, warp_idx);
            }
        }
    }

    cluster_sync();

    if (dbg_thread0) {
        printf("[DBG][B%d SQ%d CTA%d] kernel_exit cluster_sync_done\n",
               (int)blockIdx.x, s_q_idx, cta_idx);
    }

    if (warp_idx == 0 && elect_one_sync()) {
        TMEM::Allocator2Sm().free(tmem_base, tmem_cols::kNumUsedCols);
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
    auto shape_Q_nope = cute::make_shape(B_H, D_V, params.s_q);
    auto tma_Q_nope = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.q),
            cute::make_layout(
                shape_Q_nope,
                cute::make_stride(params.stride_q_h_q, cute::_1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQNoPE{}
    );

    auto shape_Q_rope = cute::make_shape(B_H, D_ROPE, params.s_q);
    auto tma_Q_rope = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.q + D_V),
            cute::make_layout(
                shape_Q_rope,
                cute::make_stride(params.stride_q_h_q, cute::_1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQRoPE{}
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
        decltype(shape_Q_nope), decltype(tma_Q_nope),
        decltype(shape_Q_rope), decltype(tma_Q_rope),
        decltype(shape_dO), decltype(tma_dO),
        decltype(shape_dQ), decltype(tma_dQ),
        decltype(shape_S), decltype(tma_S),
        decltype(shape_dS), decltype(tma_dS)
    >;

    KernelTmaParams tma_params = {
        shape_Q_nope, tma_Q_nope,
        shape_Q_rope, tma_Q_rope,
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

#pragma once

#include <cuda.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/cluster_launch.hpp>
#include <kerutils/kerutils.cuh>

#include "params.h"
#include "defines.h"

namespace sm100::bwd::head128_2kernels::dq {

using namespace cute;

template<
    typename Shape_Q, typename TMA_Q,
    typename Shape_dO, typename TMA_dO,
    typename Shape_dQ, typename TMA_dQ,
    typename Shape_S, typename TMA_S,
    typename Shape_dS, typename TMA_dS
>
struct TmaParams {
    Shape_Q shape_Q;
    TMA_Q tma_Q;
    Shape_dO shape_dO;
    TMA_dO tma_dO;
    Shape_dQ shape_dQ;
    TMA_dQ tma_dQ;
    Shape_S shape_S;
    TMA_S tma_S;
    Shape_dS shape_dS;
    TMA_dS tma_dS;
    CUtensorMap tensor_map_kv;
};

static constexpr int D_QK = 576;
static constexpr int D_Q = D_QK;
static constexpr int D_K = D_QK;
static constexpr int D_V = 512;
static constexpr int D_ROPE = D_Q - D_V;
static constexpr int B_H = 128;
static constexpr int B_TOPK = 128;
static constexpr int NUM_THREADS = 6 * 32;
static constexpr int S_DS_VEC_ELEMS = 8;
static constexpr int S_DS_ROWS_PER_CTA = B_H / 2;
static constexpr int S_DS_COLS_PER_THREAD = B_TOPK / 2;

static_assert(S_DS_ROWS_PER_CTA == B_TOPK / 2, "S/dS writer mapping assumes a 64x128 tile per CTA.");
static_assert(S_DS_COLS_PER_THREAD % S_DS_VEC_ELEMS == 0, "S/dS vectorized stores require B_TOPK/2 to be a multiple of 8.");

static constexpr int D_tQ = 384, NUM_tQ_TILES = D_tQ / 64;
static constexpr int D_sQ = D_QK - D_tQ, NUM_sQ_TILES = D_sQ / 64;
static_assert(D_sQ % 64 == 0 && D_tQ % 64 == 0 && D_sQ + D_tQ == D_Q);

struct tmem_cols {
    static constexpr int dQ = 0;
    static constexpr int P = 288;
    static constexpr int dP = 352;
    static constexpr int tQ = 416;
    static constexpr int kNumUsedCols = 512;
};

static_assert(tmem_cols::P == tmem_cols::dQ + D_Q / 2);
static_assert(tmem_cols::dP == tmem_cols::P + B_TOPK / 2);
static_assert(tmem_cols::tQ == tmem_cols::dP + B_TOPK / 2);
static_assert(tmem_cols::kNumUsedCols == tmem_cols::tQ + D_tQ / 4);

template<int NUM_TILES>
using SmemLayoutQTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H / 2>, Int<64 * NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutQ = SmemLayoutQTiles<D_Q / 64>;
using SmemLayoutSQ = SmemLayoutQTiles<NUM_sQ_TILES>;
using SmemLayoutdO = SmemLayoutQTiles<D_V / 64>;

template<int NUM_TILES>
using SmemLayoutQTilesTransposed = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_SW128_Atom<bf16>{},
    Shape<Int<64 * NUM_TILES>, Int<B_H / 2>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

using SmemLayoutdOTransposed = SmemLayoutQTilesTransposed<D_V / 64>;

template<int NUM_TILES>
using SmemLayoutKTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK / 2>, Int<64 * NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutK = SmemLayoutKTiles<D_K / 64>;
using SmemLayoutKSQ = SmemLayoutKTiles<NUM_sQ_TILES>;
using SmemLayoutKTQ = SmemLayoutKTiles<NUM_tQ_TILES>;
using SmemLayoutV = SmemLayoutKTiles<D_V / 64>;

template<int NUM_TILES>
using SmemLayoutKVTilesTransposed = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_SW128_Atom<bf16>{},
    Shape<Int<64 * NUM_TILES>, Int<B_TOPK / 2>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

using SmemLayoutKSQTransposed = SmemLayoutKVTilesTransposed<NUM_sQ_TILES>;
using SmemLayoutKTQTransposed = SmemLayoutKVTilesTransposed<NUM_tQ_TILES>;

using SmemLayoutS = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_INTER_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<B_H / 2>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

using SmemLayoutdS = SmemLayoutS;

using SmemLayoutdSTransposed = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<B_H / 2>, Int<B_TOPK>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using TiledMMA_P_tQ = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_TS_NOELECT<bf16, bf16, float, B_H, B_TOPK / 2, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_P_sQ = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_H, B_TOPK / 2, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_dP = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_H, B_TOPK / 2, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_dQ_sQ = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H / 2, D_sQ, UMMA::Major::K, UMMA::Major::MN>{}
));

using TiledMMA_dQ_tQ = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H / 2, D_tQ, UMMA::Major::K, UMMA::Major::MN>{}
));

struct alignas(128) SharedMemoryPlan {
    union {
        // Prologue first lands the full local Q tile here, then reuses the tail
        // region for the streamed K half-tiles once tQ has been copied to TMEM.
        array_aligned<bf16, cosize_v<SmemLayoutQ>> q_full;
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutSQ>> sQ;
            array_aligned<bf16, cosize_v<SmemLayoutK>> k;
        } q_k;
        array_aligned<bf16, cosize_v<SmemLayoutQ>> dq;
    } u;

    array_aligned<bf16, cosize_v<SmemLayoutdO>> dO;
    struct {
        array_aligned<bf16, cosize_v<SmemLayoutdSTransposed>> s;
        array_aligned<bf16, cosize_v<SmemLayoutdSTransposed>> ds;
    } s_ds;
    char is_k_valid[B_TOPK / 8];

    transac_bar_t bar_prologue_q;
    transac_bar_t bar_prologue_tQ;
    transac_bar_t bar_prologue_k;
    transac_bar_t bar_prologue_dO;
    transac_bar_t bar_p_ready;
    transac_bar_t bar_dp_ready;
    transac_bar_t bar_s_ready;
    transac_bar_t bar_ds_ready;
    transac_bar_t bar_k_valid_free;
    transac_bar_t bar_k_valid_ready;
    transac_bar_t bar_dq_ready;

    array_aligned<uint32_t, 1> tmem_start_addr;
};

static constexpr size_t SMEM_SIZE = sizeof(SharedMemoryPlan);

}  // namespace sm100::bwd::head128_2kernels::dq

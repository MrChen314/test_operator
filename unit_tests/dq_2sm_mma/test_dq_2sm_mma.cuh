#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cluster_launch.hpp>
#include <kerutils/kerutils.cuh>

using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;

namespace test_operator::dq_2sm_mma {

using namespace cute;
using bf16 = cutlass::bfloat16_t;

static constexpr int D_QK = 576;
static constexpr int D_Q = D_QK;
static constexpr int D_K = D_QK;
static constexpr int D_ROPE = 64;
static constexpr int B_H = 128;
static constexpr int B_TOPK = 64;
static constexpr int NUM_THREADS = 128;

using SmemLayoutS = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<B_H / 2>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutdS = SmemLayoutS;

using SmemLayoutdSTransposed = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_INTER_Atom<bf16>{},
    Shape<Int<B_H / 2>, Int<B_TOPK>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

using SmemLayoutKVTilesTransposed_KMajor = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<D_Q / 2>, Int<B_TOPK>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutKCalcDQPartNoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<128>, Int<B_TOPK>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutKCalcDQPartRoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<D_ROPE / 2>, Int<B_TOPK>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using TiledMMA_dQ_2cta = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_H, 256, UMMA::Major::MN, UMMA::Major::K>{}
));

using TiledMMA_dQ_RoPE_2cta = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_H, D_ROPE, UMMA::Major::MN, UMMA::Major::K>{}
));

struct tmem_cols {
    static constexpr int dQ = 0;
    static constexpr int dQ_RoPE = 256;
};

struct KernelParams {
    CUtensorMap tensor_map_kv;
    CUtensorMap tensor_map_kv_rope32;
};

struct alignas(128) SharedMemory {
    array_aligned<bf16, cosize_v<SmemLayoutdSTransposed>> ds_t;
    array_aligned<bf16, cosize_v<SmemLayoutKVTilesTransposed_KMajor>> k_calc_dq;

    transac_bar_t bar_kv_part0_ready;
    transac_bar_t bar_kv_part1_ready;
    transac_bar_t bar_kv_part2_ready;
    transac_bar_t bar_dq_ready;

    array_aligned<uint32_t, 1> tmem_start_addr;
};

static constexpr size_t SMEM_SIZE = sizeof(SharedMemory);

template <typename Params>
__global__ __launch_bounds__(NUM_THREADS, 1) void dq_2sm_mma_kernel(
    const bf16* __restrict__ ds,      // [B_H, B_TOPK] = [128, 64]
    const bf16* __restrict__ kv,      // [s_kv, D_K]
    const int32_t* __restrict__ indices,  // [B_TOPK] = [64]
    float* __restrict__ dQ_out,       // [B_H, D_Q] = [128, 576]
    __grid_constant__ const Params params
);

torch::Tensor run_dq_2sm_mma(
    torch::Tensor ds,      // [128, 64], bf16
    torch::Tensor kv,      // [s_kv, 576], bf16
    torch::Tensor indices  // [64], int32/int64
);

}  // namespace test_operator::dq_2sm_mma

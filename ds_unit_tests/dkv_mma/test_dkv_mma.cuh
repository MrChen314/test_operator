#pragma once

#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/cluster_launch.hpp>
#include <kerutils/kerutils.cuh>

namespace test_operator::dkv_mma {

using namespace cute;
using bf16 = cutlass::bfloat16_t;

static constexpr int B_H = 128;
static constexpr int B_TOPK = 32;
static constexpr int D_V = 512;
static constexpr int D_Q = 576;
static constexpr int D_K = D_Q;
static constexpr int D_ROPE = D_Q - D_V;
static constexpr int NUM_THREADS = 128;

template <int NUM_TILES>
using SmemLayoutQTilesTransposed = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_SW128_Atom<bf16>{},
    Shape<Int<64 * NUM_TILES>, Int<B_H / 2>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

using SmemLayoutDOTransposed = SmemLayoutQTilesTransposed<D_V / 64>;      // [512, 64]
using SmemLayoutQNoPETransposed = SmemLayoutQTilesTransposed<D_V / 64>;   // [512, 64]
using SmemLayoutQRoPETransposed = SmemLayoutQTilesTransposed<1>;          // [64, 64]

using TiledMMA_dKV = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, B_TOPK, 256, UMMA::Major::K, UMMA::Major::MN>{}
));

using TiledMMA_dKV_RoPE = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, B_TOPK, D_ROPE, UMMA::Major::K, UMMA::Major::MN>{}
));

struct tmem_cols {
    static constexpr int dKV = 288;
    static constexpr int dKV_RoPE = 416;
    static constexpr int S = 464;
    static constexpr int dS = 472;
};

struct alignas(128) SharedMemory {
    array_aligned<bf16, cosize_v<SmemLayoutDOTransposed>> dO_t;
    array_aligned<bf16, cosize_v<SmemLayoutQNoPETransposed>> q_t;
    array_aligned<bf16, cosize_v<SmemLayoutQRoPETransposed>> q_rope_t;
    alignas(4) uint32_t tmem_start_addr;
};

static constexpr size_t SMEM_SIZE = sizeof(SharedMemory);

__global__ void test_dkv_mma_kernel(
    const bf16* __restrict__ s,
    const bf16* __restrict__ ds,
    const bf16* __restrict__ dO_t,
    const bf16* __restrict__ q_t,
    const bf16* __restrict__ q_rope_t,
    float* __restrict__ dkv_out
);

void launch_test_dkv_mma(
    const bf16* s,
    const bf16* ds,
    const bf16* dO_t,
    const bf16* q_t,
    const bf16* q_rope_t,
    float* dkv_out,
    cudaStream_t stream = nullptr
);

}  // namespace test_operator::dkv_mma

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/cluster_launch.hpp>
#include <kerutils/kerutils.cuh>

namespace test_operator::dq_mma {

using namespace cute;
using bf16 = cutlass::bfloat16_t;

static constexpr int B_H = 128;
static constexpr int B_TOPK = 32;
static constexpr int D_V = 512;
static constexpr int D_Q = 576;
static constexpr int D_ROPE = D_Q - D_V;
static constexpr int NUM_THREADS = B_H;

using TiledMMA_dQ = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_TS_NOELECT<bf16, bf16, float, B_H, 256, UMMA::Major::K, UMMA::Major::MN>{}
));

using TiledMMA_dQ_RoPE = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_TS_NOELECT<bf16, bf16, float, B_H, D_ROPE, UMMA::Major::K, UMMA::Major::MN>{}
));

template <int N>
using SmemLayoutBTransposed = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_SW128_Atom<bf16>{},
    Shape<Int<N>, Int<B_TOPK>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

using SmemLayoutKNoPEPart = SmemLayoutBTransposed<256>;  // [256, 32]
using SmemLayoutKRoPE = SmemLayoutBTransposed<D_ROPE>;   // [64, 32]

struct tmem_cols {
    static constexpr int dQ = 0;
    static constexpr int dQ_RoPE = 256;
    static constexpr int dS = 472;
};

struct alignas(128) SharedMemory {
    array_aligned<bf16, cosize_v<SmemLayoutKNoPEPart>> k_nope_t_part0;
    array_aligned<bf16, cosize_v<SmemLayoutKNoPEPart>> k_nope_t_part1;
    array_aligned<bf16, cosize_v<SmemLayoutKRoPE>> k_rope_t;
    alignas(4) uint32_t tmem_start_addr;
};

static constexpr size_t SMEM_SIZE = sizeof(SharedMemory);

__global__ void test_dq_mma_kernel(
    const bf16* __restrict__ ds,
    const bf16* __restrict__ k_nope_t_part0,
    const bf16* __restrict__ k_nope_t_part1,
    const bf16* __restrict__ k_rope_t,
    float* __restrict__ dq_nope_part0,
    float* __restrict__ dq_nope_part1,
    float* __restrict__ dq_rope
);

void launch_test_dq_mma(
    const bf16* ds,
    const bf16* k_nope_t_part0,
    const bf16* k_nope_t_part1,
    const bf16* k_rope_t,
    float* dq_nope_part0,
    float* dq_nope_part1,
    float* dq_rope,
    cudaStream_t stream = nullptr
);

}  // namespace test_operator::dq_mma

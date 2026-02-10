#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/barrier.h>

#include <kerutils/kerutils.cuh>

namespace test_operator::smem_cp_async {

using namespace cute;
using bf16 = cutlass::bfloat16_t;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;

static constexpr int M_TOTAL = 128;
static constexpr int K_TOTAL = 64;
static constexpr int N_TOTAL = 256;

static constexpr int M_PER_CTA = 64;
static constexpr int K_PER_CTA = 32;

static constexpr int NUM_THREADS = 256;
static constexpr int WARP_GROUP_THREADS = 128;

template<int NUM_TILES>
using SmemLayoutATiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<M_PER_CTA>, Int<64 * NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutA = SmemLayoutATiles<K_TOTAL / 64>;

template<int NUM_TILES>
using SmemLayoutBTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<N_TOTAL>, Int<64 * NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutB = SmemLayoutBTiles<K_TOTAL / 64>;

using TiledMMA_C = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, M_PER_CTA, N_TOTAL, UMMA::Major::K, UMMA::Major::K>{}
));

struct tmem_cols {
    static constexpr int c = 256;
    static_assert(c + N_TOTAL <= 512, "TMEM column overflow");
};

struct alignas(128) SharedMemory {
    array_aligned<bf16, cosize_v<SmemLayoutA>> sA;
    array_aligned<bf16, cosize_v<SmemLayoutB>> sB_local;
    array_aligned<bf16, cosize_v<SmemLayoutB>> sB_peer;
    transac_bar_t bar_b_local_ready;
    transac_bar_t bar_b_peer_ready;
    transac_bar_t bar_cp_async;
    alignas(4) uint32_t tmem_start_addr;
};

static constexpr size_t SMEM_SIZE = sizeof(SharedMemory);

}  // namespace test_operator::smem_cp_async

__global__ void test_smem_cp_async_kernel(
    const test_operator::smem_cp_async::bf16* __restrict__ A,  // [128, 64]
    const test_operator::smem_cp_async::bf16* __restrict__ B,  // [64, 256]
    float* __restrict__ C                                       // [128, 256]
);

void launch_test_smem_cp_async(
    const test_operator::smem_cp_async::bf16* A,
    const test_operator::smem_cp_async::bf16* B,
    float* C,
    cudaStream_t stream = nullptr
);

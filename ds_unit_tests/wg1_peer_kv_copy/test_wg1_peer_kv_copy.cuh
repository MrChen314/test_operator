#pragma once

#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/cluster_launch.hpp>
#include <kerutils/kerutils.cuh>

namespace test_operator::wg1_peer_kv_copy {

using namespace cute;
using bf16 = cutlass::bfloat16_t;

static constexpr int B_TOPK = 32;
static constexpr int D_K = 576;
static constexpr int NUM_THREADS = 128;
static constexpr int KV_ROWS = B_TOPK / 2;

template<int NUM_TILES>
using SmemLayoutKVTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<KV_ROWS>, Int<64 * NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutKV = SmemLayoutKVTiles<9>;

struct alignas(128) SharedMemory {
    array_aligned<bf16, cosize_v<SmemLayoutKV>> kv;
};

static constexpr size_t SMEM_SIZE = sizeof(SharedMemory);

__global__ void test_wg1_peer_kv_copy_kernel(
    const bf16* __restrict__ init_kv,
    bf16* __restrict__ out_kv
);

void launch_test_wg1_peer_kv_copy(
    const bf16* init_kv,
    bf16* out_kv,
    cudaStream_t stream = nullptr
);

}  // namespace test_operator::wg1_peer_kv_copy

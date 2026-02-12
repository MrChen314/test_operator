#pragma once

#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/cluster_launch.hpp>
#include <cuda_bf16.h>

namespace test_operator::ds_tmem {

using bf16 = cutlass::bfloat16_t;

static constexpr int B_H = 128;
static constexpr int B_TOPK = 32;
static constexpr int D_QK = 576;
static constexpr int NUM_THREADS = B_H;

struct tmem_cols {
    static constexpr int S = 464;
    static constexpr int dS = 472;
};

struct alignas(128) SharedMemory {
    alignas(4) uint32_t tmem_start_addr;
};

static constexpr size_t SMEM_SIZE = sizeof(SharedMemory);

__global__ void test_ds_tmem_kernel(
    const float* __restrict__ p,
    const float* __restrict__ dp,
    const float* __restrict__ lse,
    const float* __restrict__ delta,
    bf16* __restrict__ s_out,
    bf16* __restrict__ ds_out
);

void launch_test_ds_tmem(
    const float* p,
    const float* dp,
    const float* lse,
    const float* delta,
    bf16* s_out,
    bf16* ds_out,
    cudaStream_t stream = nullptr
);

}  // namespace test_operator::ds_tmem

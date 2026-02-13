#pragma once

#include <cstdint>
#include <tuple>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace test_operator::red_global_add {

static constexpr int GLOBAL_ROWS = 128;
static constexpr int GLOBAL_COLS = 256;
static constexpr int ADD_ROWS = 64;
static constexpr int VEC_WIDTH = 4;
static constexpr int COLS_PER_VEC = GLOBAL_COLS / VEC_WIDTH;
static constexpr int NUM_VEC = ADD_ROWS * COLS_PER_VEC;
static constexpr int THREADS = 256;
static constexpr int WG2_THREADS = 128;

__device__ void atomic_add_float4(float* dst_ptr, const float4& v);

__global__ void accumulate_split_cta_kernel(
    const float* add1,
    const float* add2,
    const int32_t* indices,
    float* global_tensor
);

__global__ void accumulate_fused_kernel(
    const float* add1,
    const float* add2,
    const int32_t* indices,
    float* global_tensor
);

__global__ void accumulate_wg2_cluster_kernel(
    const float* add1,
    const float* add2,
    const int32_t* indices,
    float* global_tensor
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> run_red_global_add(
    torch::Tensor global_tensor2,
    torch::Tensor global_tensor3,
    torch::Tensor global_tensor4,
    torch::Tensor add1,
    torch::Tensor add2,
    torch::Tensor indices
);

}  // namespace test_operator::red_global_add

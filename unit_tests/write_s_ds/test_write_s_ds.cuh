#pragma once

#include <tuple>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>

namespace test_operator::write_s_ds {

using namespace cute;
using bf16 = cutlass::bfloat16_t;

static constexpr int B_H = 128;
static constexpr int B_TOPK = 64;
static constexpr int ROWS = B_H / 2;
static constexpr int COLS = B_TOPK;
static constexpr int HALF_COLS = COLS / 2;
static constexpr int THREADS = 128;

static constexpr int F2_PER_THREAD = HALF_COLS / 2;
static constexpr int VEC_ELEMS = 8;
static constexpr int VEC_F2 = VEC_ELEMS / 2;
static constexpr int NUM_VEC_STORES = HALF_COLS / VEC_ELEMS;
static constexpr int VEC_STRIDE = ROWS * VEC_ELEMS;
static constexpr uint32_t HALF_MASK = 0x0000FFFFu;
static constexpr float NEG_DELTA = -0.5f;
static constexpr float SM_SCALE = 0.125f;

static_assert(ROWS == 64);
static_assert(COLS == 64);
static_assert(THREADS == ROWS * 2);
static_assert(F2_PER_THREAD == 16);
static_assert(VEC_F2 == 4);

using SmemLayoutS = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<COLS>, Int<ROWS>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutdS = SmemLayoutS;

struct alignas(16) bf16x8 {
    __nv_bfloat162 a01;
    __nv_bfloat162 a23;
    __nv_bfloat162 a45;
    __nv_bfloat162 a67;
};

struct alignas(128) SharedMemory {
    array_aligned<bf16, cosize_v<SmemLayoutS>> s;
    array_aligned<bf16, cosize_v<SmemLayoutdS>> ds;
};

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> run_write_s_ds();

}  // namespace test_operator::write_s_ds

#include "test_write_s_ds.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace test_operator::write_s_ds {

namespace {

CUTE_DEVICE float make_logit(int row, int col) {
    return -3.5f + 0.03125f * static_cast<float>(col) + 0.01f * static_cast<float>(row);
}

CUTE_DEVICE float make_dp(int row, int col) {
    return -0.75f + 0.02f * static_cast<float>(col) - 0.003f * static_cast<float>(row);
}

template <bool BuggyMask>
CUTE_DEVICE void apply_half_mask(float2 (&p)[F2_PER_THREAD], uint32_t valid_mask) {
    float* p_float = reinterpret_cast<float*>(p);
    constexpr int kNumMaskIters = BuggyMask ? F2_PER_THREAD : HALF_COLS;
    CUTE_UNROLL
    for (int i = 0; i < kNumMaskIters; ++i) {
        if (!((valid_mask >> i) & 1u)) {
            p_float[i] = -CUDART_INF_F;
        }
    }
}

template <bool Vectorized, bool UseMask, bool BuggyMask>
CUTE_DEVICE void write_stage(SharedMemory& smem, int tid) {
    Tensor s_tensor = make_tensor(make_smem_ptr(smem.s.data()), SmemLayoutS{});
    Tensor ds_tensor = make_tensor(make_smem_ptr(smem.ds.data()), SmemLayoutdS{});

    const int row = tid % ROWS;
    const int col_half = tid / ROWS;
    const int col_base = col_half * HALF_COLS;

    float2 p[F2_PER_THREAD];
    float2 dp[F2_PER_THREAD];

    CUTE_UNROLL
    for (int i = 0; i < F2_PER_THREAD; ++i) {
        const int col0 = col_base + i * 2;
        const int col1 = col0 + 1;
        p[i] = make_float2(make_logit(row, col0), make_logit(row, col1));
        dp[i] = make_float2(make_dp(row, col0), make_dp(row, col1));
    }

    if constexpr (UseMask) {
        apply_half_mask<BuggyMask>(p, HALF_MASK);
    }

    if constexpr (Vectorized) {
        bf16* s_base = smem.s.data() + row * VEC_ELEMS + col_half * ROWS * HALF_COLS;
        bf16* ds_base = smem.ds.data() + row * VEC_ELEMS + col_half * ROWS * HALF_COLS;

        CUTE_UNROLL
        for (int vec = 0; vec < NUM_VEC_STORES; ++vec) {
            const int base_idx = vec * VEC_F2;
            bf16x8 s_pack;
            bf16x8 ds_pack;

            {
                float2 s_vec = make_float2(exp2f(p[base_idx + 0].x), exp2f(p[base_idx + 0].y));
                p[base_idx + 0] = s_vec;
                s_pack.a01 = __float22bfloat162_rn(s_vec);
                float2 ds_vec = make_float2(
                    s_vec.x * (dp[base_idx + 0].x + NEG_DELTA) * SM_SCALE,
                    s_vec.y * (dp[base_idx + 0].y + NEG_DELTA) * SM_SCALE
                );
                ds_pack.a01 = __float22bfloat162_rn(ds_vec);
            }
            {
                float2 s_vec = make_float2(exp2f(p[base_idx + 1].x), exp2f(p[base_idx + 1].y));
                p[base_idx + 1] = s_vec;
                s_pack.a23 = __float22bfloat162_rn(s_vec);
                float2 ds_vec = make_float2(
                    s_vec.x * (dp[base_idx + 1].x + NEG_DELTA) * SM_SCALE,
                    s_vec.y * (dp[base_idx + 1].y + NEG_DELTA) * SM_SCALE
                );
                ds_pack.a23 = __float22bfloat162_rn(ds_vec);
            }
            {
                float2 s_vec = make_float2(exp2f(p[base_idx + 2].x), exp2f(p[base_idx + 2].y));
                p[base_idx + 2] = s_vec;
                s_pack.a45 = __float22bfloat162_rn(s_vec);
                float2 ds_vec = make_float2(
                    s_vec.x * (dp[base_idx + 2].x + NEG_DELTA) * SM_SCALE,
                    s_vec.y * (dp[base_idx + 2].y + NEG_DELTA) * SM_SCALE
                );
                ds_pack.a45 = __float22bfloat162_rn(ds_vec);
            }
            {
                float2 s_vec = make_float2(exp2f(p[base_idx + 3].x), exp2f(p[base_idx + 3].y));
                p[base_idx + 3] = s_vec;
                s_pack.a67 = __float22bfloat162_rn(s_vec);
                float2 ds_vec = make_float2(
                    s_vec.x * (dp[base_idx + 3].x + NEG_DELTA) * SM_SCALE,
                    s_vec.y * (dp[base_idx + 3].y + NEG_DELTA) * SM_SCALE
                );
                ds_pack.a67 = __float22bfloat162_rn(ds_vec);
            }

            reinterpret_cast<bf16x8*>(s_base + vec * VEC_STRIDE)[0] = s_pack;
            reinterpret_cast<bf16x8*>(ds_base + vec * VEC_STRIDE)[0] = ds_pack;
        }
    } else {
        CUTE_UNROLL
        for (int i = 0; i < F2_PER_THREAD; ++i) {
            const int col = col_base + i * 2;
            float2 s_vec = make_float2(exp2f(p[i].x), exp2f(p[i].y));
            p[i] = s_vec;
            s_tensor(col + 0, row) = bf16(s_vec.x);
            s_tensor(col + 1, row) = bf16(s_vec.y);
        }

        CUTE_UNROLL
        for (int i = 0; i < F2_PER_THREAD; ++i) {
            const int col = col_base + i * 2;
            float2 ds_vec = make_float2(
                p[i].x * (dp[i].x + NEG_DELTA) * SM_SCALE,
                p[i].y * (dp[i].y + NEG_DELTA) * SM_SCALE
            );
            ds_tensor(col + 0, row) = bf16(ds_vec.x);
            ds_tensor(col + 1, row) = bf16(ds_vec.y);
        }
    }
}

CUTE_DEVICE void export_stage(
    SharedMemory& smem,
    int tid,
    bf16* s_out,
    bf16* ds_out
) {
    Tensor s_tensor = make_tensor(make_smem_ptr(smem.s.data()), SmemLayoutS{});
    Tensor ds_tensor = make_tensor(make_smem_ptr(smem.ds.data()), SmemLayoutdS{});

    for (int flat = tid; flat < ROWS * COLS; flat += THREADS) {
        const int row = flat / COLS;
        const int col = flat % COLS;
        s_out[flat] = s_tensor(col, row);
        ds_out[flat] = ds_tensor(col, row);
    }
}

}  // namespace

__global__ void write_s_ds_kernel(
    bf16* s_scalar_out,
    bf16* s_vector_out,
    bf16* s_fixed_mask_out,
    bf16* s_buggy_mask_out,
    bf16* ds_scalar_out,
    bf16* ds_vector_out,
    bf16* ds_fixed_mask_out,
    bf16* ds_buggy_mask_out
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    __shared__ SharedMemory smem;
    const int tid = threadIdx.x;

    write_stage<false, false, false>(smem, tid);
    __syncthreads();
    export_stage(smem, tid, s_scalar_out, ds_scalar_out);
    __syncthreads();

    write_stage<true, false, false>(smem, tid);
    __syncthreads();
    export_stage(smem, tid, s_vector_out, ds_vector_out);
    __syncthreads();

    write_stage<true, true, false>(smem, tid);
    __syncthreads();
    export_stage(smem, tid, s_fixed_mask_out, ds_fixed_mask_out);
    __syncthreads();

    write_stage<true, true, true>(smem, tid);
    __syncthreads();
    export_stage(smem, tid, s_buggy_mask_out, ds_buggy_mask_out);
#endif
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> run_write_s_ds() {
    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

    auto s_scalar = torch::zeros({ROWS, COLS}, options);
    auto s_vector = torch::zeros({ROWS, COLS}, options);
    auto s_fixed_mask = torch::zeros({ROWS, COLS}, options);
    auto s_buggy_mask = torch::zeros({ROWS, COLS}, options);
    auto ds_scalar = torch::zeros({ROWS, COLS}, options);
    auto ds_vector = torch::zeros({ROWS, COLS}, options);
    auto ds_fixed_mask = torch::zeros({ROWS, COLS}, options);
    auto ds_buggy_mask = torch::zeros({ROWS, COLS}, options);

    write_s_ds_kernel<<<1, THREADS>>>(
        reinterpret_cast<bf16*>(s_scalar.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16*>(s_vector.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16*>(s_fixed_mask.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16*>(s_buggy_mask.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16*>(ds_scalar.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16*>(ds_vector.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16*>(ds_fixed_mask.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16*>(ds_buggy_mask.data_ptr<at::BFloat16>())
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "write_s_ds_kernel launch failed: ", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "write_s_ds_kernel execution failed: ", cudaGetErrorString(err));

    return std::make_tuple(
        s_scalar,
        s_vector,
        s_fixed_mask,
        s_buggy_mask,
        ds_scalar,
        ds_vector,
        ds_fixed_mask,
        ds_buggy_mask
    );
}

}  // namespace test_operator::write_s_ds

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_write_s_ds", &test_operator::write_s_ds::run_write_s_ds, "Write S/dS shared-memory test");
}

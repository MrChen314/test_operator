#include "test_red_global_add.cuh"

#include <c10/cuda/CUDAStream.h>

namespace test_operator::red_global_add {

__device__ __forceinline__ float4 load_float4(const float* src) {
    return float4{src[0], src[1], src[2], src[3]};
}

__device__ void atomic_add_float4(float* dst_ptr, const float4& v) {
    asm volatile(
        "red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(dst_ptr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
        : "memory"
    );
}

__global__ void accumulate_split_cta_kernel(
    const float* add1,
    const float* add2,
    const int32_t* indices,
    float* global_tensor
) {
    const int cta_idx = blockIdx.x;
    if (cta_idx > 1) {
        return;
    }

    const float* src = (cta_idx == 0) ? add1 : add2;

    for (int linear = threadIdx.x; linear < NUM_VEC; linear += blockDim.x) {
        const int row = linear / COLS_PER_VEC;
        const int col_vec = linear % COLS_PER_VEC;
        const int dst_row = indices[row];
        if (dst_row < 0 || dst_row >= GLOBAL_ROWS) {
            continue;
        }

        const int dst_base = dst_row * GLOBAL_COLS + col_vec * VEC_WIDTH;
        const int src_base = row * GLOBAL_COLS + col_vec * VEC_WIDTH;
        const float4 v = load_float4(src + src_base);
        atomic_add_float4(global_tensor + dst_base, v);
    }
}

__global__ void accumulate_fused_kernel(
    const float* add1,
    const float* add2,
    const int32_t* indices,
    float* global_tensor
) {
    for (int linear = threadIdx.x; linear < NUM_VEC; linear += blockDim.x) {
        const int row = linear / COLS_PER_VEC;
        const int col_vec = linear % COLS_PER_VEC;
        const int dst_row = indices[row];
        if (dst_row < 0 || dst_row >= GLOBAL_ROWS) {
            continue;
        }

        const int dst_base = dst_row * GLOBAL_COLS + col_vec * VEC_WIDTH;
        const int src_base = row * GLOBAL_COLS + col_vec * VEC_WIDTH;

        const float4 a = load_float4(add1 + src_base);
        const float4 b = load_float4(add2 + src_base);
        const float4 v = float4{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
        atomic_add_float4(global_tensor + dst_base, v);
    }
}

std::tuple<torch::Tensor, torch::Tensor> run_red_global_add(
    torch::Tensor global_tensor2,
    torch::Tensor global_tensor3,
    torch::Tensor add1,
    torch::Tensor add2,
    torch::Tensor indices
) {
    TORCH_CHECK(global_tensor2.is_cuda() && global_tensor3.is_cuda(), "global tensors must be CUDA");
    TORCH_CHECK(add1.is_cuda() && add2.is_cuda(), "add tensors must be CUDA");
    TORCH_CHECK(indices.is_cuda(), "indices must be CUDA");

    TORCH_CHECK(global_tensor2.dtype() == torch::kFloat32, "global_tensor2 must be float32");
    TORCH_CHECK(global_tensor3.dtype() == torch::kFloat32, "global_tensor3 must be float32");
    TORCH_CHECK(add1.dtype() == torch::kFloat32, "add1 must be float32");
    TORCH_CHECK(add2.dtype() == torch::kFloat32, "add2 must be float32");
    TORCH_CHECK(indices.dtype() == torch::kInt32 || indices.dtype() == torch::kInt64,
                "indices must be int32 or int64");

    TORCH_CHECK(global_tensor2.dim() == 2 && global_tensor2.size(0) == GLOBAL_ROWS &&
                    global_tensor2.size(1) == GLOBAL_COLS,
                "global_tensor2 shape must be [128, 256]");
    TORCH_CHECK(global_tensor3.dim() == 2 && global_tensor3.size(0) == GLOBAL_ROWS &&
                    global_tensor3.size(1) == GLOBAL_COLS,
                "global_tensor3 shape must be [128, 256]");
    TORCH_CHECK(add1.dim() == 2 && add1.size(0) == ADD_ROWS && add1.size(1) == GLOBAL_COLS,
                "add1 shape must be [64, 256]");
    TORCH_CHECK(add2.dim() == 2 && add2.size(0) == ADD_ROWS && add2.size(1) == GLOBAL_COLS,
                "add2 shape must be [64, 256]");
    TORCH_CHECK(indices.dim() == 1 && indices.size(0) == ADD_ROWS, "indices shape must be [64]");

    TORCH_CHECK(global_tensor2.is_contiguous() && global_tensor3.is_contiguous(),
                "global tensors must be contiguous");
    TORCH_CHECK(add1.is_contiguous() && add2.is_contiguous(), "add tensors must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");

    torch::Tensor indices_i32 = (indices.dtype() == torch::kInt32) ? indices : indices.to(torch::kInt32);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    accumulate_split_cta_kernel<<<2, THREADS, 0, stream>>>(
        add1.data_ptr<float>(),
        add2.data_ptr<float>(),
        indices_i32.data_ptr<int32_t>(),
        global_tensor2.data_ptr<float>());
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "accumulate_split_cta_kernel failed: ", cudaGetErrorString(err));

    accumulate_fused_kernel<<<1, THREADS, 0, stream>>>(
        add1.data_ptr<float>(),
        add2.data_ptr<float>(),
        indices_i32.data_ptr<int32_t>(),
        global_tensor3.data_ptr<float>());
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "accumulate_fused_kernel failed: ", cudaGetErrorString(err));

    return std::make_tuple(global_tensor2, global_tensor3);
}

}  // namespace test_operator::red_global_add

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run_red_global_add",
        &test_operator::red_global_add::run_red_global_add,
        "Run minimal atomic_add_float4 red.global.add unit test kernels");
}

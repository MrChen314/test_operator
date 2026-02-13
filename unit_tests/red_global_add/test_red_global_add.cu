#include "test_red_global_add.cuh"

#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cstring>

namespace test_operator::red_global_add {

namespace cg = cooperative_groups;

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

__device__ __forceinline__ void red_add_peer_smem(float* peer_smem_ptr, float v) {
    const uint32_t peer_addr = static_cast<uint32_t>(__cvta_generic_to_shared(peer_smem_ptr));
    asm volatile(
        "red.relaxed.cluster.shared::cluster.add.f32 [%0], %1;"
        :
        : "r"(peer_addr), "f"(v)
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

__global__ void accumulate_wg2_cluster_kernel(
    const float* add1,
    const float* add2,
    const int32_t* indices,
    float* global_tensor
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cg::cluster_group cluster = cg::this_cluster();
    const int cta_idx = cluster.block_rank();
    if (cta_idx > 1) {
        return;
    }

    // Minimal WG2-style staging buffer in SMEM: [64, 256].
    extern __shared__ float s_sdkv[];

    const float* src = (cta_idx == 0) ? add1 : add2;
    constexpr int NUM_SCALAR = ADD_ROWS * GLOBAL_COLS;
    constexpr int HALF_COLS = GLOBAL_COLS / 2;
    constexpr int VEC_PER_ROW_HALF = HALF_COLS / VEC_WIDTH;
    constexpr int NUM_VEC_HALF = ADD_ROWS * VEC_PER_ROW_HALF;

    // Stage 1: emulate TMEM -> SMEM path by staging source values into SMEM.
    for (int linear = threadIdx.x; linear < NUM_SCALAR; linear += blockDim.x) {
        const float tmem_val = src[linear];
        s_sdkv[linear] = tmem_val;
    }

    __syncthreads();
    cluster.sync();

    // Stage 2: WG2 peer reduce-add in SMEM via red.cluster.shared.
    for (int linear = threadIdx.x; linear < NUM_SCALAR; linear += blockDim.x) {
        const float v = s_sdkv[linear];
        float* peer_ptr = cluster.map_shared_rank(s_sdkv + linear, cta_idx ^ 1);
        red_add_peer_smem(peer_ptr, v);
    }

    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    cluster.sync();

    // Stage 3: each CTA flushes half columns to global with red.global.add.v4.f32.
    const int half_col_start = (cta_idx == 0) ? 0 : HALF_COLS;
    for (int linear = threadIdx.x; linear < NUM_VEC_HALF; linear += blockDim.x) {
        const int row = linear / VEC_PER_ROW_HALF;
        const int col_vec = linear % VEC_PER_ROW_HALF;
        const int dst_row = indices[row];
        if (dst_row < 0 || dst_row >= GLOBAL_ROWS) {
            continue;
        }

        const int col = half_col_start + col_vec * VEC_WIDTH;
        const int src_base = row * GLOBAL_COLS + col;
        const int dst_base = dst_row * GLOBAL_COLS + col;
        const float4 v = load_float4(s_sdkv + src_base);
        atomic_add_float4(global_tensor + dst_base, v);
    }
#else
    (void)add1;
    (void)add2;
    (void)indices;
    (void)global_tensor;
#endif
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> run_red_global_add(
    torch::Tensor global_tensor2,
    torch::Tensor global_tensor3,
    torch::Tensor global_tensor4,
    torch::Tensor add1,
    torch::Tensor add2,
    torch::Tensor indices
) {
    TORCH_CHECK(global_tensor2.is_cuda() && global_tensor3.is_cuda() && global_tensor4.is_cuda(),
                "global tensors must be CUDA");
    TORCH_CHECK(add1.is_cuda() && add2.is_cuda(), "add tensors must be CUDA");
    TORCH_CHECK(indices.is_cuda(), "indices must be CUDA");

    TORCH_CHECK(global_tensor2.dtype() == torch::kFloat32, "global_tensor2 must be float32");
    TORCH_CHECK(global_tensor3.dtype() == torch::kFloat32, "global_tensor3 must be float32");
    TORCH_CHECK(global_tensor4.dtype() == torch::kFloat32, "global_tensor4 must be float32");
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
    TORCH_CHECK(global_tensor4.dim() == 2 && global_tensor4.size(0) == GLOBAL_ROWS &&
                    global_tensor4.size(1) == GLOBAL_COLS,
                "global_tensor4 shape must be [128, 256]");
    TORCH_CHECK(add1.dim() == 2 && add1.size(0) == ADD_ROWS && add1.size(1) == GLOBAL_COLS,
                "add1 shape must be [64, 256]");
    TORCH_CHECK(add2.dim() == 2 && add2.size(0) == ADD_ROWS && add2.size(1) == GLOBAL_COLS,
                "add2 shape must be [64, 256]");
    TORCH_CHECK(indices.dim() == 1 && indices.size(0) == ADD_ROWS, "indices shape must be [64]");

    TORCH_CHECK(global_tensor2.is_contiguous() && global_tensor3.is_contiguous() && global_tensor4.is_contiguous(),
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

    constexpr int WG2_SMEM_BYTES = ADD_ROWS * GLOBAL_COLS * static_cast<int>(sizeof(float));
    cudaError_t attr_err = cudaFuncSetAttribute(
        (const void*)accumulate_wg2_cluster_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        WG2_SMEM_BYTES
    );
    TORCH_CHECK(attr_err == cudaSuccess,
                "accumulate_wg2_cluster_kernel cudaFuncSetAttribute failed: ",
                cudaGetErrorString(attr_err));

    cudaLaunchConfig_t config;
    memset(&config, 0, sizeof(config));
    config.gridDim = dim3(2, 1, 1);
    config.blockDim = dim3(WG2_THREADS, 1, 1);
    config.dynamicSmemBytes = WG2_SMEM_BYTES;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    auto wg2_kernel = &accumulate_wg2_cluster_kernel;
    err = cudaLaunchKernelEx(
        &config,
        wg2_kernel,
        add1.data_ptr<float>(),
        add2.data_ptr<float>(),
        indices_i32.data_ptr<int32_t>(),
        global_tensor4.data_ptr<float>()
    );
    TORCH_CHECK(err == cudaSuccess, "accumulate_wg2_cluster_kernel failed: ", cudaGetErrorString(err));

    return std::make_tuple(global_tensor2, global_tensor3, global_tensor4);
}

}  // namespace test_operator::red_global_add

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run_red_global_add",
        &test_operator::red_global_add::run_red_global_add,
        "Run minimal atomic_add_float4 red.global.add unit test kernels");
}

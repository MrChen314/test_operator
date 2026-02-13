#include "test_red_global_add.cuh"

#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/cuda_host_adapter.hpp>
#include <cstring>

namespace test_operator::red_global_add {

namespace cg = cooperative_groups;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;
using cutlass::arch::fence_barrier_init;
using cutlass::arch::fence_view_async_shared;

struct WG2BarrierStorage {
    transac_bar_t bar_sdkv_peer_load_ready;
    transac_bar_t bar_sdkv_peer_red_done;
};

constexpr size_t WG2_SMEM_BYTES = ADD_ROWS * GLOBAL_COLS * sizeof(float);

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

    extern __shared__ float s_sdkv[];
    __shared__ __align__(16) uint8_t wg2_barrier_storage[sizeof(WG2BarrierStorage)];
    WG2BarrierStorage& wg2_barriers = *reinterpret_cast<WG2BarrierStorage*>(wg2_barrier_storage);
    transac_bar_t& bar_sdkv_peer_load_ready = wg2_barriers.bar_sdkv_peer_load_ready;
    transac_bar_t& bar_sdkv_peer_red_done = wg2_barriers.bar_sdkv_peer_red_done;

    const float* src = (cta_idx == 0) ? add1 : add2;
    constexpr int HALF_COLS = GLOBAL_COLS / 2;
    constexpr int COLS_PER_STAGE = 128;
    constexpr int COLS_PER_HALF_STAGE = COLS_PER_STAGE / 2;
    constexpr int CHUNK_SIZE = 32;
    constexpr int VEC_PER_ROW_HALF = HALF_COLS / VEC_WIDTH;
    constexpr int NUM_VEC_HALF = ADD_ROWS * VEC_PER_ROW_HALF;
    static_assert(WG2_THREADS == ADD_ROWS * 2, "WG2 launch expects one row x two-half lane mapping.");

    const int row = threadIdx.x % ADD_ROWS;
    const int half = threadIdx.x / ADD_ROWS;
    const uint32_t peer_cta_idx = static_cast<uint32_t>(cta_idx ^ 1);
    int peer_load_phase = 0;
    int peer_red_phase = 0;

    if (threadIdx.x == 0) {
        bar_sdkv_peer_load_ready.init(WG2_THREADS);
        bar_sdkv_peer_red_done.init(WG2_THREADS);
        fence_barrier_init();
    }

    auto sync_peer_load_ready = [&]() {
        bar_sdkv_peer_load_ready.arrive(peer_cta_idx);
        bar_sdkv_peer_load_ready.wait(peer_load_phase);
        peer_load_phase ^= 1;
    };

    auto sync_peer_red_done = [&]() {
        bar_sdkv_peer_red_done.arrive(peer_cta_idx);
        bar_sdkv_peer_red_done.wait(peer_red_phase);
        peer_red_phase ^= 1;
    };

    // Stage 1/2: emulate WG2 TMEM->SMEM staging and peer red without full CTA/cluster barriers.
    for (int stage = 0; stage < (GLOBAL_COLS / COLS_PER_STAGE); ++stage) {
        for (int chunk = 0; chunk < (COLS_PER_HALF_STAGE / CHUNK_SIZE); ++chunk) {
            alignas(16) float src_chunk[CHUNK_SIZE];
            const int col_base = stage * COLS_PER_STAGE + half * COLS_PER_HALF_STAGE + chunk * CHUNK_SIZE;

#pragma unroll
            for (int i = 0; i < CHUNK_SIZE; ++i) {
                const int col = col_base + i;
                const float v = src[row * GLOBAL_COLS + col];
                src_chunk[i] = v;
                s_sdkv[row * GLOBAL_COLS + col] = v;
            }

            fence_view_async_shared();
            sync_peer_load_ready();

#pragma unroll
            for (int i = 0; i < CHUNK_SIZE; ++i) {
                const int col = col_base + i;
                float* peer_ptr = cluster.map_shared_rank(s_sdkv + row * GLOBAL_COLS + col, cta_idx ^ 1);
                red_add_peer_smem(peer_ptr, src_chunk[i]);
            }
        }

        fence_view_async_shared();
        sync_peer_red_done();
    }

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

    cudaError_t attr_err = cudaFuncSetAttribute(
        (const void*)accumulate_wg2_cluster_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(WG2_SMEM_BYTES)
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

#include "test_ds_tmem.cuh"

#include <c10/cuda/CUDAStream.h>
#include <kerutils/kerutils.cuh>
#include <torch/extension.h>

#include <cmath>
#include <cstring>

namespace test_operator::ds_tmem {

__global__ void test_ds_tmem_kernel(
    const float* __restrict__ p,
    const float* __restrict__ dp,
    const float* __restrict__ lse,
    const float* __restrict__ delta,
    bf16* __restrict__ s_out,
    bf16* __restrict__ ds_out
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
    using namespace cute;

    extern __shared__ char smem_raw[];
    SharedMemory& smem = *reinterpret_cast<SharedMemory*>(smem_raw);

    const int cta_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_idx = tid / 32;

    if (warp_idx == 0) {
        TMEM::Allocator2Sm().allocate(512, &smem.tmem_start_addr);
        TMEM::Allocator2Sm().release_allocation_lock();
    }
    __syncthreads();

    const int row_in_cta = tid % (B_H / 2);     // 0..63
    const int topk_half = tid / (B_H / 2);      // 0..1
    const int global_row = cta_idx * (B_H / 2) + row_in_cta;

    const float sm_scale = rsqrtf(float(D_QK));
    const float scale_log2e = sm_scale * 1.4426950408889634f;
    const float row_lse = lse[global_row];
    const float row_delta = delta[global_row];

    uint32_t s_packed[8];
    uint32_t ds_packed[8];

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int col = topk_half * 16 + i * 2;

        const float p0 = p[global_row * B_TOPK + col + 0];
        const float p1 = p[global_row * B_TOPK + col + 1];
        const float dp0 = dp[global_row * B_TOPK + col + 0];
        const float dp1 = dp[global_row * B_TOPK + col + 1];

        const float s0 = exp2f(p0 * scale_log2e - row_lse);
        const float s1 = exp2f(p1 * scale_log2e - row_lse);

        const float ds0 = s0 * (dp0 - row_delta) * sm_scale;
        const float ds1 = s1 * (dp1 - row_delta) * sm_scale;

        const __nv_bfloat162 s_bf16 = __float22bfloat162_rn(make_float2(s0, s1));
        const __nv_bfloat162 ds_bf16 = __float22bfloat162_rn(make_float2(ds0, ds1));

        s_packed[i] = *reinterpret_cast<const uint32_t*>(&s_bf16);
        ds_packed[i] = *reinterpret_cast<const uint32_t*>(&ds_bf16);
    }

    const uint32_t tmem_base = smem.tmem_start_addr;
    const uint32_t lane_addr = tmem_base + (row_in_cta << 16);

    ku::tmem_st_32dp32bNx<8>(lane_addr + tmem_cols::S, s_packed);
    cutlass::arch::fence_view_async_tmem_store();

    ku::tmem_st_32dp32bNx<8>(lane_addr + tmem_cols::dS, ds_packed);
    cutlass::arch::fence_view_async_tmem_store();

    __syncthreads();

    uint32_t s_ld[8];
    uint32_t ds_ld[8];

    ku::tmem_ld_32dp32bNx<8>(lane_addr + tmem_cols::S, s_ld);
    cutlass::arch::fence_view_async_tmem_load();
    ku::tcgen05_before_thread_sync();

    ku::tmem_ld_32dp32bNx<8>(lane_addr + tmem_cols::dS, ds_ld);
    cutlass::arch::fence_view_async_tmem_load();
    ku::tcgen05_before_thread_sync();

    uint16_t* s_u16 = reinterpret_cast<uint16_t*>(s_out);
    uint16_t* ds_u16 = reinterpret_cast<uint16_t*>(ds_out);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int col = topk_half * 16 + i * 2;
        const int base_idx = global_row * B_TOPK + col;

        s_u16[base_idx + 0] = static_cast<uint16_t>(s_ld[i] & 0xFFFFu);
        s_u16[base_idx + 1] = static_cast<uint16_t>(s_ld[i] >> 16);

        ds_u16[base_idx + 0] = static_cast<uint16_t>(ds_ld[i] & 0xFFFFu);
        ds_u16[base_idx + 1] = static_cast<uint16_t>(ds_ld[i] >> 16);
    }

    __syncthreads();
    if (warp_idx == 0 && cute::elect_one_sync()) {
        TMEM::Allocator2Sm().free(smem.tmem_start_addr, 512);
    }
#endif
}

void launch_test_ds_tmem(
    const float* p,
    const float* dp,
    const float* lse,
    const float* delta,
    bf16* s_out,
    bf16* ds_out,
    cudaStream_t stream
) {
    dim3 grid(2, 1, 1);
    dim3 block(NUM_THREADS, 1, 1);

    cudaError_t attr_err = cudaFuncSetAttribute(
        test_ds_tmem_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(SMEM_SIZE));
    if (attr_err != cudaSuccess) {
        fprintf(stderr, "cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(attr_err));
        return;
    }

    cudaLaunchConfig_t config;
    memset(&config, 0, sizeof(config));
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = SMEM_SIZE;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    cudaError_t err = cudaLaunchKernelEx(&config, test_ds_tmem_kernel, p, dp, lse, delta, s_out, ds_out);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaLaunchKernelEx failed: %s\n", cudaGetErrorString(err));
    }
}

torch::Tensor to_bf16_tensor(const torch::Tensor& like) {
    return torch::empty(like.sizes(), torch::TensorOptions().dtype(torch::kBFloat16).device(like.device()));
}

std::tuple<torch::Tensor, torch::Tensor> run_ds_tmem(torch::Tensor p, torch::Tensor dp, torch::Tensor lse, torch::Tensor delta) {
    TORCH_CHECK(p.is_cuda() && dp.is_cuda() && lse.is_cuda() && delta.is_cuda(), "all inputs must be CUDA tensors");
    TORCH_CHECK(p.dtype() == torch::kFloat32, "p must be float32");
    TORCH_CHECK(dp.dtype() == torch::kFloat32, "dp must be float32");
    TORCH_CHECK(lse.dtype() == torch::kFloat32, "lse must be float32");
    TORCH_CHECK(delta.dtype() == torch::kFloat32, "delta must be float32");
    TORCH_CHECK(p.dim() == 2 && p.size(0) == B_H && p.size(1) == B_TOPK, "p shape must be [128, 32]");
    TORCH_CHECK(dp.dim() == 2 && dp.size(0) == B_H && dp.size(1) == B_TOPK, "dp shape must be [128, 32]");
    TORCH_CHECK(lse.dim() == 1 && lse.size(0) == B_H, "lse shape must be [128]");
    TORCH_CHECK(delta.dim() == 1 && delta.size(0) == B_H, "delta shape must be [128]");
    TORCH_CHECK(p.is_contiguous() && dp.is_contiguous() && lse.is_contiguous() && delta.is_contiguous(),
                "all inputs must be contiguous");

    torch::Tensor s_out = to_bf16_tensor(p);
    torch::Tensor ds_out = to_bf16_tensor(p);

    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_test_ds_tmem(
        p.data_ptr<float>(),
        dp.data_ptr<float>(),
        lse.data_ptr<float>(),
        delta.data_ptr<float>(),
        reinterpret_cast<bf16*>(s_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16*>(ds_out.data_ptr<at::BFloat16>()),
        stream);

    return std::make_tuple(s_out, ds_out);
}

}  // namespace test_operator::ds_tmem

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_ds_tmem", &test_operator::ds_tmem::run_ds_tmem, "Run ds->TMEM test kernel");
}

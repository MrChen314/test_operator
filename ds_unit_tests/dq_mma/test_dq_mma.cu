#include "test_dq_mma.cuh"

#include <c10/cuda/CUDAStream.h>
#include <kerutils/kerutils.cuh>
#include <torch/extension.h>

namespace test_operator::dq_mma {

__global__ void test_dq_mma_kernel(
    const bf16* __restrict__ ds,
    const bf16* __restrict__ k_nope_t_part0,
    const bf16* __restrict__ k_nope_t_part1,
    const bf16* __restrict__ k_rope_t,
    float* __restrict__ dq_nope_part0,
    float* __restrict__ dq_nope_part1,
    float* __restrict__ dq_rope
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
    using namespace cute;

    extern __shared__ char smem_raw[];
    SharedMemory& smem = *reinterpret_cast<SharedMemory*>(smem_raw);

    const int cta_idx = blockIdx.x % 2;
    const int tid = threadIdx.x;
    const int warp_idx = tid / 32;

    auto sK_nope_part0 = make_tensor(make_smem_ptr(smem.k_nope_t_part0.data()), SmemLayoutKNoPEPart{});
    auto sK_nope_part1 = make_tensor(make_smem_ptr(smem.k_nope_t_part1.data()), SmemLayoutKNoPEPart{});
    auto sK_rope = make_tensor(make_smem_ptr(smem.k_rope_t.data()), SmemLayoutKRoPE{});

    constexpr int K0_ELEMS = 256 * B_TOPK;
    constexpr int KR_ELEMS = D_ROPE * B_TOPK;
    constexpr int K0_PER_THREAD = (K0_ELEMS + NUM_THREADS - 1) / NUM_THREADS;
    constexpr int KR_PER_THREAD = (KR_ELEMS + NUM_THREADS - 1) / NUM_THREADS;

    #pragma unroll
    for (int i = 0; i < K0_PER_THREAD; ++i) {
        const int idx = tid * K0_PER_THREAD + i;
        if (idx < K0_ELEMS) {
            const int row = idx / B_TOPK;
            const int col = idx % B_TOPK;
            sK_nope_part0(row, col) = k_nope_t_part0[row * B_TOPK + col];
            sK_nope_part1(row, col) = k_nope_t_part1[row * B_TOPK + col];
        }
    }

    #pragma unroll
    for (int i = 0; i < KR_PER_THREAD; ++i) {
        const int idx = tid * KR_PER_THREAD + i;
        if (idx < KR_ELEMS) {
            const int row = idx / B_TOPK;
            const int col = idx % B_TOPK;
            sK_rope(row, col) = k_rope_t[row * B_TOPK + col];
        }
    }

    __syncthreads();
    cluster_sync();

    if (warp_idx == 0) {
        TMEM::Allocator2Sm().allocate(512, &smem.tmem_start_addr);
        TMEM::Allocator2Sm().release_allocation_lock();
    }
    __syncthreads();

    const int row_in_cta = tid % (B_H / 2);
    const int topk_half = tid / (B_H / 2);   // 0/1
    const int global_row = cta_idx * (B_H / 2) + row_in_cta;

    const uint16_t* ds_u16 = reinterpret_cast<const uint16_t*>(ds);
    uint32_t ds_packed[8];

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int col = topk_half * 16 + i * 2;
        const int base = global_row * B_TOPK + col;
        ds_packed[i] = static_cast<uint32_t>(ds_u16[base + 0]) |
                       (static_cast<uint32_t>(ds_u16[base + 1]) << 16);
    }

    const uint32_t tmem_base = smem.tmem_start_addr;
    const uint32_t lane_addr = tmem_base + (row_in_cta << 16);

    ku::tmem_st_32dp32bNx<8>(lane_addr + tmem_cols::dS, ds_packed);
    cutlass::arch::fence_view_async_tmem_store();

    __syncthreads();
    cluster_sync();

    TiledMMA_dQ tiled_mma_dQ{};
    TiledMMA_dQ_RoPE tiled_mma_dQ_rope{};

    auto tdQ_part0 = partition_fragment_C(tiled_mma_dQ, Shape<Int<B_H / 2>, Int<256>>{});
    tdQ_part0.data().get() = tmem_cols::dQ;
    auto tdQ_part1 = partition_fragment_C(tiled_mma_dQ, Shape<Int<B_H / 2>, Int<256>>{});
    tdQ_part1.data().get() = tmem_cols::dQ + 128;
    auto tdQ_rope = partition_fragment_C(tiled_mma_dQ_rope, Shape<Int<B_H / 2>, Int<D_ROPE>>{});
    tdQ_rope.data().get() = tmem_cols::dQ_RoPE;

    auto tDS_dQ = tiled_mma_dQ.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_dQ, Shape<Int<B_H>, Int<B_TOPK>>{}));
    tDS_dQ.data().get() = tmem_cols::dS;

    auto tDS_dQ_rope = tiled_mma_dQ_rope.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_dQ_rope, Shape<Int<B_H>, Int<B_TOPK>>{}));
    tDS_dQ_rope.data().get() = tmem_cols::dS;

    if (cta_idx == 0 && warp_idx == 0 && elect_one_sync()) {
        // Fixed 3-call production structure
        ku::utcmma_ts(tiled_mma_dQ, tDS_dQ, sK_nope_part0, tdQ_part0, true);
        ku::utcmma_ts(tiled_mma_dQ, tDS_dQ, sK_nope_part1, tdQ_part1, true);
        ku::utcmma_ts(tiled_mma_dQ_rope, tDS_dQ_rope, sK_rope, tdQ_rope, true);
    }

    ku::tcgen05_after_thread_sync();
    __syncthreads();
    cluster_sync();

    const int row = tid % (B_H / 2);
    const int col_half = tid / (B_H / 2);  // 0/1
    const int out_row = cta_idx * (B_H / 2) + row;

    constexpr int NOPE_FLOATS_PER_HALF = 256 / 2;
    constexpr int NOPE_CHUNKS = 8;
    constexpr int NOPE_CHUNK_FLOATS = NOPE_FLOATS_PER_HALF / NOPE_CHUNKS;  // 16
    constexpr int NOPE_CHUNK_FLOAT2 = NOPE_CHUNK_FLOATS / 2;                // 8

    const uint32_t addr_part0 = tmem_base + (row << 16) + tmem_cols::dQ;
    const uint32_t addr_part1 = tmem_base + (row << 16) + (tmem_cols::dQ + 128);

    #pragma unroll
    for (int chunk = 0; chunk < NOPE_CHUNKS; ++chunk) {
        const int chunk_col_base = chunk * NOPE_CHUNK_FLOATS;

        float2 dq_chunk0[NOPE_CHUNK_FLOAT2];
        ku::tmem_ld_32dp32bNx<NOPE_CHUNK_FLOATS>(addr_part0 + chunk_col_base, dq_chunk0);
        cutlass::arch::fence_view_async_tmem_load();
        ku::tcgen05_before_thread_sync();

        #pragma unroll
        for (int i = 0; i < NOPE_CHUNK_FLOAT2; ++i) {
            const int col = col_half * NOPE_FLOATS_PER_HALF + chunk_col_base + i * 2;
            dq_nope_part0[out_row * 256 + col + 0] = dq_chunk0[i].x;
            dq_nope_part0[out_row * 256 + col + 1] = dq_chunk0[i].y;
        }

        float2 dq_chunk1[NOPE_CHUNK_FLOAT2];
        ku::tmem_ld_32dp32bNx<NOPE_CHUNK_FLOATS>(addr_part1 + chunk_col_base, dq_chunk1);
        cutlass::arch::fence_view_async_tmem_load();
        ku::tcgen05_before_thread_sync();

        #pragma unroll
        for (int i = 0; i < NOPE_CHUNK_FLOAT2; ++i) {
            const int col = col_half * NOPE_FLOATS_PER_HALF + chunk_col_base + i * 2;
            dq_nope_part1[out_row * 256 + col + 0] = dq_chunk1[i].x;
            dq_nope_part1[out_row * 256 + col + 1] = dq_chunk1[i].y;
        }
    }

    float2 dq_rope_vals[D_ROPE / 2 / 2];  // 16 float2
    const uint32_t addr_rope = tmem_base + (row << 16) + tmem_cols::dQ_RoPE;
    ku::tmem_ld_32dp32bNx<D_ROPE / 2>(addr_rope, dq_rope_vals);
    cutlass::arch::fence_view_async_tmem_load();
    ku::tcgen05_before_thread_sync();

    #pragma unroll
    for (int i = 0; i < (D_ROPE / 2 / 2); ++i) {
        const int col = col_half * (D_ROPE / 2) + i * 2;
        dq_rope[out_row * D_ROPE + col + 0] = dq_rope_vals[i].x;
        dq_rope[out_row * D_ROPE + col + 1] = dq_rope_vals[i].y;
    }

    __syncthreads();
    cluster_sync();

    if (warp_idx == 0 && elect_one_sync()) {
        TMEM::Allocator2Sm().free(smem.tmem_start_addr, 512);
    }
#endif
}

void launch_test_dq_mma(
    const bf16* ds,
    const bf16* k_nope_t_part0,
    const bf16* k_nope_t_part1,
    const bf16* k_rope_t,
    float* dq_nope_part0,
    float* dq_nope_part1,
    float* dq_rope,
    cudaStream_t stream
) {
    dim3 grid(2, 1, 1);
    dim3 block(NUM_THREADS, 1, 1);

    cudaError_t attr_err = cudaFuncSetAttribute(
        test_dq_mma_kernel,
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

    cudaError_t err = cudaLaunchKernelEx(
        &config,
        test_dq_mma_kernel,
        ds,
        k_nope_t_part0,
        k_nope_t_part1,
        k_rope_t,
        dq_nope_part0,
        dq_nope_part1,
        dq_rope);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaLaunchKernelEx failed: %s\n", cudaGetErrorString(err));
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> run_dq_mma(
    torch::Tensor ds,
    torch::Tensor k_nope_t_part0,
    torch::Tensor k_nope_t_part1,
    torch::Tensor k_rope_t
) {
    TORCH_CHECK(ds.is_cuda() && k_nope_t_part0.is_cuda() && k_nope_t_part1.is_cuda() && k_rope_t.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(ds.dtype() == torch::kBFloat16, "ds must be bfloat16");
    TORCH_CHECK(k_nope_t_part0.dtype() == torch::kBFloat16, "k_nope_t_part0 must be bfloat16");
    TORCH_CHECK(k_nope_t_part1.dtype() == torch::kBFloat16, "k_nope_t_part1 must be bfloat16");
    TORCH_CHECK(k_rope_t.dtype() == torch::kBFloat16, "k_rope_t must be bfloat16");

    TORCH_CHECK(ds.dim() == 2 && ds.size(0) == B_H && ds.size(1) == B_TOPK, "ds shape must be [128, 32]");
    TORCH_CHECK(k_nope_t_part0.dim() == 2 && k_nope_t_part0.size(0) == 256 && k_nope_t_part0.size(1) == B_TOPK,
                "k_nope_t_part0 shape must be [256, 32]");
    TORCH_CHECK(k_nope_t_part1.dim() == 2 && k_nope_t_part1.size(0) == 256 && k_nope_t_part1.size(1) == B_TOPK,
                "k_nope_t_part1 shape must be [256, 32]");
    TORCH_CHECK(k_rope_t.dim() == 2 && k_rope_t.size(0) == D_ROPE && k_rope_t.size(1) == B_TOPK,
                "k_rope_t shape must be [64, 32]");

    TORCH_CHECK(ds.is_contiguous() && k_nope_t_part0.is_contiguous() && k_nope_t_part1.is_contiguous() && k_rope_t.is_contiguous(),
                "all inputs must be contiguous");

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(ds.device());
    torch::Tensor dq_nope_part0 = torch::empty({B_H, 256}, opts);
    torch::Tensor dq_nope_part1 = torch::empty({B_H, 256}, opts);
    torch::Tensor dq_rope = torch::empty({B_H, D_ROPE}, opts);

    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_test_dq_mma(
        reinterpret_cast<const bf16*>(ds.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16*>(k_nope_t_part0.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16*>(k_nope_t_part1.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16*>(k_rope_t.data_ptr<at::BFloat16>()),
        dq_nope_part0.data_ptr<float>(),
        dq_nope_part1.data_ptr<float>(),
        dq_rope.data_ptr<float>(),
        stream);

    return std::make_tuple(dq_nope_part0, dq_nope_part1, dq_rope);
}

}  // namespace test_operator::dq_mma

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_dq_mma", &test_operator::dq_mma::run_dq_mma, "Run TiledMMA_dQ TS unit test kernel (3 utcmma_ts calls)");
}

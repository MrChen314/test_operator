#include "test_dkv_mma.cuh"

#include <c10/cuda/CUDAStream.h>
#include <kerutils/kerutils.cuh>
#include <torch/extension.h>

using namespace test_operator::dkv_mma;

__global__ void test_dkv_mma_kernel(
    const bf16* __restrict__ s,
    const bf16* __restrict__ ds,
    const bf16* __restrict__ dO_t,
    const bf16* __restrict__ q_t,
    const bf16* __restrict__ q_rope_t,
    float* __restrict__ dkv_out
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
    using namespace cute;

    extern __shared__ char smem_raw[];
    SharedMemory& smem = *reinterpret_cast<SharedMemory*>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_idx = tid / 32;

    auto sdO_t_full = make_tensor(make_smem_ptr(smem.dO_t.data()), SmemLayoutDOTransposed{});
    auto sQ_t_full = make_tensor(make_smem_ptr(smem.q_t.data()), SmemLayoutQNoPETransposed{});
    auto sQ_rope_t = make_tensor(make_smem_ptr(smem.q_rope_t.data()), SmemLayoutQRoPETransposed{});

    constexpr int DO_ELEMS = D_V * (B_H / 2);      // 512 * 64
    constexpr int Q_ELEMS = D_V * (B_H / 2);       // 512 * 64
    constexpr int QR_ELEMS = D_ROPE * (B_H / 2);   // 64 * 64

    constexpr int DO_PER_THREAD = (DO_ELEMS + NUM_THREADS - 1) / NUM_THREADS;
    constexpr int Q_PER_THREAD = (Q_ELEMS + NUM_THREADS - 1) / NUM_THREADS;
    constexpr int QR_PER_THREAD = (QR_ELEMS + NUM_THREADS - 1) / NUM_THREADS;

    #pragma unroll
    for (int i = 0; i < DO_PER_THREAD; ++i) {
        const int idx = tid * DO_PER_THREAD + i;
        if (idx < DO_ELEMS) {
            const int row = idx / (B_H / 2);   // 0..511
            const int col = idx % (B_H / 2);   // 0..63
            sdO_t_full(row, col) = dO_t[row * (B_H / 2) + col];
        }
    }

    #pragma unroll
    for (int i = 0; i < Q_PER_THREAD; ++i) {
        const int idx = tid * Q_PER_THREAD + i;
        if (idx < Q_ELEMS) {
            const int row = idx / (B_H / 2);
            const int col = idx % (B_H / 2);
            sQ_t_full(row, col) = q_t[row * (B_H / 2) + col];
        }
    }

    #pragma unroll
    for (int i = 0; i < QR_PER_THREAD; ++i) {
        const int idx = tid * QR_PER_THREAD + i;
        if (idx < QR_ELEMS) {
            const int row = idx / (B_H / 2);   // 0..63
            const int col = idx % (B_H / 2);   // 0..63
            sQ_rope_t(row, col) = q_rope_t[row * (B_H / 2) + col];
        }
    }

    __syncthreads();

    if (warp_idx == 0) {
        TMEM::Allocator2Sm().allocate(512, &smem.tmem_start_addr);
        TMEM::Allocator2Sm().release_allocation_lock();
    }
    __syncthreads();

    const int row = tid % (B_H / 2);         // 0..63
    const int topk_half = tid / (B_H / 2);   // 0..1

    const uint16_t* s_u16 = reinterpret_cast<const uint16_t*>(s);
    const uint16_t* ds_u16 = reinterpret_cast<const uint16_t*>(ds);

    uint32_t s_pack[8];
    uint32_t ds_pack[8];

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int col = topk_half * 16 + i * 2;
        const int base = row * B_TOPK + col;
        s_pack[i] = static_cast<uint32_t>(s_u16[base + 0]) |
                    (static_cast<uint32_t>(s_u16[base + 1]) << 16);
        ds_pack[i] = static_cast<uint32_t>(ds_u16[base + 0]) |
                     (static_cast<uint32_t>(ds_u16[base + 1]) << 16);
    }

    const uint32_t tmem_base = smem.tmem_start_addr;
    const uint32_t lane_addr = tmem_base + (row << 16);
    ku::tmem_st_32dp32bNx<8>(lane_addr + tmem_cols::S, s_pack);
    cutlass::arch::fence_view_async_tmem_store();
    ku::tmem_st_32dp32bNx<8>(lane_addr + tmem_cols::dS, ds_pack);
    cutlass::arch::fence_view_async_tmem_store();

    __syncthreads();

    TiledMMA_dKV tiled_mma_dKV{};
    TiledMMA_dKV_RoPE tiled_mma_dKV_rope{};

    auto tdKV = partition_fragment_C(tiled_mma_dKV, Shape<Int<B_TOPK>, Int<256>>{});
    tdKV.data().get() = tmem_cols::dKV;
    auto tdKV_rope = partition_fragment_C(tiled_mma_dKV_rope, Shape<Int<B_TOPK>, Int<D_ROPE>>{});
    tdKV_rope.data().get() = tmem_cols::dKV_RoPE;

    auto tS_dKV = tiled_mma_dKV.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_dKV, Shape<Int<B_TOPK>, Int<B_H / 2>>{}));
    tS_dKV.data().get() = tmem_cols::S;

    auto tDS_dKV = tiled_mma_dKV.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_dKV, Shape<Int<B_TOPK>, Int<B_H / 2>>{}));
    tDS_dKV.data().get() = tmem_cols::dS;

    auto tDS_dKV_rope = tiled_mma_dKV_rope.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_dKV_rope, Shape<Int<B_TOPK>, Int<B_H / 2>>{}));
    tDS_dKV_rope.data().get() = tmem_cols::dS;

    auto sdO_t_div = flat_divide(sdO_t_full, Shape<Int<256>, Int<B_H / 2>>{});
    auto sQ_t_div = flat_divide(sQ_t_full, Shape<Int<256>, Int<B_H / 2>>{});

    if (warp_idx == 0 && elect_one_sync()) {
        // part0: [0, 256)
        ku::utcmma_ts(tiled_mma_dKV, tS_dKV, sdO_t_div(_, _, _0{}, _0{}), tdKV, true);
        ku::utcmma_ts(tiled_mma_dKV, tDS_dKV, sQ_t_div(_, _, _0{}, _0{}), tdKV, false);
    }
    ku::tcgen05_after_thread_sync();
    __syncthreads();

    if (tid < B_TOPK) {
        const int out_row = tid;
        const uint32_t row_addr = tmem_base + (out_row << 16);

        #pragma unroll
        for (int seg = 0; seg < 4; ++seg) {
            float2 dkv_data[32];
            ku::tmem_ld_32dp32bNx<64>(row_addr + tmem_cols::dKV + seg * 64, dkv_data);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();

            const float* src = reinterpret_cast<const float*>(dkv_data);
            #pragma unroll
            for (int i = 0; i < 64; ++i) {
                dkv_out[out_row * D_K + seg * 64 + i] = src[i];
            }
        }
    }
    __syncthreads();

    if (warp_idx == 0 && elect_one_sync()) {
        // part1: [256, 512)
        ku::utcmma_ts(tiled_mma_dKV, tS_dKV, sdO_t_div(_, _, _1{}, _0{}), tdKV, true);
        ku::utcmma_ts(tiled_mma_dKV, tDS_dKV, sQ_t_div(_, _, _1{}, _0{}), tdKV, false);
    }
    ku::tcgen05_after_thread_sync();
    __syncthreads();

    if (tid < B_TOPK) {
        const int out_row = tid;
        const uint32_t row_addr = tmem_base + (out_row << 16);

        #pragma unroll
        for (int seg = 0; seg < 4; ++seg) {
            float2 dkv_data[32];
            ku::tmem_ld_32dp32bNx<64>(row_addr + tmem_cols::dKV + seg * 64, dkv_data);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();

            const float* src = reinterpret_cast<const float*>(dkv_data);
            #pragma unroll
            for (int i = 0; i < 64; ++i) {
                dkv_out[out_row * D_K + 256 + seg * 64 + i] = src[i];
            }
        }
    }
    __syncthreads();

    if (warp_idx == 0 && elect_one_sync()) {
        // part2: [512, 576)
        ku::utcmma_ts(tiled_mma_dKV_rope, tDS_dKV_rope, sQ_rope_t, tdKV_rope, true);
    }
    ku::tcgen05_after_thread_sync();
    __syncthreads();

    if (tid < B_TOPK) {
        const int out_row = tid;
        const uint32_t row_addr = tmem_base + (out_row << 16);

        #pragma unroll
        for (int seg = 0; seg < 4; ++seg) {
            float2 dkv_rope_data[8];
            ku::tmem_ld_32dp32bNx<16>(row_addr + tmem_cols::dKV_RoPE + seg * 16, dkv_rope_data);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();

            const float* src = reinterpret_cast<const float*>(dkv_rope_data);
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                dkv_out[out_row * D_K + 512 + seg * 16 + i] = src[i];
            }
        }
    }

    __syncthreads();
    if (warp_idx == 0 && elect_one_sync()) {
        TMEM::Allocator2Sm().free(smem.tmem_start_addr, 512);
    }
#endif
}

void launch_test_dkv_mma(
    const bf16* s,
    const bf16* ds,
    const bf16* dO_t,
    const bf16* q_t,
    const bf16* q_rope_t,
    float* dkv_out,
    cudaStream_t stream
) {
    dim3 grid(1, 1, 1);
    dim3 block(NUM_THREADS, 1, 1);

    cudaError_t attr_err = cudaFuncSetAttribute(
        test_dkv_mma_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(SMEM_SIZE));
    if (attr_err != cudaSuccess) {
        fprintf(stderr, "cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(attr_err));
        return;
    }

    test_dkv_mma_kernel<<<grid, block, SMEM_SIZE, stream>>>(s, ds, dO_t, q_t, q_rope_t, dkv_out);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "test_dkv_mma_kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

torch::Tensor run_dkv_mma(
    torch::Tensor s,
    torch::Tensor ds,
    torch::Tensor dO_t,
    torch::Tensor q_t,
    torch::Tensor q_rope_t
) {
    TORCH_CHECK(s.is_cuda() && ds.is_cuda() && dO_t.is_cuda() && q_t.is_cuda() && q_rope_t.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(s.dtype() == torch::kBFloat16, "s must be bfloat16");
    TORCH_CHECK(ds.dtype() == torch::kBFloat16, "ds must be bfloat16");
    TORCH_CHECK(dO_t.dtype() == torch::kBFloat16, "dO_t must be bfloat16");
    TORCH_CHECK(q_t.dtype() == torch::kBFloat16, "q_t must be bfloat16");
    TORCH_CHECK(q_rope_t.dtype() == torch::kBFloat16, "q_rope_t must be bfloat16");

    TORCH_CHECK(s.dim() == 2 && s.size(0) == B_H / 2 && s.size(1) == B_TOPK, "s shape must be [64, 32]");
    TORCH_CHECK(ds.dim() == 2 && ds.size(0) == B_H / 2 && ds.size(1) == B_TOPK, "ds shape must be [64, 32]");
    TORCH_CHECK(dO_t.dim() == 2 && dO_t.size(0) == D_V && dO_t.size(1) == B_H / 2,
                "dO_t shape must be [512, 64]");
    TORCH_CHECK(q_t.dim() == 2 && q_t.size(0) == D_V && q_t.size(1) == B_H / 2,
                "q_t shape must be [512, 64]");
    TORCH_CHECK(q_rope_t.dim() == 2 && q_rope_t.size(0) == D_ROPE && q_rope_t.size(1) == B_H / 2,
                "q_rope_t shape must be [64, 64]");

    TORCH_CHECK(s.is_contiguous() && ds.is_contiguous() && dO_t.is_contiguous() && q_t.is_contiguous() && q_rope_t.is_contiguous(),
                "all inputs must be contiguous");

    auto dkv_out = torch::empty({B_TOPK, D_K}, torch::TensorOptions().dtype(torch::kFloat32).device(s.device()));
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_test_dkv_mma(
        reinterpret_cast<const bf16*>(s.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16*>(ds.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16*>(dO_t.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16*>(q_t.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16*>(q_rope_t.data_ptr<at::BFloat16>()),
        dkv_out.data_ptr<float>(),
        stream);

    return dkv_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_dkv_mma", &run_dkv_mma, "Run TiledMMA_dKV TS unit test kernel");
}

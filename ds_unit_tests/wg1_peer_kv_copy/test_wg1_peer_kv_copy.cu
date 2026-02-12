#include "test_wg1_peer_kv_copy.cuh"

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstring>

namespace test_operator::wg1_peer_kv_copy {

__global__ void test_wg1_peer_kv_copy_kernel(
    const bf16* __restrict__ init_kv,
    bf16* __restrict__ out_kv
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
    using namespace cute;

    extern __shared__ char smem_raw[];
    SharedMemory& smem = *reinterpret_cast<SharedMemory*>(smem_raw);

    const int cta_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int thread_in_wg = tid;

    Tensor sKV = make_tensor(make_smem_ptr(smem.kv.data()), SmemLayoutKV{});
    bf16* peer_kv_ptr = kerutils::get_peer_addr(smem.kv.data());
    Tensor sKVPeer = make_tensor(make_smem_ptr(peer_kv_ptr), SmemLayoutKV{});

    constexpr int KV_ELEMS = KV_ROWS * D_K;
    constexpr int KV_ELEMS_PER_THREAD = (KV_ELEMS + NUM_THREADS - 1) / NUM_THREADS;

    #pragma unroll
    for (int i = 0; i < KV_ELEMS_PER_THREAD; ++i) {
        const int linear = tid + i * NUM_THREADS;
        if (linear < KV_ELEMS) {
            const int row = linear / D_K;
            const int col = linear % D_K;
            sKV(row, col) = init_kv[(cta_idx * KV_ROWS + row) * D_K + col];
        }
    }

    __syncthreads();
    cluster_sync();

    constexpr int PEER_COLS = D_K / 2;
    constexpr int PEER_ELEMS = KV_ROWS * PEER_COLS;
    constexpr int PEER_CACHE_ELEMS = (PEER_ELEMS + 128 - 1) / 128;
    bf16 peer_cache[PEER_CACHE_ELEMS];

    const int src_col_base = (cta_idx == 0 ? 0 : D_K / 2);
    const int dst_col_base = (cta_idx == 0 ? D_K / 2 : 0);

    #pragma unroll
    for (int i = 0; i < PEER_CACHE_ELEMS; ++i) {
        int linear = thread_in_wg + i * 128;
        if (linear < PEER_ELEMS) {
            int row = linear / PEER_COLS;
            int col = linear % PEER_COLS;
            peer_cache[i] = sKVPeer(row, src_col_base + col);
        }
    }

    cluster_sync();
    ku::tcgen05_after_thread_sync();

    #pragma unroll
    for (int i = 0; i < PEER_CACHE_ELEMS; ++i) {
        int linear = thread_in_wg + i * 128;
        if (linear < PEER_ELEMS) {
            int row = linear / PEER_COLS;
            int col = linear % PEER_COLS;
            sKV(row, dst_col_base + col) = peer_cache[i];
        }
    }

    __syncthreads();
    cluster_sync();

    #pragma unroll
    for (int i = 0; i < KV_ELEMS_PER_THREAD; ++i) {
        const int linear = tid + i * NUM_THREADS;
        if (linear < KV_ELEMS) {
            const int row = linear / D_K;
            const int col = linear % D_K;
            out_kv[(cta_idx * KV_ROWS + row) * D_K + col] = sKV(row, col);
        }
    }
#else
    (void)init_kv;
    (void)out_kv;
#endif
}

void launch_test_wg1_peer_kv_copy(
    const bf16* init_kv,
    bf16* out_kv,
    cudaStream_t stream
) {
    dim3 grid(2, 1, 1);
    dim3 block(NUM_THREADS, 1, 1);

    cudaError_t attr_err = cudaFuncSetAttribute(
        test_wg1_peer_kv_copy_kernel,
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

    cudaError_t err = cudaLaunchKernelEx(&config, test_wg1_peer_kv_copy_kernel, init_kv, out_kv);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaLaunchKernelEx failed: %s\n", cudaGetErrorString(err));
    }
}

torch::Tensor run_wg1_peer_kv_copy(torch::Tensor init_kv) {
    TORCH_CHECK(init_kv.is_cuda(), "init_kv must be CUDA tensor");
    TORCH_CHECK(init_kv.dtype() == torch::kBFloat16, "init_kv must be bfloat16");
    TORCH_CHECK(init_kv.dim() == 3 &&
                init_kv.size(0) == 2 &&
                init_kv.size(1) == KV_ROWS &&
                init_kv.size(2) == D_K,
                "init_kv shape must be [2, 16, 576]");
    TORCH_CHECK(init_kv.is_contiguous(), "init_kv must be contiguous");

    torch::Tensor out_kv = torch::empty_like(init_kv);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_test_wg1_peer_kv_copy(
        reinterpret_cast<const bf16*>(init_kv.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16*>(out_kv.data_ptr<at::BFloat16>()),
        stream);

    return out_kv;
}

}  // namespace test_operator::wg1_peer_kv_copy

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_wg1_peer_kv_copy",
          &test_operator::wg1_peer_kv_copy::run_wg1_peer_kv_copy,
          "Run WG1 peer KV half-copy/backfill unit test");
}

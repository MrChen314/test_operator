#pragma once

#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/cluster_launch.hpp>
#include <kerutils/kerutils.cuh>
#include <cutlass/arch/arch.h>

// 避免使用 using namespace cute 以防止与 at::Layout 冲突
using bf16 = cutlass::bfloat16_t;

// 矩阵维度常量
static constexpr int M = 128;           // Q 行数 (2个CTA总共)
static constexpr int N = 128;           // K 行数 (2个CTA总共), 即 P 的列数
static constexpr int K_DIM = 256;       // 共享维度
static constexpr int NUM_K_TILES = K_DIM / 64;  // = 4

// SMEM Layout 定义 (从 config.h 提取)
// Q 矩阵 SMEM 布局：每个 CTA 存 [M/2, K_DIM] = [64, 256]
using SmemLayoutQ = decltype(cute::coalesce(cute::tile_to_shape(
    cute::UMMA::Layout_K_SW128_Atom<bf16>{},
    cute::Shape<cute::Int<M/2>, cute::Int<K_DIM>>{},
    cute::Step<cute::_1, cute::_2>{}
), cute::Shape<cute::_1, cute::_1>{}));

// K 矩阵 SMEM 布局：每个 CTA 存 [N/2, K_DIM] = [64, 256]
using SmemLayoutK = decltype(cute::coalesce(cute::tile_to_shape(
    cute::UMMA::Layout_K_SW128_Atom<bf16>{},
    cute::Shape<cute::Int<N/2>, cute::Int<K_DIM>>{},
    cute::Step<cute::_1, cute::_2>{}
), cute::Shape<cute::_1, cute::_1>{}));

// TMEM 列配置 (从 config.h 提取)
struct tmem_cols {
    // P 矩阵存储在 TMEM 的列 256-319 (64列)
    static constexpr int p = 256;
    static_assert(p + 64 <= 512, "TMEM column overflow");
};

// TiledMMA_P_sQ 定义 (从 config.h 提取)
// 用于计算 P = Q @ K^T，使用 SMEM-SMEM 模式
using TiledMMA_P_sQ = decltype(cute::make_tiled_mma(
    cute::SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, M, N, cute::UMMA::Major::K, cute::UMMA::Major::K>{}
));

// Shared Memory 结构
struct SharedMemory {
    alignas(128) bf16 q[M/2 * K_DIM];     // 每个 CTA 存一半 Q: [64, 256]
    alignas(128) bf16 k[N/2 * K_DIM];     // 每个 CTA 存一半 K: [64, 256]
    alignas(4) uint32_t tmem_start_addr;  // TMEM 起始地址 (由 Allocator2Sm 分配)
};

// SMEM 大小
static constexpr size_t SMEM_SIZE = sizeof(SharedMemory);

// 线程数
static constexpr int NUM_THREADS = 128;  // 每个 CTA 128 线程

// Kernel 声明 (cluster dims 在 launch 时指定)
__global__ void test_utcmma_ss_kernel(
    const bf16* __restrict__ Q,      // [M, K_DIM] = [128, 256]
    const bf16* __restrict__ K,      // [N, K_DIM] = [128, 256]
    float* __restrict__ P_out        // [M, N] = [128, 128] (矩阵乘结果)
);

// C++ wrapper 声明
void launch_test_utcmma_ss(
    const bf16* Q,
    const bf16* K,
    float* P_out,
    cudaStream_t stream = nullptr
);
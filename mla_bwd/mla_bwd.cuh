#pragma once

#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/barrier.h>
#include <kerutils/kerutils.cuh>
#include <cutlass/arch/arch.h>

using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;

namespace test_operator::mla_bwd {

using namespace cute;

using bf16 = cutlass::bfloat16_t;

// ============================================================================
// 维度常量定义
// ============================================================================
static constexpr int D_QK = 576;
static constexpr int D_Q = D_QK;                    // Query 维度
static constexpr int D_K = D_QK;                    // Key 维度  
static constexpr int D_V = 512;                     // Value/NoPE 维度
static constexpr int D_ROPE = D_Q - D_V;            // RoPE 维度 = 64 (当 D_QK=576)
static constexpr float MAX_INIT_VAL = -1e30f;       // 用于 max logits 初始化
static constexpr bool HAVE_ROPE = (D_QK == 576);    // 是否启用 RoPE

// ============================================================================
// 2CTA 相关常量定义
// ============================================================================
static constexpr int B_H = 128;                     // Query head 块大小 (2CTA 共享，每个 CTA 处理 B_H/2=64 行)
static constexpr int B_TOPK = 64;                    // TopK 块大小 (2CTA 模式每次处理 64 个 topk，每个 CTA 加载 B_TOPK/2=32 行)
static constexpr int NUM_THREADS = 4 * 128; // 4个 WarpGroup，每个128线程

// SMEM Layout definitions (from config.h)
// Q matrix SMEM layout: stores [M, K_DIM] = [64, 128]
template<int NUM_TILES>
using SmemLayoutQTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H/2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));


using SmemLayoutQNoPE = SmemLayoutQTiles<8>;
using SmemLayoutQRoPE = SmemLayoutQTiles<1>;
// Full Q layout: NoPE (8 tiles) + RoPE (1 tile) = 9 tiles
using SmemLayoutQ = SmemLayoutQTiles<9>;

// KV Layout 模板: [B_TOPK/2, 64*NUM_TILES]
// 2CTA 模式下每个 CTA 加载 B_TOPK/2 行
// 合并了 NoPE 和 RoPE，使用 SW128 处理整个 D_Q=576 维
template<int NUM_TILES>
using SmemLayoutKVTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK/2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));


using SmemLayoutKNoPE = SmemLayoutKVTiles<8>;
using SmemLayoutKRoPE = SmemLayoutKVTiles<1>;
// Full KV layout: NoPE (8 tiles) + RoPE (1 tile) = 9 tiles
using SmemLayoutKV = SmemLayoutKVTiles<9>;

using SmemLayoutV = SmemLayoutKNoPE;

// dO Layout 模板: [B_H/2, 64*NUM_TILES]
template<int NUM_TILES>
using SmemLayoutdOTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H/2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// dO 完整 Layout: [B_H/2, D_V] = [64, 512]
using SmemLayoutdO = SmemLayoutdOTiles<D_V/64>;

// S/dS 矩阵 Layout 模板: [B_H/2, 64*NUM_TILES]
// 2CTA 模式下 S 矩阵形状为 [B_H/2, B_TOPK]
template<int NUM_TILES>
using SmemLayoutSTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<B_H/2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// S/dS 完整 Layout: [B_H/2, B_TOPK] = [64, 64]
using SmemLayoutS = SmemLayoutSTiles<B_TOPK/64>;

// ============================================================================
// TiledMMA 定义 (2CTA 模式)
// ============================================================================
// TiledMMA_P: 用于计算 P = Q @ K^T
// 2CTA 模式: [B_H, B_TOPK] = [128, 64]
// 每个 CTA 处理 B_H/2 = 64 行，共同完成完整的 MMA
// 指令: utcmma_ss (SMEM-SMEM), 2x1SM 协作
using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_H, B_TOPK, UMMA::Major::K, UMMA::Major::K>{}
));

// TiledMMA_dP: 用于计算 dP = dO @ V^T
// 2CTA 模式: [B_H, B_TOPK] = [128, 64]
// dO: [B_H/2, D_V] = [64, 512], V: [B_TOPK/2, D_V] = [32, 512]
// 指令: utcmma_ss (SMEM-SMEM), 2x1SM 协作
using TiledMMA_dP = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_H, B_TOPK, UMMA::Major::K, UMMA::Major::K>{}
));


// TMEM column configuration
struct tmem_cols {
    // dQ NoPE 累加器: dQuery 梯度累加 (NoPE 部分)
    // 由于 MMA N=256 限制，分成两部分存储
    // Shape: [B_H/2, 256] × 2 = [64, 512], 需要 512*64/128 = 256 列
    static constexpr int dQ = 0;   
    
    // dQ RoPE 累加器: dQuery 梯度累加 (RoPE 部分)
    // Shape: [B_H/2, D_ROPE] = [64, 64], 需要 64*64/128 = 32 列
    static constexpr int dQ_RoPE = 256;
    
    // dKV NoPE 累加器: dKey/dValue 梯度累加 (NoPE 部分)
    // 由于 资源限制，分成两部分计算
    // Shape: [B_TOPK, 256] × 2 = [64, 512], 每部分需要 64*256/128 = 128 列
    static constexpr int dKV = 288;      

    // dKV RoPE 累加器: dKey/dValue 梯度累加 (RoPE 部分)
    // Shape: [B_TOPK, D_ROPE] = [64, 64], 需要 64*64/128 = 32 列
    static constexpr int dKV_RoPE = 416;
    
    // P 矩阵: Attention Scores
    // 2CTA 模式: 每个 CTA 存储 [B_H/2, B_TOPK] = [64, 64], 需要 64*64/128 = 32 列
    static constexpr int P = 448;

    // dP 矩阵: dAttention Scores
    // 2CTA 模式: 每个 CTA 存储 [B_H/2, B_TOPK] = [64, 64], 需要 32 列
    static constexpr int dP = 480;
};


// Shared Memory structure
struct alignas(128) SharedMemoryPlan {
    // 主联合体: dQ 与 Q+KV 空间复用
    union {
        // Q + KV 计算阶段: Q 和 KV 同时驻留
        struct {
            // Q 缓冲区 (每个 CTA 处理 B_H/2 行)
            array_aligned<bf16, cosize_v<SmemLayoutQNoPE>> q_nope;      // [B_H/2, D_V] = [64, 512] bf16
            array_aligned<bf16, cosize_v<SmemLayoutQRoPE>> q_rope;      // [B_H/2, D_ROPE] = [64, 64] bf16
            // KV 缓冲区 (每个 CTA 加载 B_TOPK/2 行)
            array_aligned<bf16, cosize_v<SmemLayoutKNoPE>> k_nope;    // [B_TOPK/2, D_V] = [32, 512] bf16
            array_aligned<bf16, cosize_v<SmemLayoutKRoPE>> k_rope;    // [B_TOPK/2, D_ROPE] = [32, 64] bf16
            array_aligned<bf16, cosize_v<SmemLayoutKV>> kv_peer;    // [B_TOPK/2, D_K] = [32, 576] bf16
        } q_kv;
        
        // dQ 输出阶段 (与 Q+KV 空间复用)
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutQ>> dq;    // [B_H/2, D_Q] = [64, 576] bf16
        } dq;
    } u;
    
    // dO 缓冲区: 全程驻留 (每个 CTA 处理 B_H/2 行)
    array_aligned<bf16, cosize_v<SmemLayoutdO>> dO;                     // [B_H/2, D_V] bf16

    struct {
        // S 矩阵：softmax值，bf16 [B_H/2, B_TOPK]
        array_aligned<bf16, cosize_v<SmemLayoutS>> s;
        // dS 矩阵: bf16 [B_H/2, B_TOPK]
        array_aligned<bf16, cosize_v<SmemLayoutS>> ds;  // dS梯度
    } s_ds;
    
    // KV 有效性掩码
    char is_kv_valid[B_TOPK/8];

    // ========================================================================
    // 同步屏障 (2CTA 同步)
    // ========================================================================
    // WG1-WG3 同步屏障
    transac_bar_t bar_qkv_loaded;               // WG1通知WG3 q/k/dO已加载到SMEM (2CTA sync)
    // WG0-WG3 同步屏障
    transac_bar_t bar_p_ready;                  // WG3通知WG0 p已准备好 (2CTA sync)
    transac_bar_t bar_dp_ready;                 // WG3通知WG0 dp已准备好 (2CTA sync)
    transac_bar_t bar_s_ready;                  // WG0通知WG3 s已准备好 (2CTA sync)
    transac_bar_t bar_ds_ready;                 // WG0通知WG3 ds已准备好 (2CTA sync)

    // TMEM 起始地址
    array_aligned<uint32_t, 1> tmem_start_addr;
    
    // Rowwise 缓冲区 (用于 softmax 和 Delta)
    float rowwise_max_buf[128];                 // max logits
    float rowwise_li_buf[128];                  // log-sum-exp
    float rowwise_delta_buf[128];               // Delta = sum(O * dO)
};

// SMEM size
static constexpr size_t SMEM_SIZE = sizeof(SharedMemoryPlan);

}  // namespace test_operator::mla_bwd

// Kernel declaration (cluster dims specified at launch time)
// Must be in global scope for CUDA kernel
__global__ void test_mla_bwd_kernel(
    const test_operator::mla_bwd::bf16* __restrict__ q,      // [B_H, D_Q] = [128, 576]
    const test_operator::mla_bwd::bf16* __restrict__ kv,      // [B_TOPK, D_K] = [64, 576]
    const test_operator::mla_bwd::bf16* __restrict__ dO,     // [B_H, D_V] = [128, 512]
    const float* __restrict__ lse,     // [B_H] = [128] (log-sum-exp for softmax)
    const test_operator::mla_bwd::bf16* __restrict__ O,     // [B_H, D_V] = [128, 512] (forward output O)
    test_operator::mla_bwd::bf16* __restrict__ q_out,        // [B_H, D_Q] = [128, 576] (output Q from SMEM)
    test_operator::mla_bwd::bf16* __restrict__ kv_out,       // [B_TOPK, D_K] = [64, 576] (output KV from SMEM)
    test_operator::mla_bwd::bf16* __restrict__ dO_out,       // [B_H, D_V] = [128, 512] (output dO from SMEM)
    float* __restrict__ P,           // [B_H, B_TOPK] = [128, 64] (P = Q @ K^T)
    float* __restrict__ dP,           // [B_H, B_TOPK] = [128, 64] (dP = dO @ V^T)
    test_operator::mla_bwd::bf16* __restrict__ s,           // [B_H, B_TOPK] = [128, 64] (softmax values)
    test_operator::mla_bwd::bf16* __restrict__ ds,           // [B_H, B_TOPK] = [128, 64] (dS gradients)
    const float* __restrict__ delta   // [B_H] = [128] (delta = sum(O * dO))
);

// C++ wrapper declaration
void launch_test_mla_bwd(
    const test_operator::mla_bwd::bf16* q,
    const test_operator::mla_bwd::bf16* kv,
    const test_operator::mla_bwd::bf16* dO,
    const float* lse,
    const test_operator::mla_bwd::bf16* O,
    test_operator::mla_bwd::bf16* q_out,
    test_operator::mla_bwd::bf16* kv_out,
    test_operator::mla_bwd::bf16* dO_out,
    float* P,
    float* dP,
    test_operator::mla_bwd::bf16* s,
    test_operator::mla_bwd::bf16* ds,
    const float* delta,
    cudaStream_t stream = nullptr
);

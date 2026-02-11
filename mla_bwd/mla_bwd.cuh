#pragma once

#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/barrier.h>
#include <kerutils/kerutils.cuh>
#include <cutlass/arch/arch.h>
#include <cuda.h>

using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;

namespace test_operator::mla_bwd {

using namespace cute;

using bf16 = cutlass::bfloat16_t;

template<
    typename Shape_QNoPE, typename TMA_QNoPE,
    typename Shape_QRoPE, typename TMA_QRoPE,
    typename Shape_KV, typename TMA_KV,
    typename Shape_dO, typename TMA_dO,
    typename Shape_dQ, typename TMA_dQ
>
struct TmaParams {
    Shape_QNoPE shape_Q_nope;
    TMA_QNoPE tma_Q_nope;
    Shape_QRoPE shape_Q_rope;
    TMA_QRoPE tma_Q_rope;
    Shape_KV shape_KV;
    TMA_KV tma_KV;
    Shape_dO shape_dO;
    TMA_dO tma_dO;
    Shape_dQ shape_dQ;
    TMA_dQ tma_dQ;
    CUtensorMap tensor_map_kv;
};

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
static constexpr int B_TOPK = 64;                    // 编译期 tile 大小。运行时 topk 由 kernel 内按 64 分块循环处理
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
// dO 完整 Layout: [B_H/2, D_V] = [64, 512]
using SmemLayoutdO = SmemLayoutQTiles<D_V/64>;

template<int NUM_TILES>
using SmemLayoutQTilesTransposed = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_SW128_Atom<bf16>{},
    Shape<Int<64*NUM_TILES>, Int<B_H/2>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

using SmemLayoutQNoPETransposed = SmemLayoutQTilesTransposed<4>;
using SmemLayoutQRoPETransposed = SmemLayoutQTilesTransposed<1>;
using SmemLayoutdOTransposed = SmemLayoutQTilesTransposed<4>;

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

template<int NUM_TILES>
using SmemLayoutKVTilesTransposed = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_SW128_Atom<bf16>{},
    Shape<Int<64*NUM_TILES>, Int<B_TOPK/2>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

using SmemLayoutKNoPETransposed = SmemLayoutKVTilesTransposed<4>;
using SmemLayoutKRoPETransposed = SmemLayoutKVTilesTransposed<1>;

// 2CTA 模式下 S 矩阵形状为 [B_H/2, B_TOPK]
using SmemLayoutS = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<B_H/2>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutdS = SmemLayoutS;

using SmemLayoutdSTransposed = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_INTER_Atom<bf16>{},
    Shape<Int<B_H/2>, Int<B_TOPK>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

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

using TiledMMA_dQ = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H/2, 256, UMMA::Major::MN, UMMA::Major::MN>{}
));

using TiledMMA_dQ_RoPE = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H/2, D_ROPE, UMMA::Major::MN, UMMA::Major::MN>{}
));

using TiledMMA_dKV = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_TOPK, 256, UMMA::Major::K, UMMA::Major::MN>{}
));

using TiledMMA_dKV_RoPE = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_TOPK, D_ROPE, UMMA::Major::K, UMMA::Major::MN>{}
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
            // KV 缓冲区 (每个 CTA 加载 B_TOPK/2 行)
            array_aligned<bf16, cosize_v<SmemLayoutKNoPE>> k_nope;    // [B_TOPK/2, D_V] = [32, 512] bf16
            array_aligned<bf16, cosize_v<SmemLayoutKRoPE>> k_rope;    // [B_TOPK/2, D_ROPE] = [32, 64] bf16
            array_aligned<bf16, cosize_v<SmemLayoutKV>> kv_peer;    // [B_TOPK/2, D_K] = [32, 576] bf16
            // Q 缓冲区 (每个 CTA 处理 B_H/2 行)
            array_aligned<bf16, cosize_v<SmemLayoutQNoPE>> q_nope;      // [B_H/2, D_V] = [64, 512] bf16
            array_aligned<bf16, cosize_v<SmemLayoutQRoPE>> q_rope;      // [B_H/2, D_ROPE] = [64, 64] bf16
        } q_kv;
        
        // dQ 输出阶段 (与 KV 空间复用；注意不能与Q复用，会影响dKV精度)
        array_aligned<bf16, cosize_v<SmemLayoutQ>> dq;    // [B_H/2, D_Q] = [64, 576] bf16
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
    char is_k_valid[B_TOPK/8];

    // ========================================================================
    // 同步屏障 (2CTA 同步)
    // ========================================================================
    // Prologue TMA load barriers
    transac_bar_t bar_prologue_q_nope;          // Q_NoPE TMA 完成
    transac_bar_t bar_prologue_q_rope;          // Q_RoPE TMA 完成
    transac_bar_t bar_prologue_kv;              // KV TMA 完成
    transac_bar_t bar_prologue_dO;              // dO TMA 完成
    // WG0-WG3 同步屏障
    transac_bar_t bar_p_ready;                  // WG3通知WG0 p已准备好 (2CTA sync)
    transac_bar_t bar_dp_ready;                 // WG3通知WG0 dp已准备好 (2CTA sync)
    transac_bar_t bar_s_ready;                  // WG0通知WG3 s已准备好 (2CTA sync)
    transac_bar_t bar_ds_ready;                 // WG0通知WG3 ds已准备好 (2CTA sync)
    transac_bar_t bar_k_valid_free;             // Reserved for future k-mask path (unused now)
    transac_bar_t bar_k_valid_ready;            // Reserved for future k-mask path (unused now)
    // WG3-WG2 同步屏障 (dKV computation)
    transac_bar_t bar_dkv_part0_ready;          // WG3通知WG2 dKV_part0计算完成
    transac_bar_t bar_dkv_part1_ready;          // WG3通知WG2 dKV_part1计算完成
    transac_bar_t bar_dkv_part2_ready;          // WG3通知WG2 dKV_part2计算完成
    transac_bar_t bar_dkv_part0_done;           // WG2通知WG3 dKV_part0传输完成
    transac_bar_t bar_dkv_part1_done;           // WG2通知WG3 dKV_part1传输完成
    transac_bar_t bar_dkv_part2_done;           // WG2通知WG3 dKV_part2传输完成
    // WG1-WG3 同步屏障 (kv_peer cp_async)
    transac_bar_t bar_kv_peer_cp_async;         // cp_async传输kv_peer的transaction barrier
    transac_bar_t bar_kv_peer_ready;            // WG1通知WG3 kv_peer加载完成
    // WG3-WG0 同步屏障 (dQ computation)
    transac_bar_t bar_dq_ready;                 // WG3通知WG0 dQ计算完成

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
template<typename TmaParamsType>
__global__ __launch_bounds__(test_operator::mla_bwd::NUM_THREADS, 1) void test_mla_bwd_kernel(
    const test_operator::mla_bwd::bf16* __restrict__ q,      // [s_q, B_H, D_Q]
    const test_operator::mla_bwd::bf16* __restrict__ kv,      // [s_kv, D_K]
    const test_operator::mla_bwd::bf16* __restrict__ dO,     // [s_q, B_H, D_V]
    const float* __restrict__ lse,     // [s_q, B_H] (log-sum-exp for softmax)
    const test_operator::mla_bwd::bf16* __restrict__ O,     // [s_q, B_H, D_V] (forward output O)
    const int32_t* __restrict__ gIndices,  // [s_q, topk_length]
    int s_kv,                        // KV sequence length
    int topk_length,                 // TopK length
    int s_q,                         // Query sequence length
    const float* __restrict__ delta,  // [s_q, B_H] (delta = sum(O * dO))
    float* __restrict__ dKV,          // [s_kv, D_K]
    test_operator::mla_bwd::bf16* __restrict__ dQ,  // [s_q, B_H, D_Q] (dQ gradient, bf16)
    __grid_constant__ const TmaParamsType tma_params
);

// C++ wrapper declaration
void launch_test_mla_bwd(
    const test_operator::mla_bwd::bf16* q,
    const test_operator::mla_bwd::bf16* kv,
    const test_operator::mla_bwd::bf16* dO,
    const float* lse,
    const test_operator::mla_bwd::bf16* O,
    const int32_t* gIndices,
    int s_kv,
    int topk_length,
    int s_q,
    const float* delta,
    float* dKV,
    test_operator::mla_bwd::bf16* dQ,
    cudaStream_t stream = nullptr
);

# SM100 资源占用分析 (Head64)

基于 `FlashMLA/csrc/sm100/prefill/sparse/fwd/head64/config.h` 的代码分析以及 `SM100.md` 的硬件规格，以下是关于 FlashMLA 在 SM100 架构上 Prefill 阶段 (head dim 64) 的 Shared Memory 和 TMEM 资源占用分析。

关键配置：`D_Q = 576`, `D_K = 576`, `D_V = 512`, `B_H = 64`, `B_TOPK = 64`.

## 1. TMEM (Tensor Memory) 占用分析

SM100 的 TMEM 总容量为 256 KB，结构为 128 行 × 512 列 × 4 字节。FlashMLA 通过 `tmem_cols` 命名空间对 TMEM 列进行了划分。

### 列分配情况
代码定义如下 (`config.h`):
```cpp
namespace tmem_cols {
    //   0 ~ 256: output
    // 256 ~ 400: Q
    // 400 ~ 464: P
    constexpr int O = 0;
    constexpr int Q = 256;
    constexpr int Q_RoPE = 256 + 128;
    constexpr int P = 400;
}
```

| 变量 | 描述 | 数据类型 | Shape | 起始列 | 结束列 | 占用列数 | Tensor 大小 (KB) | 占用大小 (KB) | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `O` | Output (O) | fp32 (accum) | [64, 512] | 0 | 256 | **256** | 128 | 128 | 用于累加输出结果 |
| `Q` | Query (Q) | bf16 | [64, D_Q] | 256 | 400 | **144** | 72 | 72 | 包含 NoPE (128列) 和 RoPE (16列) |
| `P` | P (Logits) | fp32 | [64, 64] | 400 | 464 | **64** | 16 | 32 | 用于存储 Attention Scores |
| **总计** | | | | | | **464** | **216** | **232** | |

*注：Q 部分占用 144 列。其中 Q_NoPE 为 256~384 (128列)，Q_RoPE 为 384~400 (16列)。*

### 结论
*   **TMEM 使用量**: 232 KB (464 列)
*   **TMEM 总容量**: 256 KB (512 列)
*   **使用率**: **90.6%**

相比 Head128 配置 (100% 使用率)，Head64 配置在 TMEM 上略有盈余 (48 列空闲)。

---

## 2. Shared Memory (共享内存) 占用分析

SM100 每 SM 可配置的最大共享内存为 227 KB。FlashMLA 使用 `SharedMemoryPlan` 结构体管理共享内存。

### 内存布局计算
`SharedMemoryPlan` 包含一个主要的 `union` 和其他固定成员。

#### A. Union 部分 (取最大值)
Union 包含 `q_full`, `k`, `o` 三种主要状态。其中 `k` 和 `q_full` 占用最大。

1.  **状态: `k` (Key Cache 加载与计算)**
    *   包含 3 个 buffer 的 `k_nope` 和 1 个 `k_rope`。
    *   `k_nope` (SmemLayoutKNoPE): `B_TOPK` x `64*8` = 64 x 512 bf16。
        *   大小: `64 * 512 * 2` = **65,536 bytes** (64 KB)
        *   3 个 buffer: `3 * 65,536` = **196,608 bytes**
    *   `k_rope` (SmemLayoutKRoPE): `B_TOPK` x 64 = 64 x 64 bf16。
        *   大小: `64 * 64 * 2` = **8,192 bytes** (8 KB)
    *   **Union Total**: `196,608 + 8,192` = **204,800 bytes** (200 KB)

2.  **状态: `q_full`**
    *   包含 `_k_rope_pad`, `_k_pad[2]`, `q_nope`。
    *   设计上利用了 `k` 的内存空间，`q_nope` (64KB) 复用了 `k_nope[2]` (64KB) 的位置。
    *   总大小与 `k` 状态一致，为 **204,800 bytes**。

#### B. 固定成员部分
除了 Union 外，还有必须始终存在的成员：

1.  `p_exchange_buf`: `4 * 32 * (B_TOPK/2)` floats = `4 * 32 * 32 * 4` = **16,384 bytes** (16 KB)
2.  `s_q_rope` (Union): Max(`s`, `q_rope`)
    *   `s`: `B_H * B_TOPK` = 64 * 64 bf16 = 8,192 bytes
    *   `q_rope`: 64 * 64 bf16 = 8,192 bytes
    *   占用: **8,192 bytes** (8 KB)
3.  Barriers: 约 26 个 `transac_bar_t`，26 * 8 = **208 bytes**
4.  其他 Buffers (`tmem_start_addr`, `rowwise_*`): ~1 KB (**1,028 bytes**)

**固定成员总计**: ≈ **25,812 bytes** (25.2 KB)

### 总占用量与使用率

| 组件 | 大小 (Bytes) | 大小 (KB) |
| :--- | :--- | :--- |
| Union (Max) | 204,800 | 200.0 |
| 固定成员 | 25,812 | 25.2 |
| **总计** | **230,612** | **225.2** |

*   **SM100 共享内存上限**: 227 KB (232,448 bytes)
*   **预估使用量**: ~225.2 KB
*   **使用率**: **~99.2%**

### 结论
共享内存使用率极高 (**99.2%**)，已逼近硬件物理极限 (仅剩不足 2KB)。这表明 Head64 优化的核心策略是**用满 Shared Memory 以支持 3-stage pipeline (NUM_BUFS=3)**，从而最大化掩盖内存延迟。这也是为什么 Head64 配置能够实现极高性能的原因之一——它在硬件资源允许的边缘疯狂试探。

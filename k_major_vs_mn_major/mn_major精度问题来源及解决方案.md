# MN-major 精度问题来源及解决方案

## 1. 问题现象

在 `k_major_vs_mn_major` 中，原始版本表现为：

1. `dV_cuda_k` 与 `torch_ref` 对齐。
2. `dV_cuda_mn` 与 `torch_ref` 明显不对齐（尤其是 `col128` 一侧）。
3. 输入侧 `sS / sdO_k / sdO_mn` 和直接 dot 结果一致，说明问题不在原始数据加载。

## 2. 本次实际修改了什么

修改文件：`/Users/chenql/Desktop/workspace/operator/test_operator/k_major_vs_mn_major/test_k_mn_major.cu`

### 2.1 增加了定位用打印（不改变数学公式）

1. 描述符级打印 `debug_utcmma_desc(...)`。
2. `MN` 分块对比打印：`flat_divide(..., Tile<...>)` vs `flat_divide(..., Shape<...>)`。
3. `part0` 后 TMEM 中间结果打印（phase debug）。

### 2.2 在 `part0 -> part1(accumulate)` 之间增加了明确阶段同步

核心变更是把两次 `utcmma_ss` 之间加了阶段边界：

```cpp
ku::utcmma_ss(..., true);   // part0, clear
ku::tcgen05_after_thread_sync();
__syncthreads();
// (phase debug readback + tcgen05_before_thread_sync)
ku::utcmma_ss(..., false);  // part1, accumulate
```

这部分是精度恢复的关键。

### 2.3 新增了 A/B 切换开关（用于验证，不是默认修复路径）

```cpp
#define K_MN_MAJOR_MN_USE_SHAPE_DIVIDE 0
```

默认仍走 `Tile` 路径；日志中 `Tile` 和 `Shape` 描述符一致，说明不是分块 API 选择导致。

## 3. 根因归纳

根因不是输入布局错误，而是 **MN-major 两阶段累加的时序/同步边界不充分**：

1. 输入和分块值正确：`split dot`、`full/tile/shape` 全一致。
2. 描述符正确：`MN-major part0/part1 TILE` 与 `SHAPE` 的 A/B desc 完全一致。
3. 问题出在 MMA 执行阶段：原实现中 `part0(clear)` 紧接 `part1(acc)`，缺少稳定的阶段收敛边界，导致 MN-major 路径累加出现不稳定结果。

## 4. 分析流程（可复用）

1. 先确认问题范围：`dV_cuda_k` 正常、`dV_cuda_mn` 异常。
2. 验证输入：比较 `sS/sdO` 与 reference dot，排除加载/转置错误。
3. 验证分块：检查 `part0/part1` 的 `ref vs tile vs shape`。
4. 验证描述符：打印每个 k-loop 的 A/B descriptor，排除 desc 构造错误。
5. 分阶段定位：在 `part0` 后读 TMEM，确认误差出现阶段。
6. 引入阶段同步并回归：确认 `dV_cuda_mn` 与 `dV_cuda_k`、`torch_ref` 全对齐。

## 5. 解决方案

1. 对所有 `K-major x MN-major` 的“两段 MMA（clear + accumulate）”路径，强制加入阶段同步边界。
2. 保留 `Tile/Shape` 双路径和 descriptor 打印开关，作为后续回归定位工具。
3. 将同样修复模式应用到 `mla_bwd.cu` 中同构路径（你的复现关系已说明两者同源）。

## 6. 当前验证结果（来自 `bug.txt`）

`/Users/chenql/Desktop/workspace/operator/test_operator/k_major_vs_mn_major/bug.txt` 当前结果：

1. `dV_cuda_k vs torch_ref`: `max_abs = 7.629395e-06`
2. `dV_cuda_mn vs torch_ref`: `max_abs = 7.629395e-06`
3. `dV_cuda_k vs dV_cuda_mn`: `max_abs = 0`
4. `Result: PASS`

结论：MN-major 路径精度问题已对齐，问题属于阶段同步而非数据布局本身。

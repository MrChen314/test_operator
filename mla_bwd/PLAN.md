# MLA_BWD 要求5改造 + SM100 Skill 无GPU说明更新计划

## 摘要
本计划覆盖两件事：  
1. 按 `/Users/chenql/Desktop/workspace/operator/test_operator/mla_bwd/测试需求.txt` 的“要求5”重构 WG1/WG2/WG3 的 dQ 与 dKV 路径。  
2. 在 `/Users/chenql/.codex/skills/cuda-sm100-test-pitfalls/SKILL.md` 中加入“当前环境无GPU时的处理规则”。  

本地不做编译/跑测（当前环境无 `nvidia-smi`），只交付代码改动和服务器验证步骤。

## 变更范围与实现步骤

1. 更新 skill（无GPU约束）
- 修改文件：`/Users/chenql/.codex/skills/cuda-sm100-test-pitfalls/SKILL.md`
- 新增小节 `No-GPU Environment Handling`，明确：
- 若本地无 GPU（例如 `nvidia-smi` 不可用或 `torch.cuda.is_available() == False`），不得声称已本地验证。
- 允许执行静态检查与代码改造，但构建/精度测试改为“待服务器执行”。
- 必须在交付中附带服务器验证命令：`MAX_JOBS=192 python setup.py build_ext --inplace` 与 `python test_mla_bwd.py`。
- 在 `Verify Before Finalizing` 中改成条件化流程：有GPU走完整验证；无GPU走静态检查+远端验证清单。

2. 补齐并清理 `mla_bwd.cuh` 的类型与同步定义
- 修改文件：`/Users/chenql/Desktop/workspace/operator/test_operator/mla_bwd/mla_bwd.cuh`
- 新增 `SmemLayoutsdKV` 定义，逻辑形状固定为 `[B_TOPK, D_K/4] = [64, 144]`（float staging buffer）。
- 保留并继续使用 `plan.u.k_calc_dq`（`SmemLayoutKVTilesTransposed_KMajor`）作为 WG1->WG3 的 dQ 用 K 缓冲。
- 删除旧 peer 路径 barrier 字段：
- `bar_kv_peer_cp_async`
- `bar_kv_peer_ready`
- 新增 WG1->WG3 的分段就绪 barrier：
- `bar_kv_part0_ready`
- `bar_kv_part1_ready`
- `bar_kv_part2_ready`
- 在 `TmaParams` 中新增 `CUtensorMap tensor_map_kv_rope32`，用于 32 列 RoPE 分段 TMA gather。

3. 调整 kernel 初始化与 descriptor 预取
- 修改文件：`/Users/chenql/Desktop/workspace/operator/test_operator/mla_bwd/mla_bwd.cu`
- barrier 初始化改为：
- 初始化 `bar_kv_part0_ready/1/2_ready`
- 移除 `bar_kv_peer_cp_async/bar_kv_peer_ready` 初始化
- descriptor 预取增加 `tensor_map_kv_rope32`。
- 删除所有 `kv_peer` Tensor 构造与相关注释。

4. 重写 WG1：按要求5分 6 块 TMA 传输 K 到 `k_calc_dq`
- 触发时机改为：`WG3` 完成 dP 后，`WG1` 才开始该 `k_block` 的 K 传输（`bar_dp_ready.wait(phase)`）。
- 每个 `k_block` 依序传输并通知：
- `part0`（块1+块2）：  
  - cta0: 全局列 `[0, 128)` -> 本地 `k_calc_dq` 偏移 `[0, 128)`  
  - cta1: 全局列 `[128, 256)` -> 本地 `k_calc_dq` 偏移 `[0, 128)`  
  - arrive `bar_kv_part0_ready`
- `part1`（块3+块4）：  
  - cta0: 全局列 `[256, 384)` -> 本地偏移 `[128, 256)`  
  - cta1: 全局列 `[384, 512)` -> 本地偏移 `[128, 256)`  
  - arrive `bar_kv_part1_ready`
- `part2`（块5+块6，RoPE 32列）：  
  - cta0: 全局列 `[512, 544)` -> 本地偏移 `[256, 288)`  
  - cta1: 全局列 `[544, 576)` -> 本地偏移 `[256, 288)`  
  - 使用 `tensor_map_kv_rope32`  
  - arrive `bar_kv_part2_ready`
- 保留跨 `k_block` 顺序约束：`k_block>0` 时 `WG1` 等待上一块 `bar_dq_ready`，避免覆盖 `k_calc_dq`。

5. 重写 WG3 的 dQ 计算：去掉 peer copy，改为 2CTA dQ MMA
- 删除 `sK_peer_*` 相关 Tensor 与 `bar_kv_peer_ready.wait(...)`。
- dQ MMA 切换为：
- `TiledMMA_dQ_2cta` 计算 NoPE part0 / part1
- `TiledMMA_dQ_RoPE_2cta` 计算 RoPE part2
- 消费顺序：
- wait `bar_kv_part0_ready` -> 计算 `tdQ_part0`
- wait `bar_kv_part1_ready` -> 计算 `tdQ_part1`
- wait `bar_kv_part2_ready` -> 计算 `tdQ_RoPE`
- `k_block==0` 清零累加器，`k_block>0` 累加。
- 三段 dQ 完成后统一 `arrive bar_dq_ready`。

6. 重写 WG2 的 dKV 传输：TMEM -> sdkv -> red.global.add
- WG2 每个 part 的流程改成两阶段：
- 先从 TMEM 读到 `plan.u.q_kv.sdkv`
- 再从 `sdkv` 读寄存器，用 `red.global.add.v4.f32` 累加到全局 `dKV`
- 分段策略：
- `part0` 与 `part1`：各做 2 轮，每轮处理 `[64, 128]`
- `part2`：1 轮处理 `[64, 64]`
- 同一轮内按 CTA 拆半写回：
- cta0 处理该轮前半列
- cta1 处理该轮后半列
- 保持 WG3/WG2 协议不变：`bar_dkv_part{0,1,2}_ready` / `bar_dkv_part{0,1,2}_done`。

7. host 端 tensor map 与参数打包更新
- 在 `launch_test_mla_bwd` 中新增 `tensor_map_kv_rope32` 编码（`box_size[0]=32`）。
- `KernelTmaParams` 填充新字段。
- `cudaLaunchKernelEx`、cluster 配置、Python 接口不变。

8. 代码清理
- 移除 `kv_peer` 相关 dead code。
- 仅保留一种 float4 累加实现：`atomic_add_float4`（`red.global.add.v4.f32`）。
- 注释更新为“要求5”的分段语义，确保 WG1/WG2/WG3 注释与实现一致。

## 重要接口/类型变更
- Python 暴露接口不变：`mla_bwd(q, kv, dO, lse, O, indices) -> (dQ, dKV)`。
- C++ wrapper 签名不变。
- 内部类型变更：
- `TmaParams` 新增 `tensor_map_kv_rope32` 字段。
- `SharedMemoryPlan` barrier 字段替换（移除 `kv_peer` 相关、新增 `kv_part*_ready`）。
- 新增 `SmemLayoutsdKV`。

## 测试与验收场景

1. 本地静态检查（无GPU）
- `rg -n "kv_peer|bar_kv_peer" /Users/chenql/Desktop/workspace/operator/test_operator/mla_bwd/mla_bwd.cu /Users/chenql/Desktop/workspace/operator/test_operator/mla_bwd/mla_bwd.cuh` 应无残留逻辑引用。
- `rg -n "bar_kv_part0_ready|bar_kv_part1_ready|bar_kv_part2_ready" /Users/chenql/Desktop/workspace/operator/test_operator/mla_bwd/mla_bwd.cu /Users/chenql/Desktop/workspace/operator/test_operator/mla_bwd/mla_bwd.cuh` 应覆盖 `init/wait/arrive`。
- `rg -n "SmemLayoutsdKV" /Users/chenql/Desktop/workspace/operator/test_operator/mla_bwd/mla_bwd.cuh /Users/chenql/Desktop/workspace/operator/test_operator/mla_bwd/mla_bwd.cu` 应有定义和使用。

2. 服务器验证（你后续执行）
- 在 `/Users/chenql/Desktop/workspace/operator/test_operator/mla_bwd`：
- `MAX_JOBS=192 python setup.py build_ext --inplace`
- `python test_mla_bwd.py`
- 验收标准：
- 编译通过。
- `dQ` 与 `dKV` 精度阈值通过（沿用脚本阈值 `rel_diff < 1e-2`）。
- 不出现 cluster 配置报错（launch 仍走 `cudaLaunchKernelEx`）。

## 假设与默认决策
- 默认继续以 `topk_length % 64 == 0` 为前提，不调整外部输入约束。
- 默认 `part2` 使用新增 32 列 tensor map，而不是 64 列加载后软件裁剪。
- 默认不改 `test_mla_bwd.py` 的数据规模与阈值，仅通过 kernel 逻辑修复 dKV 偏差。
- 默认 skill 更新只改 `SKILL.md` 主体，不扩展到其它 skill。

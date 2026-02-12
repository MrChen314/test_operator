### mla_bwd Kernel 重构方案（需求5，已锁定“每个 part 计算完即传输”）

#### 摘要
本次重构只改 kernel 路径，目标是把当前 `B_TOPK=32` 相关不一致全部收敛，并完成三条主线改造：
1. `dKV` 从 `atomicAdd` 切到 `cp.reduce.async.bulk`（TMA store reduce-add）+ `sdKV`。
2. `s/ds` 从“写 SMEM 给 WG3 读”改为“写 TMEM 给 WG3 做 TS MMA”。
3. `kv_peer` 改成“先从 peer SMEM 读到 RMEM，再回填本地 `kv` 对应半区”，并把 `dQ` 改成 TS 路径且仅 `cta0` 发起 MMA。

---

#### 需要修改的文件
1. `/Users/chenql/Desktop/workspace/operator/test_operator/mla_bwd/mla_bwd.cuh`
2. `/Users/chenql/Desktop/workspace/operator/test_operator/mla_bwd/mla_bwd.cu`

---

#### 公共接口/类型变更
1. Python 接口不变：`mla_bwd(q, kv, dO, lse, O, indices) -> (dQ, dKV)`。
2. Host launch 参数不变（`launch_test_mla_bwd` 入参保持一致）。
3. 仅内部实现调整：
   1. 新增/补齐内部 layout 与 helper（例如 `SmemLayoutdKV`、TMEM 存取辅助）。
   2. 删除 `atomic_addx4_tilelang` 路径，改用 `cp.reduce.async.bulk ... add.f32`。
   3. 不改你已给定的 TS TiledMMA 定义与现有 tmem/smem 主布局（按需求5.5）。

---

#### 详细实现步骤（决策已固定）

1. 全局收敛 `B_TOPK=32`
1. 清理 `/Users/chenql/Desktop/workspace/operator/test_operator/mla_bwd/mla_bwd.cu` 中所有遗留 `64` 假设：
   1. `static_assert(B_TOPK == 64)` 改为 32 逻辑。
   2. `is_k_valid` 装载 lane 数改为 `B_TOPK/8` 口径。
   3. `topk_length` 报错文案改为 `B_TOPK=32`。
2. 所有注释和索引步长统一按 `B_TOPK=32`，不再出现“按64分块”的旧描述。

2. 修复当前 `.cuh/.cu` 结构不一致（先让代码结构自洽）
1. `cu` 中对 `plan.u.q_kv.k_nope/k_rope/kv_peer`、`plan.s_ds.*` 的访问全部替换为与当前 `.cuh` 一致的视图构造。
2. 在 `/Users/chenql/Desktop/workspace/operator/test_operator/mla_bwd/mla_bwd.cuh` 补齐缺失但必要的附加 layout（仅新增，不改已有布局定义）。
3. 加 `static_assert(sizeof(SharedMemoryPlan) <= cutlass::arch::sm100_smem_capacity_bytes)`，避免后续越界。

3. WG0：`s/ds` 直接落 TMEM（替代落 SMEM）
1. 保留现有 `P/dP` 从 TMEM 读取与 softmax/ds 计算逻辑。
2. 计算出的 `s`、`ds` 转为 bf16 打包后，用 `ku::tmem_st_32dp32bNx<>` 写入 `tmem_cols::S`、`tmem_cols::dS`。
3. `bar_s_ready` / `bar_ds_ready` 继续保留，但语义改成“TMEM 中 S/dS 已可被 WG3 消费”。

4. WG1：`kv_peer` 改为 peer SMEM -> RMEM -> 本地 KV 回填
1. 不再做 `cp_async` 把 peer KV 放到本 CTA 的独立 SMEM 区。
2. 使用 `kerutils::get_peer_addr(...)` 构造 peer 视图，直接把需要的 288 维读到 RMEM：
   1. `cta0` 读 peer 的前 288 维。
   2. `cta1` 读 peer 的后 288 维。
3. 在 `dP` MMA 完成后执行回填：
   1. `cta0` 将 RMEM 写回本地 `kv` 的后 288 维。
   2. `cta1` 将 RMEM 写回本地 `kv` 的前 288 维。
4. 回填完成后通知 WG3（沿用 `bar_kv_peer_ready`，仅语义变更）。

5. WG3：切换到 TS MMA 路径并重排 dQ 计算
1. `dKV` 计算改为 TS：
   1. A 操作数从 `tmem_cols::S / tmem_cols::dS` 取 TMEM fragment。
   2. B 操作数仍取 SMEM 的 `dO^T/Q^T`。
   3. 调用从 `ku::utcmma_ss(...)` 改为 `ku::utcmma_ts(...)`。
2. `dQ` 计算改为 TS 且只由 `cta0` 发起（去掉当前 `cta0/cta1` 分支双路径）：
   1. 以 `ds:[64,32]` 与重排后的 `k:[32,288]` 做三次 MMA。
   2. N 维按 `128 + 128 + 32` 拆分（满足 `SM100_MMA_F16BF16_2x1SM_TS_NOELECT` 限制）。
3. 保持 `dQ` 累加与最终 TMEM->SMEM->GMEM 的 epilogue 流程不变。

6. WG2：`dKV` 传输改为 `sdKV + peer merge + reduce-add`
1. 保留“每个 part 计算完立即传输”的节奏（已确认采用此方案）。
2. 每个 `k_block` 的每个 part 流程：
   1. 从 TMEM 读出当前 part 到本 CTA `sdKV` 对应片段。
   2. 奇偶交替 owner：
      1. `k` 奇数：`cta0` 作为 owner，读 peer 的 `sdKV` 并累加到本地。
      2. `k` 偶数：`cta1` 作为 owner，读 peer 的 `sdKV` 并累加到本地。
   3. owner 按有效 `indices` 行发起 `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32` 到 `dKV` 全局内存对应行列范围。
   4. `tma_store_arrive + tma_store_wait<0>` 后再发 `bar_dkv_partX_done`，保证 WG3 复用缓冲前数据已落地。
3. 删除 `atomicAdd` 路径及对齐假设（保留必要的 16B 对齐检查即可）。

7. Host 侧与校验点同步
1. `TORCH_CHECK(topk_length % B_TOPK == 0, ...)` 文案改成 `B_TOPK=32`。
2. 删除与 `atomicAdd(float4)` 绑定的错误提示文案，替换为 reduce-add 对齐/可用性提示。

---

#### 测试与验收场景（服务器执行）
1. 编译验收：
   1. `python setup.py build_ext --inplace` 成功。
   2. 无 `plan.s_ds`、`kv_peer` 旧字段引用错误。
2. 功能精度：
   1. 跑 `/Users/chenql/Desktop/workspace/operator/test_operator/mla_bwd/test_mla_bwd.py`，`dQ/dKV` 相对误差与当前门限一致或更好。
3. 覆盖场景：
   1. `topk_length=32`（单 block）。
   2. `topk_length=64`（两 block，覆盖奇偶 owner 交替）。
   3. `topk_length=2048`（长序列压力）。
4. 性能回归：
   1. 重点看 `dKV` 路径耗时，确认不再受 atomicAdd 明显拖慢。

---

#### 假设与默认选择
1. 默认架构是 `sm_100a`，可用 `cp.reduce.async.bulk ... add.f32`。
2. `indices` 仍允许重复，`dKV` 需要 reduce-add 语义累加。
3. 本次不改外部 API，不改测试脚本逻辑结构。
4. 按你要求，`dKV` 采用“每个 part 计算完即传输”方案，而不是“三个 part 算完再统一传”。

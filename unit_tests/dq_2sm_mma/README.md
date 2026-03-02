# DQ 2SM MMA 单元测试

## 测试目标

根据 `phase1.cuh` 中 WG1 传输数据到 peer 的 smem 以及 WG3 通过 mma 计算 dq 的逻辑进行单测。

## 测试流程

1. **输入输出参数**
   - 输入：ds [128, 64], kv [s_kv, 576], indices [64]
   - 输出：dQ [128, 576], cuda_kv [64, 576] (用于验证)

2. **Kernel 流程**
   - 2.1 读取 ds 到 smem.ds_t (转置存储)
   - 2.2 使用 TMA gather4 加载 kv
     - CTA0 和 CTA1 各加载 [32, 576] 的 kv 数据
     - 交换 CTA0 的 k_nope[32, 256:512] 与 CTA1 的 k_nope[32, 0:256]
     - 每个线程处理 64 个 bf16 元素 (32 * 256 / 128 = 64)
     - 交换完成后写入各自的 smem
   - 2.3 将交换完成的 k_smem 输出到全局内存 cuda_kv，用于精度验证
   - 2.4 计算 dQ = dS^T @ K

3. **精度验证**
   - 验证 cuda_kv 与 ref_kv 的精度
   - 验证 dQ 与 ref_dQ 的精度

## CTA 数据交换与映射关系详解

### 交换前的数据分布

**CTA0 加载的数据** (通过 TMA gather4 从 global kv 加载):
- indices[0:32] 对应的 kv 行
- 存储在 CTA0 的 smem: [32, 576]
  - k_nope: [32, 512]
  - k_rope: [32, 64]

**CTA1 加载的数据** (通过 TMA gather4 从 global kv 加载):
- indices[32:64] 对应的 kv 行
- 存储在 CTA1 的 smem: [32, 576]
  - k_nope: [32, 512]
  - k_rope: [32, 64]

### 交换操作

**交换的数据**:
- CTA0 的 k_nope[32, 256:512] ↔ CTA1 的 k_nope[32, 0:256]
- 交换数据量: 32 * 256 = 8192 个 bf16 元素
- 每个线程处理: 8192 / 128 = 64 个 bf16 元素

**交换步骤**:
1. 每个线程从 peer CTA 的 smem 读取 64 个元素到 exchange_buffer
   - CTA0 读取 peer 的 [32, 0:256]
   - CTA1 读取 peer 的 [32, 256:512]
2. cluster_sync() 确保所有线程读取完成
3. 每个线程将 exchange_buffer 写入本地 smem
   - CTA0 写入 [32, 256:512]
   - CTA1 写入 [32, 0:256]

### 交换后的数据分布

**CTA0 的 smem.k_nope [32, 512]**:
- [32, 0:256]: 原始加载的数据 (indices[0:32] 的 kv[:, 0:256])
- [32, 256:512]: 从 CTA1 交换来的数据 (indices[32:64] 的 kv[:, 0:256])

**CTA1 的 smem.k_nope [32, 512]**:
- [32, 0:256]: 从 CTA0 交换来的数据 (indices[0:32] 的 kv[:, 256:512])
- [32, 256:512]: 原始加载的数据 (indices[32:64] 的 kv[:, 256:512])

**k_rope 不交换**:
- CTA0: [32, 64] (indices[0:32] 的 kv[:, 512:576])
- CTA1: [32, 64] (indices[32:64] 的 kv[:, 512:576])

### 输出到 Global Memory 的映射关系

输出的 cuda_kv 形状: [64, 576] = [64, 512 + 64]

**k_nope 部分 [64, 512]**:

CTA0 输出映射:
```
smem[0:32, 0:256]   → global[0:32,  0:256]   (原始数据)
smem[0:32, 256:512] → global[32:64, 0:256]   (交换来的数据)
```

CTA1 输出映射:
```
smem[0:32, 0:256]   → global[0:32,  256:512] (交换来的数据)
smem[0:32, 256:512] → global[32:64, 256:512] (原始数据)
```

**k_rope 部分 [64, 64]**:

```
CTA0: smem[0:32, 0:64] → global[0:32,  0:64]
CTA1: smem[0:32, 0:64] → global[32:64, 0:64]
```

### 输出逻辑伪代码

```cpp
// CTA0 和 CTA1 各自处理
for each element in local smem[32, 512]:
    local_row = element_idx / 512  // 0-31
    local_col = element_idx % 512  // 0-511

    if (local_col < 256):
        // 前半列 → 输出到前 32 行
        global_row = local_row
        global_col = cta_idx * 256 + local_col
    else:
        // 后半列 → 输出到后 32 行
        global_row = 32 + local_row
        global_col = cta_idx * 256 + (local_col - 256)

    cuda_kv[global_row, global_col] = smem[local_row, local_col]
```

### 为什么需要这样交换？

这种交换模式是为了让每个 CTA 在计算 dQ 时能够访问到完整的列范围:
- CTA0 负责计算 dQ[:, 0:256]，需要 K 的所有行的 [0:256] 列
- CTA1 负责计算 dQ[:, 256:512]，需要 K 的所有行的 [256:512] 列

交换后，每个 CTA 的 smem 包含了它计算所需的完整数据，无需在计算过程中再访问 peer smem。

## 构建和运行

```bash
# 构建
cd test_operator/unit_tests/dq_2sm_mma
python setup.py build_ext --inplace

# 运行测试
python test_dq_2sm_mma.py
```

## 注意事项

- 需要 SM100 架构的 GPU (Blackwell)
- 需要 CUDA 12.8+ 和对应的 CUTLASS 库
- 测试使用 2 个 CTA 组成的 cluster，每个 CTA 有 128 个线程

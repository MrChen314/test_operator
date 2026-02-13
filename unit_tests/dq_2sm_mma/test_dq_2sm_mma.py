#!/usr/bin/env python3
import torch

import dq_2sm_mma_cuda


B_H = 128
B_TOPK = 64
D_Q = 576


def calc_diff(a: torch.Tensor, b: torch.Tensor):
    abs_diff = torch.abs(a - b)
    max_diff = abs_diff.max().item()
    rel_diff = (abs_diff / (1e-4 + torch.abs(a))).mean().item()
    return max_diff, rel_diff


def run_case(case_name: str, ds: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor):
    k_sel = kv.index_select(0, indices.to(torch.int64)).float()
    dq_ref = torch.matmul(ds.float(), k_sel)
    dq_cuda = dq_2sm_mma_cuda.run_dq_2sm_mma(ds, kv, indices)
    torch.cuda.synchronize()

    max_diff, rel_diff = calc_diff(dq_cuda, dq_ref)
    print(f"[{case_name}] max_diff={max_diff:.6e}, rel_diff={rel_diff:.6e}")
    return rel_diff < 1e-3


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print("CUDA Device:", torch.cuda.get_device_name())
    print("CUDA Capability:", torch.cuda.get_device_capability())

    torch.manual_seed(42)
    s_kv = 256

    ds_rand = torch.randn(B_H, B_TOPK, device="cuda", dtype=torch.bfloat16)
    kv_rand = torch.randn(s_kv, D_Q, device="cuda", dtype=torch.bfloat16)
    indices_rand = torch.randint(0, s_kv, (B_TOPK,), device="cuda", dtype=torch.int32)

    row_id = torch.arange(s_kv, device="cuda", dtype=torch.float32).unsqueeze(1)
    col_id = torch.arange(D_Q, device="cuda", dtype=torch.float32).unsqueeze(0)
    kv_struct = (row_id * 0.01 + col_id * 0.001).to(torch.bfloat16).contiguous()
    ds_struct = torch.randn(B_H, B_TOPK, device="cuda", dtype=torch.bfloat16)
    indices_struct = torch.randperm(s_kv, device="cuda", dtype=torch.int64)[:B_TOPK].to(torch.int32).contiguous()

    ok_rand = run_case("random", ds_rand, kv_rand, indices_rand)
    ok_struct = run_case("structured_kv", ds_struct, kv_struct, indices_struct)
    ok = ok_rand and ok_struct

    print("Final:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


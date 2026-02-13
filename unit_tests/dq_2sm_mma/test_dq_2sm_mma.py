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

    seg0_max, seg0_rel = calc_diff(dq_cuda[:, :256], dq_ref[:, :256])
    seg1_max, seg1_rel = calc_diff(dq_cuda[:, 256:512], dq_ref[:, 256:512])
    seg2_max, seg2_rel = calc_diff(dq_cuda[:, 512:], dq_ref[:, 512:])
    print(
        f"[{case_name}] seg0(0:256) max={seg0_max:.6e} rel={seg0_rel:.6e} | "
        f"seg1(256:512) max={seg1_max:.6e} rel={seg1_rel:.6e} | "
        f"seg2(512:576) max={seg2_max:.6e} rel={seg2_rel:.6e}"
    )

    sample_cols = [0, 1, 2, 3, 256, 257, 512, 513]
    row0_cuda = dq_cuda[0, sample_cols].detach().cpu()
    row0_ref = dq_ref[0, sample_cols].detach().cpu()
    row64_cuda = dq_cuda[64, sample_cols].detach().cpu()
    row64_ref = dq_ref[64, sample_cols].detach().cpu()
    print(f"[{case_name}] indices[0:8]={indices[:8].detach().cpu().tolist()}")
    print(f"[{case_name}] row0 cuda={row0_cuda.tolist()}")
    print(f"[{case_name}] row0 ref ={row0_ref.tolist()}")
    print(f"[{case_name}] row64 cuda={row64_cuda.tolist()}")
    print(f"[{case_name}] row64 ref ={row64_ref.tolist()}")

    row_diff = torch.max(torch.abs(dq_cuda - dq_ref), dim=1).values
    worst_rows = torch.topk(row_diff, k=4).indices.detach().cpu().tolist()
    print(f"[{case_name}] worst_rows={worst_rows} row_max_diff={[row_diff[i].item() for i in worst_rows]}")
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

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
    _, cuda_smem_k_nope, cuda_smem_k_rope = dq_2sm_mma_cuda.run_dq_2sm_mma(ds, kv, indices)
    torch.cuda.synchronize()

    k_sel = kv.index_select(0, indices.to(torch.int64))
    smem_k_nope_ref = k_sel[:, :256].contiguous()
    smem_k_rope_ref = k_sel[:, 512:544].contiguous()

    nope_max, nope_rel = calc_diff(cuda_smem_k_nope.float(), smem_k_nope_ref.float())
    rope_max, rope_rel = calc_diff(cuda_smem_k_rope.float(), smem_k_rope_ref.float())
    print(f"[{case_name}] smem_k_nope max_diff={nope_max:.6e}, rel_diff={nope_rel:.6e}")
    print(f"[{case_name}] smem_k_rope max_diff={rope_max:.6e}, rel_diff={rope_rel:.6e}")

    # 打印前32行和前32列
    print(f"[{case_name}] cuda_smem_k_nope[:32, :32]:")
    print(cuda_smem_k_nope[:32, :32])
    print(f"[{case_name}] smem_k_nope_ref[:32, :32]:")
    print(smem_k_nope_ref[:32, :32])

    sample_nope_cols = [0, 1, 2, 3, 127, 128, 254, 255]
    sample_rope_cols = [0, 1, 2, 3, 28, 29, 30, 31]
    row0_nope_cuda = cuda_smem_k_nope[0, sample_nope_cols].detach().cpu()
    row0_nope_ref = smem_k_nope_ref[0, sample_nope_cols].detach().cpu()
    row0_rope_cuda = cuda_smem_k_rope[0, sample_rope_cols].detach().cpu()
    row0_rope_ref = smem_k_rope_ref[0, sample_rope_cols].detach().cpu()

    print(f"[{case_name}] indices[0:8]={indices[:8].detach().cpu().tolist()}")
    print(f"[{case_name}] row0 smem_k_nope cuda={row0_nope_cuda.tolist()}")
    print(f"[{case_name}] row0 smem_k_nope ref ={row0_nope_ref.tolist()}")
    print(f"[{case_name}] row0 smem_k_rope cuda={row0_rope_cuda.tolist()}")
    print(f"[{case_name}] row0 smem_k_rope ref ={row0_rope_ref.tolist()}")

    nope_row_diff = torch.max(torch.abs(cuda_smem_k_nope.float() - smem_k_nope_ref.float()), dim=1).values
    rope_row_diff = torch.max(torch.abs(cuda_smem_k_rope.float() - smem_k_rope_ref.float()), dim=1).values
    worst_nope_rows = torch.topk(nope_row_diff, k=4).indices.detach().cpu().tolist()
    worst_rope_rows = torch.topk(rope_row_diff, k=4).indices.detach().cpu().tolist()
    print(
        f"[{case_name}] worst_nope_rows={worst_nope_rows} "
        f"row_max_diff={[nope_row_diff[i].item() for i in worst_nope_rows]}"
    )
    print(
        f"[{case_name}] worst_rope_rows={worst_rope_rows} "
        f"row_max_diff={[rope_row_diff[i].item() for i in worst_rope_rows]}"
    )

    return nope_rel < 1e-3 and rope_rel < 1e-3


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

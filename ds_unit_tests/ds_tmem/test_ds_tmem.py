#!/usr/bin/env python3
import torch

import ds_tmem_cuda


B_H = 128
B_TOPK = 32
D_QK = 576


def summarize(name: str, out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    diff = (out.float() - ref.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = (diff / (ref.float().abs() + 1e-8)).max().item()
    print(f"{name}: max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, max_rel={max_rel:.6e}")
    return max_abs, mean_abs, max_rel


def run_case(name: str, p: torch.Tensor, dp: torch.Tensor, lse: torch.Tensor, delta: torch.Tensor) -> bool:
    s_out, ds_out = ds_tmem_cuda.run_ds_tmem(p, dp, lse, delta)
    torch.cuda.synchronize()

    sm_scale = 1.0 / (D_QK ** 0.5)
    scale_log2e = sm_scale * 1.4426950408889634

    s_ref = torch.exp2(p * scale_log2e - lse.unsqueeze(-1)).to(torch.bfloat16)
    ds_ref = (s_ref.float() * (dp - delta.unsqueeze(-1)) * sm_scale).to(torch.bfloat16)

    print(f"\n[{name}]")
    s_stats = summarize("s_out vs s_ref", s_out, s_ref)
    ds_stats = summarize("ds_out vs ds_ref", ds_out, ds_ref)

    max_abs_th = 2e-2
    max_rel_th = 1e-1

    ok = (
        s_stats[0] <= max_abs_th and s_stats[2] <= max_rel_th and
        ds_stats[0] <= max_abs_th and ds_stats[2] <= max_rel_th
    )
    print("PASS" if ok else "FAIL")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print("CUDA Device:", torch.cuda.get_device_name())
    print("CUDA Capability:", torch.cuda.get_device_capability())

    torch.manual_seed(42)
    p = torch.randn(B_H, B_TOPK, device="cuda", dtype=torch.float32)
    dp = torch.randn(B_H, B_TOPK, device="cuda", dtype=torch.float32)
    lse = torch.randn(B_H, device="cuda", dtype=torch.float32)
    delta = torch.randn(B_H, device="cuda", dtype=torch.float32)

    ok_random = run_case("random_normal", p, dp, lse, delta)

    p_ones = torch.ones(B_H, B_TOPK, device="cuda", dtype=torch.float32)
    dp_ones = torch.ones(B_H, B_TOPK, device="cuda", dtype=torch.float32)
    lse_ones = torch.ones(B_H, device="cuda", dtype=torch.float32)
    delta_ones = torch.ones(B_H, device="cuda", dtype=torch.float32)

    ok_ones = run_case("all_ones", p_ones, dp_ones, lse_ones, delta_ones)

    all_ok = ok_random and ok_ones
    print("\nFinal:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import torch

import dq_mma_cuda


B_H = 128
B_TOPK = 32
D_ROPE = 64


def summarize(name: str, out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    diff = (out - ref).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = (diff / (ref.abs() + 1e-8)).max().item()
    print(f"{name}: max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, max_rel={max_rel:.6e}")
    return max_abs, mean_abs, max_rel


def dq_ref(
    ds: torch.Tensor,
    k_nope_t_part0: torch.Tensor,
    k_nope_t_part1: torch.Tensor,
    k_rope_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ds_f = ds.float()
    part0 = torch.matmul(ds_f, k_nope_t_part0.float().transpose(0, 1))
    part1 = torch.matmul(ds_f, k_nope_t_part1.float().transpose(0, 1))
    rope = torch.matmul(ds_f, k_rope_t.float().transpose(0, 1))
    return part0, part1, rope


def run_case(name: str, ds: torch.Tensor, k0: torch.Tensor, k1: torch.Tensor, kr: torch.Tensor) -> bool:
    dq0_cuda, dq1_cuda, dqr_cuda = dq_mma_cuda.run_dq_mma(ds, k0, k1, kr)
    torch.cuda.synchronize()

    dq0_ref, dq1_ref, dqr_ref = dq_ref(ds, k0, k1, kr)

    print(f"\n[{name}]")
    s0 = summarize("dq_nope_part0", dq0_cuda, dq0_ref)
    s1 = summarize("dq_nope_part1", dq1_cuda, dq1_ref)
    sr = summarize("dq_rope", dqr_cuda, dqr_ref)

    ok = (
        s0[0] <= 2e-2 and s0[2] <= 1e-1 and
        s1[0] <= 2e-2 and s1[2] <= 1e-1 and
        sr[0] <= 2e-2 and sr[2] <= 1e-1
    )
    print("PASS" if ok else "FAIL")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print("CUDA Device:", torch.cuda.get_device_name())
    print("CUDA Capability:", torch.cuda.get_device_capability())

    torch.manual_seed(42)
    ds = torch.randn(B_H, B_TOPK, device="cuda", dtype=torch.bfloat16)
    k0 = torch.randn(256, B_TOPK, device="cuda", dtype=torch.bfloat16)
    k1 = torch.randn(256, B_TOPK, device="cuda", dtype=torch.bfloat16)
    kr = torch.randn(D_ROPE, B_TOPK, device="cuda", dtype=torch.bfloat16)

    ok_random = run_case("random_normal", ds, k0, k1, kr)

    ds_ones = torch.ones(B_H, B_TOPK, device="cuda", dtype=torch.bfloat16)
    k0_ones = torch.ones(256, B_TOPK, device="cuda", dtype=torch.bfloat16)
    k1_ones = torch.ones(256, B_TOPK, device="cuda", dtype=torch.bfloat16)
    kr_ones = torch.ones(D_ROPE, B_TOPK, device="cuda", dtype=torch.bfloat16)

    ok_ones = run_case("all_ones", ds_ones, k0_ones, k1_ones, kr_ones)

    all_ok = ok_random and ok_ones
    print("\nFinal:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

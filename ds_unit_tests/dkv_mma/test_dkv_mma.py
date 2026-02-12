#!/usr/bin/env python3
import torch

import dkv_mma_cuda


B_H = 128
B_TOPK = 32
D_V = 512
D_K = 576
D_ROPE = 64


def summarize(name: str, out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    diff = (out - ref).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = (diff / (ref.abs() + 1e-8)).max().item()
    print(f"{name}: max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, max_rel={max_rel:.6e}")
    return max_abs, mean_abs, max_rel


def dkv_ref(
    s: torch.Tensor,
    ds: torch.Tensor,
    dO_t: torch.Tensor,
    q_t: torch.Tensor,
    q_rope_t: torch.Tensor,
) -> torch.Tensor:
    s_t = s.float().transpose(0, 1)    # [32, 64]
    ds_t = ds.float().transpose(0, 1)  # [32, 64]

    dO_t = dO_t.float()                # [512, 64]
    q_t = q_t.float()                  # [512, 64]
    q_rope_t = q_rope_t.float()        # [64, 64]

    part0 = torch.matmul(s_t, dO_t[:256, :].transpose(0, 1)) + torch.matmul(ds_t, q_t[:256, :].transpose(0, 1))
    part1 = torch.matmul(s_t, dO_t[256:, :].transpose(0, 1)) + torch.matmul(ds_t, q_t[256:, :].transpose(0, 1))
    part2 = torch.matmul(ds_t, q_rope_t.transpose(0, 1))

    return torch.cat([part0, part1, part2], dim=1)


def run_case(name: str, s: torch.Tensor, ds: torch.Tensor, dO_t: torch.Tensor, q_t: torch.Tensor, q_rope_t: torch.Tensor) -> bool:
    dkv_cuda = dkv_mma_cuda.run_dkv_mma(s, ds, dO_t, q_t, q_rope_t)
    torch.cuda.synchronize()

    dkv_reference = dkv_ref(s, ds, dO_t, q_t, q_rope_t)

    print(f"\n[{name}]")
    stats = summarize("dkv_cuda vs dkv_ref", dkv_cuda, dkv_reference)

    ok = stats[0] <= 2e-2 and stats[2] <= 1e-1
    print("PASS" if ok else "FAIL")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print("CUDA Device:", torch.cuda.get_device_name())
    print("CUDA Capability:", torch.cuda.get_device_capability())

    torch.manual_seed(42)
    s = torch.randn(B_H // 2, B_TOPK, device="cuda", dtype=torch.bfloat16)
    ds = torch.randn(B_H // 2, B_TOPK, device="cuda", dtype=torch.bfloat16)
    dO_t = torch.randn(D_V, B_H // 2, device="cuda", dtype=torch.bfloat16)
    q_t = torch.randn(D_V, B_H // 2, device="cuda", dtype=torch.bfloat16)
    q_rope_t = torch.randn(D_ROPE, B_H // 2, device="cuda", dtype=torch.bfloat16)

    ok_random = run_case("random_normal", s, ds, dO_t, q_t, q_rope_t)

    s_ones = torch.ones(B_H // 2, B_TOPK, device="cuda", dtype=torch.bfloat16)
    ds_ones = torch.ones(B_H // 2, B_TOPK, device="cuda", dtype=torch.bfloat16)
    dO_t_ones = torch.ones(D_V, B_H // 2, device="cuda", dtype=torch.bfloat16)
    q_t_ones = torch.ones(D_V, B_H // 2, device="cuda", dtype=torch.bfloat16)
    q_rope_t_ones = torch.ones(D_ROPE, B_H // 2, device="cuda", dtype=torch.bfloat16)

    ok_ones = run_case("all_ones", s_ones, ds_ones, dO_t_ones, q_t_ones, q_rope_t_ones)

    all_ok = ok_random and ok_ones
    print("\nFinal:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Compare TiledMMA_dKV with two major modes:
1) A=K major, B=K major
2) A=K major, B=MN major
"""

import torch

import test_k_mn_major_cuda


def summarize_diff(name: str, out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    diff = (out - ref).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = (diff / (ref.abs() + 1e-8)).max().item()
    print(f"{name}:")
    print(f"  max_abs  = {max_abs:.6e}")
    print(f"  mean_abs = {mean_abs:.6e}")
    print(f"  max_rel  = {max_rel:.6e}")
    return max_abs, mean_abs, max_rel


def main() -> int:
    torch.manual_seed(42)

    # Required by spec
    s_shape = (128, 64)
    dO_shape = (128, 256)

    s_bf16 = torch.randn(*s_shape, dtype=torch.bfloat16, device="cuda")
    dO = torch.randn(*dO_shape, dtype=torch.bfloat16, device="cuda")

    # Reference: dV = s^T @ dO, shape [64, 256]
    dV_ref = torch.matmul(s_bf16.float().transpose(0, 1), dO.float())

    dV_cuda_k, dV_cuda_mn = test_k_mn_major_cuda.k_mn_major(s_bf16.contiguous(), dO.contiguous())
    torch.cuda.synchronize()

    print("=" * 64)
    print("k_major_vs_mn_major precision check")
    print("=" * 64)
    print(f"s_bf16 shape: {tuple(s_bf16.shape)}")
    print(f"dO shape:     {tuple(dO.shape)}")
    print(f"dV_ref shape: {tuple(dV_ref.shape)}")
    print()

    k_max_abs, _, k_max_rel = summarize_diff("dV_cuda_k vs torch_ref", dV_cuda_k, dV_ref)
    print()
    mn_max_abs, _, mn_max_rel = summarize_diff("dV_cuda_mn vs torch_ref", dV_cuda_mn, dV_ref)
    print()

    kmn_gap = (dV_cuda_k - dV_cuda_mn).abs()
    print("dV_cuda_k vs dV_cuda_mn:")
    print(f"  max_abs  = {kmn_gap.max().item():.6e}")
    print(f"  mean_abs = {kmn_gap.mean().item():.6e}")
    print()

    # Loose threshold for BF16 input + FP32 accumulation
    max_abs_threshold = 2e-1
    max_rel_threshold = 2e-1

    k_ok = (k_max_abs < max_abs_threshold) and (k_max_rel < max_rel_threshold)
    mn_ok = (mn_max_abs < max_abs_threshold) and (mn_max_rel < max_rel_threshold)
    ok = k_ok and mn_ok

    print("Result:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this test")
    raise SystemExit(main())

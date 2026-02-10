#!/usr/bin/env python3
"""
Test script for 2CTA SMEM cp.async exchange + 2-stage MMA.
"""

import torch

import test_smem_cp_async_cuda


M, K, N = 128, 64, 256
MAX_ABS_THRESHOLD = 1e-2
MAX_REL_THRESHOLD = 1e-1


def error_stats(out: torch.Tensor, ref: torch.Tensor):
    diff = (out - ref).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = (diff / (ref.abs() + 1e-8)).max().item()
    return max_abs, mean_abs, max_rel


def run_case(name: str, A: torch.Tensor, B: torch.Tensor) -> bool:
    print("\n" + "=" * 72)
    print(f"Case: {name}")
    print("=" * 72)

    C_ref = torch.matmul(A.float(), B.float())
    C_cuda = test_smem_cp_async_cuda.smem_cp_async(A, B)
    torch.cuda.synchronize()

    max_abs, mean_abs, max_rel = error_stats(C_cuda, C_ref)

    print(f"C_cuda vs C_ref:")
    print(f"  max_abs  = {max_abs:.6e}")
    print(f"  mean_abs = {mean_abs:.6e}")
    print(f"  max_rel  = {max_rel:.6e}")

    ok = (max_abs < MAX_ABS_THRESHOLD) and (max_rel < MAX_REL_THRESHOLD)
    print(f"Case result: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is required.")
        return 1

    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
    print(f"Shapes: A=[{M},{K}], B=[{K},{N}], C=[{M},{N}]")
    print(f"Thresholds: max_abs<{MAX_ABS_THRESHOLD}, max_rel<{MAX_REL_THRESHOLD}")

    torch.manual_seed(42)
    all_ok = True

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    all_ok = run_case("random_normal", A, B) and all_ok

    A1 = torch.ones(M, K, dtype=torch.bfloat16, device="cuda")
    B1 = torch.ones(K, N, dtype=torch.bfloat16, device="cuda")
    all_ok = run_case("all_ones", A1, B1) and all_ok

    A2 = (torch.rand(M, K, dtype=torch.bfloat16, device="cuda") * 2) - 1
    B2 = (torch.rand(K, N, dtype=torch.bfloat16, device="cuda") * 2) - 1
    all_ok = run_case("uniform_minus1_1", A2, B2) and all_ok

    print("\n" + "=" * 72)
    print("Overall:", "PASS" if all_ok else "FAIL")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

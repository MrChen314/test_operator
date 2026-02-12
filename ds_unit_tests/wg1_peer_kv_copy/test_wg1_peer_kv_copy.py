#!/usr/bin/env python3
import torch

import wg1_peer_kv_copy_cuda


B_TOPK = 32
D_K = 576
KV_ROWS = B_TOPK // 2


def build_reference(init_kv: torch.Tensor) -> torch.Tensor:
    ref = init_kv.clone()
    ref[0, :, D_K // 2 :] = init_kv[1, :, : D_K // 2]
    ref[1, :, : D_K // 2] = init_kv[0, :, D_K // 2 :]
    return ref


def summarize(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, int]:
    diff = (out.float() - ref.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    out_bits = out.contiguous().view(torch.int16)
    ref_bits = ref.contiguous().view(torch.int16)
    bit_mismatch = (out_bits != ref_bits).sum().item()

    print(f"max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, bit_mismatch={bit_mismatch}")
    return max_abs, mean_abs, bit_mismatch


def run_case(name: str, init_kv: torch.Tensor) -> bool:
    out_kv = wg1_peer_kv_copy_cuda.run_wg1_peer_kv_copy(init_kv)
    torch.cuda.synchronize()
    ref = build_reference(init_kv)

    print(f"\n[{name}]")
    max_abs, _, bit_mismatch = summarize(out_kv, ref)
    ok = (max_abs == 0.0) and (bit_mismatch == 0)
    print("PASS" if ok else "FAIL")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print("CUDA Device:", torch.cuda.get_device_name())
    print("CUDA Capability:", torch.cuda.get_device_capability())

    torch.manual_seed(42)

    init_random = torch.randn(2, KV_ROWS, D_K, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    init_ramp = (
        torch.arange(2 * KV_ROWS * D_K, device="cuda", dtype=torch.float32)
        .reshape(2, KV_ROWS, D_K)
        .div_(37.0)
        .sub_(100.0)
        .to(torch.bfloat16)
    )

    ok_random = run_case("random_bf16", init_random)
    ok_ramp = run_case("ramp_bf16", init_ramp)

    all_ok = ok_random and ok_ramp
    print("\nFinal:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

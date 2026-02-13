#!/usr/bin/env python3
import torch

import red_global_add_cuda


GLOBAL_SHAPE = (128, 256)
ADD_SHAPE = (64, 256)


def summarize(name: str, out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    diff = (out - ref).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = (diff / (ref.abs() + 1e-6)).max().item()
    print(f"{name}: max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, max_rel={max_rel:.6e}")
    return max_abs, mean_abs, max_rel


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print("CUDA Device:", torch.cuda.get_device_name())
    print("CUDA Capability:", torch.cuda.get_device_capability())

    torch.manual_seed(42)

    global_tensor = torch.randn(*GLOBAL_SHAPE, device="cuda", dtype=torch.float32)
    global_tensor1 = global_tensor.clone()
    global_tensor2 = global_tensor.clone()
    global_tensor3 = global_tensor.clone()

    add1 = torch.randn(*ADD_SHAPE, device="cuda", dtype=torch.float32)
    add2 = torch.randn(*ADD_SHAPE, device="cuda", dtype=torch.float32)
    indices = torch.randint(0, GLOBAL_SHAPE[0], (ADD_SHAPE[0],), device="cuda", dtype=torch.int32)

    torch_ref = global_tensor1.clone()
    torch_ref.index_add_(0, indices.to(torch.int64), add1)
    torch_ref.index_add_(0, indices.to(torch.int64), add2)

    out2, out3 = red_global_add_cuda.run_red_global_add(global_tensor2, global_tensor3, add1, add2, indices)
    torch.cuda.synchronize()

    print("\n[compare against torch_ref]")
    stats2 = summarize("global_tensor2 (cta0:add1, cta1:add2)", out2, torch_ref)
    stats3 = summarize("global_tensor3 (cta0:add1+add2)", out3, torch_ref)

    max_abs_th = 5e-5
    max_rel_th = 1e-4
    ok = (
        stats2[0] <= max_abs_th and stats2[2] <= max_rel_th and
        stats3[0] <= max_abs_th and stats3[2] <= max_rel_th
    )
    print("\nFinal:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

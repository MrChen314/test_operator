#!/usr/bin/env python3
import torch

import red_global_add_cuda


GLOBAL_SHAPE = (128, 256)
ADD_SHAPE = (64, 256)


def calc_diff(a: torch.Tensor, b: torch.Tensor):
    abs_diff = torch.abs(a - b)
    max_diff = abs_diff.max().item()
    rel_diff = (abs_diff / (1e-4 + torch.abs(a))).mean().item()
    return max_diff, rel_diff


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
    max_diff2, rel_diff2 = calc_diff(out2, torch_ref)
    max_diff3, rel_diff3 = calc_diff(out3, torch_ref)
    print(f"global_tensor2 (cta0:add1, cta1:add2): max_diff={max_diff2:.6e}, rel_diff={rel_diff2:.6e}")
    print(f"global_tensor3 (cta0:add1+add2): max_diff={max_diff3:.6e}, rel_diff={rel_diff3:.6e}")

    ok = (rel_diff2 < 1e-3) and (rel_diff3 < 1e-3)
    print("\nFinal:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

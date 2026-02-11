#!/usr/bin/env python3
"""
Test script for mla_bwd kernel.
Only validates dQ and dKV precision against PyTorch reference.
"""

import torch
import mla_bwd_cuda


def calc_diff(a: torch.Tensor, b: torch.Tensor):
    abs_diff = torch.abs(a - b)
    max_diff = abs_diff.max().item()
    rel_diff = (abs_diff / (1e-4 + torch.abs(a))).mean().item()
    return max_diff, rel_diff


def build_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    dO: torch.Tensor,
    lse: torch.Tensor,
    O: torch.Tensor,
    indices: torch.Tensor,
):
    B_H, D_Q = q.shape
    S_KV, D_K = kv.shape
    B_TOPK = indices.numel()
    D_V = dO.shape[1]
    D_ROPE = D_Q - D_V

    all_k = kv.float()
    all_v = kv[:, :D_V].float()
    safe_indices = indices.clamp(0, S_KV - 1).to(torch.long)
    valid_mask = (indices >= 0) & (indices < S_KV)
    k = all_k.index_select(0, safe_indices)
    v = all_v.index_select(0, safe_indices)

    # P and dP are intermediate values used to derive ds.
    P_ref = torch.matmul(q.float(), k.transpose(-2, -1))
    mask_expanded = valid_mask.unsqueeze(0).expand(B_H, B_TOPK)
    P_ref = torch.where(mask_expanded, P_ref, torch.tensor(float("-inf"), device="cuda", dtype=torch.float32))

    dP_ref = torch.matmul(dO.float(), v.transpose(-2, -1))
    delta_ref = (O.float() * dO.float()).sum(dim=-1)

    sm_scale = 1.0 / (D_Q ** 0.5)
    scale_log2e = sm_scale * 1.44269504
    s_ref = torch.exp2(P_ref * scale_log2e - lse.unsqueeze(-1))
    ds_ref = s_ref * (dP_ref - delta_ref.unsqueeze(-1)) * sm_scale
    ds_bf16 = ds_ref.to(torch.bfloat16)

    dQ_ref = torch.matmul(ds_bf16.float(), k)

    dV_ref = torch.matmul(s_ref.to(torch.bfloat16).float().T, dO.float())
    dK_nope_ref = torch.matmul(ds_bf16.float().T, q[:, :D_V].float())
    dK_rope_ref = torch.matmul(ds_bf16.float().T, q[:, D_V:].float())
    dKV_topk_ref = torch.cat([dV_ref + dK_nope_ref, dK_rope_ref], dim=-1)
    dKV_ref = torch.zeros((S_KV, D_K), device=kv.device, dtype=torch.float32)
    if valid_mask.all():
        dKV_ref.index_add_(0, safe_indices, dKV_topk_ref)
    else:
        valid_pos = valid_mask.nonzero(as_tuple=False).squeeze(-1)
        if valid_pos.numel() > 0:
            dKV_ref.index_add_(
                0,
                safe_indices.index_select(0, valid_pos),
                dKV_topk_ref.index_select(0, valid_pos),
            )

    assert dKV_ref.shape == (S_KV, D_K)
    assert dQ_ref.shape == (B_H, D_Q)
    assert dK_rope_ref.shape[1] == D_ROPE
    return dQ_ref, dKV_ref


def run_one_case(name: str, q: torch.Tensor, kv: torch.Tensor, dO: torch.Tensor, lse: torch.Tensor, O: torch.Tensor, indices: torch.Tensor):
    dQ_ref, dKV_ref = build_reference(q, kv, dO, lse, O, indices)
    dQ_cuda, dKV_cuda = mla_bwd_cuda.mla_bwd(q, kv, dO, lse, O, indices)
    torch.cuda.synchronize()

    dQ_max_diff, dQ_rel_diff = calc_diff(dQ_cuda, dQ_ref.bfloat16())
    dKV_max_diff, dKV_rel_diff = calc_diff(dKV_cuda, dKV_ref)

    print(f"{name}:")
    print(f"  dQ  max_diff={dQ_max_diff:.6e}, rel_diff={dQ_rel_diff:.6e}")
    print(f"  dKV max_diff={dKV_max_diff:.6e}, rel_diff={dKV_rel_diff:.6e}")
    return dQ_max_diff, dQ_rel_diff, dKV_max_diff, dKV_rel_diff


def test_mla_bwd():
    print("=" * 60)
    print("Testing mla_bwd kernel (dQ/dKV only)")
    print("=" * 60)

    B_H = 128
    D_Q = 576
    B_TOPK = 2 * 1024
    S_KV = 4 * 1024
    D_K = 576
    D_V = 512

    print(f"q shape: [{B_H}, {D_Q}]")
    print(f"kv shape: [{S_KV}, {D_K}]")
    print(f"dO shape: [{B_H}, {D_V}]")
    print(f"dQ shape: [{B_H}, {D_Q}]")
    print(f"dKV shape: [{S_KV}, {D_K}]")
    print(f"indices shape: [{B_TOPK}]")
    print()

    torch.manual_seed(42)
    q = torch.randn(B_H, D_Q, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(S_KV, D_K, dtype=torch.bfloat16, device="cuda")
    dO = torch.randn(B_H, D_V, dtype=torch.bfloat16, device="cuda")
    lse = torch.randn(B_H, dtype=torch.float32, device="cuda") * 2.0 + 5.0
    O = torch.randn(B_H, D_V, dtype=torch.bfloat16, device="cuda")

    indices = torch.randperm(S_KV, device="cuda")[:B_TOPK].to(torch.int32)

    dQ_max_diff, dQ_rel_diff, dKV_max_diff, dKV_rel_diff = run_one_case(
        "Random normal", q, kv, dO, lse, O, indices
    )

    rel_diff_threshold = 0.01
    ok_dQ = dQ_rel_diff < rel_diff_threshold
    ok_dKV = dKV_rel_diff < rel_diff_threshold

    print("=" * 60)
    print("Precision Validation (dQ/dKV)")
    print("=" * 60)
    print(f"{'PASS' if ok_dQ else 'FAIL'} dQ  rel_diff={dQ_rel_diff:.6e}, threshold={rel_diff_threshold:.2e}")
    print(f"{'PASS' if ok_dKV else 'FAIL'} dKV rel_diff={dKV_rel_diff:.6e}, threshold={rel_diff_threshold:.2e}")
    print(f"dQ  max_diff={dQ_max_diff:.6e}")
    print(f"dKV max_diff={dKV_max_diff:.6e}")
    return ok_dQ and ok_dKV


def test_different_inputs():
    print("\n" + "=" * 60)
    print("Testing different input patterns (dQ/dKV)")
    print("=" * 60)

    B_H = 128
    D_Q = 576
    B_TOPK = 2 * 1024
    S_KV = 4 * 1024
    D_K = 576
    D_V = 512

    test_cases = [
        ("All ones", torch.ones, torch.ones, torch.ones),
        ("All zeros", torch.zeros, torch.zeros, torch.zeros),
        (
            "Random uniform",
            lambda *args, **kwargs: torch.rand(*args, **kwargs) * 2 - 1,
            lambda *args, **kwargs: torch.rand(*args, **kwargs) * 2 - 1,
            lambda *args, **kwargs: torch.rand(*args, **kwargs) * 2 - 1,
        ),
    ]

    rel_diff_threshold = 0.01
    all_passed = True

    for name, q_gen, kv_gen, dO_gen in test_cases:
        try:
            q = q_gen(B_H, D_Q, dtype=torch.bfloat16, device="cuda")
            kv = kv_gen(S_KV, D_K, dtype=torch.bfloat16, device="cuda")
            dO = dO_gen(B_H, D_V, dtype=torch.bfloat16, device="cuda")
            lse = torch.randn(B_H, dtype=torch.float32, device="cuda") * 2.0 + 5.0
            O = torch.randn(B_H, D_V, dtype=torch.bfloat16, device="cuda")

            indices = torch.randperm(S_KV, device="cuda")[:B_TOPK].to(torch.int32)

            _, dQ_rel_diff, _, dKV_rel_diff = run_one_case(name, q, kv, dO, lse, O, indices)

            case_ok = dQ_rel_diff < rel_diff_threshold and dKV_rel_diff < rel_diff_threshold
            print(f"  threshold: rel_diff < {rel_diff_threshold:.2e}")
            if not case_ok:
                all_passed = False
        except Exception as e:
            print(f"FAIL {name}: ERROR - {e}")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print("CUDA Device:", torch.cuda.get_device_name())
    print("CUDA Capability:", torch.cuda.get_device_capability())
    print()

    passed = test_mla_bwd()
    if passed:
        passed = test_different_inputs()

    print("\n" + "=" * 60)
    print("All tests PASSED!" if passed else "Some tests FAILED!")
    print("=" * 60)
    raise SystemExit(0 if passed else 1)

#!/usr/bin/env python3

import math

import torch

import dq_phase_cuda


B_H = 128
D_QK = 576
D_V = 512
TOPK = 128


def calc_diff(a: torch.Tensor, b: torch.Tensor):
    abs_diff = torch.abs(a.float() - b.float())
    max_diff = abs_diff.max().item()
    rel_diff = (abs_diff / (1e-4 + torch.abs(b.float()))).mean().item()
    return max_diff, rel_diff


def build_inputs(s_q: int, s_kv: int, q_start_index_s: int):
    q = torch.randn(s_q, B_H, D_QK, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(s_kv, 1, D_QK, dtype=torch.bfloat16, device="cuda")
    dO = torch.randn(s_q, B_H, D_V, dtype=torch.bfloat16, device="cuda")

    indices = torch.full((s_q, 1, TOPK), s_kv, dtype=torch.int32, device="cuda")
    for t in range(s_q):
        max_valid = min(q_start_index_s + t + 1, s_kv)
        n_valid = min(max_valid, TOPK)
        if n_valid > 0:
            picked = torch.randperm(max_valid, device="cuda")[:n_valid].to(torch.int32)
            indices[t, 0, :n_valid] = picked

    return q, kv, dO, indices


def build_reference(q, kv, dO, indices, q_start_index_s: int):
    kv_2d = kv.squeeze(1).float()
    k_all = kv_2d
    v_all = kv_2d[:, :D_V]
    sm_scale = 1.0 / math.sqrt(D_QK)
    neg_inf = torch.tensor(float("-inf"), device=q.device, dtype=torch.float32)

    safe_indices = indices.squeeze(1).clamp(0, kv_2d.shape[0] - 1).to(torch.long)
    q_pos = q_start_index_s + torch.arange(q.shape[0], device=q.device, dtype=torch.int32).unsqueeze(1)
    valid_mask = (indices.squeeze(1) >= 0) & (indices.squeeze(1) < kv_2d.shape[0]) & (indices.squeeze(1) <= q_pos)

    gathered_k = k_all.index_select(0, safe_indices.reshape(-1)).view(q.shape[0], TOPK, D_QK)
    gathered_v = v_all.index_select(0, safe_indices.reshape(-1)).view(q.shape[0], TOPK, D_V)

    p = torch.einsum("shd,std->sht", q.float(), gathered_k)
    p_masked = torch.where(valid_mask.unsqueeze(1), p, neg_inf)

    logits = p_masked * sm_scale
    lse = torch.logsumexp(logits, dim=-1)
    s_ref = torch.exp(logits - lse.unsqueeze(-1))
    o = torch.einsum("sht,stv->shv", s_ref, gathered_v).to(torch.bfloat16)

    delta = (o.float() * dO.float()).sum(dim=-1)
    dp = torch.einsum("shv,stv->sht", dO.float(), gathered_v)
    ds_ref = s_ref * (dp - delta.unsqueeze(-1)) * sm_scale
    dQ_ref = torch.einsum("sht,std->shd", ds_ref.to(torch.bfloat16).float(), gathered_k)

    return o, lse.contiguous(), dQ_ref.to(torch.bfloat16), s_ref.to(torch.bfloat16), ds_ref.to(torch.bfloat16)


def run_case(name: str, s_q: int, s_kv: int, q_start_index_s: int):
    q, kv, dO, indices = build_inputs(s_q, s_kv, q_start_index_s)
    o, lse, dQ_ref, s_ref, ds_ref = build_reference(q, kv, dO, indices, q_start_index_s)

    dQ_cuda, s_cuda, ds_cuda = dq_phase_cuda.run_dq_phase(q, kv, o, dO, indices, lse, 1.0 / math.sqrt(D_QK), q_start_index_s)
    torch.cuda.synchronize()

    dQ_max, dQ_rel = calc_diff(dQ_cuda, dQ_ref)
    s_max, s_rel = calc_diff(s_cuda, s_ref)
    ds_max, ds_rel = calc_diff(ds_cuda, ds_ref)

    print(f"\n[{name}]")
    print(f"dQ: max_diff={dQ_max:.6e}, rel_diff={dQ_rel:.6e}")
    print(f"s : max_diff={s_max:.6e}, rel_diff={s_rel:.6e}")
    print(f"ds: max_diff={ds_max:.6e}, rel_diff={ds_rel:.6e}")

    threshold = 1e-2
    ok = dQ_rel < threshold and s_rel < threshold and ds_rel < threshold
    print("PASS" if ok else "FAIL")
    return ok


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print("CUDA Device:", torch.cuda.get_device_name())
    print("CUDA Capability:", torch.cuda.get_device_capability())
    print(f"Fixed TOPK={TOPK}, B_H={B_H}, D_QK={D_QK}, D_V={D_V}")

    torch.manual_seed(42)
    ok_full = run_case("full_topk_window", s_q=32, s_kv=256, q_start_index_s=127)
    ok_partial = run_case("partial_causal_window", s_q=32, s_kv=256, q_start_index_s=0)

    all_ok = ok_full and ok_partial
    print("\nFinal:", "PASS" if all_ok else "FAIL")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

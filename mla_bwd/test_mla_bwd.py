#!/usr/bin/env python3
"""
Test script for mla_bwd kernel
Tests the CUDA kernel matrix multiplication against PyTorch reference implementation.
"""

import torch
import mla_bwd_cuda  # Compiled module


def calc_diff(a, b):
    abs_diff = torch.abs(a - b)
    max_diff = abs_diff.max().item()
    rel_diff = (abs_diff / (1e-4 + torch.abs(a))).mean().item()
    return max_diff, rel_diff


def test_mla_bwd():
    """Test mla_bwd kernel precision"""
    print("=" * 60)
    print("Testing mla_bwd Kernel")
    print("=" * 60)
    
    # Matrix specifications
    B_H = 128
    D_Q = 576
    B_TOPK = 64
    D_K = 576
    D_V = 512
    
    print(f"q shape: [{B_H}, {D_Q}]")
    print(f"kv shape: [{B_TOPK}, {D_K}]")
    print(f"dO shape: [{B_H}, {D_V}]")
    print(f"Expected P shape: [{B_H}, {B_TOPK}] = Q @ K^T")
    print(f"Expected dP shape: [{B_H}, {B_TOPK}] = dO @ V^T")
    print(f"Expected dQ shape: [{B_H}, {D_Q}] = ds @ K")
    print()
    
    # Generate input data
    torch.manual_seed(42)
    q = torch.randn(B_H, D_Q, dtype=torch.bfloat16, device='cuda')
    kv = torch.randn(B_TOPK, D_K, dtype=torch.bfloat16, device='cuda')
    dO = torch.randn(B_H, D_V, dtype=torch.bfloat16, device='cuda')
    
    # Generate indices for sparse attention
    # For test purposes: B=1, S=1, HKV=1, topk=B_TOPK
    # Reference: sparse_mla_bwd.py:295-301
    S_KV = B_TOPK  # KV sequence length
    indices = torch.full((B_TOPK,), S_KV, dtype=torch.int32, device='cuda')
    # Generate random valid indices (0 to S_KV-1)
    valid_indices = torch.randperm(S_KV, device='cuda')[:B_TOPK]
    indices[:len(valid_indices)] = valid_indices
    
    print(f"q dtype: {q.dtype}, device: {q.device}")
    print(f"kv dtype: {kv.dtype}, device: {kv.device}")
    print(f"dO dtype: {dO.dtype}, device: {dO.device}")
    print(f"indices shape: {indices.shape}, dtype: {indices.dtype}")
    print(f"indices range: [{indices.min().item()}, {indices.max().item()}]")
    print()
    
    # Extract K and V from KV
    # For this test, we assume KV contains K and V concatenated
    # K: [B_TOPK, D_K] = [64, 576]
    # V: [B_TOPK, D_V] = [64, 512]
    # Since KV shape is [64, 576], we assume:
    # - K is stored in KV[:, :D_K] = KV[:, :576] (full KV for K)
    # - V is stored in KV[:, :D_V] = KV[:, :512] (first D_V columns for V)
    k = kv[:, :D_K].float()  # [64, 576]
    v = kv[:, :D_V].float()  # [64, 512]
    
    # PyTorch reference implementation: P = Q @ K^T
    print("Computing reference with PyTorch...")
    P_ref = torch.matmul(q.float(), k.transpose(-2, -1))  # [128, 576] @ [576, 64] = [128, 64]
    
    # Apply mask logic: set invalid positions to -inf
    # Reference: sparse_mla_bwd.py:175-183
    # A position is valid if: index >= 0 && index < s_kv && position < topk_length
    s_kv = S_KV
    topk_length = B_TOPK
    mask = (indices >= 0) & (indices < s_kv) & (torch.arange(B_TOPK, device='cuda') < topk_length)
    # Expand mask to [B_H, B_TOPK] for broadcasting
    mask_expanded = mask.unsqueeze(0).expand(B_H, B_TOPK)  # [128, 64]
    # Set invalid positions to -inf
    P_ref = torch.where(mask_expanded, P_ref, torch.tensor(float('-inf'), device='cuda', dtype=torch.float32))
    
    print(f"P_ref shape: {P_ref.shape}, dtype: {P_ref.dtype}")
    
    # PyTorch reference implementation: dP = dO @ V^T
    dP_ref = torch.matmul(dO.float(), v.transpose(-2, -1))  # [128, 512] @ [512, 64] = [128, 64]
    print(f"dP_ref shape: {dP_ref.shape}, dtype: {dP_ref.dtype}")
    
    # Generate lse and O for backward computation
    # lse: log-sum-exp for softmax, shape [B_H] = [128]
    lse = torch.randn(B_H, dtype=torch.float32, device='cuda') * 2.0 + 5.0  # Reasonable range for LSE
    # O: forward output, shape [B_H, D_V] = [128, 512]
    O = torch.randn(B_H, D_V, dtype=torch.bfloat16, device='cuda')
    
    # PyTorch reference: Compute delta = sum(O * dO, dim=-1)
    delta_ref = (O.float() * dO.float()).sum(dim=-1)  # [B_H] = [128]
    print(f"delta_ref shape: {delta_ref.shape}, dtype: {delta_ref.dtype}")
    
    # PyTorch reference: Compute softmax s = exp2(P*scale - LSE)
    sm_scale = 1.0 / (D_Q ** 0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504  # log2(e)
    P_scaled = P_ref * sm_scale_mul_reciprocal_log2  # [B_H, B_TOPK]
    P_scaled_minus_lse = P_scaled - lse.unsqueeze(-1)  # Broadcast LSE
    s_ref = torch.exp2(P_scaled_minus_lse)  # [B_H, B_TOPK]
    print(f"s_ref shape: {s_ref.shape}, dtype: {s_ref.dtype}")
    
    # PyTorch reference: Compute ds = s * (dP - delta) * scale
    dP_minus_delta = dP_ref - delta_ref.unsqueeze(-1)  # Broadcast delta
    ds_ref = s_ref * dP_minus_delta * sm_scale  # [B_H, B_TOPK]
    print(f"ds_ref shape: {ds_ref.shape}, dtype: {ds_ref.dtype}")
    print()
    
    # PyTorch reference: Compute dKV
    # dKV = dV + dK_nope (for dims 0:D_V) + dK_rope (for dims D_V:D_Q)
    # dV = s_bf16^T @ dO (softmax weights transposed times dO)
    # dK_nope = ds_bf16^T @ Q_nope (dS transposed times Q NoPE part)
    # dK_rope = ds_bf16^T @ Q_rope (dS transposed times Q RoPE part)
    D_ROPE = D_Q - D_V  # 64
    s_bf16 = s_ref.to(torch.bfloat16)
    ds_bf16 = ds_ref.to(torch.bfloat16)
    
    # dV = s^T @ dO: [B_TOPK, B_H] @ [B_H, D_V] = [B_TOPK, D_V]
    dV_ref = torch.matmul(s_bf16.float().T, dO.float())
    # dK_nope = ds^T @ Q[:, :D_V]: [B_TOPK, B_H] @ [B_H, D_V] = [B_TOPK, D_V]
    dK_nope_ref = torch.matmul(ds_bf16.float().T, q[:, :D_V].float())
    # dK_rope = ds^T @ Q[:, D_V:]: [B_TOPK, B_H] @ [B_H, D_ROPE] = [B_TOPK, D_ROPE]
    dK_rope_ref = torch.matmul(ds_bf16.float().T, q[:, D_V:].float())
    
    # dKV_nope = dV + dK_nope (V and K share the first D_V dimensions)
    dKV_nope_ref = dV_ref + dK_nope_ref  # [B_TOPK, D_V]
    # dKV = concat(dKV_nope, dK_rope)
    dKV_ref = torch.cat([dKV_nope_ref, dK_rope_ref], dim=-1)  # [B_TOPK, D_K]
    print(f"dKV_ref shape: {dKV_ref.shape}, dtype: {dKV_ref.dtype}")

    # PyTorch reference: Compute dQ = ds @ K
    dQ_ref = torch.matmul(ds_bf16.float(), k)  # [B_H, B_TOPK] @ [B_TOPK, D_Q] = [B_H, D_Q]
    print(f"dQ_ref shape: {dQ_ref.shape}, dtype: {dQ_ref.dtype}")
    
    # PyTorch reference: Compute sdO_t = dO^T (transpose of dO)
    # dO shape: [B_H, D_V] = [128, 512]
    # sdO_t shape: [D_V, B_H] = [512, 128]
    sdO_t_ref = dO.float().T  # [D_V, B_H] = [512, 128]
    print(f"sdO_t_ref shape: {sdO_t_ref.shape}, dtype: {sdO_t_ref.dtype}")
    print()
    
    # CUDA kernel computation
    print("Running CUDA kernel...")
    q_out, kv_out, dO_out, P_cuda, dP_cuda, s_cuda, ds_cuda, dKV_cuda, sdO_t_cuda, dQ_cuda = mla_bwd_cuda.mla_bwd(q, kv, dO, lse, O, indices)
    torch.cuda.synchronize()
    print(f"q_out shape: {q_out.shape}, dtype: {q_out.dtype}")
    print(f"kv_out shape: {kv_out.shape}, dtype: {kv_out.dtype}")
    print(f"dO_out shape: {dO_out.shape}, dtype: {dO_out.dtype}")
    print(f"P_cuda shape: {P_cuda.shape}, dtype: {P_cuda.dtype}")
    print(f"dP_cuda shape: {dP_cuda.shape}, dtype: {dP_cuda.dtype}")
    print(f"s_cuda shape: {s_cuda.shape}, dtype: {s_cuda.dtype}")
    print(f"ds_cuda shape: {ds_cuda.shape}, dtype: {ds_cuda.dtype}")
    print(f"dKV_cuda shape: {dKV_cuda.shape}, dtype: {dKV_cuda.dtype}")
    print(f"sdO_t_cuda shape: {sdO_t_cuda.shape}, dtype: {sdO_t_cuda.dtype}")
    print(f"dQ_cuda shape: {dQ_cuda.shape}, dtype: {dQ_cuda.dtype}")
    print()
    
    # Precision validation: compare CUDA result with PyTorch result
    print("=" * 60)
    print("Matrix Multiplication Precision Validation:")
    print("=" * 60)
    
    # Compare P
    P_diff = (P_cuda - P_ref).abs()
    P_max_diff, P_relative_diff = calc_diff(P_cuda, P_ref)
    
    print(f"P_cuda vs P_ref (PyTorch):")
    print(f"  Max absolute diff:     {P_max_diff:.6e}")
    print(f"  Relative diff:         {P_relative_diff:.6e}")
    print()
    
    # Compare dP
    dP_diff = (dP_cuda - dP_ref).abs()
    dP_max_diff, dP_relative_diff = calc_diff(dP_cuda, dP_ref)
    
    print(f"dP_cuda vs dP_ref (PyTorch):")
    print(f"  Max absolute diff:     {dP_max_diff:.6e}")
    print(f"  Relative diff:         {dP_relative_diff:.6e}")
    print()
    
    # Compare s
    s_diff = (s_cuda - s_ref.bfloat16()).abs()
    s_max_diff, s_relative_diff = calc_diff(s_cuda.float(), s_ref.float())
    
    print(f"s_cuda vs s_ref (PyTorch):")
    print(f"  Max absolute diff:     {s_max_diff:.6e}")
    print(f"  Relative diff:         {s_relative_diff:.6e}")
    print()
    
    # Compare ds
    ds_diff = (ds_cuda - ds_ref.bfloat16()).abs()
    ds_max_diff, ds_relative_diff = calc_diff(ds_cuda.float(), ds_ref.float())
    
    print(f"ds_cuda vs ds_ref (PyTorch):")
    print(f"  Max absolute diff:     {ds_max_diff:.6e}")
    print(f"  Relative diff:         {ds_relative_diff:.6e}")
    print()
    
    # Compare dKV
    dKV_diff = (dKV_cuda - dKV_ref).abs()
    dKV_max_diff, dKV_relative_diff = calc_diff(dKV_cuda, dKV_ref)
    
    print(f"dKV_cuda vs dKV_ref (PyTorch):")
    print(f"  Max absolute diff:     {dKV_max_diff:.6e}")
    print(f"  Relative diff:         {dKV_relative_diff:.6e}")
    
    # Breakdown: NoPE part vs RoPE part
    dKV_nope_diff = (dKV_cuda[:, :D_V] - dKV_ref[:, :D_V]).abs()
    dKV_rope_diff = (dKV_cuda[:, D_V:] - dKV_ref[:, D_V:]).abs()
    print(f"  dKV NoPE max diff:     {dKV_nope_diff.max().item():.6e}")
    print(f"  dKV RoPE max diff:     {dKV_rope_diff.max().item():.6e}")
    print()
    
    # Compare sdO_t
    sdO_t_diff = (sdO_t_cuda.float() - sdO_t_ref).abs()
    sdO_t_max_diff, sdO_t_relative_diff = calc_diff(sdO_t_cuda.float(), sdO_t_ref)
    
    print(f"sdO_t_cuda vs sdO_t_ref (PyTorch):")
    print(f"  Max absolute diff:     {sdO_t_max_diff:.6e}")
    print(f"  Relative diff:         {sdO_t_relative_diff:.6e}")
    print()

    # Compare dQ
    dQ_diff = (dQ_cuda - dQ_ref).abs()
    dQ_max_diff, dQ_relative_diff = calc_diff(dQ_cuda, dQ_ref)

    print(f"dQ_cuda vs dQ_ref (PyTorch):")
    print(f"  Max absolute diff:     {dQ_max_diff:.6e}")
    print(f"  Relative diff:         {dQ_relative_diff:.6e}")
    print()
    
    # Print sample results for debugging
    print("Sample P_cuda vs P_ref (first 5x5 block):")
    print("P_cuda:")
    print(P_cuda[:5, :5])
    print("P_ref (PyTorch):")
    print(P_ref[:5, :5])
    print("Difference:")
    print(P_diff[:5, :5])
    print()
    
    print("Sample dP_cuda vs dP_ref (first 5x5 block):")
    print("dP_cuda:")
    print(dP_cuda[:5, :5])
    print("dP_ref (PyTorch):")
    print(dP_ref[:5, :5])
    print("Difference:")
    print(dP_diff[:5, :5])
    print()
    
    print("Sample s_cuda vs s_ref (first 5x5 block):")
    print("s_cuda:")
    print(s_cuda[:5, :5])
    print("s_ref (PyTorch):")
    print(s_ref[:5, :5])
    print("Difference:")
    print(s_diff[:5, :5])
    print()
    
    print("Sample ds_cuda vs ds_ref (first 5x5 block):")
    print("ds_cuda:")
    print(ds_cuda[:5, :5])
    print("ds_ref (PyTorch):")
    print(ds_ref[:5, :5])
    print("Difference:")
    print(ds_diff[:5, :5])
    print()
    
    print("Sample dKV_cuda vs dKV_ref (first 5x5 block):")
    print("dKV_cuda:")
    print(dKV_cuda[:5, :5])
    print("dKV_ref (PyTorch):")
    print(dKV_ref[:5, :5])
    print("Difference:")
    print(dKV_diff[:5, :5])
    print()
    
    print("Sample dKV RoPE (dims 512-575, first 5x5 block):")
    print("dKV_cuda RoPE:")
    print(dKV_cuda[:5, D_V:D_V+5])
    print("dKV_ref RoPE:")
    print(dKV_ref[:5, D_V:D_V+5])
    print("Difference:")
    print(dKV_rope_diff[:5, :5])
    print()
    
    print("Sample sdO_t_cuda vs sdO_t_ref (first 5x5 block):")
    print("sdO_t_cuda:")
    print(sdO_t_cuda[:5, :5])
    print("sdO_t_ref (PyTorch):")
    print(sdO_t_ref[:5, :5])
    print("Difference:")
    print(sdO_t_diff[:5, :5])
    print()

    print("Sample dQ_cuda vs dQ_ref (first 5x5 block):")
    print("dQ_cuda:")
    print(dQ_cuda[:5, :5])
    print("dQ_ref (PyTorch):")
    print(dQ_ref[:5, :5])
    print("Difference:")
    print(dQ_diff[:5, :5])
    print()
    
    # Q, KV, dO precision validation
    print("=" * 60)
    print("Q, KV, dO Precision Validation:")
    print("=" * 60)
    
    # Compare q_out with original q
    q_diff = (q_out.float() - q.float()).abs()
    q_max_diff, q_relative_diff = calc_diff(q_out.float(), q.float())
    
    print(f"q_out vs q (original):")
    print(f"  Max absolute diff:     {q_max_diff:.6e}")
    print(f"  Relative diff:         {q_relative_diff:.6e}")
    print()
    
    # Compare kv_out with original kv
    kv_diff = (kv_out.float() - kv.float()).abs()
    kv_max_diff, kv_relative_diff = calc_diff(kv_out.float(), kv.float())
    
    print(f"kv_out vs kv (original):")
    print(f"  Max absolute diff:     {kv_max_diff:.6e}")
    print(f"  Relative diff:         {kv_relative_diff:.6e}")
    print()
    
    # Compare dO_out with original dO
    dO_diff = (dO_out.float() - dO.float()).abs()
    dO_max_diff, dO_relative_diff = calc_diff(dO_out.float(), dO.float())
    
    print(f"dO_out vs dO (original):")
    print(f"  Max absolute diff:     {dO_max_diff:.6e}")
    print(f"  Relative diff:         {dO_relative_diff:.6e}")
    print()
    
    # Print sample results for debugging
    print("Sample q_out vs q (first 5x5 block):")
    print("q_out:")
    print(q_out[:5, :5])
    print("q (original):")
    print(q[:5, :5])
    print("Difference:")
    print(q_diff[:5, :5])
    print()
    
    print("Sample kv_out vs kv (first 5x5 block):")
    print("kv_out:")
    print(kv_out[:5, :5])
    print("kv (original):")
    print(kv[:5, :5])
    print("Difference:")
    print(kv_diff[:5, :5])
    print()
    
    print("Sample dO_out vs dO (first 5x5 block):")
    print("dO_out:")
    print(dO_out[:5, :5])
    print("dO (original):")
    print(dO[:5, :5])
    print("Difference:")
    print(dO_diff[:5, :5])
    print()
    
    # Precision criterion: rel_diff < 0.01
    rel_diff_threshold = 0.01

    metrics = [
        ("P matrix multiplication", P_max_diff, P_relative_diff),
        ("dP matrix multiplication", dP_max_diff, dP_relative_diff),
        ("s (softmax)", s_max_diff, s_relative_diff),
        ("ds", ds_max_diff, ds_relative_diff),
        ("q precision", q_max_diff, q_relative_diff),
        ("kv precision", kv_max_diff, kv_relative_diff),
        ("dO precision", dO_max_diff, dO_relative_diff),
        ("dKV gradient", dKV_max_diff, dKV_relative_diff),
        ("sdO_t (dO transposed)", sdO_t_max_diff, sdO_t_relative_diff),
        ("dQ gradient", dQ_max_diff, dQ_relative_diff),
    ]

    all_passed = True
    for metric_name, max_diff, rel_diff in metrics:
        metric_passed = rel_diff < rel_diff_threshold
        status = "✓" if metric_passed else "✗"
        result = "PASSED" if metric_passed else "FAILED"
        print(f"{status} {metric_name} test {result}!")
        print(f"  (max_diff={max_diff:.6e}, rel_diff={rel_diff:.6e}, threshold={rel_diff_threshold:.2e})")
        if not metric_passed:
            all_passed = False

    return all_passed


def test_different_inputs():
    """Test with different input patterns"""
    print("\n" + "=" * 60)
    print("Testing with different input patterns")
    print("=" * 60)
    
    B_H = 128
    D_Q = 576
    B_TOPK = 64
    D_K = 576
    D_V = 512
    
    test_cases = [
        ("All ones", torch.ones, torch.ones, torch.ones),
        ("All zeros", torch.zeros, torch.zeros, torch.zeros),
        ("Random uniform", 
         lambda *args, **kwargs: torch.rand(*args, **kwargs) * 2 - 1,
         lambda *args, **kwargs: torch.rand(*args, **kwargs) * 2 - 1,
         lambda *args, **kwargs: torch.rand(*args, **kwargs) * 2 - 1),
    ]
    
    rel_diff_threshold = 0.01
    all_passed = True
    for name, q_gen, kv_gen, dO_gen in test_cases:
        try:
            q = q_gen(B_H, D_Q, dtype=torch.bfloat16, device='cuda')
            kv = kv_gen(B_TOPK, D_K, dtype=torch.bfloat16, device='cuda')
            dO = dO_gen(B_H, D_V, dtype=torch.bfloat16, device='cuda')
            
            # Generate indices for sparse attention
            S_KV = B_TOPK
            indices = torch.full((B_TOPK,), S_KV, dtype=torch.int32, device='cuda')
            valid_indices = torch.randperm(S_KV, device='cuda')[:B_TOPK]
            indices[:len(valid_indices)] = valid_indices
            
            # PyTorch reference
            k = kv[:, :D_K].float()
            v = kv[:, :D_V].float()
            P_ref = torch.matmul(q.float(), k.transpose(-2, -1))
            
            # Apply mask logic: set invalid positions to -inf
            s_kv = S_KV
            topk_length = B_TOPK
            mask = (indices >= 0) & (indices < s_kv) & (torch.arange(B_TOPK, device='cuda') < topk_length)
            mask_expanded = mask.unsqueeze(0).expand(B_H, B_TOPK)
            P_ref = torch.where(mask_expanded, P_ref, torch.tensor(float('-inf'), device='cuda', dtype=torch.float32))
            
            dP_ref = torch.matmul(dO.float(), v.transpose(-2, -1))
            
            # Generate lse and O
            lse = torch.randn(B_H, dtype=torch.float32, device='cuda') * 2.0 + 5.0
            O = torch.randn(B_H, D_V, dtype=torch.bfloat16, device='cuda')

            # PyTorch reference for dQ
            sm_scale = 1.0 / (D_Q ** 0.5)
            scale_log2e = sm_scale * 1.44269504
            delta_ref = (O.float() * dO.float()).sum(dim=-1)
            s_ref = torch.exp2(P_ref * scale_log2e - lse.unsqueeze(-1))
            ds_ref = s_ref * (dP_ref - delta_ref.unsqueeze(-1)) * sm_scale
            dQ_ref = torch.matmul(ds_ref.to(torch.bfloat16).float(), k)

            # PyTorch reference for dKV
            s_bf16 = s_ref.to(torch.bfloat16)
            ds_bf16 = ds_ref.to(torch.bfloat16)
            dV_ref = torch.matmul(s_bf16.float().T, dO.float())
            dK_nope_ref = torch.matmul(ds_bf16.float().T, q[:, :D_V].float())
            dK_rope_ref = torch.matmul(ds_bf16.float().T, q[:, D_V:].float())
            dKV_ref = torch.cat([dV_ref + dK_nope_ref, dK_rope_ref], dim=-1)
            
            # CUDA kernel
            q_out, kv_out, dO_out, P_cuda, dP_cuda, s_cuda, ds_cuda, dKV_cuda, sdO_t_cuda, dQ_cuda = mla_bwd_cuda.mla_bwd(q, kv, dO, lse, O, indices)
            
            P_max_diff, P_relative_diff = calc_diff(P_cuda, P_ref)
            dP_max_diff, dP_relative_diff = calc_diff(dP_cuda, dP_ref)
            q_max_diff, q_relative_diff = calc_diff(q_out.float(), q.float())
            kv_max_diff, kv_relative_diff = calc_diff(kv_out.float(), kv.float())
            dO_max_diff, dO_relative_diff = calc_diff(dO_out.float(), dO.float())
            dKV_max_diff, dKV_relative_diff = calc_diff(dKV_cuda, dKV_ref)
            dQ_max_diff, dQ_relative_diff = calc_diff(dQ_cuda, dQ_ref)
            
            P_test_ok = P_relative_diff < rel_diff_threshold
            dP_test_ok = dP_relative_diff < rel_diff_threshold
            q_test_ok = q_relative_diff < rel_diff_threshold
            kv_test_ok = kv_relative_diff < rel_diff_threshold
            dO_test_ok = dO_relative_diff < rel_diff_threshold
            dKV_test_ok = dKV_relative_diff < rel_diff_threshold
            dQ_test_ok = dQ_relative_diff < rel_diff_threshold
            
            test_ok = P_test_ok and dP_test_ok and q_test_ok and kv_test_ok and dO_test_ok and dKV_test_ok and dQ_test_ok
            
            status = "✓" if test_ok else "✗"
            print(f"{status} {name}:")
            print(f"    P: max_diff={P_max_diff:.6e}, rel_diff={P_relative_diff:.6e}")
            print(f"    dP: max_diff={dP_max_diff:.6e}, rel_diff={dP_relative_diff:.6e}")
            print(f"    q: max_diff={q_max_diff:.6e}, rel_diff={q_relative_diff:.6e}")
            print(f"    kv: max_diff={kv_max_diff:.6e}, rel_diff={kv_relative_diff:.6e}")
            print(f"    dO: max_diff={dO_max_diff:.6e}, rel_diff={dO_relative_diff:.6e}")
            print(f"    dKV: max_diff={dKV_max_diff:.6e}, rel_diff={dKV_relative_diff:.6e}")
            print(f"    dQ: max_diff={dQ_max_diff:.6e}, rel_diff={dQ_relative_diff:.6e}")
            print(f"    threshold: rel_diff < {rel_diff_threshold:.2e}")
            
            if not test_ok:
                all_passed = False
        except Exception as e:
            print(f"✗ {name}: ERROR - {e}")
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
    if passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
    
    exit(0 if passed else 1)

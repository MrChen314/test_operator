import torch
from test_dq_2sm_mma import run_dq_2sm_mma


def test_dq_2sm_mma():
    """
    Test dq_2sm_mma kernel:
    1. Load ds to smem
    2. Load kv using TMA gather4 and exchange data between CTA0 and CTA1
    3. Compute dQ = dS^T @ K
    """
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    B_H = 128
    B_TOPK = 64
    D_K = 576
    s_kv = 1024

    # Generate test data
    torch.manual_seed(42)
    ds = torch.randn(B_H, B_TOPK, dtype=dtype, device=device)
    kv = torch.randn(s_kv, D_K, dtype=dtype, device=device)
    indices = torch.randint(0, s_kv, (B_TOPK,), dtype=torch.int32, device=device)

    # Run CUDA kernel
    dQ_cuda, cuda_kv_nope, cuda_kv_rope, cuda_ds_t = run_dq_2sm_mma(ds, kv, indices)

    # Step 0: Verify ds_t accuracy
    # Kernel exports smem.ds_t as [B_TOPK, B_H], which should match ds^T.
    ds_ref = ds.transpose(0, 1).contiguous()

    print("Testing ds_t load accuracy...")
    ds_t_diff = (cuda_ds_t.float() - ds_ref.float()).abs()
    ds_t_max_diff = ds_t_diff.max().item()
    ds_t_mean_diff = ds_t_diff.mean().item()
    ds_t_rel_diff = (ds_t_diff / (ds_ref.float().abs() + 1e-6)).mean().item()

    print(f"ds_t max diff: {ds_t_max_diff:.6f}")
    print(f"ds_t mean diff: {ds_t_mean_diff:.6f}")
    print(f"ds_t relative diff: {ds_t_rel_diff:.6f}")

    # Compute reference
    # Step 1: Gather KV according to indices
    kv_gathered = kv[indices]  # [B_TOPK, D_K]

    # Step 2: Kernel internally exchanges SMEM tiles between CTA0/CTA1 for MMA.
    # However, the exported cuda_kv_{nope,rope} path remaps data back to
    # the original gathered row order [indices], so reference here is kv_gathered.
    ref_kv = kv_gathered.clone()

    # Verify cuda_kv matches ref_kv
    cuda_kv_full = torch.cat([cuda_kv_nope, cuda_kv_rope], dim=1)  # [B_TOPK, 576]

    print("Testing KV exchange accuracy...")
    kv_diff = (cuda_kv_full.float() - ref_kv.float()).abs()
    kv_max_diff = kv_diff.max().item()
    kv_mean_diff = kv_diff.mean().item()

    print(f"KV max diff: {kv_max_diff:.6f}")
    print(f"KV mean diff: {kv_mean_diff:.6f}")

    # Step 3: Compute reference dQ
    # dQ = dS^T @ K
    # dS: [B_H, B_TOPK], K: [B_TOPK, D_K]
    # dQ: [B_H, D_K]
    ref_dQ = torch.matmul(ds.float(), ref_kv.float())

    # Verify dQ accuracy
    print("\nTesting dQ accuracy...")
    dq_diff = (dQ_cuda - ref_dQ).abs()
    dq_max_diff = dq_diff.max().item()
    dq_mean_diff = dq_diff.mean().item()
    dq_rel_diff = (dq_diff / (ref_dQ.abs() + 1e-6)).mean().item()

    print(f"dQ max diff: {dq_max_diff:.6f}")
    print(f"dQ mean diff: {dq_mean_diff:.6f}")
    print(f"dQ relative diff: {dq_rel_diff:.6f}")

    # Check thresholds
    assert ds_t_max_diff < 1e-2, f"ds_t max diff too large: {ds_t_max_diff}"
    assert ds_t_rel_diff < 1e-3, f"ds_t relative diff too large: {ds_t_rel_diff}"
    assert kv_max_diff < 1e-2, f"KV max diff too large: {kv_max_diff}"
    assert dq_max_diff < 1.0, f"dQ max diff too large: {dq_max_diff}"
    assert dq_rel_diff < 0.01, f"dQ relative diff too large: {dq_rel_diff}"

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_dq_2sm_mma()

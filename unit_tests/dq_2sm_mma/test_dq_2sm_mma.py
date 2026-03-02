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
    dQ_cuda, cuda_kv_nope, cuda_kv_rope = run_dq_2sm_mma(ds, kv, indices)

    # Compute reference
    # Step 1: Gather KV according to indices
    kv_gathered = kv[indices]  # [B_TOPK, D_K]

    # Step 2: Exchange data between first and second half
    # CTA0 handles indices[0:32], CTA1 handles indices[32:64]
    # After exchange:
    # k_nope:
    # - CTA0's k_nope[:, 256:512] comes from CTA1's k_nope[:, 0:256]
    # - CTA1's k_nope[:, 0:256] comes from CTA0's k_nope[:, 256:512]
    # k_rope:
    # - CTA0's k_rope[:, 32:64] comes from CTA1's k_rope[:, 0:32]
    # - CTA1's k_rope[:, 0:32] comes from CTA0's k_rope[:, 32:64]

    ref_kv = kv_gathered.clone()

    # Exchange k_nope
    kv_nope_cta0 = ref_kv[0:32, 0:512].clone()
    kv_nope_cta1 = ref_kv[32:64, 0:512].clone()
    ref_kv[0:32, 256:512] = kv_nope_cta1[:, 0:256]
    ref_kv[32:64, 0:256] = kv_nope_cta0[:, 256:512]

    # Exchange k_rope
    kv_rope_cta0 = ref_kv[0:32, 512:576].clone()
    kv_rope_cta1 = ref_kv[32:64, 512:576].clone()
    ref_kv[0:32, 544:576] = kv_rope_cta1[:, 0:32]
    ref_kv[32:64, 512:544] = kv_rope_cta0[:, 32:64]

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
    assert kv_max_diff < 1e-2, f"KV max diff too large: {kv_max_diff}"
    assert dq_max_diff < 1.0, f"dQ max diff too large: {dq_max_diff}"
    assert dq_rel_diff < 0.01, f"dQ relative diff too large: {dq_rel_diff}"

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_dq_2sm_mma()

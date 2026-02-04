#!/usr/bin/env python3
"""
Test script for utcmma_ss_peer (SM100 WS_SS_NOELECT with peer SMEM access)
Tests the CUDA kernel matrix multiplication against PyTorch reference implementation.
"""

import torch
import test_2sm_mma_peer_cuda  # Compiled module


def test_utcmma_ss_peer():
    """Test utcmma_ss_peer matrix multiplication precision"""
    print("=" * 60)
    print("Testing utcmma_ss_peer Matrix Multiplication (SM100 WS_SS_NOELECT with Peer SMEM)")
    print("=" * 60)
    
    # Matrix specifications
    M, K_DIM, K_ROWS, N = 64, 128, 128, 256
    print(f"Q shape: [{M}, {K_DIM}]")
    print(f"K shape: [{K_ROWS}, {N}]")
    print(f"Expected P shape: [{M}, {N}] = Q @ K^T")
    print()
    
    # Generate input data
    torch.manual_seed(42)
    Q = torch.randn(M, K_DIM, dtype=torch.bfloat16, device='cuda')
    K_mat = torch.randn(K_ROWS, N, dtype=torch.bfloat16, device='cuda')
    
    print(f"Q dtype: {Q.dtype}, device: {Q.device}")
    print(f"K dtype: {K_mat.dtype}, device: {K_mat.device}")
    print()
    
    # PyTorch reference implementation: P = Q @ K^T
    print("Computing reference with PyTorch...")
    P_ref = torch.matmul(Q.float(), K_mat.float())
    print(f"P_ref shape: {P_ref.shape}, dtype: {P_ref.dtype}")
    print()
    
    # CUDA kernel computation
    print("Running CUDA kernel...")
    P_cuda, Q_out, K_out, Q_first_half_out, Q_second_half_out = test_2sm_mma_peer_cuda.utcmma_ss_peer(Q, K_mat)
    torch.cuda.synchronize()
    print(f"P_cuda shape: {P_cuda.shape}, dtype: {P_cuda.dtype}")
    print(f"Q_out shape: {Q_out.shape}, dtype: {Q_out.dtype}")
    print(f"K_out shape: {K_out.shape}, dtype: {K_out.dtype}")
    print(f"Q_first_half_out shape: {Q_first_half_out.shape}, dtype: {Q_first_half_out.dtype}")
    print(f"Q_second_half_out shape: {Q_second_half_out.shape}, dtype: {Q_second_half_out.dtype}")
    print()
    
    # Precision validation: compare CUDA result with PyTorch result
    print("=" * 60)
    print("Matrix Multiplication Precision Validation:")
    print("=" * 60)
    P_diff = (P_cuda - P_ref).abs()
    P_max_diff = P_diff.max().item()
    P_mean_diff = P_diff.mean().item()
    P_relative_diff = (P_diff / (P_ref.abs() + 1e-8)).max().item()
    
    print(f"P_cuda vs P_ref (PyTorch):")
    print(f"  Max absolute diff:     {P_max_diff:.6e}")
    print(f"  Mean absolute diff:    {P_mean_diff:.6e}")
    print(f"  Max relative diff:     {P_relative_diff:.6e}")
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
    
    # Q and K precision validation
    print("=" * 60)
    print("Q and K Precision Validation:")
    print("=" * 60)
    
    # Compare Q_out with original Q
    Q_diff = (Q_out.float() - Q.float()).abs()
    Q_max_diff = Q_diff.max().item()
    Q_mean_diff = Q_diff.mean().item()
    Q_relative_diff = (Q_diff / (Q.float().abs() + 1e-8)).max().item()
    
    print(f"Q_out vs Q (original):")
    print(f"  Max absolute diff:     {Q_max_diff:.6e}")
    print(f"  Mean absolute diff:    {Q_mean_diff:.6e}")
    print(f"  Max relative diff:     {Q_relative_diff:.6e}")
    print()
    
    # Compare K_out with original K
    K_diff = (K_out.float() - K_mat.float()).abs()
    K_max_diff = K_diff.max().item()
    K_mean_diff = K_diff.mean().item()
    K_relative_diff = (K_diff / (K_mat.float().abs() + 1e-8)).max().item()
    
    print(f"K_out vs K (original):")
    print(f"  Max absolute diff:     {K_max_diff:.6e}")
    print(f"  Mean absolute diff:    {K_mean_diff:.6e}")
    print(f"  Max relative diff:     {K_relative_diff:.6e}")
    print()
    
    # Print sample results for debugging
    print("Sample Q_out vs Q (first 5x5 block):")
    print("Q_out:")
    print(Q_out[:5, :5])
    print("Q (original):")
    print(Q[:5, :5])
    print("Difference:")
    print(Q_diff[:5, :5])
    print()
    
    print("Sample K_out vs K (first 5x5 block):")
    print("K_out:")
    print(K_out[:5, :5])
    print("K (original):")
    print(K_mat[:5, :5])
    print("Difference:")
    print(K_diff[:5, :5])
    print()
    
    # Compare Q_first_half_out with Q[:, :64]
    print("=" * 60)
    print("Q_first_half_out Precision Validation:")
    print("=" * 60)
    Q_first_half_ref = Q[:, :K_DIM//2]  # Q[:, :64]
    Q_first_half_diff = (Q_first_half_out.float() - Q_first_half_ref.float()).abs()
    Q_first_half_max_diff = Q_first_half_diff.max().item()
    Q_first_half_mean_diff = Q_first_half_diff.mean().item()
    Q_first_half_relative_diff = (Q_first_half_diff / (Q_first_half_ref.float().abs() + 1e-8)).max().item()
    
    print(f"Q_first_half_out vs Q[:, :{K_DIM//2}]:")
    print(f"  Max absolute diff:     {Q_first_half_max_diff:.6e}")
    print(f"  Mean absolute diff:    {Q_first_half_mean_diff:.6e}")
    print(f"  Max relative diff:     {Q_first_half_relative_diff:.6e}")
    print()
    
    # Print sample results for debugging
    print("Sample Q_first_half_out vs Q[:, :64] (first 5x5 block):")
    print("Q_first_half_out:")
    print(Q_first_half_out[:5, :5])
    print("Q[:, :64] (original):")
    print(Q_first_half_ref[:5, :5])
    print("Difference:")
    print(Q_first_half_diff[:5, :5])
    print()
    
    # Compare Q_second_half_out with Q[:, 64:]
    print("=" * 60)
    print("Q_second_half_out Precision Validation:")
    print("=" * 60)
    Q_second_half_ref = Q[:, K_DIM//2:]  # Q[:, 64:]
    Q_second_half_diff = (Q_second_half_out.float() - Q_second_half_ref.float()).abs()
    Q_second_half_max_diff = Q_second_half_diff.max().item()
    Q_second_half_mean_diff = Q_second_half_diff.mean().item()
    Q_second_half_relative_diff = (Q_second_half_diff / (Q_second_half_ref.float().abs() + 1e-8)).max().item()
    
    print(f"Q_second_half_out vs Q[:, {K_DIM//2}:]:")
    print(f"  Max absolute diff:     {Q_second_half_max_diff:.6e}")
    print(f"  Mean absolute diff:    {Q_second_half_mean_diff:.6e}")
    print(f"  Max relative diff:     {Q_second_half_relative_diff:.6e}")
    print()
    
    # Print sample results for debugging
    print("Sample Q_second_half_out vs Q[:, 64:] (first 5x5 block):")
    print("Q_second_half_out:")
    print(Q_second_half_out[:5, :5])
    print("Q[:, 64:] (original):")
    print(Q_second_half_ref[:5, :5])
    print("Difference:")
    print(Q_second_half_diff[:5, :5])
    print()
    
    # Assert precision is within reasonable range
    # For bfloat16 input and float32 output, allow some numerical error
    max_diff_threshold = 1e-2  # Maximum allowed absolute error
    relative_diff_threshold = 1e-1  # Maximum allowed relative error
    
    # For Q and K (bfloat16), allow bit-exact match or very small error
    qk_max_diff_threshold = 1e-5  # Maximum allowed absolute error for Q/K
    qk_relative_diff_threshold = 1e-4  # Maximum allowed relative error for Q/K
    
    P_test_passed = P_max_diff < max_diff_threshold and P_relative_diff < relative_diff_threshold
    Q_test_passed = Q_max_diff < qk_max_diff_threshold and Q_relative_diff < qk_relative_diff_threshold
    K_test_passed = K_max_diff < qk_max_diff_threshold and K_relative_diff < qk_relative_diff_threshold
    Q_first_half_test_passed = Q_first_half_max_diff < qk_max_diff_threshold and Q_first_half_relative_diff < qk_relative_diff_threshold
    Q_second_half_test_passed = Q_second_half_max_diff < qk_max_diff_threshold and Q_second_half_relative_diff < qk_relative_diff_threshold
    
    all_passed = P_test_passed and Q_test_passed and K_test_passed and Q_first_half_test_passed and Q_second_half_test_passed
    
    if P_test_passed:
        print(f"✓ Matrix multiplication test PASSED!")
        print(f"  (max_diff={P_max_diff:.6e} < {max_diff_threshold}, "
              f"relative_diff={P_relative_diff:.6e} < {relative_diff_threshold})")
    else:
        print(f"✗ Matrix multiplication test FAILED!")
        print(f"  (max_diff={P_max_diff:.6e} >= {max_diff_threshold} or "
              f"relative_diff={P_relative_diff:.6e} >= {relative_diff_threshold})")
    
    if Q_test_passed:
        print(f"✓ Q precision test PASSED!")
        print(f"  (max_diff={Q_max_diff:.6e} < {qk_max_diff_threshold}, "
              f"relative_diff={Q_relative_diff:.6e} < {qk_relative_diff_threshold})")
    else:
        print(f"✗ Q precision test FAILED!")
        print(f"  (max_diff={Q_max_diff:.6e} >= {qk_max_diff_threshold} or "
              f"relative_diff={Q_relative_diff:.6e} >= {qk_relative_diff_threshold})")
    
    if K_test_passed:
        print(f"✓ K precision test PASSED!")
        print(f"  (max_diff={K_max_diff:.6e} < {qk_max_diff_threshold}, "
              f"relative_diff={K_relative_diff:.6e} < {qk_relative_diff_threshold})")
    else:
        print(f"✗ K precision test FAILED!")
        print(f"  (max_diff={K_max_diff:.6e} >= {qk_max_diff_threshold} or "
              f"relative_diff={K_relative_diff:.6e} >= {qk_relative_diff_threshold})")
    
    if Q_first_half_test_passed:
        print(f"✓ Q_first_half_out precision test PASSED!")
        print(f"  (max_diff={Q_first_half_max_diff:.6e} < {qk_max_diff_threshold}, "
              f"relative_diff={Q_first_half_relative_diff:.6e} < {qk_relative_diff_threshold})")
    else:
        print(f"✗ Q_first_half_out precision test FAILED!")
        print(f"  (max_diff={Q_first_half_max_diff:.6e} >= {qk_max_diff_threshold} or "
              f"relative_diff={Q_first_half_relative_diff:.6e} >= {qk_relative_diff_threshold})")
    
    if Q_second_half_test_passed:
        print(f"✓ Q_second_half_out precision test PASSED!")
        print(f"  (max_diff={Q_second_half_max_diff:.6e} < {qk_max_diff_threshold}, "
              f"relative_diff={Q_second_half_relative_diff:.6e} < {qk_relative_diff_threshold})")
    else:
        print(f"✗ Q_second_half_out precision test FAILED!")
        print(f"  (max_diff={Q_second_half_max_diff:.6e} >= {qk_max_diff_threshold} or "
              f"relative_diff={Q_second_half_relative_diff:.6e} >= {qk_relative_diff_threshold})")
    
    return all_passed


def test_different_inputs():
    """Test with different input patterns"""
    print("\n" + "=" * 60)
    print("Testing with different input patterns")
    print("=" * 60)
    
    M, K_DIM, K_ROWS, N = 64, 128, 128, 256
    test_cases = [
        ("All ones", torch.ones, torch.ones),
        ("All zeros", torch.zeros, torch.zeros),
        ("Random uniform", lambda *args, **kwargs: torch.rand(*args, **kwargs) * 2 - 1,
                          lambda *args, **kwargs: torch.rand(*args, **kwargs) * 2 - 1),
    ]
    
    all_passed = True
    for name, q_gen, k_gen in test_cases:
        try:
            Q = q_gen(M, K_DIM, dtype=torch.bfloat16, device='cuda')
            K_mat = k_gen(K_ROWS, N, dtype=torch.bfloat16, device='cuda')
            
            # PyTorch reference
            P_ref = torch.matmul(Q.float(), K_mat.transpose(-2, -1).float())
            
            # CUDA kernel
            P_cuda, Q_out, K_out, Q_first_half_out, Q_second_half_out = test_2sm_mma_peer_cuda.utcmma_ss_peer(Q, K_mat)
            
            P_max_diff = (P_cuda - P_ref).abs().max().item()
            P_relative_diff = ((P_cuda - P_ref).abs() / (P_ref.abs() + 1e-8)).max().item()
            
            Q_max_diff = (Q_out.float() - Q.float()).abs().max().item()
            Q_relative_diff = ((Q_out.float() - Q.float()).abs() / (Q.float().abs() + 1e-8)).max().item()
            
            K_max_diff = (K_out.float() - K_mat.float()).abs().max().item()
            K_relative_diff = ((K_out.float() - K_mat.float()).abs() / (K_mat.float().abs() + 1e-8)).max().item()
            
            Q_first_half_ref = Q[:, :K_DIM//2]
            Q_first_half_max_diff = (Q_first_half_out.float() - Q_first_half_ref.float()).abs().max().item()
            Q_first_half_relative_diff = ((Q_first_half_out.float() - Q_first_half_ref.float()).abs() / (Q_first_half_ref.float().abs() + 1e-8)).max().item()
            
            Q_second_half_ref = Q[:, K_DIM//2:]
            Q_second_half_max_diff = (Q_second_half_out.float() - Q_second_half_ref.float()).abs().max().item()
            Q_second_half_relative_diff = ((Q_second_half_out.float() - Q_second_half_ref.float()).abs() / (Q_second_half_ref.float().abs() + 1e-8)).max().item()
            
            max_diff_threshold = 1e-2
            relative_diff_threshold = 1e-1
            qk_max_diff_threshold = 1e-5
            qk_relative_diff_threshold = 1e-4
            
            P_test_ok = P_max_diff < max_diff_threshold and P_relative_diff < relative_diff_threshold
            Q_test_ok = Q_max_diff < qk_max_diff_threshold and Q_relative_diff < qk_relative_diff_threshold
            K_test_ok = K_max_diff < qk_max_diff_threshold and K_relative_diff < qk_relative_diff_threshold
            Q_first_half_test_ok = Q_first_half_max_diff < qk_max_diff_threshold and Q_first_half_relative_diff < qk_relative_diff_threshold
            Q_second_half_test_ok = Q_second_half_max_diff < qk_max_diff_threshold and Q_second_half_relative_diff < qk_relative_diff_threshold
            test_ok = P_test_ok and Q_test_ok and K_test_ok and Q_first_half_test_ok and Q_second_half_test_ok
            
            status = "✓" if test_ok else "✗"
            print(f"{status} {name}:")
            print(f"    P: max_diff={P_max_diff:.6e}, relative_diff={P_relative_diff:.6e}")
            print(f"    Q: max_diff={Q_max_diff:.6e}, relative_diff={Q_relative_diff:.6e}")
            print(f"    K: max_diff={K_max_diff:.6e}, relative_diff={K_relative_diff:.6e}")
            print(f"    Q_first_half: max_diff={Q_first_half_max_diff:.6e}, relative_diff={Q_first_half_relative_diff:.6e}")
            print(f"    Q_second_half: max_diff={Q_second_half_max_diff:.6e}, relative_diff={Q_second_half_relative_diff:.6e}")
            
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
    
    passed = test_utcmma_ss_peer()
    
    if passed:
        passed = test_different_inputs()
    
    print("\n" + "=" * 60)
    if passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
    
    exit(0 if passed else 1)

#!/usr/bin/env python3
"""
Test script for utcmma_ss (SM100 2x1SM SS MMA)
Tests the CUDA kernel matrix multiplication against PyTorch reference implementation.
"""

import torch
import test_2sm_mma_cuda  # 编译后的模块


def test_utcmma_ss():
    """Test utcmma_ss matrix multiplication precision"""
    print("=" * 60)
    print("Testing utcmma_ss Matrix Multiplication (SM100 2x1SM SS)")
    print("=" * 60)
    
    # 矩阵规格
    M, N, K_DIM = 128, 128, 256
    print(f"Q shape: [{M}, {K_DIM}]")
    print(f"K shape: [{N}, {K_DIM}]")
    print(f"Expected P shape: [{M}, {N}] = Q @ K^T")
    print()
    
    # 生成输入数据
    torch.manual_seed(42)
    Q = torch.randn(M, K_DIM, dtype=torch.bfloat16, device='cuda')
    K_mat = torch.randn(N, K_DIM, dtype=torch.bfloat16, device='cuda')
    
    print(f"Q dtype: {Q.dtype}, device: {Q.device}")
    print(f"K dtype: {K_mat.dtype}, device: {K_mat.device}")
    print()
    
    # PyTorch 参考实现: P = Q @ K^T
    print("Computing reference with PyTorch...")
    P_ref = torch.matmul(Q.float(), K_mat.transpose(-2, -1).float())
    print(f"P_ref shape: {P_ref.shape}, dtype: {P_ref.dtype}")
    print()
    
    # CUDA kernel 计算
    print("Running CUDA kernel...")
    P_cuda = test_2sm_mma_cuda.utcmma_ss_debug(Q, K_mat)
    torch.cuda.synchronize()
    print(f"P_cuda shape: {P_cuda.shape}, dtype: {P_cuda.dtype}")
    print()

    before_64_col = (P_cuda[:, :64] - P_ref[:, :64]).abs().max().item()
    after_64_col = (P_cuda[:, 64:] - P_ref[:, 64:]).abs().max().item()
    print(f"before_64_col: {before_64_col}, after_64_col: {after_64_col}")
    
    # 精度校验：比较 CUDA 结果和 PyTorch 结果
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
    
    # 打印部分结果用于调试
    print("Sample P_cuda vs P_ref (first 5x5 block):")
    print("P_cuda:")
    print(P_cuda[:5, :5])
    print("P_ref (PyTorch):")
    print(P_ref[:5, :5])
    print("Difference:")
    print(P_diff[:5, :5])
    print()
    
    # 断言精度在合理范围内
    # 对于 bfloat16 输入和 float32 输出，允许一定的数值误差
    max_diff_threshold = 1e-2  # 允许的最大绝对误差
    relative_diff_threshold = 1e-1  # 允许的最大相对误差
    
    if P_max_diff < max_diff_threshold and P_relative_diff < relative_diff_threshold:
        print(f"✓ Matrix multiplication test PASSED!")
        print(f"  (max_diff={P_max_diff:.6e} < {max_diff_threshold}, "
              f"relative_diff={P_relative_diff:.6e} < {relative_diff_threshold})")
        return True
    else:
        print(f"✗ Matrix multiplication test FAILED!")
        print(f"  (max_diff={P_max_diff:.6e} >= {max_diff_threshold} or "
              f"relative_diff={P_relative_diff:.6e} >= {relative_diff_threshold})")
        return False


def test_different_inputs():
    """Test with different input patterns"""
    print("\n" + "=" * 60)
    print("Testing with different input patterns")
    print("=" * 60)
    
    M, N, K_DIM = 128, 128, 256
    test_cases = [
        ("All ones", torch.ones, torch.ones),
        ("All zeros", torch.zeros, torch.zeros),
        ("Identity-like", lambda *args, **kwargs: torch.eye(M, K_DIM, **kwargs), 
                          lambda *args, **kwargs: torch.eye(N, K_DIM, **kwargs)),
        ("Random uniform", lambda *args, **kwargs: torch.rand(*args, **kwargs) * 2 - 1,
                          lambda *args, **kwargs: torch.rand(*args, **kwargs) * 2 - 1),
    ]
    
    all_passed = True
    for name, q_gen, k_gen in test_cases:
        try:
            Q = q_gen(M, K_DIM, dtype=torch.bfloat16, device='cuda')
            K_mat = k_gen(N, K_DIM, dtype=torch.bfloat16, device='cuda')
            
            # PyTorch reference
            P_ref = torch.matmul(Q.float(), K_mat.transpose(-2, -1).float())
            
            # CUDA kernel
            P_cuda = test_2sm_mma_cuda.utcmma_ss_debug(Q, K_mat)
            
            P_max_diff = (P_cuda - P_ref).abs().max().item()
            P_relative_diff = ((P_cuda - P_ref).abs() / (P_ref.abs() + 1e-8)).max().item()
            
            max_diff_threshold = 1e-2
            relative_diff_threshold = 1e-1
            test_ok = P_max_diff < max_diff_threshold and P_relative_diff < relative_diff_threshold
            
            status = "✓" if test_ok else "✗"
            print(f"{status} {name}: max_diff={P_max_diff:.6e}, relative_diff={P_relative_diff:.6e}")
            
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
    
    passed = test_utcmma_ss()
    
    if passed:
        passed = test_different_inputs()
    
    print("\n" + "=" * 60)
    if passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
    
    exit(0 if passed else 1)

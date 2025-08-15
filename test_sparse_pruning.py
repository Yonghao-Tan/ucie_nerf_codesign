#!/usr/bin/env python3
"""
测试稀疏pruning功能
"""

import torch
import time
from ibrnet.pruning_utils import (
    apply_source_view_pruning_optimized,
    apply_source_view_pruning_sparse,
    apply_source_view_pruning_sparse_vectorized
)

def test_sparse_pruning():
    """
    测试稀疏pruning功能的正确性和性能
    """
    print("=== 测试稀疏Pruning功能 ===")
    
    # 设置测试参数
    H, W = 20, 20  # 能被5整除的尺寸
    window_size = 5
    N_rays = H * W  # 400
    N_coarse = 32
    N_importance = 32
    N_views = 8
    top_k = 6
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"图像尺寸: {H}x{W}, Total rays: {N_rays}")
    print(f"Window size: {window_size}x{window_size}")
    print(f"Number of windows: {(H//window_size) * (W//window_size)}")
    
    # 创建测试数据
    z_vals_coarse = torch.linspace(2.0, 6.0, N_coarse).unsqueeze(0).repeat(N_rays, 1).to(device)
    z_samples = 2.0 + torch.rand(N_rays, N_importance, device=device) * 4.0
    
    # 模拟稀疏情况：只有window中心有有效的blending weights
    blending_weights_valid = torch.zeros(N_rays, N_coarse, N_views, 1, device=device)
    coarse_mask = torch.zeros(N_rays, N_coarse, N_views, 1, device=device)
    
    # 只为window中心设置有效值
    center_offset = window_size // 2
    for i in range(0, H, window_size):
        for j in range(0, W, window_size):
            center_i = i + center_offset
            center_j = j + center_offset
            center_ray_idx = center_i * W + center_j
            
            # 为中心ray设置随机权重和mask
            blending_weights_valid[center_ray_idx] = torch.rand(N_coarse, N_views, 1, device=device)
            coarse_mask[center_ray_idx] = torch.ones(N_coarse, N_views, 1, device=device)
            
            # 随机设置一些views为无效
            invalid_views = torch.randperm(N_views, device=device)[:2]
            coarse_mask[center_ray_idx, :, invalid_views, :] = 0
    
    N_total = N_coarse + N_importance
    fine_mask = torch.ones(N_rays, N_total, N_views, 1, device=device)
    
    print(f"有效center rays数量: {(blending_weights_valid.sum(dim=(1,2,3)) > 0).sum().item()}")
    
    # 测试1: 标准pruning (作为baseline)
    print("\n1. 测试标准pruning...")
    start_time = time.time()
    result_standard = apply_source_view_pruning_optimized(
        z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, top_k
    )
    standard_time = time.time() - start_time
    print(f"标准pruning耗时: {standard_time:.4f}s")
    
    # 测试2: 稀疏pruning
    print("\n2. 测试稀疏pruning...")
    start_time = time.time()
    result_sparse = apply_source_view_pruning_sparse(
        z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, 
        H, W, window_size, top_k
    )
    sparse_time = time.time() - start_time
    print(f"稀疏pruning耗时: {sparse_time:.4f}s")
    
    # 测试3: 向量化稀疏pruning
    print("\n3. 测试向量化稀疏pruning...")
    start_time = time.time()
    result_sparse_vec = apply_source_view_pruning_sparse_vectorized(
        z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, 
        H, W, window_size, top_k
    )
    sparse_vec_time = time.time() - start_time
    print(f"向量化稀疏pruning耗时: {sparse_vec_time:.4f}s")
    
    # 验证结果一致性
    print("\n=== 结果验证 ===")
    
    # 对于有有效weights的rays，结果应该相同
    center_indices = []
    for i in range(0, H, window_size):
        for j in range(0, W, window_size):
            center_i = i + center_offset
            center_j = j + center_offset
            center_ray_idx = center_i * W + center_j
            center_indices.append(center_ray_idx)
    
    center_indices = torch.tensor(center_indices, device=device)
    
    # 检查中心rays的结果
    diff_sparse = torch.abs(result_standard[center_indices] - result_sparse[center_indices]).max().item()
    diff_sparse_vec = torch.abs(result_standard[center_indices] - result_sparse_vec[center_indices]).max().item()
    
    print(f"中心rays标准vs稀疏最大差异: {diff_sparse}")
    print(f"中心rays标准vs向量化稀疏最大差异: {diff_sparse_vec}")
    
    # 检查window内共享是否正确
    window_sharing_correct = True
    for i in range(0, H, window_size):
        for j in range(0, W, window_size):
            center_i = i + center_offset
            center_j = j + center_offset
            center_ray_idx = center_i * W + center_j
            
            # 检查window内所有rays是否与中心相同
            for wi in range(i, i + window_size):
                for wj in range(j, j + window_size):
                    ray_idx = wi * W + wj
                    
                    # 与fine_mask进行AND后的结果应该一致
                    expected = result_sparse[center_ray_idx] * fine_mask[ray_idx] / fine_mask[center_ray_idx]
                    expected = torch.where(torch.isnan(expected), torch.zeros_like(expected), expected)
                    
                    actual = result_sparse[ray_idx]
                    if torch.abs(expected - actual).max() > 1e-6:
                        window_sharing_correct = False
                        break
                if not window_sharing_correct:
                    break
            if not window_sharing_correct:
                break
        if not window_sharing_correct:
            break
    
    print(f"Window共享正确性: {'✓' if window_sharing_correct else '✗'}")
    
    # 性能总结
    print(f"\n=== 性能总结 ===")
    print(f"标准pruning: {standard_time:.4f}s")
    print(f"稀疏pruning: {sparse_time:.4f}s (加速比: {standard_time/sparse_time:.2f}x)")
    print(f"向量化稀疏: {sparse_vec_time:.4f}s (加速比: {standard_time/sparse_vec_time:.2f}x)")
    
    # 计算理论加速比
    num_windows = (H // window_size) * (W // window_size)
    theoretical_speedup = N_rays / num_windows
    print(f"理论加速比: {theoretical_speedup:.2f}x (只处理{num_windows}个中心rays而非{N_rays}个)")
    
    return {
        'standard_time': standard_time,
        'sparse_time': sparse_time,
        'sparse_vec_time': sparse_vec_time,
        'max_diff_sparse': diff_sparse,
        'max_diff_sparse_vec': diff_sparse_vec,
        'window_sharing_correct': window_sharing_correct
    }

def test_different_sizes():
    """
    测试不同图像尺寸的性能
    """
    print("\n=== 不同尺寸性能测试 ===")
    
    sizes = [
        (10, 10, 5),   # 小尺寸
        (20, 20, 5),   # 中等尺寸
        (40, 40, 5),   # 大尺寸
        (20, 30, 5),   # 非正方形
    ]
    
    results = []
    for H, W, window_size in sizes:
        if H % window_size != 0 or W % window_size != 0:
            print(f"跳过尺寸 {H}x{W} (不能被{window_size}整除)")
            continue
            
        print(f"\n测试尺寸: {H}x{W}")
        N_rays = H * W
        
        # 快速测试
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        z_vals_coarse = torch.randn(N_rays, 32, device=device)
        z_samples = torch.randn(N_rays, 32, device=device)
        blending_weights_valid = torch.rand(N_rays, 32, 8, 1, device=device)
        coarse_mask = torch.ones(N_rays, 32, 8, 1, device=device)
        fine_mask = torch.ones(N_rays, 64, 8, 1, device=device)
        
        # 只测试稀疏版本
        start_time = time.time()
        _ = apply_source_view_pruning_sparse_vectorized(
            z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, 
            H, W, window_size, 6
        )
        elapsed = time.time() - start_time
        
        num_windows = (H // window_size) * (W // window_size)
        speedup_ratio = N_rays / num_windows
        
        print(f"  耗时: {elapsed:.4f}s, 理论加速比: {speedup_ratio:.2f}x")
        results.append((H, W, elapsed, speedup_ratio))
    
    return results

if __name__ == "__main__":
    test_sparse_pruning()
    test_different_sizes()

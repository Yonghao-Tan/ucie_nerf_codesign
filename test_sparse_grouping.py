#!/usr/bin/env python3

import torch
import numpy as np
import time
from ibrnet.pruning_utils import apply_source_view_pruning_sparse_vectorized

def test_sparse_with_grouping():
    """测试sparse pruning + sample point grouping功能"""
    print("=== Testing Sparse Pruning + Sample Point Grouping ===")
    
    # 设置测试参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    H, W = 10, 15  # 150 rays total
    window_size = 5
    N_coarse = 8
    N_importance = 12
    N_total = N_coarse + N_importance
    N_views = 8
    top_k = 6
    group_size = 4
    
    N_rays = H * W
    
    print(f"Device: {device}")
    print(f"H={H}, W={W}, N_rays={N_rays}")
    print(f"N_coarse={N_coarse}, N_importance={N_importance}, N_total={N_total}")
    print(f"Window size: {window_size}, Group size: {group_size}")
    
    # 生成测试数据
    torch.manual_seed(42)
    z_vals_coarse = torch.sort(torch.rand(N_rays, N_coarse, device=device) * 10)[0]
    z_samples = torch.sort(torch.rand(N_rays, N_importance, device=device) * 10)[0]
    
    # 模拟真实的blending weights (从coarse网络得到)
    blending_weights_valid = torch.rand(N_rays, N_coarse, N_views, 1, device=device)
    blending_weights_valid = torch.softmax(blending_weights_valid.squeeze(-1), dim=-1).unsqueeze(-1)
    
    # 模拟mask (大部分为1，少部分为0)
    mask_coarse = torch.ones(N_rays, N_coarse, N_views, 1, device=device)
    mask_coarse[torch.rand_like(mask_coarse) < 0.1] = 0  # 10%的位置设为0
    
    fine_mask = torch.ones(N_rays, N_total, N_views, 1, device=device)
    fine_mask[torch.rand_like(fine_mask) < 0.05] = 0  # 5%的位置设为0
    
    print(f"Created test data with shapes:")
    print(f"  z_vals_coarse: {z_vals_coarse.shape}")
    print(f"  z_samples: {z_samples.shape}")
    print(f"  blending_weights_valid: {blending_weights_valid.shape}")
    print(f"  mask_coarse: {mask_coarse.shape}")
    print(f"  fine_mask: {fine_mask.shape}")
    
    # Test 1: Sparse pruning without grouping
    print("\n--- Test 1: Sparse pruning (no grouping) ---")
    start_time = time.time()
    
    result_no_group = apply_source_view_pruning_sparse_vectorized(
        z_vals_coarse, z_samples, blending_weights_valid, mask_coarse, fine_mask, 
        H, W, window_size=window_size, top_k=top_k, sample_point_group_size=None
    )
    
    time_no_group = time.time() - start_time
    print(f"Time without grouping: {time_no_group:.4f}s")
    print(f"Result shape: {result_no_group.shape}")
    
    # Test 2: Sparse pruning with grouping
    print(f"\n--- Test 2: Sparse pruning with sample point grouping (group_size={group_size}) ---")
    start_time = time.time()
    
    result_with_group = apply_source_view_pruning_sparse_vectorized(
        z_vals_coarse, z_samples, blending_weights_valid, mask_coarse, fine_mask, 
        H, W, window_size=window_size, top_k=top_k, sample_point_group_size=group_size
    )
    
    time_with_group = time.time() - start_time
    print(f"Time with grouping: {time_with_group:.4f}s")
    print(f"Result shape: {result_with_group.shape}")
    
    # Test 3: 验证grouping行为
    print(f"\n--- Test 3: Verify grouping behavior ---")
    
    # 检查几个不同rays的grouping效果
    test_ray_indices = [0, 37, 74, 149]  # 分布在不同位置的rays
    
    for ray_idx in test_ray_indices:
        print(f"\nRay {ray_idx}:")
        num_complete_groups = N_total // group_size
        
        group_consistent_count = 0
        for group_idx in range(num_complete_groups):
            start_idx = group_idx * group_size
            end_idx = start_idx + group_size
            
            # 获取组内第一个sample的mask
            first_mask = result_with_group[ray_idx, start_idx, :, :] 
            
            # 检查组内所有samples是否都有相同的mask
            group_masks = result_with_group[ray_idx, start_idx:end_idx, :, :]
            
            # 比较所有mask是否相同
            masks_equal_list = [torch.equal(group_masks[i], first_mask) for i in range(group_size)]
            masks_equal = all(masks_equal_list)
            
            if masks_equal:
                group_consistent_count += 1
            
            print(f"  Group {group_idx} (samples {start_idx}-{end_idx-1}): consistent = {masks_equal}")
        
        print(f"  Total consistent groups: {group_consistent_count}/{num_complete_groups}")
    
    # Test 4: 统计分析
    print(f"\n--- Test 4: Statistical analysis ---")
    
    # 计算pruning效果
    original_valid_views = fine_mask.sum().item()
    pruned_no_group = result_no_group.sum().item()
    pruned_with_group = result_with_group.sum().item()
    
    print(f"Original valid source views: {original_valid_views}")
    print(f"After sparse pruning (no group): {pruned_no_group} ({pruned_no_group/original_valid_views*100:.1f}%)")
    print(f"After sparse pruning (with group): {pruned_with_group} ({pruned_with_group/original_valid_views*100:.1f}%)")
    
    # 计算差异
    diff_mask = (result_no_group != result_with_group).sum().item()
    total_elements = result_no_group.numel()
    print(f"Different elements: {diff_mask}/{total_elements} ({diff_mask/total_elements*100:.2f}%)")
    
    print("\n=== Sparse Pruning + Sample Point Grouping Test Complete ===")
    return True

if __name__ == "__main__":
    test_sparse_with_grouping()

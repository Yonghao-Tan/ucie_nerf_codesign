#!/usr/bin/env python3

import torch
import numpy as np
import time
from ibrnet.pruning_utils import apply_source_view_pruning

def test_sample_point_grouping():
    """测试sample point grouping功能"""
    print("=== Testing Sample Point Grouping ===")
    
    # 设置测试参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_rays = 100
    N_coarse = 8
    N_importance = 16
    N_total = N_coarse + N_importance
    N_views = 8
    top_k = 6
    group_size = 4
    
    print(f"Device: {device}")
    print(f"N_rays: {N_rays}, N_coarse: {N_coarse}, N_importance: {N_importance}, N_total: {N_total}")
    print(f"Group size: {group_size}")
    
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
    
    # Test 1: 无grouping的标准pruning
    print("\n--- Test 1: Standard pruning (no grouping) ---")
    start_time = time.time()
    
    result_no_group = apply_source_view_pruning(
        z_vals_coarse, z_samples, blending_weights_valid, mask_coarse, fine_mask, 
        top_k=top_k, sample_point_group_size=None
    )
    
    time_no_group = time.time() - start_time
    print(f"Time without grouping: {time_no_group:.4f}s")
    print(f"Result shape: {result_no_group.shape}")
    
    # Test 2: 带grouping的pruning
    print(f"\n--- Test 2: Pruning with sample point grouping (group_size={group_size}) ---")
    start_time = time.time()
    
    result_with_group = apply_source_view_pruning(
        z_vals_coarse, z_samples, blending_weights_valid, mask_coarse, fine_mask, 
        top_k=top_k, sample_point_group_size=group_size
    )
    
    time_with_group = time.time() - start_time
    print(f"Time with grouping: {time_with_group:.4f}s")
    print(f"Result shape: {result_with_group.shape}")
    
    # Test 3: 验证grouping行为
    print(f"\n--- Test 3: Verify grouping behavior ---")
    
    # 检查第一个ray的grouping效果
    ray_idx = 0
    num_complete_groups = N_total // group_size
    print(f"Number of complete groups per ray: {num_complete_groups}")
    
    all_groups_correct = True
    for group_idx in range(num_complete_groups):
        start_idx = group_idx * group_size
        end_idx = start_idx + group_size
        
        # 获取组内第一个sample的mask
        first_mask = result_with_group[ray_idx, start_idx, :, :] 
        
        # 检查组内所有samples是否都有相同的mask
        group_masks = result_with_group[ray_idx, start_idx:end_idx, :, :]
        
        # 比较所有mask是否相同
        masks_equal = torch.all(torch.eq(group_masks, first_mask.unsqueeze(0)))
        
        print(f"  Group {group_idx} (samples {start_idx}-{end_idx-1}): masks_equal = {masks_equal.item()}")
        
        if not masks_equal:
            all_groups_correct = False
            print(f"    First mask sum: {first_mask.sum().item()}")
            for i in range(group_size):
                sample_mask = group_masks[i]
                print(f"    Sample {start_idx+i} mask sum: {sample_mask.sum().item()}")
    
    print(f"All groups have consistent masks: {all_groups_correct}")
    
    # Test 4: 统计分析
    print(f"\n--- Test 4: Statistical analysis ---")
    
    # 计算pruning效果
    original_valid_views = fine_mask.sum().item()
    pruned_no_group = result_no_group.sum().item()
    pruned_with_group = result_with_group.sum().item()
    
    print(f"Original valid source views: {original_valid_views}")
    print(f"After pruning (no group): {pruned_no_group} ({pruned_no_group/original_valid_views*100:.1f}%)")
    print(f"After pruning (with group): {pruned_with_group} ({pruned_with_group/original_valid_views*100:.1f}%)")
    
    # Test 5: 验证group size边界情况
    print(f"\n--- Test 5: Edge cases ---")
    
    # 测试group_size=1 (应该和no grouping相同)
    result_group_1 = apply_source_view_pruning(
        z_vals_coarse, z_samples, blending_weights_valid, mask_coarse, fine_mask, 
        top_k=top_k, sample_point_group_size=1
    )
    
    group_1_same_as_no_group = torch.allclose(result_group_1, result_no_group)
    print(f"Group size 1 same as no grouping: {group_1_same_as_no_group}")
    
    # 测试大的group_size
    large_group_size = N_total + 5  # 比总数还大
    result_large_group = apply_source_view_pruning(
        z_vals_coarse, z_samples, blending_weights_valid, mask_coarse, fine_mask, 
        top_k=top_k, sample_point_group_size=large_group_size
    )
    
    print(f"Large group size ({large_group_size}) handled gracefully: {result_large_group.shape == result_no_group.shape}")
    
    print("\n=== Sample Point Grouping Test Complete ===")
    return True

if __name__ == "__main__":
    test_sample_point_grouping()

#!/usr/bin/env python3

import torch
import numpy as np

def test_grouping_logic_directly():
    """直接测试grouping逻辑，不涉及complex pruning"""
    print("=== Testing Grouping Logic Directly ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_rays = 2
    N_total = 8
    N_views = 4
    group_size = 4
    
    # 创建简单的测试mask - pruned_mask
    pruned_mask = torch.zeros(N_rays, N_total, N_views, 1, device=device)
    
    # 为每个sample设置不同的pattern，这样便于验证grouping
    for ray_idx in range(N_rays):
        for sample_idx in range(N_total):
            # 每个sample有不同数量的active views
            num_active = (sample_idx % N_views) + 1
            pruned_mask[ray_idx, sample_idx, :num_active, 0] = 1.0
    
    print("Original pruned mask:")
    for ray_idx in range(N_rays):
        print(f"Ray {ray_idx}:")
        for sample_idx in range(N_total):
            active_views = pruned_mask[ray_idx, sample_idx, :, 0].sum().item()
            print(f"  Sample {sample_idx}: {active_views} active views")
    
    # 应用grouping
    print(f"\nApplying grouping with group_size={group_size}...")
    
    for ray_idx in range(N_rays):
        num_complete_groups = N_total // group_size
        print(f"Ray {ray_idx} has {num_complete_groups} complete groups")
        
        for group_idx in range(num_complete_groups):
            start_idx = group_idx * group_size
            end_idx = start_idx + group_size
            
            print(f"  Processing group {group_idx} (samples {start_idx}-{end_idx-1})")
            
            # 使用组内第一个sample的pruning mask
            first_sample_mask = pruned_mask[ray_idx, start_idx, :, :].clone()
            print(f"    First sample ({start_idx}) has {first_sample_mask.sum().item()} active views")
            
            # 将第一个sample的mask应用到组内所有samples
            pruned_mask[ray_idx, start_idx:end_idx, :, :] = first_sample_mask.unsqueeze(0)
            
            # 验证组内一致性
            for i in range(group_size):
                sample_mask = pruned_mask[ray_idx, start_idx + i, :, :]
                is_same = torch.equal(sample_mask, first_sample_mask)
                print(f"    Sample {start_idx + i}: {sample_mask.sum().item()} active views, same_as_first={is_same}")
    
    print("\nAfter grouping:")
    for ray_idx in range(N_rays):
        print(f"Ray {ray_idx}:")
        for sample_idx in range(N_total):
            active_views = pruned_mask[ray_idx, sample_idx, :, 0].sum().item()
            print(f"  Sample {sample_idx}: {active_views} active views")
    
    # 现在创建fine_mask并应用AND操作
    print(f"\nCreating fine_mask and applying AND operation...")
    
    fine_mask = torch.ones_like(pruned_mask)
    # 在某些位置设置为0来模拟fine_mask的约束
    fine_mask[0, 1, 1, 0] = 0  # Ray 0, Sample 1, View 1
    fine_mask[0, 2, 2, 0] = 0  # Ray 0, Sample 2, View 2
    fine_mask[1, 5, 0, 0] = 0  # Ray 1, Sample 5, View 0
    
    print("Fine mask constraints:")
    for ray_idx in range(N_rays):
        for sample_idx in range(N_total):
            zero_positions = torch.where(fine_mask[ray_idx, sample_idx, :, 0] == 0)[0]
            if len(zero_positions) > 0:
                print(f"  Ray {ray_idx}, Sample {sample_idx}: views {zero_positions.tolist()} set to 0")
    
    # 应用AND操作
    final_mask = pruned_mask * fine_mask
    
    print("\nFinal mask after AND operation:")
    for ray_idx in range(N_rays):
        print(f"Ray {ray_idx}:")
        for sample_idx in range(N_total):
            active_views = final_mask[ray_idx, sample_idx, :, 0].sum().item()
            print(f"  Sample {sample_idx}: {active_views} active views")
    
    # 验证grouping的效果是否被fine_mask破坏
    print(f"\nVerifying group consistency after AND operation:")
    for ray_idx in range(N_rays):
        num_complete_groups = N_total // group_size
        for group_idx in range(num_complete_groups):
            start_idx = group_idx * group_size
            end_idx = start_idx + group_size
            
            print(f"  Ray {ray_idx}, Group {group_idx} (samples {start_idx}-{end_idx-1}):")
            first_final_mask = final_mask[ray_idx, start_idx, :, :]
            
            all_same = True
            for i in range(group_size):
                sample_final_mask = final_mask[ray_idx, start_idx + i, :, :]
                is_same = torch.equal(sample_final_mask, first_final_mask)
                if not is_same:
                    all_same = False
                print(f"    Sample {start_idx + i}: same_as_first={is_same}")
            
            print(f"    Group consistent: {all_same}")

if __name__ == "__main__":
    test_grouping_logic_directly()

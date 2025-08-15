#!/usr/bin/env python3

"""
Sample Point Grouping 功能演示

这个脚本展示了如何使用新的sample point grouping功能，
包括标准pruning和稀疏pruning两种模式。
"""

import torch
import time
from ibrnet.pruning_utils import (
    apply_source_view_pruning,
    apply_source_view_pruning_sparse_vectorized
)

def demo_sample_point_grouping():
    """演示sample point grouping功能"""
    
    print("=" * 60)
    print("IBRNet Sample Point Grouping 功能演示")
    print("=" * 60)
    
    # 设置参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_rays = 200
    N_coarse = 8
    N_importance = 16
    N_total = N_coarse + N_importance
    N_views = 8
    top_k = 6
    
    print(f"测试环境: {device}")
    print(f"Rays: {N_rays}, Coarse: {N_coarse}, Fine: {N_importance}, Total: {N_total}")
    print(f"Source views: {N_views}, Top-k: {top_k}")
    
    # 生成测试数据
    torch.manual_seed(42)
    z_vals_coarse = torch.sort(torch.rand(N_rays, N_coarse, device=device) * 10)[0]
    z_samples = torch.sort(torch.rand(N_rays, N_importance, device=device) * 10)[0]
    blending_weights_valid = torch.rand(N_rays, N_coarse, N_views, 1, device=device)
    blending_weights_valid = torch.softmax(blending_weights_valid.squeeze(-1), dim=-1).unsqueeze(-1)
    mask_coarse = torch.ones(N_rays, N_coarse, N_views, 1, device=device)
    fine_mask = torch.ones(N_rays, N_total, N_views, 1, device=device)
    
    # 添加一些随机的mask约束
    mask_coarse[torch.rand_like(mask_coarse) < 0.1] = 0
    fine_mask[torch.rand_like(fine_mask) < 0.05] = 0
    
    print("\n" + "=" * 40)
    print("1. 标准 Source View Pruning 测试")
    print("=" * 40)
    
    # 测试不同的group sizes
    group_sizes = [None, 1, 2, 4, 8]
    
    results = {}
    for group_size in group_sizes:
        print(f"\n--- Group Size: {group_size} ---")
        
        start_time = time.time()
        result = apply_source_view_pruning(
            z_vals_coarse, z_samples, blending_weights_valid, mask_coarse, fine_mask,
            top_k=top_k, sample_point_group_size=group_size
        )
        elapsed_time = time.time() - start_time
        
        active_views = result.sum().item()
        total_views = result.numel()
        
        print(f"执行时间: {elapsed_time:.4f}s")
        print(f"Active views: {active_views}/{total_views} ({active_views/total_views*100:.1f}%)")
        
        results[group_size] = {
            'result': result,
            'time': elapsed_time,
            'active_views': active_views
        }
    
    print("\n" + "=" * 40)
    print("2. 稀疏 Source View Pruning 测试")
    print("=" * 40)
    
    # 稀疏测试参数
    H, W = 10, 20  # 200 rays total
    window_size = 5
    
    print(f"稀疏模式: H={H}, W={W}, window_size={window_size}")
    
    sparse_results = {}
    for group_size in [None, 4]:
        print(f"\n--- 稀疏模式 Group Size: {group_size} ---")
        
        start_time = time.time()
        result = apply_source_view_pruning_sparse_vectorized(
            z_vals_coarse, z_samples, blending_weights_valid, mask_coarse, fine_mask,
            H, W, window_size=window_size, top_k=top_k, sample_point_group_size=group_size
        )
        elapsed_time = time.time() - start_time
        
        active_views = result.sum().item()
        total_views = result.numel()
        
        print(f"执行时间: {elapsed_time:.4f}s")
        print(f"Active views: {active_views}/{total_views} ({active_views/total_views*100:.1f}%)")
        
        sparse_results[group_size] = {
            'result': result,
            'time': elapsed_time,
            'active_views': active_views
        }
    
    print("\n" + "=" * 40)
    print("3. Grouping 一致性验证")
    print("=" * 40)
    
    # 验证grouping效果
    group_size = 4
    result_grouped = results[group_size]['result']
    
    print(f"验证group_size={group_size}的一致性效果:")
    
    consistent_groups = 0
    total_groups = 0
    
    # 检查前几个rays
    for ray_idx in range(min(5, N_rays)):
        num_complete_groups = N_total // group_size
        ray_consistent_groups = 0
        
        for group_idx in range(num_complete_groups):
            start_idx = group_idx * group_size
            end_idx = start_idx + group_size
            
            # 检查组内第一个sample与其他samples的一致性
            first_mask = result_grouped[ray_idx, start_idx, :, :]
            group_consistent = True
            
            for sample_idx in range(start_idx + 1, end_idx):
                sample_mask = result_grouped[ray_idx, sample_idx, :, :]
                if not torch.equal(first_mask, sample_mask):
                    group_consistent = False
                    break
            
            if group_consistent:
                ray_consistent_groups += 1
            total_groups += 1
        
        consistent_groups += ray_consistent_groups
        print(f"  Ray {ray_idx}: {ray_consistent_groups}/{num_complete_groups} groups consistent")
    
    print(f"\n总体一致性: {consistent_groups}/{total_groups} groups ({consistent_groups/total_groups*100:.1f}%)")
    print("注意: 由于fine_mask的AND操作，某些组可能不完全一致，这是预期行为")
    
    print("\n" + "=" * 40)
    print("4. 性能对比总结")
    print("=" * 40)
    
    baseline_time = results[None]['time']
    
    print("标准模式性能对比:")
    for group_size in group_sizes:
        if group_size is None:
            print(f"  No grouping: {results[group_size]['time']:.4f}s (baseline)")
        else:
            speedup = baseline_time / results[group_size]['time']
            print(f"  Group size {group_size}: {results[group_size]['time']:.4f}s ({speedup:.2f}x)")
    
    print("\n稀疏模式性能对比:")
    sparse_baseline = sparse_results[None]['time']
    for group_size in [None, 4]:
        if group_size is None:
            print(f"  No grouping: {sparse_results[group_size]['time']:.4f}s (baseline)")
        else:
            speedup = sparse_baseline / sparse_results[group_size]['time']
            print(f"  Group size {group_size}: {sparse_results[group_size]['time']:.4f}s ({speedup:.2f}x)")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    
    return results, sparse_results

if __name__ == "__main__":
    demo_sample_point_grouping()

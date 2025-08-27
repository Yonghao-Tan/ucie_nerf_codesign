#!/usr/bin/env python3
"""
测试动态threshold-based pruning功能
"""

import torch
import sys
import os

# 添加路径以便导入模块
sys.path.append('/home/ytanaz/access/IBRNet')

from ibrnet.pruning_utils.source_view_pruning import apply_source_view_pruning_2x2_windows_threshold_based

def test_dynamic_threshold_pruning():
    """
    测试动态阈值pruning功能
    """
    print("=== 测试动态threshold-based pruning ===")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 测试参数 - 确保可以被2x window_size整除
    H, W = 20, 20  # 2x5 = 10，所以 H=W=20 可以被10整除
    window_size = 5
    N_rays = H * W  # 400
    N_coarse = 8
    N_importance = 40
    N_total = N_coarse + N_importance  # 48
    N_views = 8
    sample_point_group_size = 8  # 48/8 = 6 groups per ray
    
    print(f"Test dimensions: H={H}, W={W}, N_rays={N_rays}")
    print(f"N_coarse={N_coarse}, N_importance={N_importance}, N_total={N_total}")
    print(f"N_views={N_views}, group_size={sample_point_group_size}")
    print(f"Groups per ray: {N_total // sample_point_group_size}")
    print(f"Total groups: {N_rays * (N_total // sample_point_group_size) // (4 * 4)} = {N_rays * (N_total // sample_point_group_size) // 16}")
    
    # 创建测试数据
    z_vals_coarse = torch.linspace(0.1, 1.0, N_coarse, device=device).unsqueeze(0).expand(N_rays, -1)
    z_samples = torch.linspace(1.1, 2.0, N_importance, device=device).unsqueeze(0).expand(N_rays, -1)
    
    # 创建模拟的blending weights - 有些权重大于阈值，有些小于阈值
    blending_weights_valid = torch.zeros(N_rays, N_coarse, N_views, 1, device=device)
    
    # 为不同rays设置不同的权重分布，模拟真实场景
    for ray_idx in range(N_rays):
        for sample_idx in range(N_coarse):
            # 随机生成权重，有些大于0.08，有些小于0.08
            weights = torch.rand(N_views, device=device) * 0.15  # 0到0.15之间
            weights = weights / weights.sum()  # 归一化
            blending_weights_valid[ray_idx, sample_idx, :, 0] = weights
    
    # 创建coarse_mask（所有都有效）
    coarse_mask = torch.ones(N_rays, N_coarse, N_views, 1, device=device)
    
    # 创建fine_mask（所有都有效）
    fine_mask = torch.ones(N_rays, N_total, N_views, 1, device=device)
    
    # 测试不同的阈值
    thresholds = [0.05, 0.08, 0.12, 0.15]
    
    for threshold in thresholds:
        print(f"\n--- 测试阈值 {threshold} ---")
        
        try:
            result_mask = apply_source_view_pruning_2x2_windows_threshold_based(
                z_vals_coarse=z_vals_coarse,
                z_samples=z_samples, 
                blending_weights_valid=blending_weights_valid,
                coarse_mask=coarse_mask,
                fine_mask=fine_mask,
                H=H, W=W,
                window_size=window_size,
                sample_point_group_size=sample_point_group_size,
                weight_threshold=threshold
            )
            
            print(f"输出shape: {result_mask.shape}")
            print(f"非零元素比例: {result_mask.sum().item() / result_mask.numel():.3f}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n=== 动态threshold pruning测试完成 ===")

if __name__ == "__main__":
    test_dynamic_threshold_pruning()

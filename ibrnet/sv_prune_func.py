#!/usr/bin/env python3
"""
简单可视化IBRNet中coarse和fine采样点的深度分布
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_depth_samples(z_vals_coarse, z_samples, save_path="depth_samples_distribution.png"):
    """
    可视化coarse和fine采样点的深度分布
    
    Args:
        z_vals_coarse: coarse网络的深度采样点，shape: [N_rays, N_samples] 或 [N_samples]
        z_samples: fine网络新增的深度采样点，shape: [N_rays, N_importance] 或 [N_importance]
        save_path: 保存图片的路径
    """
    
    # 转换为numpy数组
    if isinstance(z_vals_coarse, torch.Tensor):
        z_coarse = z_vals_coarse.detach().cpu().numpy().flatten()
    else:
        z_coarse = np.array(z_vals_coarse).flatten()
        
    if isinstance(z_samples, torch.Tensor):
        z_fine = z_samples.detach().cpu().numpy().flatten()
    else:
        z_fine = np.array(z_samples).flatten()
    
    # 创建图形
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # 图1: 散点图显示采样点位置
    ax1.scatter(z_coarse, np.zeros_like(z_coarse), 
               c='blue', s=30, alpha=0.7, marker='|', label=f'Coarse samples ({len(z_coarse)})')
    ax1.scatter(z_fine, np.ones(len(z_fine)) * 0.1, 
               c='red', s=20, alpha=0.7, marker='x', label=f'Fine samples ({len(z_fine)})')
    
    ax1.set_ylim(-0.05, 0.15)
    ax1.set_xlabel('Depth (meters)')
    ax1.set_ylabel('Sample Type')
    ax1.set_title('Depth Sampling Points Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 直方图对比
    depth_min = min(z_coarse.min(), z_fine.min())
    depth_max = max(z_coarse.max(), z_fine.max())
    bins = np.linspace(depth_min, depth_max, 50)
    
    ax2.hist(z_coarse, bins=bins, alpha=0.7, color='blue', 
             label=f'Coarse samples', density=True, edgecolor='black', linewidth=0.5)
    ax2.hist(z_fine, bins=bins, alpha=0.7, color='red', 
             label=f'Fine samples', density=True, edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Depth (meters)')
    ax2.set_ylabel('Density')
    ax2.set_title('Depth Distribution Histogram')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: 合并后的采样点分布
    z_combined = np.concatenate([z_coarse, z_fine])
    z_combined_sorted = np.sort(z_combined)
    
    # 标记哪些是coarse点，哪些是fine点
    coarse_mask = np.isin(z_combined_sorted, z_coarse)
    fine_mask = ~coarse_mask
    
    coarse_positions = z_combined_sorted[coarse_mask]
    fine_positions = z_combined_sorted[fine_mask]
    
    ax3.scatter(coarse_positions, np.zeros_like(coarse_positions), 
               c='blue', s=30, alpha=0.7, marker='|', label=f'Coarse samples')
    ax3.scatter(fine_positions, np.zeros_like(fine_positions), 
               c='red', s=20, alpha=0.7, marker='x', label=f'Fine samples')
    
    ax3.set_ylim(-0.05, 0.05)
    ax3.set_xlabel('Depth (meters)')
    ax3.set_ylabel('Combined')
    ax3.set_title(f'Combined Sampling Points (Total: {len(z_combined)})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存到: {save_path}")
    
    # 打印统计信息
    print("\n=== 采样统计 ===")
    print(f"Coarse samples: {len(z_coarse)}")
    print(f"Fine samples: {len(z_fine)}")
    print(f"Total samples: {len(z_combined)}")
    print(f"Depth range: {depth_min:.3f} - {depth_max:.3f} meters")
    print(f"Coarse depth range: {z_coarse.min():.3f} - {z_coarse.max():.3f}")
    print(f"Fine depth range: {z_fine.min():.3f} - {z_fine.max():.3f}")
    
    return fig

# 测试函数
if __name__ == "__main__":
    # 创建一些测试数据
    print("创建测试数据...")
    
    # 模拟coarse采样点 (均匀分布)
    N_coarse = 64
    z_vals_coarse_test = torch.linspace(2.0, 6.0, N_coarse)
    
    # 模拟fine采样点 (在某些区域密集采样)
    N_importance = 64
    # 在depth 3.0-3.5和4.5-5.0区域密集采样
    z_samples_test1 = torch.rand(N_importance//2) * 0.5 + 3.0  # 3.0-3.5
    z_samples_test2 = torch.rand(N_importance//2) * 0.5 + 4.5  # 4.5-5.0
    z_samples_test = torch.cat([z_samples_test1, z_samples_test2])
    
    # 可视化
    visualize_depth_samples(z_vals_coarse_test, z_samples_test, 
                           save_path="/home/ytanaz/access/IBRNet/test_depth_distribution.png")

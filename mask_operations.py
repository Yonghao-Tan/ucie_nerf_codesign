#!/usr/bin/env python3
"""
实现IBRNet中coarse到fine采样点的mask共享和pruning功能
"""

import torch
import numpy as np

def find_fine_sample_assignment(z_vals_coarse, z_samples):
    """
    找到每个fine采样点属于哪两个coarse采样点之间
    
    Args:
        z_vals_coarse: [N_rays, N_coarse_samples] coarse采样点深度
        z_samples: [N_rays, N_importance] fine采样点深度
    
    Returns:
        assignment: [N_rays, N_importance, 2] 每个fine点对应的两个coarse点的索引
        weights: [N_rays, N_importance, 2] 插值权重
    """
    N_rays, N_coarse = z_vals_coarse.shape
    N_rays2, N_importance = z_samples.shape
    assert N_rays == N_rays2, "Ray数量必须匹配"
    
    # 为每个fine sample找到对应的coarse interval
    assignment = torch.zeros(N_rays, N_importance, 2, dtype=torch.long, device=z_vals_coarse.device)
    weights = torch.zeros(N_rays, N_importance, 2, device=z_vals_coarse.device)
    
    for ray_idx in range(N_rays):
        z_coarse = z_vals_coarse[ray_idx]  # [N_coarse]
        z_fine = z_samples[ray_idx]        # [N_importance]
        
        for fine_idx in range(N_importance):
            z_target = z_fine[fine_idx]
            
            # 找到z_target在z_coarse中的位置
            # searchsorted找到插入位置
            insert_pos = torch.searchsorted(z_coarse, z_target)
            
            if insert_pos == 0:
                # 在最前面，使用前两个点
                left_idx, right_idx = 0, 1
            elif insert_pos >= N_coarse:
                # 在最后面，使用后两个点
                left_idx, right_idx = N_coarse-2, N_coarse-1
            else:
                # 在中间，使用前后两个点
                left_idx, right_idx = insert_pos-1, insert_pos
            
            assignment[ray_idx, fine_idx, 0] = left_idx
            assignment[ray_idx, fine_idx, 1] = right_idx
            
            # 计算插值权重
            z_left = z_coarse[left_idx]
            z_right = z_coarse[right_idx]
            
            if z_right - z_left > 1e-8:  # 避免除零
                w_right = (z_target - z_left) / (z_right - z_left)
                w_left = 1.0 - w_right
            else:
                w_left, w_right = 0.5, 0.5
            
            weights[ray_idx, fine_idx, 0] = w_left
            weights[ray_idx, fine_idx, 1] = w_right
    
    return assignment, weights

def create_pruned_mask(blending_weights_valid, coarse_mask, top_k=6):
    """
    基于blending_weights创建pruned mask，只保留top-k个source views
    
    Args:
        blending_weights_valid: [N_rays, N_coarse_samples, N_sourceviews, 1]
        coarse_mask: [N_rays, N_coarse_samples, N_sourceviews, 1] 
        top_k: 保留的top source views数量
    
    Returns:
        pruned_mask: [N_rays, N_coarse_samples, N_sourceviews, 1]
    """
    N_rays, N_coarse, N_views, _ = blending_weights_valid.shape
    
    # 对每个ray和每个sample point，找到top-k个source views
    pruned_mask = torch.zeros_like(coarse_mask)
    
    for ray_idx in range(N_rays):
        for sample_idx in range(N_coarse):
            # 获取当前点的权重 [N_views, 1]
            weights = blending_weights_valid[ray_idx, sample_idx, :, 0]  # [N_views]
            mask = coarse_mask[ray_idx, sample_idx, :, 0]               # [N_views]
            
            # 只考虑mask为True的views
            valid_views = mask.bool()
            if valid_views.sum() == 0:
                continue
                
            # 在valid views中找到top-k个权重最大的
            valid_weights = weights[valid_views]
            valid_indices = torch.where(valid_views)[0]
            
            # 选择top-k个（如果valid views少于k个，就全选）
            k = min(top_k, len(valid_indices))
            if k > 0:
                _, top_indices_in_valid = torch.topk(valid_weights, k)
                top_indices = valid_indices[top_indices_in_valid]
                
                # 设置pruned mask
                pruned_mask[ray_idx, sample_idx, top_indices, 0] = 1.0
    
    return pruned_mask

def extend_mask_to_fine(pruned_mask_coarse, assignment, fine_mask):
    """
    将coarse的pruned mask扩展到fine网络
    
    Args:
        pruned_mask_coarse: [N_rays, N_coarse_samples, N_sourceviews, 1]
        assignment: [N_rays, N_importance, 2] fine点对应的coarse点索引
        fine_mask: [N_rays, N_total_samples, N_sourceviews, 1] fine网络原始mask
    
    Returns:
        final_mask: [N_rays, N_total_samples, N_sourceviews, 1] 最终mask
    """
    N_rays, N_coarse, N_views, _ = pruned_mask_coarse.shape
    N_rays2, N_importance, _ = assignment.shape
    N_rays3, N_total, N_views2, _ = fine_mask.shape
    
    assert N_rays == N_rays2 == N_rays3, "Ray数量必须匹配"
    assert N_views == N_views2, "Source view数量必须匹配"
    assert N_total == N_coarse + N_importance, "Total samples数量不匹配"
    
    # 创建扩展的pruned mask
    extended_pruned_mask = torch.zeros_like(fine_mask)
    
    # 先复制coarse部分
    extended_pruned_mask[:, :N_coarse, :, :] = pruned_mask_coarse
    
    # 为fine部分分配mask（简单策略：使用左边coarse点的mask）
    for ray_idx in range(N_rays):
        for fine_idx in range(N_importance):
            left_coarse_idx = assignment[ray_idx, fine_idx, 0]  # 使用左边的coarse点
            
            # 复制左边coarse点的mask到对应的fine点
            fine_sample_idx = N_coarse + fine_idx
            extended_pruned_mask[ray_idx, fine_sample_idx, :, :] = \
                pruned_mask_coarse[ray_idx, left_coarse_idx, :, :]
    
    # 与原始fine mask进行AND运算
    final_mask = extended_pruned_mask * fine_mask
    
    return final_mask, extended_pruned_mask

def visualize_mask_assignment(z_vals_coarse, z_samples, assignment, ray_idx=0, save_path="mask_assignment.png"):
    """
    可视化fine采样点的assignment关系
    """
    import matplotlib.pyplot as plt
    
    z_coarse = z_vals_coarse[ray_idx].cpu().numpy()
    z_fine = z_samples[ray_idx].cpu().numpy()
    assign = assignment[ray_idx].cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 画coarse点
    ax.scatter(z_coarse, np.zeros_like(z_coarse), 
               c='blue', s=100, marker='|', label=f'Coarse samples ({len(z_coarse)})', alpha=0.8)
    
    # 给coarse点编号
    for i, z in enumerate(z_coarse):
        ax.text(z, 0.02, f'{i}', ha='center', fontsize=8, color='blue')
    
    # 画fine点和它们的assignment
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(z_fine)))
    for i, z in enumerate(z_fine):
        left_idx, right_idx = assign[i]
        ax.scatter(z, 0.05, c=[colors[i]], s=60, marker='x', alpha=0.8)
        
        # 画连接线显示assignment
        ax.plot([z_coarse[left_idx], z], [0, 0.05], 'r--', alpha=0.5, linewidth=1)
        ax.plot([z_coarse[right_idx], z], [0, 0.05], 'r--', alpha=0.5, linewidth=1)
        
        # 标注assignment
        ax.text(z, 0.07, f'({left_idx},{right_idx})', ha='center', fontsize=6, color='red')
    
    ax.scatter([], [], c='red', s=60, marker='x', label=f'Fine samples ({len(z_fine)})')
    
    ax.set_ylim(-0.02, 0.1)
    ax.set_xlabel('Depth (meters)')
    ax.set_title(f'Fine Sample Assignment to Coarse Intervals (Ray {ray_idx})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Assignment可视化已保存到: {save_path}")

# 测试函数
def test_mask_operations():
    """
    测试mask操作功能
    """
    print("=== 测试Mask操作功能 ===")
    
    # 创建测试数据
    N_rays = 2
    N_coarse = 8
    N_importance = 4
    N_views = 10
    
    # 模拟coarse采样点
    z_vals_coarse = torch.linspace(2.0, 6.0, N_coarse).unsqueeze(0).repeat(N_rays, 1)
    z_vals_coarse[1] = torch.linspace(1.5, 5.5, N_coarse)  # 第二条ray不同的深度范围
    
    # 模拟fine采样点（在coarse范围内）
    z_samples = torch.zeros(N_rays, N_importance)
    z_samples[0] = torch.tensor([2.5, 3.2, 4.1, 5.3])  # 在coarse范围内
    z_samples[1] = torch.tensor([2.0, 2.8, 3.5, 4.2])  # 在coarse范围内
    
    # 模拟blending weights和mask
    blending_weights_valid = torch.rand(N_rays, N_coarse, N_views, 1)
    coarse_mask = torch.ones(N_rays, N_coarse, N_views, 1)
    
    # 随机设置一些mask为0
    coarse_mask[0, :, 5:, :] = 0  # 第一条ray的后5个views无效
    coarse_mask[1, :, 7:, :] = 0  # 第二条ray的后3个views无效
    
    print(f"输入数据:")
    print(f"  z_vals_coarse shape: {z_vals_coarse.shape}")
    print(f"  z_samples shape: {z_samples.shape}")
    print(f"  blending_weights_valid shape: {blending_weights_valid.shape}")
    print(f"  coarse_mask shape: {coarse_mask.shape}")
    
    # 1. 找到fine采样点的assignment
    assignment, weights = find_fine_sample_assignment(z_vals_coarse, z_samples)
    print(f"\n1. Fine sample assignment:")
    print(f"  assignment shape: {assignment.shape}")
    print(f"  weights shape: {weights.shape}")
    print(f"  Ray 0 assignments: {assignment[0]}")
    
    # 2. 创建pruned mask
    pruned_mask = create_pruned_mask(blending_weights_valid, coarse_mask, top_k=6)
    print(f"\n2. Pruned mask:")
    print(f"  pruned_mask shape: {pruned_mask.shape}")
    print(f"  Ray 0, Sample 0 有效views: {pruned_mask[0, 0, :, 0].sum().item()}")
    
    # 3. 创建fine网络的原始mask
    N_total = N_coarse + N_importance
    fine_mask = torch.ones(N_rays, N_total, N_views, 1)
    fine_mask[:, :, 8:, :] = 0  # 模拟一些views在fine阶段无效
    
    # 4. 扩展mask到fine网络
    final_mask, extended_pruned_mask = extend_mask_to_fine(pruned_mask, assignment, fine_mask)
    print(f"\n3. 扩展到fine网络:")
    print(f"  final_mask shape: {final_mask.shape}")
    print(f"  extended_pruned_mask shape: {extended_pruned_mask.shape}")
    
    # 5. 可视化assignment
    visualize_mask_assignment(z_vals_coarse, z_samples, assignment, 
                            ray_idx=0, save_path="/home/ytanaz/access/IBRNet/mask_assignment.png")
    
    return {
        'assignment': assignment,
        'weights': weights,
        'pruned_mask': pruned_mask,
        'final_mask': final_mask,
        'extended_pruned_mask': extended_pruned_mask
    }

if __name__ == "__main__":
    results = test_mask_operations()
    print("\n测试完成!")

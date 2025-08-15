#!/usr/bin/env python3
"""
Source View Pruning 功能
包含原始版本和优化版本的pruning函数
"""

import torch

def apply_source_view_pruning_original(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, top_k=6):
    """
    原始版本的source view pruning (较慢，但易理解)
    
    Args:
        z_vals_coarse: [N_rays, N_coarse_samples] coarse采样点深度
        z_samples: [N_rays, N_importance] fine网络新增采样点深度  
        blending_weights_valid: [N_rays, N_coarse_samples, N_sourceviews, 1] coarse网络的混合权重
        coarse_mask: [N_rays, N_coarse_samples, N_sourceviews, 1] coarse网络的mask
        fine_mask: [N_rays, N_total_samples, N_sourceviews, 1] fine网络原始mask 
        top_k: 保留的top source views数量，默认6
    
    Returns:
        final_fine_mask: [N_rays, N_total_samples, N_sourceviews, 1] 经过pruning的final mask
    """
    
    N_rays, N_coarse = z_vals_coarse.shape
    N_rays2, N_importance = z_samples.shape  
    N_rays3, N_total, N_views, _ = fine_mask.shape
    
    assert N_rays == N_rays2 == N_rays3, "Ray数量必须匹配"
    assert N_total == N_coarse + N_importance, f"Total samples数量不匹配: {N_total} != {N_coarse} + {N_importance}"
    
    # Step 1: 创建基于权重的pruned mask (只针对coarse samples)
    pruned_mask_coarse = torch.zeros_like(coarse_mask)
    
    for ray_idx in range(N_rays):
        for sample_idx in range(N_coarse):
            # 获取当前点的权重和mask
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
                pruned_mask_coarse[ray_idx, sample_idx, top_indices, 0] = 1.0
    
    # Step 2: 找到fine采样点与coarse采样点的assignment关系
    assignment = torch.zeros(N_rays, N_importance, dtype=torch.long, device=z_vals_coarse.device)
    
    for ray_idx in range(N_rays):
        z_coarse = z_vals_coarse[ray_idx]  # [N_coarse]
        z_fine = z_samples[ray_idx]        # [N_importance]
        
        for fine_idx in range(N_importance):
            z_target = z_fine[fine_idx]
            
            # 找到z_target在z_coarse中的位置
            insert_pos = torch.searchsorted(z_coarse, z_target)
            
            if insert_pos == 0:
                # 在最前面，使用第一个coarse点
                nearest_coarse_idx = 0
            elif insert_pos >= N_coarse:
                # 在最后面，使用最后一个coarse点
                nearest_coarse_idx = N_coarse - 1
            else:
                # 在中间，选择距离更近的coarse点
                left_idx = insert_pos - 1
                right_idx = insert_pos
                
                dist_left = abs(z_target - z_coarse[left_idx])
                dist_right = abs(z_target - z_coarse[right_idx])
                
                nearest_coarse_idx = left_idx if dist_left <= dist_right else right_idx
            
            assignment[ray_idx, fine_idx] = nearest_coarse_idx
    
    # Step 3: 构建完整的pruned mask (coarse + fine)
    extended_pruned_mask = torch.zeros_like(fine_mask)
    
    # 复制coarse部分的pruned mask
    extended_pruned_mask[:, :N_coarse, :, :] = pruned_mask_coarse
    
    # 为fine部分分配mask (基于assignment)
    for ray_idx in range(N_rays):
        for fine_idx in range(N_importance):
            nearest_coarse_idx = assignment[ray_idx, fine_idx]
            fine_sample_idx = N_coarse + fine_idx
            
            # 复制最近coarse点的pruned mask到fine点
            extended_pruned_mask[ray_idx, fine_sample_idx, :, :] = \
                pruned_mask_coarse[ray_idx, nearest_coarse_idx, :, :]
    
    # Step 4: 与原始fine mask进行AND运算得到最终mask
    final_fine_mask = extended_pruned_mask * fine_mask
    
    return final_fine_mask


def apply_source_view_pruning_optimized(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, top_k=6):
    """
    优化版本的source view pruning (向量化操作，显著提速)
    
    Args:
        z_vals_coarse: [N_rays, N_coarse_samples] coarse采样点深度
        z_samples: [N_rays, N_importance] fine网络新增采样点深度  
        blending_weights_valid: [N_rays, N_coarse_samples, N_sourceviews, 1] coarse网络的混合权重
        coarse_mask: [N_rays, N_coarse_samples, N_sourceviews, 1] coarse网络的mask
        fine_mask: [N_rays, N_total_samples, N_sourceviews, 1] fine网络原始mask 
        top_k: 保留的top source views数量，默认6
    
    Returns:
        final_fine_mask: [N_rays, N_total_samples, N_sourceviews, 1] 经过pruning的final mask
    """
    
    N_rays, N_coarse = z_vals_coarse.shape
    N_rays2, N_importance = z_samples.shape  
    N_rays3, N_total, N_views, _ = fine_mask.shape
    
    assert N_rays == N_rays2 == N_rays3, "Ray数量必须匹配"
    assert N_total == N_coarse + N_importance, f"Total samples数量不匹配: {N_total} != {N_coarse} + {N_importance}"
    
    device = z_vals_coarse.device
    
    # Step 1: 向量化的pruned mask创建
    weights = blending_weights_valid[..., 0]  # [N_rays, N_coarse, N_views]
    mask = coarse_mask[..., 0]                # [N_rays, N_coarse, N_views]
    
    # 将无效views的权重设为负无穷，这样topk就不会选到它们
    masked_weights = torch.where(mask.bool(), weights, torch.tensor(float('-inf'), device=device))
    
    # 批量topk操作
    _, top_indices = torch.topk(masked_weights, min(top_k, N_views), dim=-1, largest=True, sorted=False)
    # top_indices: [N_rays, N_coarse, top_k]
    
    # 创建pruned mask
    pruned_mask_coarse = torch.zeros_like(coarse_mask)
    
    # 使用scatter操作批量设置mask
    ray_indices = torch.arange(N_rays, device=device).view(-1, 1, 1).expand(-1, N_coarse, min(top_k, N_views))
    sample_indices = torch.arange(N_coarse, device=device).view(1, -1, 1).expand(N_rays, -1, min(top_k, N_views))
    
    # 只有当权重不是-inf时才设置mask (即原本mask为True的位置)
    valid_top_mask = masked_weights.gather(-1, top_indices) != float('-inf')
    
    # 设置pruned mask
    pruned_mask_coarse[ray_indices[valid_top_mask], sample_indices[valid_top_mask], top_indices[valid_top_mask], 0] = 1.0
    
    # Step 2: 向量化的assignment计算
    # 批量searchsorted
    z_samples_flat = z_samples.reshape(-1)  # [N_rays * N_importance]
    z_coarse_flat = z_vals_coarse.reshape(-1, N_coarse)  # [N_rays, N_coarse]
    
    # 为每个ray分别计算assignment
    assignment = torch.zeros(N_rays, N_importance, dtype=torch.long, device=device)
    
    for ray_idx in range(N_rays):
        z_coarse_ray = z_vals_coarse[ray_idx]  # [N_coarse]
        z_fine_ray = z_samples[ray_idx]        # [N_importance]
        
        # 批量searchsorted
        insert_positions = torch.searchsorted(z_coarse_ray, z_fine_ray)  # [N_importance]
        
        # 处理边界情况
        insert_positions = torch.clamp(insert_positions, 0, N_coarse-1)
        
        # 对于在中间的点，选择距离更近的coarse点
        left_indices = torch.clamp(insert_positions - 1, 0, N_coarse-1)
        right_indices = insert_positions
        
        left_distances = torch.abs(z_fine_ray - z_coarse_ray[left_indices])
        right_distances = torch.abs(z_fine_ray - z_coarse_ray[right_indices])
        
        # 选择距离更近的点
        assignment[ray_idx] = torch.where(left_distances <= right_distances, left_indices, right_indices)
    
    # Step 3: 向量化的mask扩展
    extended_pruned_mask = torch.zeros_like(fine_mask)
    
    # 复制coarse部分
    extended_pruned_mask[:, :N_coarse, :, :] = pruned_mask_coarse
    
    # 向量化的fine部分分配
    ray_indices_fine = torch.arange(N_rays, device=device).view(-1, 1).expand(-1, N_importance)
    fine_sample_indices = torch.arange(N_coarse, N_total, device=device).view(1, -1).expand(N_rays, -1)
    
    # 使用assignment进行批量索引
    extended_pruned_mask[ray_indices_fine, fine_sample_indices, :, :] = \
        pruned_mask_coarse[ray_indices_fine, assignment, :, :]
    
    # Step 4: 最终mask
    final_fine_mask = extended_pruned_mask * fine_mask
    
    return final_fine_mask


# 为了向后兼容，默认使用优化版本
def apply_source_view_pruning(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, top_k=6):
    """
    默认使用优化版本的source view pruning
    """
    return apply_source_view_pruning_optimized(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, top_k)


def apply_source_view_pruning_sparse(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, 
                                    H, W, window_size=5, top_k=6):
    """
    稀疏采样专用的source view pruning (用于sample_point_sparsity模式)
    
    只对window中心的rays进行pruning计算，然后将结果共享给window内的其他rays。
    这样可以显著减少计算量，因为实际上只有中心rays有有效的blending_weights。
    
    Args:
        z_vals_coarse: [N_rays, N_coarse_samples] coarse采样点深度  
        z_samples: [N_rays, N_importance] fine网络新增采样点深度
        blending_weights_valid: [N_rays, N_coarse_samples, N_sourceviews, 1] coarse网络的混合权重
        coarse_mask: [N_rays, N_coarse_samples, N_sourceviews, 1] coarse网络的mask
        fine_mask: [N_rays, N_total_samples, N_sourceviews, 1] fine网络原始mask
        H: 图像高度 (rays的高度维度)
        W: 图像宽度 (rays的宽度维度)  
        window_size: 窗口大小，默认5
        top_k: 保留的top source views数量，默认6
    
    Returns:
        final_fine_mask: [N_rays, N_total_samples, N_sourceviews, 1] 经过pruning的final mask
    """
    
    N_rays, N_coarse = z_vals_coarse.shape
    N_rays2, N_importance = z_samples.shape  
    N_rays3, N_total, N_views, _ = fine_mask.shape
    
    assert N_rays == N_rays2 == N_rays3, "Ray数量必须匹配"
    assert N_total == N_coarse + N_importance, f"Total samples数量不匹配: {N_total} != {N_coarse} + {N_importance}"
    assert H * W == N_rays, f"H*W必须等于N_rays: {H}*{W} != {N_rays}"
    assert H % window_size == 0 and W % window_size == 0, f"H和W必须能被window_size整除: {H}%{window_size}!=0 or {W}%{window_size}!=0"
    
    device = z_vals_coarse.device
    center_offset = window_size // 2  # 中心偏移，对于5x5窗口就是2
    
    # Step 1: 收集所有window中心points的索引
    center_indices = []
    center_ray_to_window = {}  # 中心ray索引 -> (window_i, window_j)
    
    for i in range(0, H, window_size):
        for j in range(0, W, window_size):
            center_i = i + center_offset
            center_j = j + center_offset
            center_ray_idx = center_i * W + center_j
            center_indices.append(center_ray_idx)
            center_ray_to_window[center_ray_idx] = (i, j)
    
    center_indices = torch.tensor(center_indices, device=device)  # [num_windows]
    num_windows = len(center_indices)
    
    # Step 2: 只对中心rays进行pruning计算
    # 提取中心rays的数据
    center_z_vals_coarse = z_vals_coarse[center_indices]  # [num_windows, N_coarse]
    center_z_samples = z_samples[center_indices]          # [num_windows, N_importance]
    center_blending_weights = blending_weights_valid[center_indices]  # [num_windows, N_coarse, N_views, 1]
    center_coarse_mask = coarse_mask[center_indices]      # [num_windows, N_coarse, N_views, 1]
    center_fine_mask = fine_mask[center_indices]          # [num_windows, N_total, N_views, 1]
    
    # 对中心rays应用标准的向量化pruning
    center_final_mask = apply_source_view_pruning_optimized(
        center_z_vals_coarse, center_z_samples, center_blending_weights, 
        center_coarse_mask, center_fine_mask, top_k
    )  # [num_windows, N_total, N_views, 1]
    
    # Step 3: 将中心rays的pruning结果共享给整个window
    final_fine_mask = torch.zeros_like(fine_mask)
    
    # 向量化的window共享操作
    window_idx = 0
    for i in range(0, H, window_size):
        for j in range(0, W, window_size):
            # 当前window的ray索引范围
            ray_start_i = i
            ray_end_i = i + window_size
            ray_start_j = j  
            ray_end_j = j + window_size
            
            # 转换为1D ray索引
            window_ray_indices = []
            for wi in range(ray_start_i, ray_end_i):
                for wj in range(ray_start_j, ray_end_j):
                    ray_idx = wi * W + wj
                    window_ray_indices.append(ray_idx)
            
            window_ray_indices = torch.tensor(window_ray_indices, device=device)  # [window_size^2]
            
            # 将中心ray的mask复制给整个window
            center_mask = center_final_mask[window_idx]  # [N_total, N_views, 1]
            final_fine_mask[window_ray_indices] = center_mask.unsqueeze(0).expand(window_size*window_size, -1, -1, -1)
            
            window_idx += 1
    
    # Step 4: 与原始fine mask进行AND运算（保持原有的view可见性约束）
    final_fine_mask = final_fine_mask * fine_mask
    
    return final_fine_mask


def apply_source_view_pruning_sparse_vectorized(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, 
                                               H, W, window_size=5, top_k=6):
    """
    完全向量化的稀疏pruning版本（更快）
    支持边缘情况处理，不要求H和W必须被window_size整除
    
    Args:
        z_vals_coarse: [N_rays, N_coarse_samples] coarse采样点深度  
        z_samples: [N_rays, N_importance] fine网络新增采样点深度
        blending_weights_valid: [N_rays, N_coarse_samples, N_sourceviews, 1] coarse网络的混合权重
        coarse_mask: [N_rays, N_coarse_samples, N_sourceviews, 1] coarse网络的mask
        fine_mask: [N_rays, N_total_samples, N_sourceviews, 1] fine网络原始mask
        H: 图像高度
        W: 图像宽度  
        window_size: 窗口大小，默认5
        top_k: 保留的top source views数量，默认6
    
    Returns:
        final_fine_mask: [N_rays, N_total_samples, N_sourceviews, 1] 经过pruning的final mask
    """
    
    N_rays, N_coarse = z_vals_coarse.shape
    N_rays2, N_importance = z_samples.shape  
    N_rays3, N_total, N_views, _ = fine_mask.shape
    
    assert N_rays == N_rays2 == N_rays3, "Ray数量必须匹配"
    assert N_total == N_coarse + N_importance, f"Total samples数量不匹配"
    assert H * W == N_rays, f"H*W必须等于N_rays"
    
    # 检查是否可以整除，如果不行就回退到标准pruning
    if H % window_size != 0 or W % window_size != 0:
        print(f"Warning: H={H}, W={W} 无法被window_size={window_size}整除，回退到标准pruning")
        return apply_source_view_pruning_optimized(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, top_k)
    
    device = z_vals_coarse.device
    center_offset = window_size // 2
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    num_windows = num_windows_h * num_windows_w
    
    # Step 1: 向量化生成中心indices
    window_starts_i = torch.arange(0, H, window_size, device=device)  # [num_windows_h]
    window_starts_j = torch.arange(0, W, window_size, device=device)  # [num_windows_w]
    
    # 使用meshgrid生成所有window的起始位置
    grid_i, grid_j = torch.meshgrid(window_starts_i, window_starts_j, indexing='ij')  # [num_windows_h, num_windows_w]
    
    # 计算中心位置
    center_i = (grid_i + center_offset).flatten()  # [num_windows]
    center_j = (grid_j + center_offset).flatten()  # [num_windows]
    center_indices = center_i * W + center_j       # [num_windows]
    
    # Step 2: 批量提取中心rays数据并进行pruning
    center_final_mask = apply_source_view_pruning_optimized(
        z_vals_coarse[center_indices], 
        z_samples[center_indices], 
        blending_weights_valid[center_indices], 
        coarse_mask[center_indices], 
        fine_mask[center_indices], 
        top_k
    )  # [num_windows, N_total, N_views, 1]
    
    # Step 3: 向量化的window共享
    # 创建window到ray的映射矩阵
    window_to_rays = torch.zeros(num_windows, window_size * window_size, dtype=torch.long, device=device)
    
    window_idx = 0
    for i in range(num_windows_h):
        for j in range(num_windows_w):
            # 当前window内的相对位置
            rel_positions = torch.arange(window_size * window_size, device=device)
            rel_i = rel_positions // window_size
            rel_j = rel_positions % window_size
            
            # 转换为绝对ray索引
            abs_i = i * window_size + rel_i
            abs_j = j * window_size + rel_j
            ray_indices = abs_i * W + abs_j
            
            window_to_rays[window_idx] = ray_indices
            window_idx += 1
    
    # 批量复制中心mask到所有window rays
    final_fine_mask = torch.zeros_like(fine_mask)
    
    # 使用高级索引进行批量赋值
    for window_idx in range(num_windows):
        rays_in_window = window_to_rays[window_idx]  # [window_size^2]
        center_mask = center_final_mask[window_idx]  # [N_total, N_views, 1]
        
        # 广播中心mask到window内所有rays
        final_fine_mask[rays_in_window] = center_mask.unsqueeze(0).expand(window_size*window_size, -1, -1, -1)
    
    # Step 4: 与原始fine mask进行AND运算
    final_fine_mask = final_fine_mask * fine_mask
    
    return final_fine_mask

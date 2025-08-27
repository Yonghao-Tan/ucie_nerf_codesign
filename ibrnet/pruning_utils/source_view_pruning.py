#!/usr/bin/env python3
"""
Source View Pruning 功能
包含原始版本和优化版本的pruning函数
"""

import torch


def apply_source_view_pruning_optimized(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, top_k=6, sample_point_group_size=None):
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
    
    # Step 4: 应用sample point grouping (如果启用) - 在pruned mask上操作
    if sample_point_group_size is not None and sample_point_group_size > 1:
        # 对每个ray独立进行grouping
        for ray_idx in range(N_rays):
            # 计算完整的groups数量
            num_complete_groups = N_total // sample_point_group_size
            
            for group_idx in range(num_complete_groups):
                start_idx = group_idx * sample_point_group_size
                end_idx = start_idx + sample_point_group_size
                
                # 使用组内第一个sample的pruning mask
                first_sample_mask = extended_pruned_mask[ray_idx, start_idx, :, :].clone()
                
                # 将第一个sample的mask应用到组内所有samples
                extended_pruned_mask[ray_idx, start_idx:end_idx, :, :] = first_sample_mask.unsqueeze(0)
        
        # print(f"Applied sample point grouping: group_size={sample_point_group_size}, {num_complete_groups} complete groups per ray")
    
    # Step 5: 最终与fine_mask进行AND操作 (fine_mask是雷打不动的)
    final_fine_mask = extended_pruned_mask * fine_mask
    
    return final_fine_mask


def apply_source_view_pruning_optimized_aggregated(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, top_k=6, sample_point_group_size=None):
    """
    优化版本的source view pruning，支持组内权重聚合
    
    与原版本的不同：在sample point grouping阶段，先聚合组内所有sample的blending weights，
    然后在聚合权重基础上进行top-k选择，而不是简单复制第一个sample的mask。
    
    Args:
        z_vals_coarse: [N_rays, N_coarse_samples] coarse采样点深度
        z_samples: [N_rays, N_importance] fine网络新增采样点深度  
        blending_weights_valid: [N_rays, N_coarse_samples, N_sourceviews, 1] coarse网络的混合权重
        coarse_mask: [N_rays, N_coarse_samples, N_sourceviews, 1] coarse网络的mask
        fine_mask: [N_rays, N_total_samples, N_sourceviews, 1] fine网络原始mask 
        top_k: 保留的top source views数量，默认6
        sample_point_group_size: 采样点分组大小，启用权重聚合功能
    
    Returns:
        final_fine_mask: [N_rays, N_total_samples, N_sourceviews, 1] 经过pruning的final mask
    """
    
    N_rays, N_coarse = z_vals_coarse.shape
    N_rays2, N_importance = z_samples.shape  
    N_rays3, N_total, N_views, _ = fine_mask.shape
    
    assert N_rays == N_rays2 == N_rays3, "Ray数量必须匹配"
    assert N_total == N_coarse + N_importance, f"Total samples数量不匹配: {N_total} != {N_coarse} + {N_importance}"
    
    device = z_vals_coarse.device
    
    # Step 1: 向量化的pruned mask创建（与原版本相同）
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
    
    # Step 2: 向量化的assignment计算（与原版本相同）
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
    
    # Step 3: 创建扩展的blending weights（包含coarse+fine）
    # 这里需要重构原始的blending weights到完整的N_total采样点
    extended_blending_weights = torch.zeros(N_rays, N_total, N_views, device=device)
    
    # 复制coarse部分的权重
    extended_blending_weights[:, :N_coarse, :] = blending_weights_valid[..., 0]
    
    # 为fine部分分配权重（基于assignment）
    ray_indices_fine = torch.arange(N_rays, device=device).view(-1, 1).expand(-1, N_importance)
    fine_sample_indices = torch.arange(N_coarse, N_total, device=device).view(1, -1).expand(N_rays, -1)
    
    # 使用assignment进行批量索引，将coarse权重分配给对应的fine points
    extended_blending_weights[ray_indices_fine, fine_sample_indices, :] = \
        blending_weights_valid[ray_indices_fine, assignment, :, 0]
    
    # Step 4: 应用sample point grouping with weight aggregation
    extended_pruned_mask = torch.zeros_like(fine_mask)
    
    if sample_point_group_size is not None and sample_point_group_size > 1:
        # 对每个ray独立进行grouping with aggregation
        for ray_idx in range(N_rays):
            # 计算完整的groups数量
            num_complete_groups = N_total // sample_point_group_size
            
            for group_idx in range(num_complete_groups):
                start_idx = group_idx * sample_point_group_size
                end_idx = start_idx + sample_point_group_size
                
                # 获取组内所有sample points的权重和mask
                group_weights = extended_blending_weights[ray_idx, start_idx:end_idx, :]  # [group_size, N_views]
                group_mask = fine_mask[ray_idx, start_idx:end_idx, :, 0]  # [group_size, N_views]
                
                # 聚合组内权重：对所有有效mask的权重求和
                # 只有当至少一个sample在某个view上有效时，才考虑该view
                valid_mask = group_mask.bool()  # [group_size, N_views]
                
                # 计算每个view在组内的聚合权重
                aggregated_weights = torch.zeros(N_views, device=device)
                for view_idx in range(N_views):
                    # 找到在当前view上有效的samples
                    valid_samples = valid_mask[:, view_idx]
                    if valid_samples.sum() > 0:
                        # 对有效samples的权重求和
                        aggregated_weights[view_idx] = group_weights[valid_samples, view_idx].sum()
                
                # 在聚合权重基础上进行top-k选择
                # 只考虑有有效mask的views
                has_valid_mask = valid_mask.any(dim=0)  # [N_views] - 哪些views在组内至少有一个有效sample
                
                if has_valid_mask.sum() == 0:
                    # 如果组内没有任何有效的view，跳过
                    continue
                
                # 将无效views的权重设为负无穷
                masked_aggregated_weights = torch.where(
                    has_valid_mask, 
                    aggregated_weights, 
                    torch.tensor(float('-inf'), device=device)
                )
                
                # 在聚合权重基础上选择top-k
                k = min(top_k, has_valid_mask.sum().item())
                if k > 0:
                    _, top_view_indices = torch.topk(masked_aggregated_weights, k, largest=True)
                    
                    # 为组内所有samples设置相同的pruned mask
                    group_pruned_mask = torch.zeros_like(group_mask)
                    group_pruned_mask[:, top_view_indices] = 1.0
                    
                    # 应用到extended_pruned_mask
                    extended_pruned_mask[ray_idx, start_idx:end_idx, :, 0] = group_pruned_mask
            
            # 处理剩余的不完整组（保持原样）
            remaining_start = num_complete_groups * sample_point_group_size
            if remaining_start < N_total:
                # 对剩余samples使用原始的单点pruning逻辑
                for sample_idx in range(remaining_start, N_total):
                    sample_weights = extended_blending_weights[ray_idx, sample_idx, :]  # [N_views]
                    sample_mask = fine_mask[ray_idx, sample_idx, :, 0]  # [N_views]
                    
                    valid_views = sample_mask.bool()
                    if valid_views.sum() > 0:
                        masked_sample_weights = torch.where(
                            valid_views, 
                            sample_weights, 
                            torch.tensor(float('-inf'), device=device)
                        )
                        
                        k = min(top_k, valid_views.sum().item())
                        if k > 0:
                            _, top_indices_single = torch.topk(masked_sample_weights, k, largest=True)
                            extended_pruned_mask[ray_idx, sample_idx, top_indices_single, 0] = 1.0
        
        # print(f"Applied aggregated sample point grouping: group_size={sample_point_group_size}, {num_complete_groups} complete groups per ray")
    
    else:
        # 如果没有启用grouping，使用原始的单点pruning逻辑
        extended_pruned_mask = torch.zeros_like(fine_mask)
        
        # 复制coarse部分
        extended_pruned_mask[:, :N_coarse, :, :] = pruned_mask_coarse
        
        # 向量化的fine部分分配
        ray_indices_fine = torch.arange(N_rays, device=device).view(-1, 1).expand(-1, N_importance)
        fine_sample_indices = torch.arange(N_coarse, N_total, device=device).view(1, -1).expand(N_rays, -1)
        
        # 使用assignment进行批量索引
        extended_pruned_mask[ray_indices_fine, fine_sample_indices, :, :] = \
            pruned_mask_coarse[ray_indices_fine, assignment, :, :]
    
    # Step 5: 最终与fine_mask进行AND操作 (fine_mask是雷打不动的)
    final_fine_mask = extended_pruned_mask * fine_mask
    
    return final_fine_mask


# 为了向后兼容，默认使用优化版本
def apply_source_view_pruning(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, top_k=6, sample_point_group_size=None):
    """
    默认使用优化版本的source view pruning
    """
    return apply_source_view_pruning_optimized(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, top_k, sample_point_group_size)


def apply_source_view_pruning_sparse_vectorized_aggregated(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, 
                                                        H, W, window_size=5, top_k=6, sample_point_group_size=None):
    """
    完全向量化的稀疏pruning版本，支持权重聚合的sample point grouping
    
    与原版本的不同：在sample point grouping阶段，先聚合组内所有sample的blending weights，
    然后在聚合权重基础上进行top-k选择。
    
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
        sample_point_group_size: 采样点分组大小，启用权重聚合功能
    
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
        print(f"Warning: H={H}, W={W} 无法被window_size={window_size}整除，回退到聚合标准pruning")
        return apply_source_view_pruning_optimized_aggregated(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, top_k, sample_point_group_size)
    
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
    
    # Step 2: 批量提取中心rays数据并进行pruning（使用聚合版本）
    center_final_mask = apply_source_view_pruning_optimized_aggregated(
        z_vals_coarse[center_indices], 
        z_samples[center_indices], 
        blending_weights_valid[center_indices], 
        coarse_mask[center_indices], 
        fine_mask[center_indices], 
        top_k,
        sample_point_group_size  # 传递grouping参数
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
    pruned_mask = torch.zeros_like(fine_mask)
    
    # 使用高级索引进行批量赋值
    for window_idx in range(num_windows):
        rays_in_window = window_to_rays[window_idx]  # [window_size^2]
        center_mask = center_final_mask[window_idx]  # [N_total, N_views, 1]
        
        # 广播中心mask到window内所有rays
        pruned_mask[rays_in_window] = center_mask.unsqueeze(0).expand(window_size*window_size, -1, -1, -1)
    
    # 注意：在稀疏版本中，sample point grouping已经在中心ray处理时完成了
    # 这里不需要重复应用grouping，因为所有rays都共享中心ray的结果
    
    # Step 4: 最终与fine_mask进行AND操作 (fine_mask是雷打不动的)
    final_fine_mask = pruned_mask * fine_mask
    
    return final_fine_mask


def apply_source_view_pruning_sparse_vectorized(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, 
                                               H, W, window_size=5, top_k=6, sample_point_group_size=None):
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
    pruned_mask = torch.zeros_like(fine_mask)
    
    # 使用高级索引进行批量赋值
    for window_idx in range(num_windows):
        rays_in_window = window_to_rays[window_idx]  # [window_size^2]
        center_mask = center_final_mask[window_idx]  # [N_total, N_views, 1]
        
        # 广播中心mask到window内所有rays
        pruned_mask[rays_in_window] = center_mask.unsqueeze(0).expand(window_size*window_size, -1, -1, -1)
    
    # Step 4: 应用sample point grouping (如果启用) - 在pruned mask上操作
    if sample_point_group_size is not None and sample_point_group_size > 1:
        N_total = pruned_mask.shape[1]
        num_complete_groups = N_total // sample_point_group_size
        
        # 对每个ray独立进行grouping
        for ray_idx in range(N_rays):
            for group_idx in range(num_complete_groups):
                start_idx = group_idx * sample_point_group_size
                end_idx = start_idx + sample_point_group_size
                
                # 使用组内第一个sample的pruning mask
                first_sample_mask = pruned_mask[ray_idx, start_idx, :, :].clone()
                
                # 将第一个sample的mask应用到组内所有samples
                pruned_mask[ray_idx, start_idx:end_idx, :, :] = first_sample_mask.unsqueeze(0)
        
        # print(f"Applied sparse sample point grouping: group_size={sample_point_group_size}, {num_complete_groups} complete groups per ray")
    
    # Step 5: 最终与fine_mask进行AND操作 (fine_mask是雷打不动的)
    final_fine_mask = pruned_mask * fine_mask
    
    return final_fine_mask


def apply_source_view_pruning_2x2_windows_variance_based(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, 
                                                        H, W, window_size=5, sample_point_group_size=8, **kwargs):
    """
    基于2x2 windows和方差排名的source view pruning
    
    工作流程：
    1. 对2x2个窗口分别进行权重聚合（每个sample group内聚合）
    2. 对聚合后的权重进行归一化
    3. 每个group pruning掉末2个权重（8->6）
    4. 计算所有24个groups的权重方差
    5. 按方差排名，分4档进行不同程度的pruning
    
    Args:
        z_vals_coarse: [N_rays, N_coarse_samples] coarse采样点深度
        z_samples: [N_rays, N_importance] fine网络新增采样点深度  
        blending_weights_valid: [N_rays, N_coarse_samples, N_sourceviews, 1] coarse网络的混合权重
        coarse_mask: [N_rays, N_coarse_samples, N_sourceviews, 1] coarse网络的mask
        fine_mask: [N_rays, N_total_samples, N_sourceviews, 1] fine网络原始mask
        H: 图像高度
        W: 图像宽度  
        window_size: 窗口大小，默认5
        sample_point_group_size: 采样点分组大小，默认8
    
    Returns:
        final_fine_mask: [N_rays, N_total_samples, N_sourceviews, 1] 经过pruning的final mask
    """
    
    N_rays, N_coarse = z_vals_coarse.shape
    N_rays2, N_importance = z_samples.shape  
    N_rays3, N_total, N_views, _ = fine_mask.shape
    
    assert N_rays == N_rays2 == N_rays3, "Ray数量必须匹配"
    assert N_total == N_coarse + N_importance, f"Total samples数量不匹配"
    assert H * W == N_rays, f"H*W必须等于N_rays"
    assert sample_point_group_size == 8, "当前实现要求sample_point_group_size=8"
    
    # 检查是否可以处理2x2 windows
    if H % (window_size * 2) != 0 or W % (window_size * 2) != 0:
        print(f"Warning: H={H}, W={W} 无法被2x window_size={window_size*2}整除，回退到标准pruning")
        return apply_source_view_pruning_sparse_vectorized(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, H, W, window_size=window_size, sample_point_group_size=sample_point_group_size, **kwargs)
    
    device = z_vals_coarse.device
    center_offset = window_size // 2
    
    # 计算2x2 window的数量
    num_2x2_windows_h = H // (window_size * 2)
    num_2x2_windows_w = W // (window_size * 2)
    total_2x2_windows = num_2x2_windows_h * num_2x2_windows_w
    
    # Step 1: 构建扩展的blending weights（包含coarse+fine）
    # 首先计算assignment关系
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
    
    # 创建扩展的blending weights
    extended_blending_weights = torch.zeros(N_rays, N_total, N_views, device=device)
    
    # 复制coarse部分的权重
    extended_blending_weights[:, :N_coarse, :] = blending_weights_valid[..., 0]
    
    # 为fine部分分配权重（基于assignment）
    ray_indices_fine = torch.arange(N_rays, device=device).view(-1, 1).expand(-1, N_importance)
    fine_sample_indices = torch.arange(N_coarse, N_total, device=device).view(1, -1).expand(N_rays, -1)
    
    extended_blending_weights[ray_indices_fine, fine_sample_indices, :] = \
        blending_weights_valid[ray_indices_fine, assignment, :, 0]
    
    # Step 2: 处理每个2x2 window
    all_group_weights = []  # 存储所有24个groups的权重 [24, N_views]
    all_group_masks = []    # 存储所有24个groups的mask [24, N_views]
    group_to_ray_mapping = []  # 记录每个group对应的(ray_idx, group_idx)
    
    for big_window_i in range(num_2x2_windows_h):
        for big_window_j in range(num_2x2_windows_w):
            # 当前2x2 window包含的4个小window
            for sub_i in range(2):
                for sub_j in range(2):
                    # 计算当前小window的中心ray位置
                    window_start_i = big_window_i * window_size * 2 + sub_i * window_size
                    window_start_j = big_window_j * window_size * 2 + sub_j * window_size
                    center_i = window_start_i + center_offset
                    center_j = window_start_j + center_offset
                    center_ray_idx = center_i * W + center_j
                    
                    # 获取该中心ray的数据
                    ray_weights = extended_blending_weights[center_ray_idx]  # [N_total, N_views]
                    ray_mask = fine_mask[center_ray_idx, :, :, 0]  # [N_total, N_views]
                    
                    # 按group_size=8分组处理（假设N_total=48）
                    num_groups_per_ray = N_total // sample_point_group_size
                    
                    for group_idx in range(num_groups_per_ray):
                        start_idx = group_idx * sample_point_group_size
                        end_idx = start_idx + sample_point_group_size
                        
                        # 获取组内数据
                        group_weights = ray_weights[start_idx:end_idx, :]  # [8, N_views]
                        group_mask = ray_mask[start_idx:end_idx, :]  # [8, N_views]
                        
                        # Step 2.1: 聚合组内权重
                        valid_mask = group_mask.bool()  # [8, N_views]
                        aggregated_weights = torch.zeros(N_views, device=device)
                        
                        for view_idx in range(N_views):
                            valid_samples = valid_mask[:, view_idx]
                            if valid_samples.sum() > 0:
                                # 对有效samples的权重求和
                                aggregated_weights[view_idx] = group_weights[valid_samples, view_idx].sum()
                        
                        # Step 2.2: 归一化（除以权重总和）
                        weight_sum = aggregated_weights.sum()
                        if weight_sum > 0:
                            normalized_weights = aggregated_weights / weight_sum
                        else:
                            normalized_weights = aggregated_weights.clone()
                        
                        # Step 2.3: 初步pruning（8->6，去掉末2个）
                        has_valid_mask = valid_mask.any(dim=0)  # [N_views] - 哪些views有效
                        if has_valid_mask.sum() > 2:
                            # 将无效views的权重设为负无穷
                            masked_weights = torch.where(
                                has_valid_mask, 
                                normalized_weights, 
                                torch.tensor(float('-inf'), device=device)
                            )
                            
                            # 选择top-6（去掉末2个）
                            num_to_keep = min(6, has_valid_mask.sum().item())
                            _, top_6_indices = torch.topk(masked_weights, num_to_keep, largest=True)
                            
                            # 创建pruned权重
                            pruned_weights = torch.zeros_like(normalized_weights)
                            pruned_weights[top_6_indices] = normalized_weights[top_6_indices]
                            
                            all_group_weights.append(pruned_weights)
                        else:
                            # 如果有效views<=2，保持原样
                            all_group_weights.append(normalized_weights)
                        
                        # 记录对应关系
                        group_to_ray_mapping.append((center_ray_idx, group_idx))
                        
                        # 记录group mask（用于最终应用）
                        all_group_masks.append(has_valid_mask)
    
    # Step 3: 计算所有24个groups的方差并排名
    group_variances = []
    for group_weights in all_group_weights:
        # 只计算非零权重的方差
        non_zero_weights = group_weights[group_weights > 0]
        if len(non_zero_weights) > 1:
            variance = torch.var(non_zero_weights)
        else:
            variance = torch.tensor(0.0, device=device)
        group_variances.append(variance)
    
    group_variances = torch.stack(group_variances)  # [24]
    
    # 按方差排序（从小到大）
    _, sorted_indices = torch.sort(group_variances)
    
    # Step 4: 按方差排名分档进行不同程度的pruning
    # 排名1-6: 留6个, 排名7-12: 留5个, 排名13-18: 留4个, 排名19-24: 留3个
    keep_counts = [6, 5, 4, 3]
    
    final_group_masks = []
    for i, group_idx in enumerate(sorted_indices):
        # 确定当前group应该保留几个source view
        tier = i // 6  # 0,1,2,3
        tier = min(tier, 3)  # 确保不越界
        num_keep = keep_counts[tier]
        
        group_weights = all_group_weights[group_idx]
        group_mask = all_group_masks[group_idx]
        
        # 在当前group的权重基础上再次选择top-num_keep
        if group_mask.sum() > 0:
            masked_weights = torch.where(
                group_mask & (group_weights > 0), 
                group_weights, 
                torch.tensor(float('-inf'), device=device)
            )
            
            actual_keep = min(num_keep, (group_weights > 0).sum().item())
            if actual_keep > 0:
                _, final_top_indices = torch.topk(masked_weights, actual_keep, largest=True)
                
                # 创建最终mask
                final_mask = torch.zeros_like(group_weights, dtype=torch.bool)
                final_mask[final_top_indices] = True
                final_group_masks.append(final_mask)
            else:
                final_group_masks.append(torch.zeros_like(group_weights, dtype=torch.bool))
        else:
            final_group_masks.append(torch.zeros_like(group_weights, dtype=torch.bool))
    
    # Step 5: 应用最终的pruning结果到中心rays
    center_pruned_masks = {}  # 存储每个中心ray的pruning结果
    
    # 重新按原始顺序应用mask到中心rays
    for orig_idx, (center_ray_idx, group_idx) in enumerate(group_to_ray_mapping):
        final_mask = final_group_masks[sorted_indices.tolist().index(orig_idx)]
        
        # 计算该group在ray中的位置
        start_idx = group_idx * sample_point_group_size
        end_idx = start_idx + sample_point_group_size
        
        # 初始化center ray的mask（如果还没有）
        if center_ray_idx not in center_pruned_masks:
            center_pruned_masks[center_ray_idx] = torch.zeros_like(fine_mask[center_ray_idx, :, :, 0])
        
        # 应用到中心ray的对应group
        for sample_idx in range(start_idx, end_idx):
            center_pruned_masks[center_ray_idx][sample_idx, final_mask] = 1.0
    
    # Step 6: 扩展到整个2x2 window区域（完整实现window共享逻辑）
    extended_pruned_mask = torch.zeros_like(fine_mask)
    
    # 为每个2x2 window创建window到ray的映射
    for big_window_i in range(num_2x2_windows_h):
        for big_window_j in range(num_2x2_windows_w):
            # 当前2x2 window包含的4个小window
            for sub_i in range(2):
                for sub_j in range(2):
                    # 计算当前小window的中心ray位置
                    window_start_i = big_window_i * window_size * 2 + sub_i * window_size
                    window_start_j = big_window_j * window_size * 2 + sub_j * window_size
                    center_i = window_start_i + center_offset
                    center_j = window_start_j + center_offset
                    center_ray_idx = center_i * W + center_j
                    
                    # 获取该中心ray的pruning结果
                    if center_ray_idx in center_pruned_masks:
                        center_mask = center_pruned_masks[center_ray_idx]  # [N_total, N_views]
                        
                        # 计算该window内所有rays的索引
                        window_ray_indices = []
                        for wi in range(window_start_i, window_start_i + window_size):
                            for wj in range(window_start_j, window_start_j + window_size):
                                if wi < H and wj < W:  # 边界检查
                                    ray_idx = wi * W + wj
                                    window_ray_indices.append(ray_idx)
                        
                        # 将中心ray的mask共享给window内所有rays
                        for ray_idx in window_ray_indices:
                            extended_pruned_mask[ray_idx, :, :, 0] = center_mask
    
    # Step 7: 最终与fine_mask进行AND操作
    final_fine_mask = extended_pruned_mask * fine_mask
    
    return final_fine_mask


def apply_source_view_pruning_2x2_windows_threshold_based(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, 
                                                         H, W, window_size=5, sample_point_group_size=8, weight_threshold=0.04, **kwargs):
    """
    基于2x2 windows和权重阈值数量排名的source view pruning
    
    工作流程：
    1. 对2x2个窗口分别进行权重聚合（每个sample group内聚合）
    2. 对聚合后的权重进行归一化
    3. 计算每个group中大于阈值的权重数量
    4. 按阈值数量排名，分4档进行不同程度的pruning
    
    Args:
        z_vals_coarse: [N_rays, N_coarse_samples] coarse采样点深度
        z_samples: [N_rays, N_importance] fine网络新增采样点深度  
        blending_weights_valid: [N_rays, N_coarse_samples, N_sourceviews, 1] coarse网络的混合权重
        coarse_mask: [N_rays, N_coarse_samples, N_sourceviews, 1] coarse网络的mask
        fine_mask: [N_rays, N_total_samples, N_sourceviews, 1] fine网络原始mask
        H: 图像高度
        W: 图像宽度  
        window_size: 窗口大小，默认5
        sample_point_group_size: 采样点分组大小，默认8
        weight_threshold: 权重阈值，默认0.08
    
    Returns:
        final_fine_mask: [N_rays, N_total_samples, N_sourceviews, 1] 经过pruning的final mask
    """
    
    N_rays, N_coarse = z_vals_coarse.shape
    N_rays2, N_importance = z_samples.shape  
    N_rays3, N_total, N_views, _ = fine_mask.shape
    
    assert N_rays == N_rays2 == N_rays3, "Ray数量必须匹配"
    assert N_total == N_coarse + N_importance, f"Total samples数量不匹配"
    assert H * W == N_rays, f"H*W必须等于N_rays"
    assert sample_point_group_size == 8, "当前实现要求sample_point_group_size=8"
    
    # 检查是否可以处理2x2 windows
    if H % (window_size * 2) != 0 or W % (window_size * 2) != 0:
        print(f"Warning: H={H}, W={W} 无法被2x window_size={window_size*2}整除，回退到标准pruning")
        return apply_source_view_pruning_sparse_vectorized(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, H, W, window_size=window_size, sample_point_group_size=sample_point_group_size, **kwargs)

    device = z_vals_coarse.device
    center_offset = window_size // 2
    
    # 计算2x2 window的数量
    num_2x2_windows_h = H // (window_size * 2)
    num_2x2_windows_w = W // (window_size * 2)
    total_2x2_windows = num_2x2_windows_h * num_2x2_windows_w
    
    # Step 1: 构建扩展的blending weights（包含coarse+fine）
    # 首先计算assignment关系
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
    
    # 创建扩展的blending weights
    extended_blending_weights = torch.zeros(N_rays, N_total, N_views, device=device)
    
    # 复制coarse部分的权重
    extended_blending_weights[:, :N_coarse, :] = blending_weights_valid[..., 0]
    
    # 为fine部分分配权重（基于assignment）
    ray_indices_fine = torch.arange(N_rays, device=device).view(-1, 1).expand(-1, N_importance)
    fine_sample_indices = torch.arange(N_coarse, N_total, device=device).view(1, -1).expand(N_rays, -1)
    
    extended_blending_weights[ray_indices_fine, fine_sample_indices, :] = \
        blending_weights_valid[ray_indices_fine, assignment, :, 0]
    
    # Step 2: 处理每个2x2 window
    all_group_weights = []  # 存储所有24个groups的归一化权重 [24, N_views]
    all_group_masks = []    # 存储所有24个groups的mask [24, N_views]
    group_to_ray_mapping = []  # 记录每个group对应的(ray_idx, group_idx)
    
    for big_window_i in range(num_2x2_windows_h):
        for big_window_j in range(num_2x2_windows_w):
            # 当前2x2 window包含的4个小window
            for sub_i in range(2):
                for sub_j in range(2):
                    # 计算当前小window的中心ray位置
                    window_start_i = big_window_i * window_size * 2 + sub_i * window_size
                    window_start_j = big_window_j * window_size * 2 + sub_j * window_size
                    center_i = window_start_i + center_offset
                    center_j = window_start_j + center_offset
                    center_ray_idx = center_i * W + center_j
                    
                    # 获取该中心ray的数据
                    ray_weights = extended_blending_weights[center_ray_idx]  # [N_total, N_views]
                    ray_mask = fine_mask[center_ray_idx, :, :, 0]  # [N_total, N_views]
                    
                    # 按group_size=8分组处理（假设N_total=48）
                    num_groups_per_ray = N_total // sample_point_group_size
                    
                    for group_idx in range(num_groups_per_ray):
                        start_idx = group_idx * sample_point_group_size
                        end_idx = start_idx + sample_point_group_size
                        
                        # 获取组内数据
                        group_weights = ray_weights[start_idx:end_idx, :]  # [8, N_views]
                        group_mask = ray_mask[start_idx:end_idx, :]  # [8, N_views]
                        
                        # Step 2.1: 聚合组内权重
                        valid_mask = group_mask.bool()  # [8, N_views]
                        aggregated_weights = torch.zeros(N_views, device=device)
                        
                        for view_idx in range(N_views):
                            valid_samples = valid_mask[:, view_idx]
                            if valid_samples.sum() > 0:
                                # 对有效samples的权重求和
                                aggregated_weights[view_idx] = group_weights[valid_samples, view_idx].sum()
                        
                        # Step 2.2: 归一化（除以权重总和）
                        weight_sum = aggregated_weights.sum()
                        if weight_sum > 0:
                            normalized_weights = aggregated_weights / weight_sum
                        else:
                            normalized_weights = aggregated_weights.clone()
                        
                        # 记录权重和mask
                        has_valid_mask = valid_mask.any(dim=0)  # [N_views] - 哪些views有效
                        all_group_weights.append(normalized_weights)
                        all_group_masks.append(has_valid_mask)
                        
                        # 记录对应关系
                        group_to_ray_mapping.append((center_ray_idx, group_idx))
    
    # Step 3: 计算所有24个groups中大于阈值的权重数量并排名
    group_threshold_counts = []
    for group_weights in all_group_weights:
        # 计算大于阈值的权重数量
        above_threshold_count = (group_weights > weight_threshold).sum().item()
        group_threshold_counts.append(above_threshold_count)
    
    group_threshold_counts = torch.tensor(group_threshold_counts, device=device)  # [24]
    
    # 按阈值数量排序（从多到少，即数量越多排名越靠前）
    _, sorted_indices = torch.sort(group_threshold_counts, descending=True)
    
    # Step 4: 按阈值数量排名分档进行不同程度的pruning
    # 排名1-6: 留6个, 排名7-12: 留5个, 排名13-18: 留4个, 排名19-24: 留3个
    keep_counts = [6, 5, 4, 3] # TODO
    
    final_group_masks = []
    for i, group_idx in enumerate(sorted_indices):
        # 确定当前group应该保留几个source view
        tier = i // 6  # 0,1,2,3
        tier = min(tier, 3)  # 确保不越界
        num_keep = keep_counts[tier]
        
        group_weights = all_group_weights[group_idx]
        group_mask = all_group_masks[group_idx]
        
        # 在当前group的权重基础上选择top-num_keep
        if group_mask.sum() > 0:
            masked_weights = torch.where(
                group_mask, 
                group_weights, 
                torch.tensor(float('-inf'), device=device)
            )
            
            actual_keep = min(num_keep, group_mask.sum().item())
            if actual_keep > 0:
                _, final_top_indices = torch.topk(masked_weights, actual_keep, largest=True)
                
                # 创建最终mask
                final_mask = torch.zeros_like(group_weights, dtype=torch.bool)
                final_mask[final_top_indices] = True
                final_group_masks.append(final_mask)
            else:
                final_group_masks.append(torch.zeros_like(group_weights, dtype=torch.bool))
        else:
            final_group_masks.append(torch.zeros_like(group_weights, dtype=torch.bool))
    
    # Step 5: 应用最终的pruning结果到中心rays
    center_pruned_masks = {}  # 存储每个中心ray的pruning结果
    
    # 重新按原始顺序应用mask到中心rays
    for orig_idx, (center_ray_idx, group_idx) in enumerate(group_to_ray_mapping):
        final_mask = final_group_masks[sorted_indices.tolist().index(orig_idx)]
        
        # 计算该group在ray中的位置
        start_idx = group_idx * sample_point_group_size
        end_idx = start_idx + sample_point_group_size
        
        # 初始化center ray的mask（如果还没有）
        if center_ray_idx not in center_pruned_masks:
            center_pruned_masks[center_ray_idx] = torch.zeros_like(fine_mask[center_ray_idx, :, :, 0])
        
        # 应用到中心ray的对应group
        for sample_idx in range(start_idx, end_idx):
            center_pruned_masks[center_ray_idx][sample_idx, final_mask] = 1.0
    
    # Step 6: 扩展到整个2x2 window区域（完整实现window共享逻辑）
    extended_pruned_mask = torch.zeros_like(fine_mask)
    
    # 为每个2x2 window创建window到ray的映射
    for big_window_i in range(num_2x2_windows_h):
        for big_window_j in range(num_2x2_windows_w):
            # 当前2x2 window包含的4个小window
            for sub_i in range(2):
                for sub_j in range(2):
                    # 计算当前小window的中心ray位置
                    window_start_i = big_window_i * window_size * 2 + sub_i * window_size
                    window_start_j = big_window_j * window_size * 2 + sub_j * window_size
                    center_i = window_start_i + center_offset
                    center_j = window_start_j + center_offset
                    center_ray_idx = center_i * W + center_j
                    
                    # 获取该中心ray的pruning结果
                    if center_ray_idx in center_pruned_masks:
                        center_mask = center_pruned_masks[center_ray_idx]  # [N_total, N_views]
                        
                        # 计算该window内所有rays的索引
                        window_ray_indices = []
                        for wi in range(window_start_i, window_start_i + window_size):
                            for wj in range(window_start_j, window_start_j + window_size):
                                if wi < H and wj < W:  # 边界检查
                                    ray_idx = wi * W + wj
                                    window_ray_indices.append(ray_idx)
                        
                        # 将中心ray的mask共享给window内所有rays
                        for ray_idx in window_ray_indices:
                            extended_pruned_mask[ray_idx, :, :, 0] = center_mask
    
    # Step 7: 最终与fine_mask进行AND操作
    final_fine_mask = extended_pruned_mask * fine_mask
    
    return final_fine_mask

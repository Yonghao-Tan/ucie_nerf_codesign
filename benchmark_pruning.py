#!/usr/bin/env python3
"""
测试source view pruning函数的性能优化效果
"""

import torch
import time
import numpy as np

def apply_source_view_pruning_old(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, top_k=6):
    """
    原始版本 (有嵌套循环的版本)
    """
    N_rays, N_coarse = z_vals_coarse.shape
    N_rays2, N_importance = z_samples.shape  
    N_rays3, N_total, N_views, _ = fine_mask.shape
    
    assert N_rays == N_rays2 == N_rays3
    assert N_total == N_coarse + N_importance
    
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
                nearest_coarse_idx = 0
            elif insert_pos >= N_coarse:
                nearest_coarse_idx = N_coarse - 1
            else:
                left_idx = insert_pos - 1
                right_idx = insert_pos
                
                dist_left = abs(z_target - z_coarse[left_idx])
                dist_right = abs(z_target - z_coarse[right_idx])
                
                nearest_coarse_idx = left_idx if dist_left <= dist_right else right_idx
            
            assignment[ray_idx, fine_idx] = nearest_coarse_idx
    
    # Step 3: 构建完整的pruned mask
    extended_pruned_mask = torch.zeros_like(fine_mask)
    
    # 复制coarse部分的pruned mask
    extended_pruned_mask[:, :N_coarse, :, :] = pruned_mask_coarse
    
    # 为fine部分分配mask
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

def benchmark_pruning_functions():
    """
    对比新旧版本的性能
    """
    print("=== Source View Pruning Performance Benchmark ===")
    
    # 设置不同的测试参数
    test_configs = [
        {"N_rays": 1024, "N_coarse": 32, "N_importance": 32, "N_views": 8, "name": "Small"},
        {"N_rays": 2048, "N_coarse": 64, "N_importance": 64, "N_views": 8, "name": "Medium"},
        {"N_rays": 4096, "N_coarse": 64, "N_importance": 64, "N_views": 8, "name": "Large"},
        {"N_rays": 4096, "N_coarse": 64, "N_importance": 64, "N_views": 12, "name": "Large+Views"},
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    for config in test_configs:
        N_rays = config["N_rays"]
        N_coarse = config["N_coarse"]
        N_importance = config["N_importance"]
        N_views = config["N_views"]
        N_total = N_coarse + N_importance
        
        print(f"\n--- {config['name']} Test: {N_rays} rays, {N_coarse}+{N_importance} samples, {N_views} views ---")
        
        # 创建测试数据
        z_vals_coarse = torch.linspace(2.0, 6.0, N_coarse).unsqueeze(0).repeat(N_rays, 1).to(device)
        z_samples = 2.0 + torch.rand(N_rays, N_importance, device=device) * 4.0
        
        blending_weights_valid = torch.rand(N_rays, N_coarse, N_views, 1, device=device)
        coarse_mask = torch.ones(N_rays, N_coarse, N_views, 1, device=device)
        fine_mask = torch.ones(N_rays, N_total, N_views, 1, device=device)
        
        # 随机设置一些mask为无效
        invalid_prob = 0.2
        coarse_mask = torch.where(torch.rand_like(coarse_mask) > invalid_prob, coarse_mask, torch.zeros_like(coarse_mask))
        fine_mask = torch.where(torch.rand_like(fine_mask) > invalid_prob, fine_mask, torch.zeros_like(fine_mask))
        
        # 预热GPU
        if device.type == 'cuda':
            for _ in range(3):
                _ = apply_source_view_pruning_old(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, top_k=6)
                torch.cuda.synchronize()
        
        # 测试原始版本
        n_runs = 5
        times_old = []
        
        for _ in range(n_runs):
            start_time = time.time()
            result_old = apply_source_view_pruning_old(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, top_k=6)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times_old.append(end_time - start_time)
        
        avg_time_old = np.mean(times_old)
        std_time_old = np.std(times_old)
        
        # 导入新版本函数
        try:
            from ibrnet.render_ray import apply_source_view_pruning as apply_source_view_pruning_new
            
            # 预热
            if device.type == 'cuda':
                for _ in range(3):
                    _ = apply_source_view_pruning_new(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, top_k=6)
                    torch.cuda.synchronize()
            
            # 测试新版本
            times_new = []
            
            for _ in range(n_runs):
                start_time = time.time()
                result_new = apply_source_view_pruning_new(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, top_k=6)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times_new.append(end_time - start_time)
            
            avg_time_new = np.mean(times_new)
            std_time_new = np.std(times_new)
            
            # 验证结果一致性
            diff = torch.abs(result_old - result_new).max().item()
            results_match = diff < 1e-5
            
            # 计算加速比
            speedup = avg_time_old / avg_time_new
            
            print(f"  原始版本: {avg_time_old*1000:.2f}±{std_time_old*1000:.2f} ms")
            print(f"  优化版本: {avg_time_new*1000:.2f}±{std_time_new*1000:.2f} ms")
            print(f"  加速比: {speedup:.2f}x")
            print(f"  结果一致: {'✓' if results_match else '✗'} (max diff: {diff:.2e})")
            
        except Exception as e:
            print(f"  新版本测试失败: {e}")
            print(f"  原始版本: {avg_time_old*1000:.2f}±{std_time_old*1000:.2f} ms")

if __name__ == "__main__":
    benchmark_pruning_functions()

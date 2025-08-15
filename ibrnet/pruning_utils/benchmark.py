#!/usr/bin/env python3
"""
性能基准测试
对比原始版本和优化版本的pruning性能
"""

import torch
import time
import numpy as np
from .source_view_pruning import apply_source_view_pruning_original, apply_source_view_pruning_optimized

def benchmark_pruning_methods(N_rays=4096, N_coarse=64, N_importance=64, N_views=8, top_k=6, num_runs=5):
    """
    基准测试pruning方法性能
    
    Args:
        N_rays: ray数量
        N_coarse: coarse采样点数量
        N_importance: fine采样点数量  
        N_views: source view数量
        top_k: 保留的top views数量
        num_runs: 运行次数用于平均
    
    Returns:
        benchmark_results: dict包含性能结果
    """
    
    print(f"=== Pruning Performance Benchmark ===")
    print(f"Test Configuration:")
    print(f"  N_rays: {N_rays}")
    print(f"  N_coarse: {N_coarse}")  
    print(f"  N_importance: {N_importance}")
    print(f"  N_views: {N_views}")
    print(f"  top_k: {top_k}")
    print(f"  num_runs: {num_runs}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  device: {device}")
    
    # 创建测试数据
    z_vals_coarse = torch.linspace(2.0, 6.0, N_coarse).unsqueeze(0).repeat(N_rays, 1).to(device)
    z_samples = 2.0 + torch.rand(N_rays, N_importance, device=device) * 4.0
    
    blending_weights_valid = torch.rand(N_rays, N_coarse, N_views, 1, device=device)
    coarse_mask = torch.ones(N_rays, N_coarse, N_views, 1, device=device)
    
    # 随机设置一些mask为无效
    invalid_ratio = 0.2  # 20%的views无效
    for ray_idx in range(0, N_rays, 10):  # 每10个ray设置一些无效views
        invalid_count = int(N_views * invalid_ratio)
        invalid_views = torch.randperm(N_views, device=device)[:invalid_count]
        coarse_mask[ray_idx, :, invalid_views, :] = 0
    
    N_total = N_coarse + N_importance
    fine_mask = torch.ones(N_rays, N_total, N_views, 1, device=device)
    
    print(f"\n开始性能测试...")
    
    # 预热GPU
    if device.type == 'cuda':
        print("预热GPU...")
        for _ in range(3):
            _ = apply_source_view_pruning_optimized(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, top_k)
            torch.cuda.synchronize()
    
    # 测试原始版本
    print("测试原始版本...")
    original_times = []
    
    for run in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        result_original = apply_source_view_pruning_original(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, top_k)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        original_times.append(end_time - start_time)
        print(f"  Run {run+1}: {original_times[-1]:.4f}s")
    
    # 测试优化版本
    print("测试优化版本...")
    optimized_times = []
    
    for run in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        result_optimized = apply_source_view_pruning_optimized(z_vals_coarse, z_samples, blending_weights_valid, coarse_mask, fine_mask, top_k)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        optimized_times.append(end_time - start_time)
        print(f"  Run {run+1}: {optimized_times[-1]:.4f}s")
    
    # 验证结果一致性
    print("验证结果一致性...")
    diff = torch.abs(result_original - result_optimized).max().item()
    print(f"最大差异: {diff}")
    
    if diff < 1e-5:
        print("✓ 结果一致!")
    else:
        print("✗ 结果不一致，可能存在bug")
    
    # 计算统计数据
    original_mean = np.mean(original_times)
    original_std = np.std(original_times)
    optimized_mean = np.mean(optimized_times)
    optimized_std = np.std(optimized_times)
    
    speedup = original_mean / optimized_mean
    
    # 打印结果
    print(f"\n=== Performance Results ===")
    print(f"Original method:")
    print(f"  Mean time: {original_mean:.4f}s ± {original_std:.4f}s")
    print(f"  Best time: {min(original_times):.4f}s")
    print(f"  Worst time: {max(original_times):.4f}s")
    
    print(f"Optimized method:")
    print(f"  Mean time: {optimized_mean:.4f}s ± {optimized_std:.4f}s") 
    print(f"  Best time: {min(optimized_times):.4f}s")
    print(f"  Worst time: {max(optimized_times):.4f}s")
    
    print(f"Performance improvement:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time reduction: {(1-1/speedup)*100:.1f}%")
    
    # 内存使用分析
    if device.type == 'cuda':
        print(f"\nGPU Memory Analysis:")
        print(f"  Current allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")
    
    return {
        'original_times': original_times,
        'optimized_times': optimized_times,
        'original_mean': original_mean,
        'optimized_mean': optimized_mean,
        'speedup': speedup,
        'max_diff': diff,
        'config': {
            'N_rays': N_rays,
            'N_coarse': N_coarse,
            'N_importance': N_importance,
            'N_views': N_views,
            'top_k': top_k,
            'device': str(device)
        }
    }

def run_scaling_benchmark():
    """
    运行不同规模的性能测试
    """
    print("=== Scaling Benchmark ===")
    
    configurations = [
        (1024, 32, 32, 8, 6),   # 小规模
        (2048, 48, 48, 8, 6),   # 中等规模
        (4096, 64, 64, 8, 6),   # 标准规模
        (8192, 64, 64, 8, 6),   # 大规模rays
        (4096, 96, 96, 8, 6),   # 大规模samples
        (4096, 64, 64, 12, 8),  # 大规模views
    ]
    
    results = []
    
    for config in configurations:
        N_rays, N_coarse, N_importance, N_views, top_k = config
        print(f"\n测试配置: N_rays={N_rays}, N_coarse={N_coarse}, N_importance={N_importance}, N_views={N_views}")
        
        result = benchmark_pruning_methods(N_rays, N_coarse, N_importance, N_views, top_k, num_runs=3)
        result['config_tuple'] = config
        results.append(result)
    
    # 打印汇总表格
    print(f"\n=== Scaling Results Summary ===")
    print(f"{'Config':<25} {'Original(s)':<12} {'Optimized(s)':<12} {'Speedup':<8}")
    print("-" * 60)
    
    for result in results:
        config = result['config_tuple']
        config_str = f"{config[0]},{config[1]},{config[2]},{config[3]}"
        print(f"{config_str:<25} {result['original_mean']:<12.4f} {result['optimized_mean']:<12.4f} {result['speedup']:<8.2f}x")
    
    return results

if __name__ == "__main__":
    # 运行基本benchmark
    benchmark_pruning_methods()
    
    # 运行scaling benchmark
    run_scaling_benchmark()

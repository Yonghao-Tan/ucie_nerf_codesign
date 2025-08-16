#!/usr/bin/env python3
"""
分析SR对PSNR的影响
分析不同tile对精度损失的贡献

使用方法:
python analyze_psnr.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
import shutil
from scipy import ndimage, stats
import pandas as pd

def load_image(img_path):
    """
    加载图片并转换为numpy数组
    返回 (H, W, 3) 的float32数组，值范围[0, 1]
    """
    # 使用PIL加载图片
    img = Image.open(img_path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    return img

def calculate_psnr(pred_img, gt_img):
    """
    计算PSNR
    参数:
        pred_img: 预测图片，numpy数组 (H, W, 3)，值范围[0, 1]
        gt_img: 真实图片，numpy数组 (H, W, 3)，值范围[0, 1]
    返回:
        psnr: PSNR值
    """
    # 确保输入维度正确
    if pred_img.shape != gt_img.shape:
        raise ValueError(f"图片尺寸不匹配: {pred_img.shape} vs {gt_img.shape}")
    
    # 计算MSE
    mse = np.mean((pred_img - gt_img) ** 2)
    
    # 处理特殊情况
    if mse == 0:
        return float('inf')  # 如果两张图像完全一样，则 PSNR 是无限大
    
    # 计算PSNR
    max_pixel = 1.0  # 归一化后的最大像素值
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr

def calculate_tile_wise_psnr(pred_img, gt_img, tile_size=16):
    """
    计算每个tile的PSNR
    参数:
        pred_img: 预测图片 (H, W, 3)
        gt_img: 真实图片 (H, W, 3)  
        tile_size: tile大小
    返回:
        tile_psnr_map: 每个tile的PSNR值 (h_tiles, w_tiles)
        tile_coords: 每个tile的坐标信息
    """
    H, W, C = pred_img.shape
    
    # 计算tile数量
    h_tiles = (H + tile_size - 1) // tile_size
    w_tiles = (W + tile_size - 1) // tile_size
    
    tile_psnr_map = np.zeros((h_tiles, w_tiles))
    tile_coords = []
    
    for i in range(h_tiles):
        for j in range(w_tiles):
            # 计算tile边界
            h_start = i * tile_size
            h_end = min(h_start + tile_size, H)
            w_start = j * tile_size
            w_end = min(w_start + tile_size, W)
            
            # 提取tile
            pred_tile = pred_img[h_start:h_end, w_start:w_end, :]
            gt_tile = gt_img[h_start:h_end, w_start:w_end, :]
            
            # 计算该tile的PSNR
            tile_psnr = calculate_psnr(pred_tile, gt_tile)
            tile_psnr_map[i, j] = tile_psnr
            
            # 保存坐标信息
            tile_coords.append({
                'tile_id': (i, j),
                'coords': (h_start, h_end, w_start, w_end),
                'psnr': tile_psnr
            })
    
    return tile_psnr_map, tile_coords

def visualize_tile_psnr(tile_psnr_map, save_path=None, title="Tile-wise PSNR"):
    """
    可视化tile-wise PSNR
    """
    plt.figure(figsize=(12, 8))
    
    # 创建热力图
    im = plt.imshow(tile_psnr_map, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, label='PSNR (dB)')
    plt.title(title)
    plt.xlabel('Width Tiles')
    plt.ylabel('Height Tiles')
    
    # 在每个tile上显示PSNR值
    h_tiles, w_tiles = tile_psnr_map.shape
    for i in range(h_tiles):
        for j in range(w_tiles):
            psnr_val = tile_psnr_map[i, j]
            if np.isfinite(psnr_val):
                plt.text(j, i, f'{psnr_val:.1f}', 
                        ha='center', va='center', 
                        color='white' if psnr_val < np.nanmean(tile_psnr_map) else 'black',
                        fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PSNR热力图保存到: {save_path}")
    
    plt.close()

def save_tile_blocks(org_img, sr_img, gt_img, tile_info_list, save_dir="../results/images/tile_analysis"):
    """
    保存tile图片块到文件夹
    """
    for i, tile_info in enumerate(tile_info_list):
        tile_id = tile_info['tile_id']
        coords = tile_info['coords']
        h_start, h_end, w_start, w_end = coords
        
        # 提取tile
        org_tile = org_img[h_start:h_end, w_start:w_end, :]
        sr_tile = sr_img[h_start:h_end, w_start:w_end, :]
        gt_tile = gt_img[h_start:h_end, w_start:w_end, :]
        
        # 保存GT tile
        gt_path = os.path.join(save_dir, f'rank_{i+1:02d}_tile_{tile_id[0]}_{tile_id[1]}_gt.png')
        plt.imsave(gt_path, gt_tile)
        
        # 保存ORG tile
        org_path = os.path.join(save_dir, f'rank_{i+1:02d}_tile_{tile_id[0]}_{tile_id[1]}_org.png')
        plt.imsave(org_path, org_tile)
        
        # 保存SR tile
        sr_path = os.path.join(save_dir, f'rank_{i+1:02d}_tile_{tile_id[0]}_{tile_id[1]}_sr.png')
        plt.imsave(sr_path, sr_tile)
        
        print(f"保存Rank {i+1} Tile{tile_id}: GT, ORG, SR 图片块")

def visualize_worst_tiles_on_gt(gt_img, tile_info_list, top_k=30, save_path="../results/images/worst_tiles_overlay.png"):
    """
    在GT图片上框出损失最大的tiles
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    
    # 显示GT图片
    ax.imshow(gt_img)
    
    # 为最差的tiles添加框和标注
    colors = plt.cm.Reds(np.linspace(0.3, 1.0, top_k))  # 颜色从浅红到深红
    
    for i, tile_info in enumerate(tile_info_list[:top_k]):
        tile_id = tile_info['tile_id']
        coords = tile_info['coords']
        loss = tile_info['loss']
        h_start, h_end, w_start, w_end = coords
        
        # 绘制半透明矩形框
        alpha = 0.3 + 0.5 * (i / top_k)  # 排名越靠前，透明度越低（越明显）
        rect = plt.Rectangle((w_start, h_start), w_end-w_start, h_end-h_start,
                           linewidth=2, edgecolor=colors[i], facecolor=colors[i], alpha=alpha)
        ax.add_patch(rect)
        
        # 添加文本标注
        text_x = w_start + (w_end - w_start) / 2
        text_y = h_start + (h_end - h_start) / 2
        
        # 根据排名调整字体大小
        fontsize = max(6, 10 - i // 5)
        
        ax.text(text_x, text_y, f'{i+1}\n{loss:.1f}dB', 
               ha='center', va='center', 
               fontsize=fontsize, fontweight='bold',
               color='white', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    ax.set_title(f'GT Image with Top {top_k} Worst Tiles Highlighted\n(Number = Rank, Value = PSNR Loss)', 
                fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"GT图片上的最差tiles可视化保存到: {save_path}")
    plt.close()

def replace_worst_tiles_with_org(sr_img, org_img, tile_info_list, top_k=30, tile_size=16):
    """
    将SR图片中最差的top_k个tiles替换成ORG版本
    
    参数:
        sr_img: SR图片 (H, W, 3)
        org_img: ORG图片 (H, W, 3)
        tile_info_list: 按损失排序的tile信息列表
        top_k: 要替换的最差tiles数量
        tile_size: tile大小
    
    返回:
        hybrid_img: 混合图片（SR + ORG的最差tiles）
        replaced_tiles_info: 被替换的tiles信息
    """
    # 复制SR图片作为基础
    hybrid_img = sr_img.copy()
    replaced_tiles_info = []
    
    print(f"\n=== 替换最差的{top_k}个tiles ===")
    
    for i, tile_info in enumerate(tile_info_list[:top_k]):
        tile_id = tile_info['tile_id']
        coords = tile_info['coords']
        loss = tile_info['loss']
        h_start, h_end, w_start, w_end = coords
        
        # 从ORG图片中提取对应的tile
        org_tile = org_img[h_start:h_end, w_start:w_end, :]
        
        # 替换SR图片中的对应区域
        hybrid_img[h_start:h_end, w_start:w_end, :] = org_tile
        
        replaced_tiles_info.append({
            'rank': i + 1,
            'tile_id': tile_id,
            'coords': coords,
            'loss': loss
        })
        
        print(f"替换Rank {i+1}: Tile{tile_id} - 坐标[{h_start}:{h_end}, {w_start}:{w_end}] - 损失: {loss:.2f}dB")
    
    return hybrid_img, replaced_tiles_info

def analyze_hybrid_performance(hybrid_img, sr_img, org_img, gt_img, replaced_tiles_info):
    """
    分析混合图片的性能表现
    
    参数:
        hybrid_img: 混合图片（SR + ORG的最差tiles）
        sr_img: 原始SR图片
        org_img: ORG图片
        gt_img: GT图片
        replaced_tiles_info: 被替换的tiles信息
    
    返回:
        performance_report: 性能报告字典
    """
    print(f"\n=== 混合图片性能分析 ===")
    
    # 计算各种PSNR
    sr_psnr = calculate_psnr(sr_img, gt_img)
    org_psnr = calculate_psnr(org_img, gt_img) 
    hybrid_psnr = calculate_psnr(hybrid_img, gt_img)
    
    # 计算提升
    hybrid_vs_sr_gain = hybrid_psnr - sr_psnr
    hybrid_vs_org_diff = hybrid_psnr - org_psnr
    
    print(f"原始SR PSNR: {sr_psnr:.2f} dB")
    print(f"原始ORG PSNR: {org_psnr:.2f} dB") 
    print(f"混合图片 PSNR: {hybrid_psnr:.2f} dB")
    print(f"混合图片相比SR提升: {hybrid_vs_sr_gain:.2f} dB")
    print(f"混合图片相比ORG差距: {hybrid_vs_org_diff:.2f} dB")
    
    # 计算理论最大提升（如果所有替换的tiles都达到ORG的性能）
    total_pixels = gt_img.shape[0] * gt_img.shape[1]
    replaced_pixels = len(replaced_tiles_info) * 16 * 16  # 假设都是16x16的tile
    replaced_ratio = replaced_pixels / total_pixels
    
    print(f"替换的tile数量: {len(replaced_tiles_info)}")
    print(f"替换的像素比例: {replaced_ratio:.1%}")
    
    # 保存混合图片
    hybrid_path = "../results/images/hybrid_sr_org.png"
    plt.imsave(hybrid_path, hybrid_img)
    print(f"混合图片保存到: {hybrid_path}")
    
    # 创建对比可视化
    create_comparison_visualization(sr_img, hybrid_img, org_img, gt_img, 
                                  sr_psnr, hybrid_psnr, org_psnr)
    
    performance_report = {
        'sr_psnr': sr_psnr,
        'org_psnr': org_psnr,
        'hybrid_psnr': hybrid_psnr,
        'gain_vs_sr': hybrid_vs_sr_gain,
        'diff_vs_org': hybrid_vs_org_diff,
        'replaced_tiles_count': len(replaced_tiles_info),
        'replaced_pixel_ratio': replaced_ratio
    }
    
    return performance_report

def create_comparison_visualization(sr_img, hybrid_img, org_img, gt_img, 
                                  sr_psnr, hybrid_psnr, org_psnr):
    """
    创建四张图片的对比可视化
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # GT
    axes[0, 0].imshow(gt_img)
    axes[0, 0].set_title('Ground Truth', fontsize=14)
    axes[0, 0].axis('off')
    
    # ORG
    axes[0, 1].imshow(org_img)
    axes[0, 1].set_title(f'ORG\nPSNR: {org_psnr:.2f} dB', fontsize=14)
    axes[0, 1].axis('off')
    
    # SR
    axes[1, 0].imshow(sr_img)
    axes[1, 0].set_title(f'SR\nPSNR: {sr_psnr:.2f} dB', fontsize=14)
    axes[1, 0].axis('off')
    
    # Hybrid
    axes[1, 1].imshow(hybrid_img)
    axes[1, 1].set_title(f'Hybrid (SR + ORG tiles)\nPSNR: {hybrid_psnr:.2f} dB\n(+{hybrid_psnr - sr_psnr:.2f} dB vs SR)', fontsize=14)
    axes[1, 1].axis('off')
    
    plt.suptitle('Performance Comparison: Original vs Hybrid Approach', fontsize=16)
    plt.tight_layout()
    plt.savefig('../results/images/comparison_sr_hybrid_org.png', dpi=300, bbox_inches='tight')
    print(f"对比可视化保存到: ../results/images/comparison_sr_hybrid_org.png")
    plt.close()

def calculate_tile_features(img, coords, feature_type="all"):
    """
    计算单个tile的各种特征
    """
    h_start, h_end, w_start, w_end = coords
    tile = img[h_start:h_end, w_start:w_end, :]
    
    features = {}
    
    if feature_type in ["all", "basic"]:
        # 基础统计特征
        features['mean'] = np.mean(tile)
        features['std'] = np.std(tile)
        features['min'] = np.min(tile)
        features['max'] = np.max(tile)
        features['range'] = features['max'] - features['min']
        
    if feature_type in ["all", "frequency"]:
        # 频域特征
        gray_tile = np.mean(tile, axis=2)  # 转灰度
        f_transform = np.fft.fft2(gray_tile)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # 高频能量比例
        h, w = gray_tile.shape
        center_h, center_w = h//2, w//2
        
        # 定义高频区域（距离中心较远的区域）
        y, x = np.ogrid[:h, :w]
        mask_high = ((x - center_w)**2 + (y - center_h)**2) > (min(h, w)//4)**2
        
        high_freq_energy = np.sum(magnitude_spectrum[mask_high])
        total_energy = np.sum(magnitude_spectrum)
        features['high_freq_ratio'] = high_freq_energy / (total_energy + 1e-8)
        
    if feature_type in ["all", "edge"]:
        # 边缘特征
        gray_tile = np.mean(tile, axis=2)
        
        # Sobel算子计算梯度
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # 使用scipy.ndimage的方式计算梯度
        from scipy import ndimage
        grad_x = ndimage.convolve(gray_tile, sobel_x)
        grad_y = ndimage.convolve(gray_tile, sobel_y)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features['edge_density'] = np.mean(gradient_magnitude)
        features['edge_max'] = np.max(gradient_magnitude)
        features['edge_std'] = np.std(gradient_magnitude)
        
    if feature_type in ["all", "texture"]:
        # 纹理特征 - 局部二值模式(LBP)近似
        gray_tile = np.mean(tile, axis=2)
        
        # 计算局部方差作为纹理复杂度指标
        from scipy import ndimage
        local_mean = ndimage.uniform_filter(gray_tile, size=3)
        local_variance = ndimage.uniform_filter(gray_tile**2, size=3) - local_mean**2
        features['texture_complexity'] = np.mean(local_variance)
        features['texture_max'] = np.max(local_variance)
        
    if feature_type in ["all", "color"]:
        # 颜色特征
        for i, channel in enumerate(['r', 'g', 'b']):
            features[f'{channel}_mean'] = np.mean(tile[:, :, i])
            features[f'{channel}_std'] = np.std(tile[:, :, i])
        
        # 颜色饱和度
        rgb_max = np.max(tile, axis=2)
        rgb_min = np.min(tile, axis=2)
        saturation = (rgb_max - rgb_min) / (rgb_max + 1e-8)
        features['saturation_mean'] = np.mean(saturation)
        features['saturation_std'] = np.std(saturation)
        
    return features

def analyze_worst_tiles_features(gt_img, org_img, sr_img, tile_loss_info, top_k=100):
    """
    分析最差的K个tiles的特征
    """
    print(f"\n{'='*60}")
    print(f"{'='*15} 最差{top_k}个Tiles特征分析 {'='*15}")
    print(f"{'='*60}")
    
    worst_tiles = tile_loss_info[:top_k]
    best_tiles = sorted(tile_loss_info, key=lambda x: x['loss'])[:top_k]  # 损失最小的100个
    
    print(f"分析最差的{top_k}个tiles vs 最好的{top_k}个tiles")
    
    # 收集特征
    worst_features = []
    best_features = []
    
    print("计算最差tiles的特征...")
    for i, tile_info in enumerate(worst_tiles):
        coords = tile_info['coords']
        
        # 计算GT tile的特征
        gt_features = calculate_tile_features(gt_img, coords, "all")
        gt_features['tile_id'] = tile_info['tile_id']
        gt_features['loss'] = tile_info['loss']
        gt_features['org_psnr'] = tile_info['org_psnr']
        gt_features['sr_psnr'] = tile_info['sr_psnr']
        
        worst_features.append(gt_features)
        
        if (i + 1) % 20 == 0:
            print(f"  已处理 {i+1}/{top_k} 个最差tiles")
    
    print("计算最好tiles的特征...")
    for i, tile_info in enumerate(best_tiles):
        coords = tile_info['coords']
        
        gt_features = calculate_tile_features(gt_img, coords, "all")
        gt_features['tile_id'] = tile_info['tile_id']
        gt_features['loss'] = tile_info['loss']
        gt_features['org_psnr'] = tile_info['org_psnr']
        gt_features['sr_psnr'] = tile_info['sr_psnr']
        
        best_features.append(gt_features)
        
        if (i + 1) % 20 == 0:
            print(f"  已处理 {i+1}/{top_k} 个最好tiles")
    
    # 统计分析
    feature_names = [k for k in worst_features[0].keys() if k not in ['tile_id', 'loss', 'org_psnr', 'sr_psnr']]
    
    print(f"\n=== 特征对比分析 ===")
    comparison_results = {}
    
    for feature_name in feature_names:
        worst_values = [f[feature_name] for f in worst_features]
        best_values = [f[feature_name] for f in best_features]
        
        worst_mean = np.mean(worst_values)
        best_mean = np.mean(best_values)
        worst_std = np.std(worst_values)
        best_std = np.std(best_values)
        
        # 计算统计显著性 (简单的t-test近似)
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(worst_values, best_values)
        
        comparison_results[feature_name] = {
            'worst_mean': worst_mean,
            'worst_std': worst_std,
            'best_mean': best_mean,
            'best_std': best_std,
            'difference': worst_mean - best_mean,
            'relative_diff': (worst_mean - best_mean) / (best_mean + 1e-8),
            't_stat': t_stat,
            'p_value': p_value
        }
        
        # 打印显著差异的特征
        if p_value < 0.01:  # 高显著性
            significance = "***"
        elif p_value < 0.05:  # 显著性
            significance = "**"
        elif p_value < 0.1:  # 边缘显著性
            significance = "*"
        else:
            significance = ""
            
        print(f"{feature_name:20s}: 最差={worst_mean:.4f}±{worst_std:.4f}, "
              f"最好={best_mean:.4f}±{best_std:.4f}, "
              f"差异={worst_mean-best_mean:+.4f} ({(worst_mean-best_mean)/(best_mean+1e-8)*100:+.1f}%) {significance}")
    
    # 保存详细分析结果
    save_feature_analysis_results(comparison_results, worst_features, best_features, top_k)
    
    # 创建特征可视化
    create_feature_visualizations(comparison_results, worst_features, best_features, top_k)
    
    return comparison_results, worst_features, best_features

def save_feature_analysis_results(comparison_results, worst_features, best_features, top_k):
    """
    保存特征分析结果到文件
    """
    report_path = f"../results/reports/tile_feature_analysis_top{top_k}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"=== 最差{top_k}个Tiles vs 最好{top_k}个Tiles 特征分析报告 ===\n\n")
        
        # 按显著性排序
        sorted_features = sorted(comparison_results.items(), 
                               key=lambda x: abs(x[1]['relative_diff']), reverse=True)
        
        f.write("特征重要性排序 (按相对差异绝对值排序):\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'特征名':<20} {'最差均值':<12} {'最好均值':<12} {'绝对差异':<12} {'相对差异':<12} {'P值':<10} {'显著性':<8}\n")
        f.write("-" * 100 + "\n")
        
        for feature_name, stats in sorted_features:
            significance = ""
            if stats['p_value'] < 0.01:
                significance = "***"
            elif stats['p_value'] < 0.05:
                significance = "**"
            elif stats['p_value'] < 0.1:
                significance = "*"
                
            f.write(f"{feature_name:<20} {stats['worst_mean']:<12.4f} {stats['best_mean']:<12.4f} "
                   f"{stats['difference']:<12.4f} {stats['relative_diff']*100:<11.1f}% "
                   f"{stats['p_value']:<10.4f} {significance:<8}\n")
        
        f.write("\n显著性标记: *** p<0.01, ** p<0.05, * p<0.1\n")
        
        # 关键发现总结
        f.write("\n=== 关键发现总结 ===\n")
        highly_significant = [name for name, stats in comparison_results.items() if stats['p_value'] < 0.01]
        f.write(f"高度显著差异的特征 (p<0.01): {len(highly_significant)}个\n")
        for feature in highly_significant[:10]:  # 只列出前10个
            stats = comparison_results[feature]
            f.write(f"  - {feature}: 相对差异 {stats['relative_diff']*100:.1f}%\n")
    
    print(f"特征分析报告保存到: {report_path}")

def create_feature_visualizations(comparison_results, worst_features, best_features, top_k):
    """
    创建特征可视化图表
    """
    # 选择最重要的特征进行可视化
    sorted_features = sorted(comparison_results.items(), 
                           key=lambda x: abs(x[1]['relative_diff']), reverse=True)
    
    top_features = [name for name, _ in sorted_features[:12]]  # 选择前12个最重要的特征
    
    # 创建子图
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.ravel()
    
    for i, feature_name in enumerate(top_features):
        ax = axes[i]
        
        worst_values = [f[feature_name] for f in worst_features]
        best_values = [f[feature_name] for f in best_features]
        
        # 创建箱型图
        data = [worst_values, best_values]
        bp = ax.boxplot(data, labels=[f'最差{top_k}个', f'最好{top_k}个'], patch_artist=True)
        
        # 设置颜色
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightblue')
        
        ax.set_title(f'{feature_name}\n(相对差异: {comparison_results[feature_name]["relative_diff"]*100:.1f}%)')
        ax.grid(True, alpha=0.3)
        
        # 添加统计显著性标记
        p_value = comparison_results[feature_name]['p_value']
        if p_value < 0.01:
            ax.text(0.5, 0.95, '***', transform=ax.transAxes, ha='center', fontsize=16, color='red')
        elif p_value < 0.05:
            ax.text(0.5, 0.95, '**', transform=ax.transAxes, ha='center', fontsize=14, color='orange')
        elif p_value < 0.1:
            ax.text(0.5, 0.95, '*', transform=ax.transAxes, ha='center', fontsize=12, color='blue')
    
    plt.tight_layout()
    plt.savefig(f'../results/images/tile_feature_comparison_top{top_k}.png', dpi=300, bbox_inches='tight')
    print(f"特征对比可视化保存到: ../results/images/tile_feature_comparison_top{top_k}.png")
    plt.close()
    
    # 创建特征相关性热力图
    create_feature_correlation_heatmap(worst_features, best_features, top_k)

def create_feature_correlation_heatmap(worst_features, best_features, top_k):
    """
    创建特征相关性热力图
    """
    import pandas as pd
    
    # 准备数据
    feature_names = [k for k in worst_features[0].keys() 
                    if k not in ['tile_id', 'loss', 'org_psnr', 'sr_psnr']]
    
    # 创建数据框
    worst_df = pd.DataFrame(worst_features)[feature_names]
    best_df = pd.DataFrame(best_features)[feature_names]
    
    # 计算相关性矩阵
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # 最差tiles的特征相关性
    worst_corr = worst_df.corr()
    im1 = ax1.imshow(worst_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(range(len(feature_names)))
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_xticklabels(feature_names, rotation=45, ha='right')
    ax1.set_yticklabels(feature_names)
    ax1.set_title(f'最差{top_k}个Tiles - 特征相关性')
    
    # 添加相关性数值
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            ax1.text(j, i, f'{worst_corr.iloc[i, j]:.2f}', 
                    ha='center', va='center', color='black' if abs(worst_corr.iloc[i, j]) < 0.5 else 'white',
                    fontsize=8)
    
    # 最好tiles的特征相关性
    best_corr = best_df.corr()
    im2 = ax2.imshow(best_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xticks(range(len(feature_names)))
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_xticklabels(feature_names, rotation=45, ha='right')
    ax2.set_yticklabels(feature_names)
    ax2.set_title(f'最好{top_k}个Tiles - 特征相关性')
    
    # 添加相关性数值
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            ax2.text(j, i, f'{best_corr.iloc[i, j]:.2f}', 
                    ha='center', va='center', color='black' if abs(best_corr.iloc[i, j]) < 0.5 else 'white',
                    fontsize=8)
    
    # 添加颜色条
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(f'../results/images/tile_feature_correlation_top{top_k}.png', dpi=300, bbox_inches='tight')
    print(f"特征相关性热力图保存到: ../results/images/tile_feature_correlation_top{top_k}.png")
    plt.close()

def test_hybrid_with_different_k(sr_img, org_img, gt_img, tile_loss_info, k_values=[10, 30, 50, 100, 200]):
    """
    测试不同K值下的混合图片性能
    """
    print(f"\n{'='*60}")
    print(f"{'='*15} 不同K值混合图片性能测试 {'='*15}")
    print(f"{'='*60}")
    
    original_sr_psnr = calculate_psnr(sr_img, gt_img)
    original_org_psnr = calculate_psnr(org_img, gt_img)
    
    results = []
    
    for k in k_values:
        if k > len(tile_loss_info):
            print(f"K={k} 超过了总tile数量，跳过")
            continue
            
        # 创建混合图片
        hybrid_img, _ = replace_worst_tiles_with_org(sr_img, org_img, tile_loss_info, top_k=k, tile_size=32)
        hybrid_psnr = calculate_psnr(hybrid_img, gt_img)
        
        gain = hybrid_psnr - original_sr_psnr
        pixel_ratio = k * 32 * 32 / (sr_img.shape[0] * sr_img.shape[1])
        
        results.append({
            'k': k,
            'hybrid_psnr': hybrid_psnr,
            'gain_vs_sr': gain,
            'pixel_ratio': pixel_ratio,
            'efficiency': gain / pixel_ratio if pixel_ratio > 0 else 0  # 增益/像素比例
        })
        
        print(f"K={k:3d}: 混合PSNR={hybrid_psnr:.2f}dB, 增益={gain:.2f}dB, "
              f"像素比例={pixel_ratio:.1%}, 效率={gain/pixel_ratio:.2f}")
    
    # 保存K值测试结果
    save_k_test_results(results, original_sr_psnr, original_org_psnr)
    
    # 可视化K值测试结果
    visualize_k_test_results(results)
    
    return results

def save_k_test_results(results, sr_psnr, org_psnr):
    """
    保存K值测试结果
    """
    report_path = "../results/reports/k_value_test_results.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 不同K值混合图片性能测试结果 ===\n\n")
        f.write(f"原始SR PSNR: {sr_psnr:.2f} dB\n")
        f.write(f"原始ORG PSNR: {org_psnr:.2f} dB\n\n")
        
        f.write(f"{'K值':<6} {'混合PSNR':<10} {'增益':<8} {'像素比例':<10} {'效率':<8}\n")
        f.write("-" * 50 + "\n")
        
        for result in results:
            f.write(f"{result['k']:<6} {result['hybrid_psnr']:<10.2f} "
                   f"{result['gain_vs_sr']:<8.2f} {result['pixel_ratio']:<10.1%} "
                   f"{result['efficiency']:<8.2f}\n")
        
        # 找到最佳效率点
        best_efficiency = max(results, key=lambda x: x['efficiency'])
        f.write(f"\n最佳效率点: K={best_efficiency['k']}, 效率={best_efficiency['efficiency']:.2f}\n")
        
    print(f"K值测试结果保存到: {report_path}")

def visualize_k_test_results(results):
    """
    可视化K值测试结果
    """
    k_values = [r['k'] for r in results]
    gains = [r['gain_vs_sr'] for r in results]
    pixel_ratios = [r['pixel_ratio'] * 100 for r in results]  # 转换为百分比
    efficiencies = [r['efficiency'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 增益 vs K值
    ax1.plot(k_values, gains, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('替换的Tiles数量 (K)')
    ax1.set_ylabel('PSNR增益 (dB)')
    ax1.set_title('PSNR增益 vs 替换Tiles数量')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(k_values) * 1.05)
    
    # 添加像素比例作为第二个y轴
    ax1_twin = ax1.twinx()
    ax1_twin.plot(k_values, pixel_ratios, 's--', color='red', alpha=0.7)
    ax1_twin.set_ylabel('替换像素比例 (%)', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    # 效率 vs K值
    ax2.plot(k_values, efficiencies, 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('替换的Tiles数量 (K)')
    ax2.set_ylabel('效率 (增益/像素比例)')
    ax2.set_title('替换效率 vs Tiles数量')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(k_values) * 1.05)
    
    # 标注最佳效率点
    best_idx = np.argmax(efficiencies)
    ax2.annotate(f'最佳效率\nK={k_values[best_idx]}', 
                xy=(k_values[best_idx], efficiencies[best_idx]),
                xytext=(k_values[best_idx] + max(k_values)*0.1, efficiencies[best_idx]),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig('../results/images/k_value_performance_analysis.png', dpi=300, bbox_inches='tight')
    print(f"K值性能分析图保存到: k_value_performance_analysis.png")
    plt.close()

def main():
    # 创建tile分析文件夹
    tile_analysis_dir = "../results/images/tile_analysis"
    # 如果文件夹已存在，则先清空
    if os.path.exists(tile_analysis_dir):
        shutil.rmtree(tile_analysis_dir)  # 删除文件夹及其内容
        print(f"清空并删除原有tile分析文件夹: {tile_analysis_dir}")
    os.makedirs(tile_analysis_dir, exist_ok=True)
    print(f"创建tile分析文件夹: {tile_analysis_dir}")
    
    gt_path = '/home/ytanaz/access/IBRNet/eval/llff_test/eval_llff_sr/horns/0_gt_rgb_hr.png'
    org_path = '/home/ytanaz/access/IBRNet/eval/llff_test/eval_llff/horns/0_pred_fine.png'
    sr_path = '/home/ytanaz/access/IBRNet/eval/llff_test/eval_llff_sr/horns/0_pred_sr.png'

    # 检查文件是否存在
    for path, name in [(gt_path, "GT"), (org_path, "ORG"), (sr_path, "SR")]:
        if not os.path.exists(path):
            print(f"错误: {name} 图片不存在: {path}")
            return
    
    print("=== SR精度分析 ===")
    print(f"加载图片...")
    
    # 加载图片
    try:
        gt_img = load_image(gt_path)
        org_img = load_image(org_path)
        sr_img = load_image(sr_path)
        
        print(f"GT图片尺寸: {gt_img.shape}")
        print(f"ORG图片尺寸: {org_img.shape}")
        print(f"SR图片尺寸: {sr_img.shape}")
        
    except Exception as e:
        print(f"加载图片时出错: {e}")
        return
    
    # 检查尺寸一致性
    if gt_img.shape != org_img.shape or gt_img.shape != sr_img.shape:
        print("警告: 图片尺寸不一致，尝试resize...")
        target_shape = gt_img.shape[:2]
        
        if org_img.shape[:2] != target_shape:
            org_img_pil = Image.fromarray((org_img * 255).astype(np.uint8))
            org_img_pil = org_img_pil.resize((target_shape[1], target_shape[0]))
            org_img = np.array(org_img_pil).astype(np.float32) / 255.0
            print(f"ORG图片已resize到: {org_img.shape}")
            
        if sr_img.shape[:2] != target_shape:
            sr_img_pil = Image.fromarray((sr_img * 255).astype(np.uint8))
            sr_img_pil = sr_img_pil.resize((target_shape[1], target_shape[0]))
            sr_img = np.array(sr_img_pil).astype(np.float32) / 255.0
            print(f"SR图片已resize到: {sr_img.shape}")
    
    # 计算整体PSNR
    print("\n=== 整体PSNR对比 ===")
    
    org_psnr = calculate_psnr(org_img, gt_img)
    sr_psnr = calculate_psnr(sr_img, gt_img)
    
    print(f"ORG vs GT PSNR: {org_psnr:.2f} dB")
    print(f"SR vs GT PSNR: {sr_psnr:.2f} dB")
    print(f"SR相对损失: {org_psnr - sr_psnr:.2f} dB")
    
    # Tile-wise分析
    print("\n=== Tile-wise PSNR分析 ===")
    
    tile_size = 32
    print(f"使用tile大小: {tile_size}x{tile_size}")
    
    # 计算每个tile的PSNR
    org_tile_psnr, org_tile_coords = calculate_tile_wise_psnr(org_img, gt_img, tile_size)
    sr_tile_psnr, sr_tile_coords = calculate_tile_wise_psnr(sr_img, gt_img, tile_size)
    
    # 计算PSNR差异
    psnr_diff = org_tile_psnr - sr_tile_psnr
    
    print(f"Tile数量: {org_tile_psnr.shape[0]} x {org_tile_psnr.shape[1]} = {org_tile_psnr.size}")
    print(f"ORG平均tile PSNR: {np.nanmean(org_tile_psnr):.2f} dB")
    print(f"SR平均tile PSNR: {np.nanmean(sr_tile_psnr):.2f} dB")
    print(f"平均PSNR损失: {np.nanmean(psnr_diff):.2f} dB")
    print(f"最大PSNR损失: {np.nanmax(psnr_diff):.2f} dB")
    print(f"最小PSNR损失: {np.nanmin(psnr_diff):.2f} dB")
    
    # 可视化结果
    print("\n=== 生成可视化结果 ===")
    
    # 可视化ORG tile PSNR
    visualize_tile_psnr(org_tile_psnr, 
                       save_path="../results/images/org_tile_psnr.png",
                       title="ORG vs GT - Tile-wise PSNR")
    
    # 可视化SR tile PSNR  
    visualize_tile_psnr(sr_tile_psnr,
                       save_path="../results/images/sr_tile_psnr.png", 
                       title="SR vs GT - Tile-wise PSNR")
    
    # 可视化PSNR差异
    visualize_tile_psnr(psnr_diff,
                       save_path="../results/images/psnr_diff.png",
                       title="PSNR Difference (ORG - SR)")
    
    # 分析最差的tiles (SR相比ORG损失最大的)
    # 计算每个tile的损失
    tile_loss_info = []
    for i, (org_info, sr_info) in enumerate(zip(org_tile_coords, sr_tile_coords)):
        loss = org_info['psnr'] - sr_info['psnr']
        tile_loss_info.append({
            'tile_id': org_info['tile_id'],
            'coords': org_info['coords'],
            'org_psnr': org_info['psnr'],
            'sr_psnr': sr_info['psnr'],
            'loss': loss
        })
    
    # 按损失排序
    tile_loss_info.sort(key=lambda x: x['loss'], reverse=True)
    
    # 在GT图片上可视化最差的30个tiles
    visualize_worst_tiles_on_gt(gt_img, tile_loss_info, top_k=30, 
                               save_path="../results/images/worst_tiles_overlay.png")
    
    print("\n=== 损失最大的10个tiles ===")
    for i, info in enumerate(tile_loss_info[:10]):
        tile_id = info['tile_id']
        coords = info['coords'] 
        org_psnr = info['org_psnr']
        sr_psnr = info['sr_psnr']
        loss = info['loss']
        
        h_start, h_end, w_start, w_end = coords
        print(f"Rank {i+1}: Tile{tile_id} - 坐标[{h_start}:{h_end}, {w_start}:{w_end}]")
        print(f"  ORG PSNR: {org_psnr:.2f} dB, SR PSNR: {sr_psnr:.2f} dB, 损失: {loss:.2f} dB")
    
    # 保存最差的10个tiles的图片块
    print(f"\n=== 保存最差10个tiles的图片块到 {tile_analysis_dir} ===")
    save_tile_blocks(org_img, sr_img, gt_img, tile_loss_info[:10], tile_analysis_dir)
    
    # 新增：混合图片分析 - 用ORG替换最差的30个tiles
    print(f"\n{'='*60}")
    print(f"{'='*20} 混合图片分析 {'='*20}")
    print(f"{'='*60}")
    K_mix = 100
    
    # 创建混合图片：用ORG替换SR中最差的30个tiles
    hybrid_img, replaced_tiles_info = replace_worst_tiles_with_org(
        sr_img, org_img, tile_loss_info, top_k=K_mix, tile_size=tile_size)
    
    # 分析混合图片的性能
    performance_report = analyze_hybrid_performance(
        hybrid_img, sr_img, org_img, gt_img, replaced_tiles_info)
    
    # 保存性能报告
    report_path = "hybrid_performance_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 混合图片性能分析报告 ===\n\n")
        f.write(f"原始SR PSNR: {performance_report['sr_psnr']:.2f} dB\n")
        f.write(f"原始ORG PSNR: {performance_report['org_psnr']:.2f} dB\n")
        f.write(f"混合图片 PSNR: {performance_report['hybrid_psnr']:.2f} dB\n")
        f.write(f"混合图片相比SR提升: {performance_report['gain_vs_sr']:.2f} dB\n")
        f.write(f"混合图片相比ORG差距: {performance_report['diff_vs_org']:.2f} dB\n")
        f.write(f"替换的tile数量: {performance_report['replaced_tiles_count']}\n")
        f.write(f"替换的像素比例: {performance_report['replaced_pixel_ratio']:.1%}\n\n")
        
        f.write("被替换的tiles详情:\n")
        for info in replaced_tiles_info:
            rank = info['rank']
            tile_id = info['tile_id']
            coords = info['coords']
            loss = info['loss']
            h_start, h_end, w_start, w_end = coords
            f.write(f"Rank {rank}: Tile{tile_id} - 坐标[{h_start}:{h_end}, {w_start}:{w_end}] - 损失: {loss:.2f}dB\n")
    
    print(f"混合图片性能报告保存到: {report_path}")
    
    # 新增：深度特征分析
    print(f"\n{'='*60}")
    print(f"{'='*15} 深度特征分析部分 {'='*15}")
    print(f"{'='*60}")
    
    # 1. 分析最差100个tiles的特征
    feature_comparison, worst_features, best_features = analyze_worst_tiles_features(
        gt_img, org_img, sr_img, tile_loss_info, top_k=100)
    
    # 2. 测试不同K值的混合图片性能
    k_test_results = test_hybrid_with_different_k(
        sr_img, org_img, gt_img, tile_loss_info, k_values=[10, 30, 50, 100, 200, 300])
    
    # 保存详细结果
    np.savetxt("../results/reports/org_tile_psnr.txt", org_tile_psnr, fmt='%.2f')
    np.savetxt("../results/reports/sr_tile_psnr.txt", sr_tile_psnr, fmt='%.2f')
    np.savetxt("../results/reports/psnr_diff.txt", psnr_diff, fmt='%.2f')
    
    print(f"\n结果已保存:")
    print(f"- org_tile_psnr.png/txt: ORG方法的tile-wise PSNR")
    print(f"- sr_tile_psnr.png/txt: SR方法的tile-wise PSNR") 
    print(f"- psnr_diff.png/txt: PSNR差异热力图")
    print(f"- worst_tiles_overlay.png: GT图片上标注最差30个tiles")
    print(f"- {tile_analysis_dir}/: 损失最大的10个tiles的GT/ORG/SR图片块")
    print(f"- hybrid_sr_org.png: 混合图片（SR + ORG的最差30个tiles）")
    print(f"- comparison_sr_hybrid_org.png: 四张图片的对比可视化")
    print(f"- {report_path}: 混合图片性能分析报告")

if __name__ == "__main__":
    main()
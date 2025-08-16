#!/usr/bin/env python3
"""
低分辨率边缘检测 + 高分辨率映射方法
在fine分辨率图像上进行边缘检测，然后将决策映射到高分辨率图像
模拟真实SR流程：低分辨率输入 → 边缘检测 → 高分辨率输出
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.transform import resize

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class LowResEdgePredictor:
    """低分辨率边缘检测预测器"""
    
    def __init__(self, method='canny', mode='balanced', scale_factor=2):
        """
        Args:
            method: 边缘检测方法
            mode: 预设模式
            scale_factor: 超分倍数（fine → sr的放大倍数）
        """
        self.method = method
        self.mode = mode
        self.scale_factor = scale_factor
        self.fine_tile_size = 16  # fine图像上的tile大小（原来32的一半）
        self.sr_tile_size = 32    # 对应SR图像上的tile大小
        
        # 预设阈值（与之前相同）
        self.preset_thresholds = {
            'canny': {
                'aggressive': 0.047,
                'balanced': 0.094,
                'conservative': 0.141
            },
            'sobel': {
                'aggressive': 29.5,
                'balanced': 58.9,
                'conservative': 88.4
            },
            'gradient': {
                'aggressive': 4.0,
                'balanced': 8.0,
                'conservative': 12.0
            },
            'combined': {
                'aggressive': 0.20,
                'balanced': 0.40,
                'conservative': 0.60
            }
        }
        
        self.threshold = self.preset_thresholds[method][mode]
        
        print(f"🔍 低分辨率边缘检测预测器初始化")
        print(f"方法: {method}")
        print(f"模式: {mode}")
        print(f"阈值: {self.threshold}")
        print(f"超分倍数: {scale_factor}×")
        print(f"Fine tile大小: {self.fine_tile_size}×{self.fine_tile_size}")
        print(f"SR tile大小: {self.sr_tile_size}×{self.sr_tile_size}")
    
    def extract_edge_score(self, tile):
        """提取单个tile的边缘复杂度得分"""
        if tile.max() > 1.0:
            tile = tile.astype(np.float32) / 255.0
            
        # 转换为灰度图
        if len(tile.shape) == 3:
            gray = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (tile * 255).astype(np.uint8)
            
        if self.method == 'canny':
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / edges.size
            return edge_ratio
            
        elif self.method == 'sobel':
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            return np.mean(sobel_magnitude)
            
        elif self.method == 'gradient':
            grad_x = np.gradient(gray.astype(np.float32), axis=1)
            grad_y = np.gradient(gray.astype(np.float32), axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            return np.mean(gradient_magnitude)
            
        elif self.method == 'combined':
            # Canny边缘比例
            edges = cv2.Canny(gray, 50, 150)
            canny_score = np.sum(edges > 0) / edges.size
            
            # Sobel边缘密度
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_score = min(np.mean(sobel_magnitude) / 50.0, 1.0)
            
            # 梯度统计
            grad_x = np.gradient(gray.astype(np.float32), axis=1)
            grad_y = np.gradient(gray.astype(np.float32), axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_score = min(np.mean(gradient_magnitude) / 30.0, 1.0)
            
            # 加权组合
            combined_score = 0.5 * canny_score + 0.3 * sobel_score + 0.2 * gradient_score
            return combined_score
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def should_replace_tile(self, fine_tile):
        """在低分辨率tile上判断是否需要替换"""
        edge_score = self.extract_edge_score(fine_tile)
        return edge_score > self.threshold
    
    def create_replacement_mask(self, fine_img):
        """
        在fine图像上创建替换掩码
        
        Returns:
            replacement_mask: 与fine图像同尺寸的布尔掩码
        """
        h, w = fine_img.shape[:2]
        tile_h, tile_w = h // self.fine_tile_size, w // self.fine_tile_size
        
        # 创建掩码
        replacement_mask = np.zeros((h, w), dtype=bool)
        tile_decisions = []
        
        print(f"在fine图像({h}×{w})上分析 {tile_h}×{tile_w} = {tile_h*tile_w} 个tiles...")
        
        for i in range(tile_h):
            for j in range(tile_w):
                h_start = i * self.fine_tile_size
                h_end = (i + 1) * self.fine_tile_size
                w_start = j * self.fine_tile_size
                w_end = (j + 1) * self.fine_tile_size
                
                fine_tile = fine_img[h_start:h_end, w_start:w_end]
                
                # 在fine分辨率上判断
                edge_score = self.extract_edge_score(fine_tile)
                should_replace = edge_score > self.threshold
                
                tile_decisions.append({
                    'tile_coord': (i, j),
                    'edge_score': edge_score,
                    'should_replace': should_replace
                })
                
                if should_replace:
                    replacement_mask[h_start:h_end, w_start:w_end] = True
        
        replaced_tiles = sum(1 for d in tile_decisions if d['should_replace'])
        print(f"Fine图像分析完成: {replaced_tiles}/{len(tile_decisions)} tiles需要替换 ({replaced_tiles/len(tile_decisions)*100:.1f}%)")
        
        return replacement_mask, tile_decisions
    
    def upscale_mask(self, fine_mask, target_shape):
        """将fine掩码上采样到目标分辨率"""
        # 使用最近邻插值保持布尔值
        upscaled_mask = cv2.resize(
            fine_mask.astype(np.uint8), 
            (target_shape[1], target_shape[0]), 
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        
        return upscaled_mask

def test_lowres_edge_method(method='canny', mode='balanced'):
    """测试低分辨率边缘检测方法"""
    
    print(f"🔍 测试低分辨率边缘检测方法")
    print(f"方法: {method}, 模式: {mode}")
    print("="*60)
    
    # 加载图像
    fine_path = "../data/no2_pred_fine.png"
    sr_path = "../data/no2_pred_sr.png" 
    org_path = "../data/no2_pred_org.png"
    gt_path = "../data/no2_gt.png"
    
    fine_img = np.array(Image.open(fine_path))
    sr_img = np.array(Image.open(sr_path))
    org_img = np.array(Image.open(org_path))
    gt_img = np.array(Image.open(gt_path))
    
    print(f"图像尺寸:")
    print(f"- Fine: {fine_img.shape}")
    print(f"- SR: {sr_img.shape}")
    print(f"- ORG: {org_img.shape}")
    print(f"- GT: {gt_img.shape}")
    
    # 计算原始PSNR
    sr_psnr = psnr(gt_img, sr_img)
    print(f"\n原始SR PSNR: {sr_psnr:.2f} dB")
    
    # 创建预测器
    scale_factor = sr_img.shape[0] // fine_img.shape[0]
    predictor = LowResEdgePredictor(method=method, mode=mode, scale_factor=scale_factor)
    
    # 1. 在fine图像上进行边缘检测
    print(f"\n步骤1: 在Fine图像上进行边缘检测...")
    fine_mask, tile_decisions = predictor.create_replacement_mask(fine_img)
    
    # 2. 将掩码上采样到SR分辨率
    print(f"\n步骤2: 将掩码上采样到SR分辨率...")
    sr_mask = predictor.upscale_mask(fine_mask, sr_img.shape[:2])
    
    print(f"Fine掩码形状: {fine_mask.shape}")
    print(f"SR掩码形状: {sr_mask.shape}")
    print(f"SR掩码中True像素比例: {np.sum(sr_mask) / sr_mask.size * 100:.1f}%")
    
    # 3. 在SR图像上应用掩码进行替换
    print(f"\n步骤3: 在SR图像上应用掩码...")
    hybrid_img = sr_img.copy()
    hybrid_img[sr_mask] = org_img[sr_mask]
    
    # 计算最终效果
    hybrid_psnr = psnr(gt_img, hybrid_img)
    improvement = hybrid_psnr - sr_psnr
    pixel_ratio = np.sum(sr_mask) / sr_mask.size
    efficiency = improvement / pixel_ratio if pixel_ratio > 0 else 0
    
    print(f"\n📊 低分辨率边缘检测结果:")
    print(f"替换像素比例: {pixel_ratio*100:.1f}%")
    print(f"PSNR: {sr_psnr:.2f} → {hybrid_psnr:.2f} dB (+{improvement:.2f}dB)")
    print(f"效率: {efficiency:.1f}")
    
    # 保存结果
    result_path = f"../results/images/lowres_{method}_{mode}.png"
    Image.fromarray(hybrid_img.astype(np.uint8)).save(result_path)
    
    # 保存掩码可视化
    mask_vis = np.zeros_like(sr_img)
    mask_vis[sr_mask] = [255, 0, 0]  # 红色表示替换区域
    mask_vis[~sr_mask] = sr_img[~sr_mask]  # 其他区域保持原图
    mask_vis_path = f"../results/images/lowres_mask_{method}_{mode}.png"
    Image.fromarray(mask_vis.astype(np.uint8)).save(mask_vis_path)
    
    print(f"\n💾 结果保存:")
    print(f"- 混合图像: {result_path}")
    print(f"- 掩码可视化: {mask_vis_path}")
    
    return {
        'method': method,
        'mode': mode,
        'scale_factor': scale_factor,
        'fine_tiles_replaced': sum(1 for d in tile_decisions if d['should_replace']),
        'total_fine_tiles': len(tile_decisions),
        'pixel_ratio': pixel_ratio,
        'sr_psnr': sr_psnr,
        'hybrid_psnr': hybrid_psnr,
        'improvement': improvement,
        'efficiency': efficiency,
        'tile_decisions': tile_decisions
    }

def compare_lowres_vs_highres():
    """对比低分辨率边缘检测 vs 高分辨率边缘检测"""
    
    print("🔬 对比低分辨率 vs 高分辨率边缘检测")
    print("="*70)
    
    method = 'canny'
    mode = 'balanced'
    
    # 测试低分辨率方法
    print(f"\n🔍 测试低分辨率方法...")
    lowres_result = test_lowres_edge_method(method=method, mode=mode)
    
    # 对比高分辨率方法（从之前的结果文件中读取，或重新计算）
    print(f"\n📊 对比结果:")
    print(f"{'方法':<15} {'像素替换率':<12} {'PSNR提升':<10} {'效率':<8}")
    print("-" * 50)
    print(f"{'低分辨率':<15} {lowres_result['pixel_ratio']*100:<11.1f}% {lowres_result['improvement']:<9.2f}dB {lowres_result['efficiency']:<7.1f}")
    
    # 从之前的实验中，我们知道高分辨率balanced模式的结果
    highres_replacement = 12.9  # %
    highres_improvement = 0.82  # dB  
    highres_efficiency = 6.4
    
    print(f"{'高分辨率':<15} {highres_replacement:<11.1f}% {highres_improvement:<9.2f}dB {highres_efficiency:<7.1f}")
    
    # 分析差异
    print(f"\n🔍 分析:")
    replacement_diff = lowres_result['pixel_ratio']*100 - highres_replacement
    improvement_diff = lowres_result['improvement'] - highres_improvement
    efficiency_diff = lowres_result['efficiency'] - highres_efficiency
    
    print(f"替换率差异: {replacement_diff:+.1f}%")
    print(f"PSNR提升差异: {improvement_diff:+.2f}dB")
    print(f"效率差异: {efficiency_diff:+.1f}")
    
    if abs(replacement_diff) < 5:
        print("✅ 替换率相近，低分辨率方法有效")
    elif replacement_diff > 0:
        print("⚠️ 低分辨率方法替换过多")
    else:
        print("⚠️ 低分辨率方法替换过少")
    
    # 保存对比报告
    with open('../results/reports/lowres_vs_highres_comparison.txt', 'w') as f:
        f.write("=== 低分辨率 vs 高分辨率边缘检测对比报告 ===\n\n")
        f.write("测试设置:\n")
        f.write(f"- 方法: {method.upper()}\n")
        f.write(f"- 模式: {mode.upper()}\n")
        f.write(f"- 超分倍数: {lowres_result['scale_factor']}×\n\n")
        
        f.write("低分辨率方法:\n")
        f.write(f"- Fine tiles: {lowres_result['fine_tiles_replaced']}/{lowres_result['total_fine_tiles']}\n")
        f.write(f"- 像素替换率: {lowres_result['pixel_ratio']*100:.1f}%\n")
        f.write(f"- PSNR提升: +{lowres_result['improvement']:.2f}dB\n")
        f.write(f"- 效率: {lowres_result['efficiency']:.1f}\n\n")
        
        f.write("高分辨率方法 (参考):\n")
        f.write(f"- 像素替换率: {highres_replacement:.1f}%\n")
        f.write(f"- PSNR提升: +{highres_improvement:.2f}dB\n")
        f.write(f"- 效率: {highres_efficiency:.1f}\n\n")
        
        f.write("差异分析:\n")
        f.write(f"- 替换率差异: {replacement_diff:+.1f}%\n")
        f.write(f"- PSNR提升差异: {improvement_diff:+.2f}dB\n")
        f.write(f"- 效率差异: {efficiency_diff:+.1f}\n")
    
    return lowres_result

if __name__ == "__main__":
    print("🔍 低分辨率边缘检测 + 高分辨率映射实验")
    print("模拟真实SR流程：fine输入 → 边缘检测 → SR输出优化")
    print("="*70)
    
    # 测试所有模式
    modes = ['aggressive', 'balanced', 'conservative']
    method = 'canny'
    
    results = []
    
    for mode in modes:
        print(f"\n{'='*20} {mode.upper()} 模式 {'='*20}")
        result = test_lowres_edge_method(method=method, mode=mode)
        results.append(result)
    
    # 生成总结报告
    print(f"\n📊 所有模式对比:")
    print(f"{'模式':<12} {'Fine Tiles':<12} {'像素替换率':<12} {'PSNR提升':<10} {'效率':<8}")
    print("-" * 65)
    for r in results:
        fine_ratio = r['fine_tiles_replaced'] / r['total_fine_tiles'] * 100
        print(f"{r['mode']:<12} {r['fine_tiles_replaced']:<3}/{r['total_fine_tiles']:<7} {r['pixel_ratio']*100:<11.1f}% {r['improvement']:<9.2f}dB {r['efficiency']:<7.1f}")
    
    # 对比分析
    print(f"\n🔬 进行低分辨率 vs 高分辨率对比分析...")
    compare_lowres_vs_highres()
    
    print(f"\n✅ 实验完成！")
    print(f"\n📁 输出文件:")
    print("- ../results/images/lowres_*.png: 低分辨率方法结果图像")
    print("- ../results/images/lowres_mask_*.png: 替换掩码可视化")
    print("- ../results/reports/lowres_vs_highres_comparison.txt: 详细对比报告")
    
    # 推荐
    balanced_result = next(r for r in results if r['mode'] == 'balanced')
    print(f"\n🏆 推荐配置 (低分辨率方法):")
    print(f"效果: +{balanced_result['improvement']:.2f}dB, {balanced_result['pixel_ratio']*100:.1f}%像素替换, 效率{balanced_result['efficiency']:.1f}")
    print(f"优势: 在fine分辨率上分析，计算量更小，更适合实时应用")

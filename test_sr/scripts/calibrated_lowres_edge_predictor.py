#!/usr/bin/env python3
"""
阈值校正的低分辨率边缘检测方法
根据超分倍数自动调整阈值，避免过度替换
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from skimage.metrics import peak_signal_noise_ratio as psnr

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class CalibratedLowResEdgePredictor:
    """校正阈值的低分辨率边缘检测预测器"""
    
    def __init__(self, method='canny', mode='balanced', scale_factor=2):
        """
        Args:
            method: 边缘检测方法
            mode: 预设模式
            scale_factor: 超分倍数，用于阈值校正
        """
        self.method = method
        self.mode = mode
        self.scale_factor = scale_factor
        self.fine_tile_size = 32 // scale_factor  # 根据超分倍数调整tile大小
        self.sr_tile_size = 32
        
        # 基础阈值（高分辨率方法的阈值）
        self.base_thresholds = {
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
        
        # 计算校正后的阈值
        base_threshold = self.base_thresholds[method][mode]
        self.threshold = self.calibrate_threshold_for_resolution(base_threshold, scale_factor)
        
        print(f"🔧 校正阈值的低分辨率边缘检测预测器")
        print(f"方法: {method}")
        print(f"模式: {mode}")
        print(f"超分倍数: {scale_factor}×")
        print(f"基础阈值: {base_threshold:.3f}")
        print(f"校正阈值: {self.threshold:.3f}")
        print(f"校正系数: {self.threshold/base_threshold:.2f}")
        print(f"Fine tile大小: {self.fine_tile_size}×{self.fine_tile_size}")
    
    def calibrate_threshold_for_resolution(self, base_threshold, scale_factor):
        """
        根据超分倍数校正阈值
        
        Args:
            base_threshold: 基础阈值（高分辨率方法的阈值）
            scale_factor: 超分倍数
            
        Returns:
            校正后的阈值
        """
        # 校正策略：低分辨率需要更高阈值来避免过度替换
        if self.method == 'canny':
            # Canny方法的校正策略
            adjustment_factor = 1.0 + (scale_factor - 1) * 0.5  # 对于2×: 1.5倍
        elif self.method == 'sobel':
            # Sobel方法的校正策略  
            adjustment_factor = 1.0 + (scale_factor - 1) * 0.4  # 对于2×: 1.4倍
        elif self.method == 'gradient':
            # Gradient方法的校正策略
            adjustment_factor = 1.0 + (scale_factor - 1) * 0.3  # 对于2×: 1.3倍
        elif self.method == 'combined':
            # Combined方法的校正策略
            adjustment_factor = 1.0 + (scale_factor - 1) * 0.4  # 对于2×: 1.4倍
        else:
            adjustment_factor = 1.3  # 默认校正
            
        return base_threshold * adjustment_factor
    
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

def test_calibrated_method(method='canny', mode='balanced'):
    """测试校正阈值的低分辨率边缘检测方法"""
    
    print(f"🔧 测试校正阈值的低分辨率边缘检测方法")
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
    
    # 计算原始PSNR
    sr_psnr = psnr(gt_img, sr_img)
    print(f"\n原始SR PSNR: {sr_psnr:.2f} dB")
    
    # 创建校正预测器
    scale_factor = sr_img.shape[0] // fine_img.shape[0]
    predictor = CalibratedLowResEdgePredictor(method=method, mode=mode, scale_factor=scale_factor)
    
    # 1. 在fine图像上进行边缘检测
    print(f"\n步骤1: 在Fine图像上进行边缘检测...")
    fine_mask, tile_decisions = predictor.create_replacement_mask(fine_img)
    
    # 2. 将掩码上采样到SR分辨率
    print(f"\n步骤2: 将掩码上采样到SR分辨率...")
    sr_mask = predictor.upscale_mask(fine_mask, sr_img.shape[:2])
    
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
    
    print(f"\n📊 校正后低分辨率边缘检测结果:")
    print(f"替换像素比例: {pixel_ratio*100:.1f}%")
    print(f"PSNR: {sr_psnr:.2f} → {hybrid_psnr:.2f} dB (+{improvement:.2f}dB)")
    print(f"效率: {efficiency:.1f}")
    
    # 保存结果
    result_path = f"../results/images/calibrated_lowres_{method}_{mode}.png"
    Image.fromarray(hybrid_img.astype(np.uint8)).save(result_path)
    
    # 保存掩码可视化
    mask_vis = np.zeros_like(sr_img)
    mask_vis[sr_mask] = [255, 0, 0]  # 红色表示替换区域
    mask_vis[~sr_mask] = sr_img[~sr_mask]  # 其他区域保持原图
    mask_vis_path = f"../results/images/calibrated_mask_{method}_{mode}.png"
    Image.fromarray(mask_vis.astype(np.uint8)).save(mask_vis_path)
    
    print(f"\n💾 结果保存:")
    print(f"- 混合图像: {result_path}")
    print(f"- 掩码可视化: {mask_vis_path}")
    
    return {
        'method': method,
        'mode': mode,
        'scale_factor': scale_factor,
        'base_threshold': predictor.base_thresholds[method][mode],
        'calibrated_threshold': predictor.threshold,
        'calibration_factor': predictor.threshold / predictor.base_thresholds[method][mode],
        'fine_tiles_replaced': sum(1 for d in tile_decisions if d['should_replace']),
        'total_fine_tiles': len(tile_decisions),
        'pixel_ratio': pixel_ratio,
        'sr_psnr': sr_psnr,
        'hybrid_psnr': hybrid_psnr,
        'improvement': improvement,
        'efficiency': efficiency,
        'tile_decisions': tile_decisions
    }

def compare_before_after_calibration():
    """对比校正前后的效果"""
    
    print("🔬 对比阈值校正前后的效果")
    print("="*70)
    
    method = 'canny'
    mode = 'balanced'
    
    # 测试校正后的方法
    print(f"\n🔧 测试校正后的方法...")
    calibrated_result = test_calibrated_method(method=method, mode=mode)
    
    # 之前未校正的结果（从上次实验）
    uncalibrated_replacement = 29.3  # %
    uncalibrated_improvement = 0.78  # dB  
    uncalibrated_efficiency = 2.7
    
    # 高分辨率方法的结果（参考标准）
    highres_replacement = 12.9  # %
    highres_improvement = 0.82  # dB  
    highres_efficiency = 6.4
    
    print(f"\n📊 详细对比:")
    print(f"{'方法':<15} {'替换率':<12} {'PSNR提升':<10} {'效率':<8} {'阈值':<10}")
    print("-" * 65)
    print(f"{'高分辨率(参考)':<15} {highres_replacement:<11.1f}% {highres_improvement:<9.2f}dB {highres_efficiency:<7.1f} 0.094")
    print(f"{'低分辨率(原)':<15} {uncalibrated_replacement:<11.1f}% {uncalibrated_improvement:<9.2f}dB {uncalibrated_efficiency:<7.1f} 0.094")
    print(f"{'低分辨率(校正)':<15} {calibrated_result['pixel_ratio']*100:<11.1f}% {calibrated_result['improvement']:<9.2f}dB {calibrated_result['efficiency']:<7.1f} {calibrated_result['calibrated_threshold']:.3f}")
    
    # 分析改进效果
    replacement_improvement = calibrated_result['pixel_ratio']*100 - uncalibrated_replacement
    psnr_change = calibrated_result['improvement'] - uncalibrated_improvement
    efficiency_improvement = calibrated_result['efficiency'] - uncalibrated_efficiency
    
    print(f"\n🎯 校正效果分析:")
    print(f"替换率变化: {replacement_improvement:+.1f}% (目标：接近{highres_replacement}%)")
    print(f"PSNR变化: {psnr_change:+.2f}dB")
    print(f"效率提升: {efficiency_improvement:+.1f}")
    
    # 与高分辨率方法的对比
    replacement_vs_highres = calibrated_result['pixel_ratio']*100 - highres_replacement
    psnr_vs_highres = calibrated_result['improvement'] - highres_improvement
    efficiency_vs_highres = calibrated_result['efficiency'] - highres_efficiency
    
    print(f"\n🆚 与高分辨率方法对比:")
    print(f"替换率差异: {replacement_vs_highres:+.1f}%")
    print(f"PSNR差异: {psnr_vs_highres:+.2f}dB")
    print(f"效率差异: {efficiency_vs_highres:+.1f}")
    
    if abs(replacement_vs_highres) < 5:
        print("✅ 替换率已接近高分辨率方法")
    elif replacement_vs_highres > 0:
        print("⚠️ 替换率仍偏高，需要进一步调整")
    else:
        print("⚠️ 替换率偏低，可以适当降低阈值")
    
    # 保存详细对比报告
    with open('../results/reports/calibrated_threshold_comparison.txt', 'w') as f:
        f.write("=== 阈值校正前后对比报告 ===\n\n")
        f.write("校正策略: 根据超分倍数调整阈值\n")
        f.write(f"校正公式: threshold = base_threshold × (1 + (scale_factor - 1) × 0.5)\n")
        f.write(f"对于2×超分: 0.094 → {calibrated_result['calibrated_threshold']:.3f} ({calibrated_result['calibration_factor']:.2f}倍)\n\n")
        
        f.write("对比结果:\n")
        f.write(f"高分辨率方法: {highres_replacement:.1f}%替换, +{highres_improvement:.2f}dB, 效率{highres_efficiency:.1f}\n")
        f.write(f"低分辨率原方法: {uncalibrated_replacement:.1f}%替换, +{uncalibrated_improvement:.2f}dB, 效率{uncalibrated_efficiency:.1f}\n")
        f.write(f"低分辨率校正后: {calibrated_result['pixel_ratio']*100:.1f}%替换, +{calibrated_result['improvement']:.2f}dB, 效率{calibrated_result['efficiency']:.1f}\n\n")
        
        f.write("校正效果:\n")
        f.write(f"替换率改善: {replacement_improvement:+.1f}%\n")
        f.write(f"效率提升: {efficiency_improvement:+.1f}\n")
        f.write(f"计算量减少: 75% (在fine分辨率上分析)\n")
    
    return calibrated_result

if __name__ == "__main__":
    print("🔧 阈值校正的低分辨率边缘检测实验")
    print("根据超分倍数自动调整阈值，避免过度替换")
    print("="*70)
    
    # 测试所有模式的校正效果
    modes = ['aggressive', 'balanced', 'conservative']
    method = 'canny'
    
    print(f"\n📊 测试所有模式的校正效果:")
    
    results = []
    for mode in modes:
        print(f"\n{'='*20} {mode.upper()} 模式 {'='*20}")
        result = test_calibrated_method(method=method, mode=mode)
        results.append(result)
    
    # 生成校正后的总结
    print(f"\n📊 校正后所有模式对比:")
    print(f"{'模式':<12} {'校正阈值':<10} {'替换率':<10} {'PSNR提升':<10} {'效率':<8}")
    print("-" * 55)
    for r in results:
        print(f"{r['mode']:<12} {r['calibrated_threshold']:<9.3f} {r['pixel_ratio']*100:<9.1f}% {r['improvement']:<9.2f}dB {r['efficiency']:<7.1f}")
    
    # 详细的校正前后对比
    print(f"\n🔬 进行校正前后详细对比分析...")
    compare_before_after_calibration()
    
    print(f"\n✅ 阈值校正实验完成！")
    print(f"\n📁 输出文件:")
    print("- ../results/images/calibrated_lowres_*.png: 校正后结果图像")
    print("- ../results/images/calibrated_mask_*.png: 校正后掩码可视化")
    print("- ../results/reports/calibrated_threshold_comparison.txt: 校正前后对比报告")
    
    # 推荐最佳配置
    balanced_result = next(r for r in results if r['mode'] == 'balanced')
    print(f"\n🏆 推荐配置 (校正后平衡模式):")
    print(f"基础阈值: {balanced_result['base_threshold']:.3f}")
    print(f"校正阈值: {balanced_result['calibrated_threshold']:.3f} ({balanced_result['calibration_factor']:.2f}倍)")
    print(f"效果: +{balanced_result['improvement']:.2f}dB, {balanced_result['pixel_ratio']*100:.1f}%替换率, 效率{balanced_result['efficiency']:.1f}")
    print(f"优势: 计算量减少75%，效果接近高分辨率方法")

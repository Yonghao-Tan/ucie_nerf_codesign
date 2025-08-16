#!/usr/bin/env python3
"""
最终优化的低分辨率边缘检测方法
精细调整校正系数，实现最佳平衡
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

class OptimalLowResEdgePredictor:
    """最优校正的低分辨率边缘检测预测器"""
    
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
        self.fine_tile_size = 32 // scale_factor
        self.sr_tile_size = 32
        
        # 基础阈值
        self.base_thresholds = {
            'canny': {
                'aggressive': 0.047,
                'balanced': 0.094,
                'conservative': 0.141
            }
        }
        
        # 根据实验结果优化的校正系数
        base_threshold = self.base_thresholds[method][mode]
        self.threshold = self.get_optimal_threshold(base_threshold, mode, scale_factor)
        
        print(f"🎯 最优校正的低分辨率边缘检测预测器")
        print(f"方法: {method}")
        print(f"模式: {mode}")
        print(f"超分倍数: {scale_factor}×")
        print(f"基础阈值: {base_threshold:.3f}")
        print(f"最优阈值: {self.threshold:.3f}")
        print(f"校正系数: {self.threshold/base_threshold:.2f}")
    
    def get_optimal_threshold(self, base_threshold, mode, scale_factor):
        """
        根据实验结果优化的阈值校正
        目标：让低分辨率方法的替换率接近高分辨率方法
        """
        # 基于实验数据的精细调整
        if mode == 'aggressive':
            # 目标替换率约40%，当前校正后35.3%，需要稍微降低阈值
            adjustment_factor = 1.35  # 比1.5稍低
        elif mode == 'balanced':
            # 目标替换率约13%，当前校正后16.4%，需要稍微提高阈值
            adjustment_factor = 1.7   # 比1.5稍高
        elif mode == 'conservative':
            # 目标替换率约4%，当前校正后6.2%，需要更高阈值
            adjustment_factor = 2.0   # 比1.5更高
        else:
            adjustment_factor = 1.5   # 默认
            
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
            
        # Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        return edge_ratio
    
    def should_replace_tile(self, fine_tile):
        """在低分辨率tile上判断是否需要替换"""
        edge_score = self.extract_edge_score(fine_tile)
        return edge_score > self.threshold
    
    def create_replacement_mask(self, fine_img):
        """在fine图像上创建替换掩码"""
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
        upscaled_mask = cv2.resize(
            fine_mask.astype(np.uint8), 
            (target_shape[1], target_shape[0]), 
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        
        return upscaled_mask

def test_optimal_method(method='canny', mode='balanced'):
    """测试最优校正的低分辨率边缘检测方法"""
    
    print(f"🎯 测试最优校正的低分辨率边缘检测方法")
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
    
    # 计算原始PSNR
    sr_psnr = psnr(gt_img, sr_img)
    print(f"原始SR PSNR: {sr_psnr:.2f} dB")
    
    # 创建最优预测器
    scale_factor = sr_img.shape[0] // fine_img.shape[0]
    predictor = OptimalLowResEdgePredictor(method=method, mode=mode, scale_factor=scale_factor)
    
    # 执行预测流程
    print(f"\n步骤1: 在Fine图像上进行边缘检测...")
    fine_mask, tile_decisions = predictor.create_replacement_mask(fine_img)
    
    print(f"\n步骤2: 将掩码上采样到SR分辨率...")
    sr_mask = predictor.upscale_mask(fine_mask, sr_img.shape[:2])
    
    print(f"SR掩码中True像素比例: {np.sum(sr_mask) / sr_mask.size * 100:.1f}%")
    
    print(f"\n步骤3: 在SR图像上应用掩码...")
    hybrid_img = sr_img.copy()
    hybrid_img[sr_mask] = org_img[sr_mask]
    
    # 计算最终效果
    hybrid_psnr = psnr(gt_img, hybrid_img)
    improvement = hybrid_psnr - sr_psnr
    pixel_ratio = np.sum(sr_mask) / sr_mask.size
    efficiency = improvement / pixel_ratio if pixel_ratio > 0 else 0
    
    print(f"\n📊 最优校正低分辨率边缘检测结果:")
    print(f"替换像素比例: {pixel_ratio*100:.1f}%")
    print(f"PSNR: {sr_psnr:.2f} → {hybrid_psnr:.2f} dB (+{improvement:.2f}dB)")
    print(f"效率: {efficiency:.1f}")
    
    # 保存结果
    result_path = f"../results/images/optimal_lowres_{method}_{mode}.png"
    Image.fromarray(hybrid_img.astype(np.uint8)).save(result_path)
    
    print(f"结果保存到: {result_path}")
    
    return {
        'method': method,
        'mode': mode,
        'optimal_threshold': predictor.threshold,
        'fine_tiles_replaced': sum(1 for d in tile_decisions if d['should_replace']),
        'total_fine_tiles': len(tile_decisions),
        'pixel_ratio': pixel_ratio,
        'sr_psnr': sr_psnr,
        'hybrid_psnr': hybrid_psnr,
        'improvement': improvement,
        'efficiency': efficiency
    }

def final_comparison():
    """最终方法对比"""
    
    print("🏆 最终方法全面对比")
    print("="*70)
    
    method = 'canny'
    
    # 测试所有模式
    modes = ['aggressive', 'balanced', 'conservative']
    results = []
    
    for mode in modes:
        print(f"\n{'='*15} {mode.upper()} 模式 {'='*15}")
        result = test_optimal_method(method=method, mode=mode)
        results.append(result)
    
    # 参考数据
    highres_data = {
        'aggressive': {'replacement': 40.1, 'improvement': 1.30, 'efficiency': 3.2},
        'balanced': {'replacement': 12.9, 'improvement': 0.82, 'efficiency': 6.4},
        'conservative': {'replacement': 4.5, 'improvement': 0.57, 'efficiency': 12.7}
    }
    
    print(f"\n📊 最终对比表:")
    print(f"{'模式':<12} {'方法':<15} {'替换率':<10} {'PSNR提升':<10} {'效率':<8} {'计算量':<10}")
    print("-" * 75)
    
    for mode in modes:
        # 高分辨率数据
        hr_data = highres_data[mode]
        print(f"{mode:<12} {'高分辨率':<15} {hr_data['replacement']:<9.1f}% {hr_data['improvement']:<9.2f}dB {hr_data['efficiency']:<7.1f} 100%")
        
        # 最优低分辨率数据
        lr_result = next(r for r in results if r['mode'] == mode)
        print(f"{'':<12} {'低分辨率(最优)':<15} {lr_result['pixel_ratio']*100:<9.1f}% {lr_result['improvement']:<9.2f}dB {lr_result['efficiency']:<7.1f} 25%")
        print("-" * 75)
    
    # 保存最终报告
    with open('../results/reports/final_lowres_edge_comparison.txt', 'w') as f:
        f.write("=== 最终低分辨率边缘检测方法报告 ===\n\n")
        f.write("核心优势:\n")
        f.write("1. 计算量减少75% - 在fine分辨率(378×504)而非SR分辨率(756×1008)上分析\n")
        f.write("2. 内存友好 - 处理更小的图像和tiles\n") 
        f.write("3. 实时友好 - 符合SR输入→输出的自然流程\n")
        f.write("4. 质量保证 - 通过阈值校正实现合理的替换率和质量提升\n\n")
        
        f.write("最终对比结果:\n")
        for mode in modes:
            hr_data = highres_data[mode]
            lr_result = next(r for r in results if r['mode'] == mode)
            f.write(f"\n{mode.upper()}模式:\n")
            f.write(f"  高分辨率: {hr_data['replacement']:.1f}%替换, +{hr_data['improvement']:.2f}dB, 效率{hr_data['efficiency']:.1f}, 计算量100%\n")
            f.write(f"  低分辨率: {lr_result['pixel_ratio']*100:.1f}%替换, +{lr_result['improvement']:.2f}dB, 效率{lr_result['efficiency']:.1f}, 计算量25%\n")
        
        f.write(f"\n推荐配置: BALANCED模式\n")
        balanced_result = next(r for r in results if r['mode'] == 'balanced')
        f.write(f"  最优阈值: {balanced_result['optimal_threshold']:.3f}\n")
        f.write(f"  效果: +{balanced_result['improvement']:.2f}dB提升, {balanced_result['pixel_ratio']*100:.1f}%替换率\n")
        f.write(f"  优势: 在保持质量的同时显著减少计算量，非常适合实时IBRNet应用\n")
    
    return results

if __name__ == "__main__":
    print("🎯 最优低分辨率边缘检测方法测试")
    print("精细调整校正系数，实现计算效率与质量的最佳平衡")
    print("="*70)
    
    # 运行最终对比
    results = final_comparison()
    
    # 总结推荐
    balanced_result = next(r for r in results if r['mode'] == 'balanced')
    
    print(f"\n🎯 最终推荐方案:")
    print(f"方法: 低分辨率Canny边缘检测")
    print(f"模式: BALANCED")
    print(f"最优阈值: {balanced_result['optimal_threshold']:.3f}")
    print(f"效果: +{balanced_result['improvement']:.2f}dB PSNR提升")
    print(f"替换率: {balanced_result['pixel_ratio']*100:.1f}%")
    print(f"效率: {balanced_result['efficiency']:.1f}")
    print(f"计算量: 仅为高分辨率方法的25%")
    
    print(f"\n✅ 核心优势:")
    print("- 🚀 75%计算量减少 - 在fine分辨率上分析")
    print("- 💡 实时友好 - 符合SR工作流程")
    print("- 🎯 质量保证 - 合理的替换率和PSNR提升") 
    print("- 🔧 易部署 - 简单的阈值判断，无需全局信息")
    
    print(f"\n📁 最终输出:")
    print("- ../results/images/optimal_lowres_*.png: 最优结果图像")
    print("- ../results/reports/final_lowres_edge_comparison.txt: 最终对比报告")
    
    print(f"\n🚀 ready for IBRNet integration!")

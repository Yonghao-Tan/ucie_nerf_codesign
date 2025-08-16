#!/usr/bin/env python3
"""
真正的实时边缘检测Tile替换方法
使用预设阈值，无需预知全局信息，适用于IBRNet逐tile渲染场景
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

class TrueRealtimeEdgePredictor:
    """真正的实时边缘检测质量预测器 - 使用预设阈值"""
    
    def __init__(self, method='canny', mode='balanced'):
        """
        Args:
            method: 边缘检测方法 ('canny', 'sobel', 'gradient', 'combined')
            mode: 预设模式 ('aggressive', 'balanced', 'conservative')
        """
        self.method = method
        self.tile_size = 32
        
        # 预设阈值 - 基于之前校准实验的统计结果
        self.preset_thresholds = {
            'canny': {
                'aggressive': 0.047,    # 约38%替换率，最高质量
                'balanced': 0.094,      # 约13%替换率，推荐配置
                'conservative': 0.141   # 约4%替换率，高效率
            },
            'sobel': {
                'aggressive': 29.5,     # 对应Canny aggressive
                'balanced': 58.9,       # 对应Canny balanced  
                'conservative': 88.4    # 对应Canny conservative
            },
            'gradient': {
                'aggressive': 4.0,      # 对应Canny aggressive
                'balanced': 8.0,        # 对应Canny balanced
                'conservative': 12.0    # 对应Canny conservative
            },
            'combined': {
                'aggressive': 0.20,     # 对应Canny aggressive
                'balanced': 0.40,       # 对应Canny balanced
                'conservative': 0.60    # 对应Canny conservative
            }
        }
        
        self.mode = mode
        self.threshold = self.preset_thresholds[method][mode]
        
        print(f"🚀 真实时{method}预测器初始化")
        print(f"模式: {mode}")
        print(f"预设阈值: {self.threshold}")
        
        # 预期效果说明
        expected_effects = {
            'aggressive': "高质量模式 - 更多替换，更高PSNR提升",
            'balanced': "平衡模式 - 质量与效率兼顾，推荐使用",
            'conservative': "高效率模式 - 少量替换，高计算效率"
        }
        print(f"预期效果: {expected_effects[mode]}")
    
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
            # Canny边缘检测 - 返回边缘像素比例
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / edges.size
            return edge_ratio
            
        elif self.method == 'sobel':
            # Sobel边缘检测 - 返回边缘密度
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            return np.mean(sobel_magnitude)
            
        elif self.method == 'gradient':
            # 梯度统计 - 返回梯度均值
            grad_x = np.gradient(gray.astype(np.float32), axis=1)
            grad_y = np.gradient(gray.astype(np.float32), axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            return np.mean(gradient_magnitude)
            
        elif self.method == 'combined':
            # 组合多种边缘特征
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
    
    def should_replace_tile(self, sr_tile):
        """
        判断单个tile是否需要替换 - 真正的实时判断
        
        Args:
            sr_tile: 32x32的SR tile
            
        Returns:
            bool: True表示需要替换，False表示保持SR
        """
        edge_score = self.extract_edge_score(sr_tile)
        return edge_score > self.threshold
    
    def process_tile_with_info(self, sr_tile):
        """
        处理tile并返回详细信息（调试用）
        
        Returns:
            dict: 包含edge_score, should_replace, confidence等信息
        """
        edge_score = self.extract_edge_score(sr_tile)
        should_replace = edge_score > self.threshold
        
        # 计算置信度（距离阈值的程度）
        if should_replace:
            confidence = min((edge_score - self.threshold) / self.threshold, 1.0)
        else:
            confidence = min((self.threshold - edge_score) / self.threshold, 1.0)
        
        return {
            'edge_score': edge_score,
            'threshold': self.threshold,
            'should_replace': should_replace,
            'confidence': confidence,
            'method': self.method,
            'mode': self.mode
        }

def simulate_realtime_rendering(method='canny', mode='balanced'):
    """模拟真实的IBRNet逐tile渲染过程"""
    sr_path = "../data/no2_pred_sr.png"
    org_path = "../data/no2_pred_org.png"
    gt_path = "../data/no2_gt.png"
    
    # 加载图像
    sr_img = np.array(Image.open(sr_path))
    org_img = np.array(Image.open(org_path))
    gt_img = np.array(Image.open(gt_path))
    
    # 计算原始SR的PSNR
    sr_psnr = psnr(gt_img, sr_img)
    
    # 创建真实时预测器
    predictor = TrueRealtimeEdgePredictor(method=method, mode=mode)
    
    # 准备输出图像
    h, w = sr_img.shape[:2]
    tile_size = 32
    tile_h, tile_w = h // tile_size, w // tile_size
    hybrid_img = sr_img.copy()
    
    # 统计信息
    replaced_tiles = 0
    total_tiles = tile_h * tile_w
    tile_decisions = []
    
    print(f"\n🎬 模拟IBRNet逐tile渲染...")
    print(f"图像尺寸: {h}x{w}")
    print(f"Tile尺寸: {tile_size}x{tile_size}")
    print(f"总tiles: {total_tiles}")
    print(f"\n开始逐个处理tiles...")
    
    # 模拟逐个tile的处理过程
    for i in range(tile_h):
        for j in range(tile_w):
            tile_idx = i * tile_w + j
            h_start, h_end = i * tile_size, (i + 1) * tile_size
            w_start, w_end = j * tile_size, (j + 1) * tile_size
            
            # 1. 渲染SR tile（IBRNet当前渲染）
            sr_tile = sr_img[h_start:h_end, w_start:w_end]
            
            # 2. 实时判断是否需要替换（无需全局信息！）
            decision_info = predictor.process_tile_with_info(sr_tile)
            tile_decisions.append(decision_info)
            
            # 3. 如果需要替换，渲染高质量版本
            if decision_info['should_replace']:
                # 在实际应用中，这里会调用高质量渲染
                org_tile = org_img[h_start:h_end, w_start:w_end]
                hybrid_img[h_start:h_end, w_start:w_end] = org_tile
                replaced_tiles += 1
                
                if tile_idx % 100 == 0:  # 每100个tile打印一次进度
                    print(f"Tile {tile_idx}: 替换 (edge_score={decision_info['edge_score']:.3f})")
            else:
                if tile_idx % 100 == 0:
                    print(f"Tile {tile_idx}: 保持 (edge_score={decision_info['edge_score']:.3f})")
    
    # 计算最终效果
    hybrid_psnr = psnr(gt_img, hybrid_img)
    improvement = hybrid_psnr - sr_psnr
    replacement_ratio = replaced_tiles / total_tiles
    pixel_ratio = replacement_ratio  # tiles替换比例等于像素替换比例
    efficiency = improvement / pixel_ratio if pixel_ratio > 0 else 0
    
    print(f"\n📊 真实时渲染结果:")
    print(f"替换tiles: {replaced_tiles}/{total_tiles} ({replacement_ratio*100:.1f}%)")
    print(f"PSNR: {sr_psnr:.2f} → {hybrid_psnr:.2f} dB (+{improvement:.2f}dB)")
    print(f"效率: {efficiency:.1f}")
    
    # 保存结果图像
    result_path = f"../results/images/true_realtime_{method}_{mode}.png"
    Image.fromarray(hybrid_img.astype(np.uint8)).save(result_path)
    print(f"结果图像保存到: {result_path}")
    
    return {
        'method': method,
        'mode': mode,
        'threshold': predictor.threshold,
        'replaced_tiles': replaced_tiles,
        'total_tiles': total_tiles,
        'replacement_ratio': replacement_ratio,
        'sr_psnr': sr_psnr,
        'hybrid_psnr': hybrid_psnr,
        'improvement': improvement,
        'efficiency': efficiency,
        'tile_decisions': tile_decisions
    }

def compare_all_modes():
    """比较所有预设模式的效果"""
    method = 'canny'  # 使用最佳方法
    modes = ['aggressive', 'balanced', 'conservative']
    
    print("🎯 比较所有预设模式效果")
    print("="*60)
    
    results = []
    for mode in modes:
        print(f"\n{'='*20} {mode.upper()} 模式 {'='*20}")
        result = simulate_realtime_rendering(method=method, mode=mode)
        results.append(result)
    
    # 生成对比报告
    print(f"\n📊 预设模式对比:")
    print("模式\t\t阈值\t替换率\tPSNR提升\t效率")
    print("-" * 60)
    for r in results:
        print(f"{r['mode']:<12}\t{r['threshold']:.3f}\t{r['replacement_ratio']*100:.1f}%\t+{r['improvement']:.2f}dB\t\t{r['efficiency']:.1f}")
    
    # 保存详细报告
    with open('../results/reports/true_realtime_comparison.txt', 'w') as f:
        f.write("=== 真实时边缘检测Tile替换方法对比报告 ===\n\n")
        f.write("特点: 使用预设阈值，无需全局信息，适合IBRNet逐tile渲染\n\n")
        
        for r in results:
            f.write(f"模式: {r['mode'].upper()}\n")
            f.write(f"方法: {r['method'].upper()}\n")
            f.write(f"预设阈值: {r['threshold']:.3f}\n")
            f.write(f"替换tiles: {r['replaced_tiles']}/{r['total_tiles']} ({r['replacement_ratio']*100:.1f}%)\n")
            f.write(f"PSNR: {r['sr_psnr']:.2f} → {r['hybrid_psnr']:.2f} dB (+{r['improvement']:.2f}dB)\n")
            f.write(f"效率: {r['efficiency']:.1f}\n")
            f.write("-" * 50 + "\n")
    
    return results

if __name__ == "__main__":
    print("🚀 真实时边缘检测Tile替换方法测试")
    print("无需全局信息，使用预设阈值，适合IBRNet逐tile渲染")
    print("="*70)
    
    # 比较所有预设模式
    results = compare_all_modes()
    
    # 推荐最佳配置
    balanced_result = next(r for r in results if r['mode'] == 'balanced')
    
    print(f"\n🏆 推荐配置 (平衡模式):")
    print(f"方法: {balanced_result['method'].upper()}")
    print(f"模式: {balanced_result['mode'].upper()}")
    print(f"预设阈值: {balanced_result['threshold']:.3f}")
    print(f"效果: +{balanced_result['improvement']:.2f}dB, {balanced_result['replacement_ratio']*100:.1f}%替换率, 效率{balanced_result['efficiency']:.1f}")
    
    print(f"\n✅ 真实时方法的优势:")
    print("- ✅ 无需预知全局信息")
    print("- ✅ 每个tile独立决策，O(1)复杂度")
    print("- ✅ 完美适合IBRNet逐tile渲染")
    print("- ✅ 可根据场景选择aggressive/balanced/conservative模式")
    
    print(f"\n📁 输出文件:")
    print("- ../results/images/true_realtime_canny_*.png: 各模式结果图像")
    print("- ../results/reports/true_realtime_comparison.txt: 详细对比报告")

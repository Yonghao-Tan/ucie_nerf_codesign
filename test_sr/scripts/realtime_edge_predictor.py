#!/usr/bin/env python3
"""
实时边缘检测Tile替换方法
适用于IBRNet实际渲染场景：逐个tile判断是否需要替换，无需预知全局信息
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

class RealtimeEdgePredictor:
    """实时边缘检测质量预测器"""
    
    def __init__(self, method='canny', threshold=None):
        """
        Args:
            method: 边缘检测方法 ('canny', 'sobel', 'gradient', 'combined')
            threshold: 替换阈值，如果为None则使用默认值
        """
        self.method = method
        self.tile_size = 32
        
        # 根据方法设置默认阈值
        default_thresholds = {
            'canny': 0.15,      # Canny边缘比例阈值
            'sobel': 20.0,      # Sobel边缘密度阈值  
            'gradient': 12.0,   # 梯度均值阈值
            'combined': 0.4     # 组合得分阈值
        }
        
        self.threshold = threshold if threshold is not None else default_thresholds.get(method, 0.15)
        print(f"使用{method}方法，阈值={self.threshold}")
    
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
        判断单个tile是否需要替换
        
        Args:
            sr_tile: 32x32的SR tile
            
        Returns:
            bool: True表示需要替换，False表示保持SR
        """
        edge_score = self.extract_edge_score(sr_tile)
        return edge_score > self.threshold
    
    def get_replacement_stats(self, sr_tile):
        """
        获取tile的详细信息（用于调试和分析）
        
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
            'method': self.method
        }

def calibrate_threshold(sr_path, org_path, gt_path, method='canny', target_replace_ratio=0.13):
    """
    校准阈值，使替换比例接近目标比例
    
    Args:
        target_replace_ratio: 目标替换像素比例（如0.13表示13%）
    """
    print(f"=== 校准{method}方法的阈值 ===")
    print(f"目标替换比例: {target_replace_ratio*100:.1f}%")
    
    # 加载图像
    sr_img = np.array(Image.open(sr_path))
    h, w = sr_img.shape[:2]
    tile_size = 32
    tile_h, tile_w = h // tile_size, w // tile_size
    total_tiles = tile_h * tile_w
    
    # 计算所有tiles的边缘得分
    scores = []
    for i in range(tile_h):
        for j in range(tile_w):
            h_start, h_end = i * tile_size, (i + 1) * tile_size
            w_start, w_end = j * tile_size, (j + 1) * tile_size
            tile = sr_img[h_start:h_end, w_start:w_end]
            
            predictor = RealtimeEdgePredictor(method=method, threshold=0)  # 临时阈值
            score = predictor.extract_edge_score(tile)
            scores.append(score)
    
    scores = np.array(scores)
    
    # 根据目标替换比例确定阈值
    target_tiles = int(total_tiles * target_replace_ratio)
    optimal_threshold = np.percentile(scores, 100 - target_replace_ratio * 100)
    
    print(f"所有tiles得分范围: {scores.min():.3f} ~ {scores.max():.3f}")
    print(f"建议阈值: {optimal_threshold:.3f}")
    print(f"使用此阈值将替换 {target_tiles} 个tiles ({target_replace_ratio*100:.1f}%)")
    
    return optimal_threshold, scores

def test_realtime_method(method='canny', threshold=None, target_ratio=0.13):
    """测试实时方法的效果"""
    sr_path = "../data/no2_pred_sr.png"
    org_path = "../data/no2_pred_org.png"
    gt_path = "../data/no2_gt.png"
    
    # 加载图像
    sr_img = np.array(Image.open(sr_path))
    org_img = np.array(Image.open(org_path))
    gt_img = np.array(Image.open(gt_path))
    
    sr_psnr = psnr(gt_img, sr_img)
    
    # 如果没有提供阈值，先校准
    if threshold is None:
        threshold, all_scores = calibrate_threshold(sr_path, org_path, gt_path, 
                                                  method=method, target_replace_ratio=target_ratio)
    
    # 创建预测器
    predictor = RealtimeEdgePredictor(method=method, threshold=threshold)
    
    # 逐个tile进行判断和替换
    h, w = sr_img.shape[:2]
    tile_size = 32
    tile_h, tile_w = h // tile_size, w // tile_size
    
    hybrid_img = sr_img.copy()
    replaced_tiles = 0
    replaced_pixels = 0
    
    tile_decisions = []  # 记录每个tile的决策信息
    
    print(f"\n=== 实时{method}方法测试 ===")
    print(f"逐个tile判断...")
    
    for i in range(tile_h):
        for j in range(tile_w):
            h_start, h_end = i * tile_size, (i + 1) * tile_size
            w_start, w_end = j * tile_size, (j + 1) * tile_size
            
            sr_tile = sr_img[h_start:h_end, w_start:w_end]
            
            # 实时判断是否需要替换
            stats = predictor.get_replacement_stats(sr_tile)
            tile_decisions.append(stats)
            
            if stats['should_replace']:
                # 替换为ORG tile
                hybrid_img[h_start:h_end, w_start:w_end] = org_img[h_start:h_end, w_start:w_end]
                replaced_tiles += 1
                replaced_pixels += tile_size * tile_size
    
    # 计算结果
    hybrid_psnr = psnr(gt_img, hybrid_img)
    improvement = hybrid_psnr - sr_psnr
    pixel_ratio = replaced_pixels / (sr_img.shape[0] * sr_img.shape[1])
    efficiency = improvement / pixel_ratio if pixel_ratio > 0 else 0
    
    print(f"替换tiles: {replaced_tiles}/{tile_h * tile_w} ({replaced_tiles/(tile_h * tile_w)*100:.1f}%)")
    print(f"替换像素: {pixel_ratio*100:.1f}%")
    print(f"PSNR提升: {sr_psnr:.2f} → {hybrid_psnr:.2f} dB (+{improvement:.2f}dB)")
    print(f"效率: {efficiency:.1f}")
    
    # 保存结果
    hybrid_path = f"../results/images/realtime_{method}_threshold{threshold:.3f}.png"
    Image.fromarray(hybrid_img.astype(np.uint8)).save(hybrid_path)
    
    return {
        'method': method,
        'threshold': threshold,
        'replaced_tiles': replaced_tiles,
        'total_tiles': tile_h * tile_w,
        'pixel_ratio': pixel_ratio,
        'sr_psnr': sr_psnr,
        'hybrid_psnr': hybrid_psnr,
        'improvement': improvement,
        'efficiency': efficiency,
        'tile_decisions': tile_decisions
    }

def compare_threshold_sensitivity():
    """测试不同阈值的敏感性"""
    method = 'canny'  # 使用最佳的Canny方法
    sr_path = "../data/no2_pred_sr.png"
    org_path = "../data/no2_pred_org.png"
    gt_path = "../data/no2_gt.png"
    
    print("=== 阈值敏感性分析 ===")
    
    # 校准基准阈值
    base_threshold, all_scores = calibrate_threshold(sr_path, org_path, gt_path, 
                                                   method=method, target_replace_ratio=0.13)
    
    # 测试不同阈值
    thresholds = [
        base_threshold * 0.5,   # 更激进
        base_threshold * 0.75,  # 较激进
        base_threshold,         # 基准
        base_threshold * 1.25,  # 较保守
        base_threshold * 1.5    # 更保守
    ]
    
    results = []
    
    for threshold in thresholds:
        print(f"\n--- 测试阈值: {threshold:.3f} ---")
        result = test_realtime_method(method=method, threshold=threshold)
        results.append(result)
    
    # 可视化阈值效果
    visualize_threshold_effects(results)
    
    return results

def visualize_threshold_effects(results):
    """可视化不同阈值的效果"""
    thresholds = [r['threshold'] for r in results]
    improvements = [r['improvement'] for r in results]
    pixel_ratios = [r['pixel_ratio'] * 100 for r in results]
    efficiencies = [r['efficiency'] for r in results]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    # PSNR改善
    ax1.plot(thresholds, improvements, 'bo-')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('PSNR Improvement (dB)')
    ax1.set_title('PSNR vs Threshold')
    ax1.grid(True)
    
    # 替换比例
    ax2.plot(thresholds, pixel_ratios, 'ro-')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Replaced Pixels (%)')
    ax2.set_title('Replacement Ratio vs Threshold')
    ax2.grid(True)
    
    # 效率
    ax3.plot(thresholds, efficiencies, 'go-')
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Efficiency')
    ax3.set_title('Efficiency vs Threshold')
    ax3.grid(True)
    
    plt.tight_layout()
    save_path = '../results/images/threshold_sensitivity_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"阈值敏感性分析图保存到: {save_path}")
    plt.close()

if __name__ == "__main__":
    print("🚀 开始实时边缘检测方法测试...")
    
    # 1. 校准和测试不同方法的阈值
    methods = ['canny', 'sobel', 'gradient', 'combined']
    target_ratio = 0.13  # 目标替换13%像素（与之前实验对比）
    
    all_results = []
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"测试{method.upper()}方法")
        print(f"{'='*50}")
        
        result = test_realtime_method(method=method, target_ratio=target_ratio)
        all_results.append(result)
    
    # 2. 阈值敏感性分析（使用最佳的Canny方法）
    print(f"\n{'='*50}")
    print("阈值敏感性分析")
    print(f"{'='*50}")
    
    threshold_results = compare_threshold_sensitivity()
    
    # 3. 保存详细报告
    with open('../results/reports/realtime_edge_method_report.txt', 'w') as f:
        f.write("=== 实时边缘检测Tile替换方法实验报告 ===\n\n")
        f.write("目标替换比例: 13%\n\n")
        
        for result in all_results:
            f.write(f"方法: {result['method'].upper()}\n")
            f.write(f"阈值: {result['threshold']:.3f}\n")
            f.write(f"替换tiles: {result['replaced_tiles']}/{result['total_tiles']} ({result['replaced_tiles']/result['total_tiles']*100:.1f}%)\n")
            f.write(f"替换像素: {result['pixel_ratio']*100:.1f}%\n") 
            f.write(f"PSNR: {result['sr_psnr']:.2f} → {result['hybrid_psnr']:.2f} dB (+{result['improvement']:.2f}dB)\n")
            f.write(f"效率: {result['efficiency']:.1f}\n")
            f.write("-" * 50 + "\n")
        
        f.write("\n阈值敏感性分析:\n")
        for result in threshold_results:
            f.write(f"阈值{result['threshold']:.3f}: +{result['improvement']:.2f}dB, {result['pixel_ratio']*100:.1f}%像素, 效率{result['efficiency']:.1f}\n")
    
    print("\n✅ 实时边缘检测方法测试完成！")
    print("\n📊 结果文件:")
    print("- ../results/images/realtime_*_threshold*.png: 不同阈值的结果图像")
    print("- ../results/images/threshold_sensitivity_analysis.png: 阈值敏感性分析")
    print("- ../results/reports/realtime_edge_method_report.txt: 详细实验报告")
    
    # 打印最佳推荐
    best_result = max(all_results, key=lambda x: x['efficiency'])
    print(f"\n🏆 推荐配置:")
    print(f"方法: {best_result['method'].upper()}")
    print(f"阈值: {best_result['threshold']:.3f}")
    print(f"效果: +{best_result['improvement']:.2f}dB, {best_result['pixel_ratio']*100:.1f}%像素, 效率{best_result['efficiency']:.1f}")

#!/usr/bin/env python3
"""
基于边缘检测的SR质量预测和tile替换方法
核心思想：边缘密集的区域细节丰富，SR容易失败，应该替换为ORG
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import os
from skimage.metrics import peak_signal_noise_ratio as psnr

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class EdgeBasedPredictor:
    """基于边缘检测的质量预测器"""
    
    def __init__(self):
        self.tile_size = 32
        
    def extract_edge_features(self, tile):
        """提取边缘相关特征"""
        if tile.max() > 1.0:
            tile = tile.astype(np.float32) / 255.0
            
        # 转换为灰度图
        if len(tile.shape) == 3:
            gray = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (tile * 255).astype(np.uint8)
            
        features = {}
        
        # 1. Sobel边缘检测
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        features['edge_density'] = np.mean(sobel_magnitude)
        features['edge_std'] = np.std(sobel_magnitude)
        features['edge_max'] = np.max(sobel_magnitude)
        features['strong_edge_ratio'] = np.sum(sobel_magnitude > np.percentile(sobel_magnitude, 90)) / sobel_magnitude.size
        
        # 2. Canny边缘检测
        edges_canny = cv2.Canny(gray, 50, 150)
        features['canny_edge_ratio'] = np.sum(edges_canny > 0) / edges_canny.size
        features['canny_edge_count'] = np.sum(edges_canny > 0)
        
        # 3. Laplacian边缘检测（衡量细节复杂度）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['laplacian_variance'] = np.var(laplacian)
        features['laplacian_mean'] = np.mean(np.abs(laplacian))
        
        # 4. 梯度统计
        grad_x = np.gradient(gray.astype(np.float32), axis=1)
        grad_y = np.gradient(gray.astype(np.float32), axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features['gradient_mean'] = np.mean(gradient_magnitude)
        features['gradient_std'] = np.std(gradient_magnitude)
        
        return features
    
    def predict_tile_quality(self, tile, method='combined'):
        """
        预测tile的质量（越高表示越可能需要替换）
        
        Args:
            tile: 32x32的图像块
            method: 预测方法
                - 'sobel': 基于Sobel边缘密度
                - 'canny': 基于Canny边缘比例  
                - 'laplacian': 基于Laplacian方差
                - 'gradient': 基于梯度统计
                - 'combined': 组合多种边缘特征
        """
        features = self.extract_edge_features(tile)
        
        if method == 'sobel':
            # Sobel边缘密度作为质量指标
            return features['edge_density']
            
        elif method == 'canny':
            # Canny边缘比例作为质量指标
            return features['canny_edge_ratio']
            
        elif method == 'laplacian':
            # Laplacian方差作为质量指标
            return features['laplacian_variance']
            
        elif method == 'gradient':
            # 梯度均值作为质量指标
            return features['gradient_mean']
            
        elif method == 'combined':
            # 组合多种边缘特征
            # 归一化各个特征到[0,1]范围
            edge_score = min(features['edge_density'] / 50.0, 1.0)  # Sobel
            canny_score = features['canny_edge_ratio']  # 本身就是比例
            laplacian_score = min(features['laplacian_variance'] / 1000.0, 1.0)  # Laplacian
            gradient_score = min(features['gradient_mean'] / 30.0, 1.0)  # 梯度
            
            # 加权组合
            combined_score = (0.3 * edge_score + 
                            0.3 * canny_score + 
                            0.2 * laplacian_score + 
                            0.2 * gradient_score)
            return combined_score
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def predict_image_quality(self, image_path, method='combined'):
        """预测整个图像的tile-wise质量"""
        img = np.array(Image.open(image_path))
        
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
            
        h, w = img.shape[:2]
        tile_h, tile_w = h // self.tile_size, w // self.tile_size
        
        quality_map = np.zeros((tile_h, tile_w))
        
        for i in range(tile_h):
            for j in range(tile_w):
                h_start, h_end = i * self.tile_size, (i + 1) * self.tile_size
                w_start, w_end = j * self.tile_size, (j + 1) * self.tile_size
                
                tile = img[h_start:h_end, w_start:w_end]
                quality_map[i, j] = self.predict_tile_quality(tile, method)
                
        return quality_map

def replace_random_tiles(sr_path, org_path, top_k=100, random_seed=42):
    """
    随机替换K个tiles作为baseline对比
    
    Args:
        sr_path: SR图像路径
        org_path: ORG图像路径  
        top_k: 替换的tile数量
        random_seed: 随机种子，确保结果可重复
    """
    np.random.seed(random_seed)
    
    # 加载图像
    sr_img = np.array(Image.open(sr_path))
    org_img = np.array(Image.open(org_path))
    
    # 计算总的tile数量
    h, w = sr_img.shape[:2]
    tile_size = 32
    tile_h, tile_w = h // tile_size, w // tile_size
    total_tiles = tile_h * tile_w
    
    # 随机选择K个tiles
    all_indices = list(range(total_tiles))
    random_indices = np.random.choice(all_indices, size=min(top_k, total_tiles), replace=False)
    random_tiles = [(idx // tile_w, idx % tile_w) for idx in random_indices]
    
    # 创建混合图像
    hybrid_img = sr_img.copy()
    replaced_pixels = 0
    
    for i, j in random_tiles:
        h_start, h_end = i * tile_size, (i + 1) * tile_size
        w_start, w_end = j * tile_size, (j + 1) * tile_size
        
        # 替换为ORG tile
        hybrid_img[h_start:h_end, w_start:w_end] = org_img[h_start:h_end, w_start:w_end]
        replaced_pixels += tile_size * tile_size
    
    return hybrid_img, random_tiles, replaced_pixels

def replace_worst_tiles_edge(sr_path, org_path, top_k=100, method='combined'):
    """
    基于边缘检测替换最差的K个tiles
    
    Args:
        sr_path: SR图像路径
        org_path: ORG图像路径  
        top_k: 替换的tile数量
        method: 边缘检测方法
    """
    predictor = EdgeBasedPredictor()
    
    # 加载图像
    sr_img = np.array(Image.open(sr_path))
    org_img = np.array(Image.open(org_path))
    
    # 预测质量
    quality_map = predictor.predict_image_quality(sr_path, method)
    
    # 找到最差的K个tiles
    h, w = quality_map.shape
    flat_indices = np.argsort(quality_map.flatten())[-top_k:]  # 质量分数高的认为需要替换
    worst_tiles = [(idx // w, idx % w) for idx in flat_indices]
    
    # 创建混合图像
    hybrid_img = sr_img.copy()
    tile_size = 32
    
    replaced_pixels = 0
    for i, j in worst_tiles:
        h_start, h_end = i * tile_size, (i + 1) * tile_size
        w_start, w_end = j * tile_size, (j + 1) * tile_size
        
        # 替换为ORG tile
        hybrid_img[h_start:h_end, w_start:w_end] = org_img[h_start:h_end, w_start:w_end]
        replaced_pixels += tile_size * tile_size
    
    return hybrid_img, quality_map, worst_tiles, replaced_pixels

def test_edge_methods():
    """测试不同边缘检测方法的效果，包括随机baseline"""
    sr_path = "../data/no2_pred_sr.png"
    org_path = "../data/no2_pred_org.png"
    gt_path = "../data/no2_gt.png"
    
    # 检查文件是否存在
    for path, name in [(sr_path, "SR"), (org_path, "ORG"), (gt_path, "GT")]:
        if not os.path.exists(path):
            print(f"错误: {name} 图片不存在: {path}")
            return
    
    # 加载图像
    sr_img = np.array(Image.open(sr_path))
    org_img = np.array(Image.open(org_path))
    gt_img = np.array(Image.open(gt_path))
    
    # 计算基础PSNR
    sr_psnr = psnr(gt_img, sr_img)
    org_psnr = psnr(gt_img, org_img)
    
    print("=== 基于边缘检测的Tile替换测试 ===")
    print(f"原始SR PSNR: {sr_psnr:.2f} dB")
    print(f"原始ORG PSNR: {org_psnr:.2f} dB")
    print()
    
    # 首先测试随机替换baseline
    print("=== BASELINE: 随机替换 ===")
    k_values = [30, 50, 100]
    random_results = {}
    
    for k in k_values:
        # 多次随机替换取平均值
        random_psnrs = []
        for seed in range(5):  # 5次随机实验
            hybrid_img, _, replaced_pixels = replace_random_tiles(
                sr_path, org_path, top_k=k, random_seed=seed
            )
            random_psnr = psnr(gt_img, hybrid_img)
            random_psnrs.append(random_psnr)
        
        avg_random_psnr = np.mean(random_psnrs)
        std_random_psnr = np.std(random_psnrs)
        improvement = avg_random_psnr - sr_psnr
        pixel_ratio = replaced_pixels / (sr_img.shape[0] * sr_img.shape[1])
        efficiency = improvement / pixel_ratio if pixel_ratio > 0 else 0
        
        random_results[k] = {
            'psnr': avg_random_psnr,
            'std': std_random_psnr,
            'improvement': improvement,
            'pixel_ratio': pixel_ratio,
            'efficiency': efficiency
        }
        
        print(f"K={k:3d}: PSNR={avg_random_psnr:.2f}±{std_random_psnr:.2f}dB "
              f"(+{improvement:+.2f}dB), 像素比例={pixel_ratio*100:.1f}%, "
              f"效率={efficiency:.1f}")
    
    print()
    
    # 测试不同边缘检测方法
    methods = ['sobel', 'canny', 'laplacian', 'gradient', 'combined']
    
    results = {'random': random_results}
    
    for method in methods:
        print(f"=== 测试方法: {method.upper()} ===")
        method_results = {}
        
        for k in k_values:
            hybrid_img, quality_map, worst_tiles, replaced_pixels = replace_worst_tiles_edge(
                sr_path, org_path, top_k=k, method=method
            )
            
            # 计算混合图像PSNR
            hybrid_psnr = psnr(gt_img, hybrid_img)
            improvement = hybrid_psnr - sr_psnr
            pixel_ratio = replaced_pixels / (sr_img.shape[0] * sr_img.shape[1])
            efficiency = improvement / pixel_ratio if pixel_ratio > 0 else 0
            
            method_results[k] = {
                'psnr': hybrid_psnr,
                'improvement': improvement,
                'pixel_ratio': pixel_ratio,
                'efficiency': efficiency
            }
            
            print(f"K={k:3d}: PSNR={hybrid_psnr:.2f}dB (+{improvement:+.2f}dB), "
                  f"像素比例={pixel_ratio*100:.1f}%, 效率={efficiency:.1f}")
            
            # 保存混合图像
            hybrid_path = f"../results/images/edge_{method}_k{k}.png"
            Image.fromarray(hybrid_img.astype(np.uint8)).save(hybrid_path)
        
        results[method] = method_results
        print()
    
    return results

def visualize_edge_quality_map(image_path, method='combined', save_path=None):
    """可视化边缘质量预测图"""
    predictor = EdgeBasedPredictor()
    quality_map = predictor.predict_image_quality(image_path, method)
    
    plt.figure(figsize=(12, 5))
    
    # 原图
    plt.subplot(1, 2, 1)
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title("Original Image", fontsize=14)
    plt.axis('off')
    
    # 质量预测图
    plt.subplot(1, 2, 2)
    plt.imshow(quality_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Edge Complexity Score')
    plt.title(f'Edge-based Quality Prediction ({method})', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"边缘质量预测图保存到: {save_path}")
    
    plt.close()  # 关闭图像避免显示
    return quality_map

def compare_with_ml_method():
    """与机器学习方法和随机baseline对比"""
    sr_path = "../data/no2_pred_sr.png"
    org_path = "../data/no2_pred_org.png"
    gt_path = "../data/no2_gt.png"
    
    # 加载图像
    sr_img = np.array(Image.open(sr_path))
    gt_img = np.array(Image.open(gt_path))
    sr_psnr = psnr(gt_img, sr_img)
    
    print("=== 方法对比总结 (K=100) ===")
    print(f"基础SR PSNR: {sr_psnr:.2f} dB\n")
    
    # 随机baseline
    hybrid_img_random, _, replaced_pixels_random = replace_random_tiles(
        sr_path, org_path, top_k=100, random_seed=42
    )
    random_psnr = psnr(gt_img, hybrid_img_random)
    random_improvement = random_psnr - sr_psnr
    random_pixel_ratio = replaced_pixels_random / (sr_img.shape[0] * sr_img.shape[1])
    random_efficiency = random_improvement / random_pixel_ratio
    
    print("随机替换方法 (K=100):")
    print(f"  PSNR: {random_psnr:.2f} dB (+{random_improvement:+.2f} dB)")
    print(f"  像素比例: {random_pixel_ratio*100:.1f}%")
    print(f"  效率: {random_efficiency:.1f}")
    
    # 边缘方法 (最佳：Canny)
    hybrid_img_edge, _, _, replaced_pixels_edge = replace_worst_tiles_edge(
        sr_path, org_path, top_k=100, method='canny'
    )
    edge_psnr = psnr(gt_img, hybrid_img_edge)
    edge_improvement = edge_psnr - sr_psnr
    edge_pixel_ratio = replaced_pixels_edge / (sr_img.shape[0] * sr_img.shape[1])
    edge_efficiency = edge_improvement / edge_pixel_ratio
    
    print("\n边缘检测方法 (Canny, K=100):")
    print(f"  PSNR: {edge_psnr:.2f} dB (+{edge_improvement:+.2f} dB)")
    print(f"  像素比例: {edge_pixel_ratio*100:.1f}%")
    print(f"  效率: {edge_efficiency:.1f}")
    
    # 机器学习方法的结果（从之前的实验）
    print("\n机器学习方法 (K=100) [参考]:")
    print(f"  PSNR: ~{sr_psnr + 1.00:.2f} dB (+1.00 dB)")
    print(f"  像素比例: 13.0%")
    print(f"  效率: 7.7")
    
    # 真实PSNR方法（理论最优）
    print("\n真实PSNR方法 (K=100) [理论最优]:")
    print(f"  PSNR: ~{sr_psnr + 0.95:.2f} dB (+0.95 dB)")
    print(f"  像素比例: 3.4%")
    print(f"  效率: 28.0")
    
    # 保存对比图
    save_method_comparison(sr_img, hybrid_img_random, hybrid_img_edge, gt_img, 
                          sr_psnr, random_psnr, edge_psnr)
    
    return {
        'random': {'psnr': random_psnr, 'improvement': random_improvement, 'efficiency': random_efficiency},
        'edge': {'psnr': edge_psnr, 'improvement': edge_improvement, 'efficiency': edge_efficiency}
    }

def save_method_comparison(sr_img, random_img, edge_img, gt_img, sr_psnr, random_psnr, edge_psnr):
    """保存方法对比可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].imshow(sr_img)
    axes[0,0].set_title(f'SR Original\nPSNR: {sr_psnr:.2f} dB', fontsize=12)
    axes[0,0].axis('off')
    
    axes[0,1].imshow(random_img)
    axes[0,1].set_title(f'Random Replacement\nPSNR: {random_psnr:.2f} dB', fontsize=12)
    axes[0,1].axis('off')
    
    axes[1,0].imshow(edge_img)
    axes[1,0].set_title(f'Edge-based Replacement\nPSNR: {edge_psnr:.2f} dB', fontsize=12)
    axes[1,0].axis('off')
    
    axes[1,1].imshow(gt_img)
    axes[1,1].set_title('Ground Truth', fontsize=12)
    axes[1,1].axis('off')
    
    plt.tight_layout()
    save_path = f'../results/images/method_comparison_all.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"方法对比图保存到: {save_path}")
    plt.close()

def save_comparison_visualization(sr_img, hybrid_img, gt_img, sr_psnr, hybrid_psnr, method_name):
    """保存对比可视化"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(sr_img)
    axes[0].set_title(f'SR Original\nPSNR: {sr_psnr:.2f} dB', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(hybrid_img)
    axes[1].set_title(f'{method_name} Hybrid\nPSNR: {hybrid_psnr:.2f} dB', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(gt_img)
    axes[2].set_title('Ground Truth', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    save_path = f'../results/images/edge_method_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"对比图保存到: {save_path}")
    plt.close()

if __name__ == "__main__":
    print("🚀 开始边缘检测方法测试...")
    
    # 1. 生成边缘质量预测图
    sr_path = "../data/no2_pred_sr.png"
    print("1. 生成边缘质量预测图...")
    visualize_edge_quality_map(sr_path, method='combined', 
                              save_path='../results/images/edge_quality_prediction_map.png')
    
    # 2. 测试不同边缘检测方法
    print("2. 测试不同边缘检测方法...")
    results = test_edge_methods()
    
    # 3. 与机器学习方法对比  
    print("3. 与机器学习方法对比...")
    compare_with_ml_method()
    
    # 4. 保存详细报告
    with open('../results/reports/edge_method_report.txt', 'w') as f:
        f.write("=== 基于边缘检测的Tile替换方法实验报告 ===\n\n")
        
        # 首先写入随机baseline
        f.write("BASELINE: 随机替换\n")
        f.write("-" * 50 + "\n")
        if 'random' in results:
            random_results = results['random']
            for k, result in random_results.items():
                f.write(f"K={k}: PSNR={result['psnr']:.2f}±{result['std']:.2f}dB "
                       f"(+{result['improvement']:+.2f}dB), "
                       f"像素比例={result['pixel_ratio']*100:.1f}%, "
                       f"效率={result['efficiency']:.1f}\n")
            f.write("\n")
        
        # 然后写入边缘检测方法
        for method, method_results in results.items():
            if method == 'random':
                continue
            f.write(f"方法: {method.upper()}\n")
            f.write("-" * 50 + "\n")
            for k, result in method_results.items():
                f.write(f"K={k}: PSNR={result['psnr']:.2f}dB "
                       f"(+{result['improvement']:+.2f}dB), "
                       f"像素比例={result['pixel_ratio']*100:.1f}%, "
                       f"效率={result['efficiency']:.1f}\n")
            f.write("\n")
    
    print("✅ 边缘检测方法测试完成！")
    print("\n📊 结果文件:")
    print("- ../results/images/edge_quality_prediction_map.png: 边缘质量预测图")
    print("- ../results/images/edge_*_k*.png: 不同方法的混合图像")
    print("- ../results/images/edge_method_comparison.png: 与其他方法对比")
    print("- ../results/reports/edge_method_report.txt: 详细实验报告")

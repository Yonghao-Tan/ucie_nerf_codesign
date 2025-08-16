#!/usr/bin/env python3
"""
简化版实际SR质量预测器
仅使用启发式方法，无需训练模型
基于前面分析的关键失败模式
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

class SimpleSRQualityPredictor:
    """简化版SR质量预测器 - 基于启发式方法"""
    
    def predict_tile_quality(self, tile):
        """
        预测单个tile的质量分数（越高表示质量越差）
        基于前面分析的关键SR失败模式
        """
        if tile.max() > 1.0:
            tile = tile.astype(np.float32) / 255.0
        
        gray = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        # 基于前面分析的关键失败模式
        score = 0
        
        # 1. 亮度因子（最重要 - 16.8%差异）
        brightness_mean = np.mean(gray)
        brightness_factor = brightness_mean * 4.0  # 高权重
        
        # 高亮度区域加重惩罚
        if brightness_mean > 0.6:
            brightness_factor *= 1.5
        
        # 2. 边缘密度因子（23.8%差异）
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_density = np.mean(edge_magnitude)
        edge_factor = edge_density * 5.0  # 高权重
        
        # 3. 高频因子（13.2%差异）
        # 简化的高频检测
        fft = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft)
        h, w = fft_magnitude.shape
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4
        
        y, x = np.ogrid[:h, :w]
        mask = (x - center_w)**2 + (y - center_h)**2 > radius**2
        
        high_freq_energy = np.sum(fft_magnitude[mask])
        total_energy = np.sum(fft_magnitude)
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        freq_factor = high_freq_ratio * 3.0
        
        # 4. 纹理复杂度因子（22.7%差异）
        kernel = np.ones((3, 3), np.float32) / 9
        local_mean = cv2.filter2D(gray, -1, kernel)
        local_variance = cv2.filter2D(gray**2, -1, kernel) - local_mean**2
        texture_complexity = np.mean(local_variance)
        texture_factor = texture_complexity * 1000  # 放大到合适范围
        
        # 5. 综合评分
        score = brightness_factor + edge_factor + freq_factor + texture_factor
        
        # 6. 特殊情况加权
        # 高亮像素比例
        bright_pixel_ratio = np.sum(gray > 0.7) / gray.size
        if bright_pixel_ratio > 0.3:
            score *= 1.3
        
        # 强边缘比例
        strong_edge_ratio = np.sum(edge_magnitude > 0.1) / edge_magnitude.size
        if strong_edge_ratio > 0.1:
            score *= 1.2
        
        # 7. 颜色单调性（最差tiles RGB标准差更低）
        r_std = np.std(tile[:, :, 0])
        g_std = np.std(tile[:, :, 1])
        b_std = np.std(tile[:, :, 2])
        avg_color_std = (r_std + g_std + b_std) / 3
        
        # 色彩变化小的高亮区域更容易SR失败
        if brightness_mean > 0.5 and avg_color_std < 0.12:
            score *= 1.25
        
        return score
    
    def predict_image_quality(self, image_path, tile_size=32):
        """预测整张图像的质量"""
        img = np.array(Image.open(image_path))
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        
        h, w = img.shape[:2]
        quality_scores = []
        tile_positions = []
        
        for i in range(0, h, tile_size):
            for j in range(0, w, tile_size):
                if i + tile_size <= h and j + tile_size <= w:
                    tile = img[i:i+tile_size, j:j+tile_size]
                    if tile.shape[:2] == (tile_size, tile_size):
                        score = self.predict_tile_quality(tile)
                        quality_scores.append(score)
                        tile_positions.append((i//tile_size, j//tile_size))
        
        return np.array(quality_scores), tile_positions
    
    def identify_poor_quality_tiles(self, image_path, top_k=100, tile_size=32):
        """识别质量最差的K个tiles"""
        quality_scores, positions = self.predict_image_quality(image_path, tile_size)
        
        worst_indices = np.argsort(quality_scores)[-top_k:]
        worst_scores = quality_scores[worst_indices]
        worst_positions = [positions[i] for i in worst_indices]
        
        return {
            'worst_indices': worst_indices,
            'worst_scores': worst_scores,
            'worst_positions': worst_positions,
            'all_scores': quality_scores,
            'all_positions': positions
        }
    
    def visualize_quality_map(self, image_path, save_path="../results/images/simple_quality_heatmap.png", tile_size=32):
        """可视化质量热力图"""
        quality_scores, positions = self.predict_image_quality(image_path, tile_size)
        
        max_row = max(pos[0] for pos in positions) + 1
        max_col = max(pos[1] for pos in positions) + 1
        
        heatmap = np.zeros((max_row, max_col))
        for i, (row, col) in enumerate(positions):
            heatmap[row, col] = quality_scores[i]
        
        plt.figure(figsize=(15, 6))
        
        # 原图
        plt.subplot(1, 2, 1)
        img = Image.open(image_path)
        plt.imshow(img)
        plt.title('SR Image')
        plt.axis('off')
        
        # 质量热力图
        plt.subplot(1, 2, 2)
        im = plt.imshow(heatmap, cmap='viridis')
        plt.title('Predicted Quality Map\n(Yellow = Poor Quality)')
        plt.xlabel('Tile Column')
        plt.ylabel('Tile Row')
        plt.colorbar(im, label='Quality Score')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"质量热力图保存到: {save_path}")
        plt.show()
        
        return heatmap

def test_against_ground_truth():
    """对比启发式预测与真实情况"""
    print("=== 启发式预测 vs 真实质量对比 ===")
    
    # 加载真实的PSNR损失数据
    import pickle
    
    if not os.path.exists('sr_quality_predictor.pkl'):
        print("错误：需要先运行sr_quality_predictor.py生成训练数据")
        return
    
    # 加载之前计算的真实PSNR损失
    sr_path = "../data/no2_pred_sr.png"
    gt_path = "../data/no2_gt.png"
    
    # 计算真实PSNR损失
    sr_img = np.array(Image.open(sr_path))
    gt_img = np.array(Image.open(gt_path))
    
    if sr_img.max() > 1.0:
        sr_img = sr_img.astype(np.float32) / 255.0
    if gt_img.max() > 1.0:
        gt_img = gt_img.astype(np.float32) / 255.0
    
    h, w = sr_img.shape[:2]
    tile_size = 32
    
    true_losses = []
    sr_tiles = []
    
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            if i + tile_size <= h and j + tile_size <= w:
                sr_tile = sr_img[i:i+tile_size, j:j+tile_size]
                gt_tile = gt_img[i:i+tile_size, j:j+tile_size]
                
                if sr_tile.shape[:2] == (tile_size, tile_size):
                    mse = np.mean((sr_tile - gt_tile) ** 2)
                    if mse == 0:
                        psnr_loss = 0
                    else:
                        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
                        psnr_loss = max(0, 50 - psnr)
                    
                    true_losses.append(psnr_loss)
                    sr_tiles.append(sr_tile)
    
    true_losses = np.array(true_losses)
    
    # 使用启发式方法预测
    predictor = SimpleSRQualityPredictor()
    predicted_scores = []
    
    print("预测tiles质量...")
    for i, tile in enumerate(sr_tiles):
        if i % 100 == 0:
            print(f"  已处理 {i}/{len(sr_tiles)} 个tiles")
        score = predictor.predict_tile_quality(tile)
        predicted_scores.append(score)
    
    predicted_scores = np.array(predicted_scores)
    
    # 计算准确性
    print("\n=== 启发式方法预测准确性 ===")
    
    correlation = np.corrcoef(true_losses, predicted_scores)[0, 1]
    print(f"整体相关性: {correlation:.4f}")
    
    # 测试不同K值的准确性
    k_values = [10, 30, 50, 100, 200, 300]
    
    print(f"\n不同K值下的预测准确性:")
    print(f"{'K值':<5} {'重叠数量':<8} {'精确率':<8}")
    print("-" * 25)
    
    for k in k_values:
        if k <= len(sr_tiles):
            true_worst_k = set(np.argsort(true_losses)[-k:])
            pred_worst_k = set(np.argsort(predicted_scores)[-k:])
            
            overlap_k = len(true_worst_k & pred_worst_k)
            precision = overlap_k / k
            
            print(f"{k:<5} {overlap_k:<8} {precision:<8.3f}")
    
    # 可视化对比
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(true_losses, predicted_scores, alpha=0.6, s=15)
    plt.plot([min(true_losses), max(true_losses)], 
             [min(predicted_scores), max(predicted_scores)], 'r--', alpha=0.8)
    plt.xlabel('True PSNR Loss (dB)')
    plt.ylabel('Predicted Quality Score')
    plt.title(f'Heuristic Prediction vs Truth\nCorrelation: {correlation:.3f}')
    plt.grid(True, alpha=0.3)
    
    # 排名对比
    true_ranks = np.argsort(np.argsort(true_losses))
    pred_ranks = np.argsort(np.argsort(predicted_scores))
    
    plt.subplot(1, 2, 2)
    plt.scatter(true_ranks, pred_ranks, alpha=0.6, s=15)
    plt.plot([0, len(true_ranks)], [0, len(true_ranks)], 'r--', alpha=0.8)
    plt.xlabel('True Quality Rank')
    plt.ylabel('Predicted Quality Rank')
    plt.title('Rank Correlation')
    plt.grid(True, alpha=0.3)
    
    rank_correlation = np.corrcoef(true_ranks, pred_ranks)[0, 1]
    plt.text(0.05, 0.95, f'Rank Corr: {rank_correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('../results/images/heuristic_prediction_validation.png', dpi=300, bbox_inches='tight')
    print(f"\n验证结果图保存到: ../results/images/heuristic_prediction_validation.png")
    plt.show()
    
    return correlation, predicted_scores, true_losses

def main():
    print("=== 简化版SR质量预测器演示 ===")
    
    sr_path = "../data/no2_pred_sr.png"
    if not os.path.exists(sr_path):
        print(f"错误：找不到 {sr_path}")
        return
    
    predictor = SimpleSRQualityPredictor()
    
    # 1. 预测质量
    print("1. 预测图像质量...")
    quality_scores, positions = predictor.predict_image_quality(sr_path)
    
    print(f"总tiles数量: {len(quality_scores)}")
    print(f"质量分数范围: {quality_scores.min():.3f} - {quality_scores.max():.3f}")
    print(f"平均质量分数: {quality_scores.mean():.3f}")
    
    # 2. 识别最差tiles
    top_k = 100
    print(f"\n2. 识别质量最差的 {top_k} 个tiles...")
    worst_results = predictor.identify_poor_quality_tiles(sr_path, top_k=top_k)
    
    print(f"最差tiles质量分数范围: {worst_results['worst_scores'].min():.3f} - {worst_results['worst_scores'].max():.3f}")
    
    # 显示前10个最差tiles
    print(f"\n最差的10个tiles位置和分数:")
    for i in range(min(10, len(worst_results['worst_positions']))):
        pos = worst_results['worst_positions'][-(i+1)]
        score = worst_results['worst_scores'][-(i+1)]
        row_pixel = pos[0] * 32
        col_pixel = pos[1] * 32
        print(f"  Rank {i+1}: Tile({pos[0]}, {pos[1]}) - 像素坐标[{row_pixel}:{row_pixel+32}, {col_pixel}:{col_pixel+32}] - 分数: {score:.3f}")
    
    # 3. 可视化
    print(f"\n3. 生成质量热力图...")
    predictor.visualize_quality_map(sr_path)
    
    # 4. 与真实情况对比（如果有数据）
    if os.path.exists('sr_quality_predictor.pkl'):
        print(f"\n4. 验证预测准确性...")
        correlation, pred_scores, true_losses = test_against_ground_truth()
        
        print(f"\n=== 最终结果 ===")
        print(f"启发式方法相关性: {correlation:.4f}")
        print(f"这个结果说明：")
        if correlation > 0.7:
            print("  ✅ 启发式方法预测准确性很高！")
        elif correlation > 0.5:
            print("  ⚠️  启发式方法预测准确性中等")
        else:
            print("  ❌ 启发式方法需要改进")
    
    # 5. 保存结果
    print(f"\n5. 保存预测结果...")
    with open('../results/reports/simple_quality_prediction_report.txt', 'w') as f:
        f.write("=== 简化版SR质量预测报告 ===\n\n")
        f.write(f"图像路径: {sr_path}\n")
        f.write(f"总tiles数量: {len(quality_scores)}\n")
        f.write(f"质量分数范围: {quality_scores.min():.3f} - {quality_scores.max():.3f}\n")
        f.write(f"平均质量分数: {quality_scores.mean():.3f}\n\n")
        
        f.write(f"质量最差的 {top_k} 个tiles:\n")
        f.write("排名   Tile位置    像素坐标             质量分数\n")
        f.write("-" * 50 + "\n")
        
        for i in range(len(worst_results['worst_positions'])):
            rank = len(worst_results['worst_positions']) - i
            pos = worst_results['worst_positions'][-(i+1)]
            score = worst_results['worst_scores'][-(i+1)]
            row_pixel = pos[0] * 32
            col_pixel = pos[1] * 32
            f.write(f"{rank:3d}   ({pos[0]:2d}, {pos[1]:2d})    [{row_pixel:3d}:{row_pixel+32:3d}, {col_pixel:3d}:{col_pixel+32:3d}]     {score:.3f}\n")
    
    print("报告保存到: simple_quality_prediction_report.txt")

if __name__ == "__main__":
    main()

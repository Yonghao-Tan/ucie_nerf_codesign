#!/usr/bin/env python3
"""
实用SR质量预测器 - 专门用于实际部署
仅使用SR图像预测质量，不需要GT参考
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import os

class SRQualityPredictor:
    """SR质量预测器类"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.feature_names = []
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def extract_features(self, tile):
        """提取SR质量相关特征"""
        if tile.max() > 1.0:
            tile = tile.astype(np.float32) / 255.0
        
        features = {}
        gray = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        # 1. 亮度特征（关键失败指标）
        features['brightness_mean'] = np.mean(gray)
        features['brightness_max'] = np.max(gray)
        features['brightness_range'] = np.max(gray) - np.min(gray)
        features['bright_pixel_ratio'] = np.sum(gray > 0.7) / gray.size
        
        # 2. 边缘特征
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        features['edge_density'] = np.mean(edge_magnitude)
        features['edge_max'] = np.max(edge_magnitude)
        features['strong_edge_ratio'] = np.sum(edge_magnitude > 0.1) / edge_magnitude.size
        
        # 3. 高频成分
        fft = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft)
        h, w = fft_magnitude.shape
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4
        
        y, x = np.ogrid[:h, :w]
        mask = (x - center_w)**2 + (y - center_h)**2 > radius**2
        
        high_freq_energy = np.sum(fft_magnitude[mask])
        total_energy = np.sum(fft_magnitude)
        features['high_freq_ratio'] = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # 4. 纹理复杂度
        kernel = np.ones((3, 3), np.float32) / 9
        local_mean = cv2.filter2D(gray, -1, kernel)
        local_variance = cv2.filter2D(gray**2, -1, kernel) - local_mean**2
        features['texture_complexity'] = np.mean(local_variance)
        
        # 5. 颜色特征
        for i, channel in enumerate(['r', 'g', 'b']):
            channel_data = tile[:, :, i]
            features[f'{channel}_mean'] = np.mean(channel_data)
            features[f'{channel}_std'] = np.std(channel_data)
        
        # 6. 失真检测
        gray_uint8 = (gray * 255).astype(np.uint8)
        laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
        features['sharpness'] = np.var(laplacian)
        
        # 7. 质量启发式指标
        features['brightness_edge_interaction'] = features['brightness_mean'] * features['edge_density']
        features['complexity_score'] = features['high_freq_ratio'] * features['texture_complexity']
        
        return features
    
    def predict_tile_quality(self, tile):
        """预测单个tile的质量分数（越高表示质量越差）"""
        if self.model is None:
            # 如果没有训练好的模型，使用启发式方法
            return self._heuristic_quality_score(tile)
        
        features = self.extract_features(tile)
        feature_vector = np.array([features[name] for name in self.feature_names]).reshape(1, -1)
        
        return self.model.predict(feature_vector)[0]
    
    def _heuristic_quality_score(self, tile):
        """基于特征分析的启发式质量评分"""
        features = self.extract_features(tile)
        
        # 基于之前分析的关键失败模式
        score = 0
        
        # 亮度因子（亮度高的区域SR容易失败）
        brightness_factor = features['brightness_mean'] * 2.0
        if features['brightness_mean'] > 0.6:
            brightness_factor *= 1.5  # 高亮度区域加重惩罚
        
        # 边缘密度因子（边缘多的区域SR难处理）
        edge_factor = features['edge_density'] * 3.0
        
        # 高频因子（高频内容SR容易丢失）
        freq_factor = features['high_freq_ratio'] * 2.5
        
        # 纹理复杂度因子
        texture_factor = features['texture_complexity'] * 1000  # 放大到合适范围
        
        # 综合评分
        score = brightness_factor + edge_factor + freq_factor + texture_factor
        
        # 特殊情况加权
        if features['bright_pixel_ratio'] > 0.3:  # 大量高亮像素
            score *= 1.3
        
        if features['strong_edge_ratio'] > 0.1:  # 大量强边缘
            score *= 1.2
        
        return score
    
    def predict_image_quality(self, image_path, tile_size=32, return_details=False):
        """预测整张图像的质量，返回每个tile的质量分数"""
        # 加载图像
        img = np.array(Image.open(image_path))
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        
        h, w = img.shape[:2]
        quality_scores = []
        tile_positions = []
        tiles = []
        
        # 提取tiles并预测质量
        for i in range(0, h, tile_size):
            for j in range(0, w, tile_size):
                if i + tile_size <= h and j + tile_size <= w:
                    tile = img[i:i+tile_size, j:j+tile_size]
                    if tile.shape[:2] == (tile_size, tile_size):
                        score = self.predict_tile_quality(tile)
                        quality_scores.append(score)
                        tile_positions.append((i//tile_size, j//tile_size))
                        if return_details:
                            tiles.append(tile)
        
        quality_scores = np.array(quality_scores)
        
        if return_details:
            return quality_scores, tile_positions, tiles
        else:
            return quality_scores, tile_positions
    
    def identify_poor_quality_tiles(self, image_path, top_k=100, tile_size=32):
        """识别质量最差的K个tiles"""
        quality_scores, positions = self.predict_image_quality(image_path, tile_size)
        
        # 找到质量最差的K个tiles
        worst_indices = np.argsort(quality_scores)[-top_k:]
        worst_scores = quality_scores[worst_indices]
        worst_positions = [positions[i] for i in worst_indices]
        
        results = {
            'worst_indices': worst_indices,
            'worst_scores': worst_scores,
            'worst_positions': worst_positions,
            'all_scores': quality_scores,
            'all_positions': positions
        }
        
        return results
    
    def visualize_quality_map(self, image_path, save_path="../results/images/quality_heatmap.png", tile_size=32):
        """可视化质量热力图"""
        quality_scores, positions = self.predict_image_quality(image_path, tile_size)
        
        # 重建热力图
        max_row = max(pos[0] for pos in positions) + 1
        max_col = max(pos[1] for pos in positions) + 1
        
        heatmap = np.zeros((max_row, max_col))
        for i, (row, col) in enumerate(positions):
            heatmap[row, col] = quality_scores[i]
        
        # 可视化
        plt.figure(figsize=(12, 8))
        
        # 原图
        plt.subplot(1, 2, 1)
        img = Image.open(image_path)
        plt.imshow(img)
        plt.title('Original SR Image')
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
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']
        print(f"模型加载完成: {model_path}")
    
    def save_model(self, model_path):
        """保存模型"""
        data = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        with open(model_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"模型保存完成: {model_path}")

def demonstrate_sr_quality_prediction():
    """演示SR质量预测"""
    print("=== SR质量预测演示 ===")
    
    # 初始化预测器（尝试加载训练好的模型）
    predictor = SRQualityPredictor()
    
    if os.path.exists('sr_quality_predictor.pkl'):
        predictor.load_model('sr_quality_predictor.pkl')
        print("使用训练好的模型进行预测")
    else:
        print("使用启发式方法进行预测")
    
    # 检查文件是否存在
    sr_path = "../data/no2_pred_sr.png"
    if not os.path.exists(sr_path):
        print(f"错误：找不到 {sr_path}")
        print("请确保SR图像文件存在")
        return
    
    print(f"分析图像: {sr_path}")
    
    # 1. 预测整张图像的质量
    print("1. 预测整张图像质量...")
    quality_scores, positions = predictor.predict_image_quality(sr_path)
    
    print(f"总tiles数量: {len(quality_scores)}")
    print(f"质量分数范围: {quality_scores.min():.3f} - {quality_scores.max():.3f}")
    print(f"平均质量分数: {quality_scores.mean():.3f}")
    
    # 2. 识别最差的tiles
    top_k = 100
    print(f"\n2. 识别质量最差的 {top_k} 个tiles...")
    worst_results = predictor.identify_poor_quality_tiles(sr_path, top_k=top_k)
    
    print(f"最差tiles质量分数范围: {worst_results['worst_scores'].min():.3f} - {worst_results['worst_scores'].max():.3f}")
    
    # 显示前10个最差tiles的位置
    print(f"\n最差的10个tiles位置和分数:")
    for i in range(min(10, len(worst_results['worst_positions']))):
        pos = worst_results['worst_positions'][-(i+1)]  # 从最差开始
        score = worst_results['worst_scores'][-(i+1)]
        row_pixel = pos[0] * 32
        col_pixel = pos[1] * 32
        print(f"  Rank {i+1}: Tile({pos[0]}, {pos[1]}) - 像素坐标[{row_pixel}:{row_pixel+32}, {col_pixel}:{col_pixel+32}] - 分数: {score:.3f}")
    
    # 3. 可视化质量图
    print(f"\n3. 生成质量热力图...")
    heatmap = predictor.visualize_quality_map(sr_path, "../results/images/predicted_quality_heatmap.png")
    
    # 4. 保存预测结果
    print(f"\n4. 保存预测结果...")
    
    # 保存详细报告
    with open('predicted_quality_report.txt', 'w') as f:
        f.write("=== SR质量预测报告（无GT参考）===\n\n")
        f.write(f"图像路径: {sr_path}\n")
        f.write(f"总tiles数量: {len(quality_scores)}\n")
        f.write(f"质量分数范围: {quality_scores.min():.3f} - {quality_scores.max():.3f}\n")
        f.write(f"平均质量分数: {quality_scores.mean():.3f}\n")
        f.write(f"质量分数标准差: {quality_scores.std():.3f}\n\n")
        
        f.write(f"质量最差的 {top_k} 个tiles:\n")
        f.write("排名   Tile位置    像素坐标             质量分数\n")
        f.write("-" * 50 + "\n")
        
        for i in range(len(worst_results['worst_positions'])):
            rank = len(worst_results['worst_positions']) - i  # 从最差开始排名
            pos = worst_results['worst_positions'][-(i+1)]
            score = worst_results['worst_scores'][-(i+1)]
            row_pixel = pos[0] * 32
            col_pixel = pos[1] * 32
            f.write(f"{rank:3d}   ({pos[0]:2d}, {pos[1]:2d})    [{row_pixel:3d}:{row_pixel+32:3d}, {col_pixel:3d}:{col_pixel+32:3d}]     {score:.3f}\n")
    
    print("预测报告保存到: predicted_quality_report.txt")
    
    # 5. 生成推荐的替换策略
    print(f"\n5. 生成替换策略建议...")
    
    replacement_ratios = [0.5, 1.0, 2.0, 3.0, 5.0]  # 不同的替换像素比例
    
    print(f"建议的替换策略（基于质量预测）:")
    print(f"{'像素比例':<8} {'Tiles数量':<10} {'预期改善':<12}")
    print("-" * 35)
    
    total_pixels = len(quality_scores) * 32 * 32
    
    for ratio in replacement_ratios:
        target_pixels = int(total_pixels * ratio / 100)
        target_tiles = target_pixels // (32 * 32)
        target_tiles = min(target_tiles, len(quality_scores))
        
        if target_tiles > 0:
            # 估计改善程度（基于质量分数）
            worst_scores_subset = np.sort(quality_scores)[-target_tiles:]
            avg_worst_score = np.mean(worst_scores_subset)
            avg_all_score = np.mean(quality_scores)
            expected_improvement = (avg_worst_score - avg_all_score) / avg_all_score * 100
            
            print(f"{ratio:5.1f}%   {target_tiles:6d}     {expected_improvement:6.1f}%")
    
    print(f"\n=== 预测完成 ===")
    print(f"生成文件:")
    print(f"- ../results/images/predicted_quality_heatmap.png: 质量热力图")
    print(f"- predicted_quality_report.txt: 详细预测报告")

if __name__ == "__main__":
    demonstrate_sr_quality_prediction()

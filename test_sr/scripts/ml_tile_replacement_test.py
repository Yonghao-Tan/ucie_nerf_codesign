#!/usr/bin/env python3
"""
使用机器学习模型进行SR质量预测和tile替换测试
完整的端到端评估流程
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from scipy.fftpack import dct

def load_trained_model(model_path='../models/sr_quality_predictor.pkl'):
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        print("请先运行 sr_quality_predictor.py 训练模型")
        return None, None
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        feature_names = data['feature_names']
    
    print(f"模型加载成功: {model_path}")
    print(f"特征数量: {len(feature_names)}")
    return model, feature_names

def extract_ml_features(tile):
    """
    提取与训练模型一致的特征
    这必须与 sr_quality_predictor.py 中的 extract_sr_only_features 函数完全一致
    """
    if tile.max() > 1.0:
        tile = tile.astype(np.float32) / 255.0
    
    features = {}
    
    # 1. 亮度相关特征
    gray = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    features['brightness_mean'] = np.mean(gray)
    features['brightness_std'] = np.std(gray)
    features['brightness_max'] = np.max(gray)
    features['brightness_min'] = np.min(gray)
    features['brightness_range'] = features['brightness_max'] - features['brightness_min']
    
    features['bright_pixel_ratio'] = np.sum(gray > 0.7) / gray.size
    features['very_bright_pixel_ratio'] = np.sum(gray > 0.85) / gray.size
    
    features['brightness_skewness'] = stats.skew(gray.flatten())
    features['brightness_kurtosis'] = stats.kurtosis(gray.flatten())
    
    # 2. 边缘和细节特征
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    features['edge_density'] = np.mean(edge_magnitude)
    features['edge_max'] = np.max(edge_magnitude)
    features['edge_std'] = np.std(edge_magnitude)
    features['strong_edge_ratio'] = np.sum(edge_magnitude > 0.1) / edge_magnitude.size
    
    # Canny边缘检测
    gray_uint8 = (gray * 255).astype(np.uint8)
    canny = cv2.Canny(gray_uint8, 50, 150)
    features['canny_edge_ratio'] = np.sum(canny > 0) / canny.size
    
    # 3. 频域特征
    dct_coeff = dct(dct(gray.T).T)
    h, w = dct_coeff.shape
    high_freq_region = dct_coeff[h//2:, w//2:]
    features['high_freq_energy'] = np.sum(high_freq_region**2) / np.sum(dct_coeff**2)
    
    fft = np.fft.fft2(gray)
    fft_magnitude = np.abs(fft)
    h, w = fft_magnitude.shape
    center_h, center_w = h // 2, w // 2
    radius = min(h, w) // 4
    
    y, x = np.ogrid[:h, :w]
    mask = (x - center_w)**2 + (y - center_h)**2 > radius**2
    
    high_freq_energy = np.sum(fft_magnitude[mask])
    total_energy = np.sum(fft_magnitude)
    features['fft_high_freq_ratio'] = high_freq_energy / total_energy if total_energy > 0 else 0
    
    # 4. 纹理复杂度特征
    kernel = np.ones((3, 3), np.float32) / 9
    local_mean = cv2.filter2D(gray, -1, kernel)
    local_variance = cv2.filter2D(gray**2, -1, kernel) - local_mean**2
    
    features['texture_complexity'] = np.mean(local_variance)
    features['texture_max'] = np.max(local_variance)
    features['texture_std'] = np.std(local_variance)
    
    # GLCM纹理特征
    def compute_glcm_features(image, distance=1):
        quantized = (image * 15).astype(np.uint8)
        h, w = quantized.shape
        
        glcm = np.zeros((16, 16))
        for i in range(h):
            for j in range(w - distance):
                glcm[quantized[i, j], quantized[i, j + distance]] += 1
        
        glcm = glcm / np.sum(glcm) if np.sum(glcm) > 0 else glcm
        
        contrast = 0
        homogeneity = 0
        for i in range(16):
            for j in range(16):
                contrast += glcm[i, j] * (i - j)**2
                homogeneity += glcm[i, j] / (1 + abs(i - j))
        
        return contrast, homogeneity
    
    contrast, homogeneity = compute_glcm_features(gray)
    features['glcm_contrast'] = contrast
    features['glcm_homogeneity'] = homogeneity
    
    # 5. 颜色特征
    for i, channel in enumerate(['r', 'g', 'b']):
        channel_data = tile[:, :, i]
        features[f'{channel}_mean'] = np.mean(channel_data)
        features[f'{channel}_std'] = np.std(channel_data)
        features[f'{channel}_max'] = np.max(channel_data)
        features[f'{channel}_min'] = np.min(channel_data)
    
    # 饱和度特征
    hsv = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    features['saturation_mean'] = np.mean(saturation)
    features['saturation_std'] = np.std(saturation)
    
    features['color_variance'] = np.var(tile.reshape(-1, 3), axis=0).mean()
    
    # 6. 失真检测特征
    gray_uint8 = (gray * 255).astype(np.uint8)
    laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
    features['laplacian_variance'] = np.var(laplacian)
    features['sharpness'] = np.var(laplacian)
    
    # 块效应检测
    def detect_blocking_artifacts(image):
        h, w = image.shape
        block_size = 8
        
        h_diff = 0
        count_h = 0
        for i in range(block_size, h, block_size):
            if i < h:
                diff = np.mean(np.abs(image[i-1, :] - image[i, :]))
                h_diff += diff
                count_h += 1
        
        v_diff = 0
        count_v = 0
        for j in range(block_size, w, block_size):
            if j < w:
                diff = np.mean(np.abs(image[:, j-1] - image[:, j]))
                v_diff += diff
                count_v += 1
        
        avg_h_diff = h_diff / count_h if count_h > 0 else 0
        avg_v_diff = v_diff / count_v if count_v > 0 else 0
        
        return (avg_h_diff + avg_v_diff) / 2
    
    features['blocking_artifacts'] = detect_blocking_artifacts(gray)
    
    # 振铃效应检测
    def detect_ringing_artifacts(image):
        kernel = np.array([[-1, -1, -1], 
                          [-1,  8, -1], 
                          [-1, -1, -1]])
        filtered = cv2.filter2D(image, -1, kernel)
        return np.std(filtered)
    
    features['ringing_artifacts'] = detect_ringing_artifacts(gray)
    
    # 7. 局部一致性特征
    grad_x = np.gradient(gray, axis=1)
    grad_y = np.gradient(gray, axis=0)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    features['gradient_variance'] = np.var(grad_magnitude)
    features['gradient_mean'] = np.mean(grad_magnitude)
    
    local_std_map = cv2.filter2D(gray**2, -1, kernel) - (cv2.filter2D(gray, -1, kernel))**2
    features['local_std_variance'] = np.var(local_std_map)
    
    return features

def ml_predict_image_quality(model, feature_names, image_path, tile_size=32):
    """使用机器学习模型预测图像质量"""
    img = np.array(Image.open(image_path))
    if img.max() > 1.0:
        img = img.astype(np.float32) / 255.0
    
    h, w = img.shape[:2]
    quality_scores = []
    tile_positions = []
    tiles = []
    
    print("提取tiles并预测质量...")
    
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            if i + tile_size <= h and j + tile_size <= w:
                tile = img[i:i+tile_size, j:j+tile_size]
                if tile.shape[:2] == (tile_size, tile_size):
                    # 提取特征
                    features = extract_ml_features(tile)
                    feature_vector = np.array([features[name] for name in feature_names]).reshape(1, -1)
                    
                    # 预测质量分数
                    score = model.predict(feature_vector)[0]
                    
                    quality_scores.append(score)
                    tile_positions.append((i//tile_size, j//tile_size))
                    tiles.append(tile)
    
    return np.array(quality_scores), tile_positions, tiles

def replace_worst_tiles_ml(sr_path, org_path, gt_path, model, feature_names, top_k=100):
    """
    使用机器学习模型识别最差tiles并替换
    """
    print(f"\n=== 使用机器学习模型替换最差的{top_k}个tiles ===")
    
    # 1. 加载图像
    sr_img = np.array(Image.open(sr_path))
    org_img = np.array(Image.open(org_path))
    gt_img = np.array(Image.open(gt_path))
    
    if sr_img.max() > 1.0:
        sr_img = sr_img.astype(np.float32) / 255.0
    if org_img.max() > 1.0:
        org_img = org_img.astype(np.float32) / 255.0
    if gt_img.max() > 1.0:
        gt_img = gt_img.astype(np.float32) / 255.0
    
    # 2. 使用机器学习模型预测质量
    quality_scores, positions, sr_tiles = ml_predict_image_quality(model, feature_names, sr_path)
    
    print(f"预测完成，总tiles数量: {len(quality_scores)}")
    print(f"质量分数范围: {quality_scores.min():.3f} - {quality_scores.max():.3f}")
    
    # 3. 找到最差的K个tiles
    worst_indices = np.argsort(quality_scores)[-top_k:]
    worst_positions = [positions[i] for i in worst_indices]
    worst_scores = quality_scores[worst_indices]
    
    print(f"最差{top_k}个tiles的质量分数范围: {worst_scores.min():.3f} - {worst_scores.max():.3f}")
    
    # 4. 计算原始PSNR
    def calculate_psnr(pred_img, gt_img):
        mse = np.mean((pred_img - gt_img) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(1.0 / np.sqrt(mse))
    
    original_sr_psnr = calculate_psnr(sr_img, gt_img)
    original_org_psnr = calculate_psnr(org_img, gt_img)
    
    print(f"\n原始PSNR:")
    print(f"  SR PSNR: {original_sr_psnr:.2f} dB")
    print(f"  ORG PSNR: {original_org_psnr:.2f} dB")
    print(f"  ORG比SR高: {original_org_psnr - original_sr_psnr:.2f} dB")
    
    # 5. 替换最差的tiles
    hybrid_img = sr_img.copy()
    tile_size = 32
    replaced_count = 0
    
    print(f"\n替换最差的{top_k}个tiles:")
    for i, pos in enumerate(worst_positions):
        row, col = pos
        row_start = row * tile_size
        col_start = col * tile_size
        row_end = row_start + tile_size
        col_end = col_start + tile_size
        
        if row_end <= org_img.shape[0] and col_end <= org_img.shape[1]:
            # 替换为ORG版本
            hybrid_img[row_start:row_end, col_start:col_end] = org_img[row_start:row_end, col_start:col_end]
            replaced_count += 1
            
            if i < 10:  # 显示前10个
                rank = len(worst_positions) - i
                score = worst_scores[-(i+1)]
                print(f"  Rank {rank}: Tile({row}, {col}) - 坐标[{row_start}:{row_end}, {col_start}:{col_end}] - 预测分数: {score:.3f}")
    
    print(f"实际替换的tiles数量: {replaced_count}")
    
    # 6. 计算混合图像PSNR
    hybrid_psnr = calculate_psnr(hybrid_img, gt_img)
    
    print(f"\n=== 机器学习模型替换结果 ===")
    print(f"原始SR PSNR: {original_sr_psnr:.2f} dB")
    print(f"混合图像PSNR: {hybrid_psnr:.2f} dB")
    print(f"PSNR提升: {hybrid_psnr - original_sr_psnr:.2f} dB")
    print(f"替换像素比例: {replaced_count * tile_size * tile_size / (sr_img.shape[0] * sr_img.shape[1]) * 100:.1f}%")
    print(f"效率 (dB提升/像素比例): {(hybrid_psnr - original_sr_psnr) / (replaced_count * tile_size * tile_size / (sr_img.shape[0] * sr_img.shape[1]) * 100):.2f}")
    
    # 7. 保存混合图像
    hybrid_path = f"../results/images/ml_hybrid_k{top_k}.png"
    hybrid_img_uint8 = (hybrid_img * 255).astype(np.uint8)
    Image.fromarray(hybrid_img_uint8).save(hybrid_path)
    print(f"混合图像保存到: {hybrid_path}")
    
    # 8. 计算真实PSNR损失进行验证
    tile_true_losses = []
    tile_predicted_scores = []
    
    for i, pos in enumerate(positions):
        row, col = pos
        row_start = row * tile_size
        col_start = col * tile_size
        row_end = row_start + tile_size
        col_end = col_start + tile_size
        
        if row_end <= gt_img.shape[0] and col_end <= gt_img.shape[1]:
            sr_tile = sr_img[row_start:row_end, col_start:col_end]
            gt_tile = gt_img[row_start:row_end, col_start:col_end]
            
            mse = np.mean((sr_tile - gt_tile) ** 2)
            if mse == 0:
                psnr_loss = 0
            else:
                psnr = 20 * np.log10(1.0 / np.sqrt(mse))
                psnr_loss = max(0, 50 - psnr)  # 假设完美PSNR为50dB
            
            tile_true_losses.append(psnr_loss)
            tile_predicted_scores.append(quality_scores[i])
    
    # 验证预测准确性
    tile_true_losses = np.array(tile_true_losses)
    tile_predicted_scores = np.array(tile_predicted_scores)
    
    correlation = np.corrcoef(tile_true_losses, tile_predicted_scores)[0, 1]
    
    # 计算真实最差K个tiles的重叠率
    true_worst_indices = np.argsort(tile_true_losses)[-top_k:]
    pred_worst_indices = np.argsort(tile_predicted_scores)[-top_k:]
    overlap = len(set(true_worst_indices) & set(pred_worst_indices))
    overlap_rate = overlap / top_k
    
    print(f"\n=== 预测验证 ===")
    print(f"预测与真实损失相关性: {correlation:.4f}")
    print(f"Top {top_k} 重叠率: {overlap_rate:.3f} ({overlap_rate*100:.1f}%)")
    
    return {
        'hybrid_img': hybrid_img,
        'hybrid_psnr': hybrid_psnr,
        'original_sr_psnr': original_sr_psnr,
        'psnr_improvement': hybrid_psnr - original_sr_psnr,
        'replaced_count': replaced_count,
        'correlation': correlation,
        'overlap_rate': overlap_rate,
        'worst_positions': worst_positions,
        'worst_scores': worst_scores
    }

def compare_ml_vs_previous_method():
    """对比机器学习方法与之前方法的效果"""
    print("\n" + "="*60)
    print("=== 机器学习方法 vs 真实PSNR方法对比 ===")
    print("="*60)
    
    # 文件路径
    sr_path = "../data/no2_pred_sr.png"
    org_path = "../data/no2_pred_org.png"
    gt_path = "../data/no2_gt.png"
    
    # 加载训练好的模型
    model, feature_names = load_trained_model()
    if model is None:
        return
    
    # 测试不同的K值
    k_values = [10, 30, 50, 100]
    
    results = {}
    
    for k in k_values:
        print(f"\n{'='*40}")
        print(f"测试 K = {k}")
        print(f"{'='*40}")
        
        result = replace_worst_tiles_ml(sr_path, org_path, gt_path, model, feature_names, top_k=k)
        results[k] = result
    
    # 汇总结果
    print(f"\n" + "="*80)
    print(f"=== 机器学习方法性能汇总 ===")
    print(f"="*80)
    
    print(f"{'K值':<5} {'PSNR提升':<10} {'像素比例':<10} {'效率':<8} {'重叠率':<8} {'相关性':<8}")
    print("-" * 60)
    
    for k in k_values:
        result = results[k]
        pixel_ratio = result['replaced_count'] * 32 * 32 / (768 * 1024) * 100  # 假设图像尺寸
        efficiency = result['psnr_improvement'] / pixel_ratio if pixel_ratio > 0 else 0
        
        print(f"{k:<5} {result['psnr_improvement']:<10.2f} {pixel_ratio:<10.1f}% {efficiency:<8.2f} {result['overlap_rate']:<8.3f} {result['correlation']:<8.4f}")
    
    # 读取之前真实PSNR方法的结果进行对比
    if os.path.exists('hybrid_performance_report.txt'):
        print(f"\n=== 与真实PSNR方法对比 ===")
        print("(读取之前analyze_psnr.py的结果)")
        
        try:
            with open('../results/reports/hybrid_performance_report.txt', 'r') as f:
                content = f.read()
                print("之前真实PSNR方法结果:")
                for line in content.split('\n'):
                    if 'PSNR' in line or '提升' in line or '比例' in line:
                        print(f"  {line}")
        except:
            print("无法读取之前的结果文件")
    
    return results

def visualize_ml_prediction_results(model, feature_names):
    """可视化机器学习预测结果"""
    sr_path = "../data/no2_pred_sr.png"
    
    # 预测质量
    quality_scores, positions, tiles = ml_predict_image_quality(model, feature_names, sr_path)
    
    # 重建热力图
    max_row = max(pos[0] for pos in positions) + 1
    max_col = max(pos[1] for pos in positions) + 1
    
    heatmap = np.zeros((max_row, max_col))
    for i, (row, col) in enumerate(positions):
        heatmap[row, col] = quality_scores[i]
    
    # 可视化
    plt.figure(figsize=(15, 6))
    
    # 原图
    plt.subplot(1, 2, 1)
    img = Image.open(sr_path)
    plt.imshow(img)
    plt.title('SR Image')
    plt.axis('off')
    
    # 机器学习预测的质量热力图
    plt.subplot(1, 2, 2)
    im = plt.imshow(heatmap, cmap='viridis')
    plt.title('ML Predicted Quality Map\n(Yellow = Poor Quality)')
    plt.xlabel('Tile Column')
    plt.ylabel('Tile Row')
    plt.colorbar(im, label='Predicted PSNR Loss (dB)')
    
    plt.tight_layout()
    plt.savefig('../results/images/ml_quality_prediction_map.png', dpi=300, bbox_inches='tight')
    print(f"机器学习质量预测图保存到: ../results/images/ml_quality_prediction_map.png")
    plt.show()

def main():
    print("=== 机器学习SR质量预测和Tile替换测试 ===")
    
    # 1. 加载训练好的模型
    model, feature_names = load_trained_model()
    if model is None:
        print("请先运行 sr_quality_predictor.py 训练模型")
        return
    
    # 2. 可视化预测结果
    print("\n1. 生成机器学习质量预测图...")
    visualize_ml_prediction_results(model, feature_names)
    
    # 3. 使用机器学习方法进行tile替换测试
    print("\n2. 使用机器学习方法进行tile替换测试...")
    results = compare_ml_vs_previous_method()
    
    # 4. 生成详细报告
    print("\n3. 生成详细报告...")
    with open('../results/reports/ml_tile_replacement_report.txt', 'w') as f:
        f.write("=== 机器学习SR质量预测和Tile替换报告 ===\n\n")
        f.write("方法说明:\n")
        f.write("- 使用随机森林回归模型预测SR图像中每个tile的质量损失\n")
        f.write("- 模型基于43个特征进行训练，包括亮度、边缘、纹理、频域等\n")
        f.write("- 识别质量最差的K个tiles，用ORG方法替换\n\n")
        
        f.write("测试结果:\n")
        f.write(f"{'K值':<5} {'PSNR提升':<10} {'像素比例':<10} {'效率':<8} {'重叠率':<8}\n")
        f.write("-" * 50 + "\n")
        
        if results:
            for k, result in results.items():
                pixel_ratio = result['replaced_count'] * 32 * 32 / (768 * 1024) * 100
                efficiency = result['psnr_improvement'] / pixel_ratio if pixel_ratio > 0 else 0
                f.write(f"{k:<5} {result['psnr_improvement']:<10.2f} {pixel_ratio:<10.1f}% {efficiency:<8.2f} {result['overlap_rate']:<8.3f}\n")
    
    print("详细报告保存到: ml_tile_replacement_report.txt")

if __name__ == "__main__":
    main()

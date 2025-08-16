#!/usr/bin/env python3
"""
SR质量预测器 - 仅基于SR图像预测tile质量
基于之前的特征分析，设计无参考质量评估方法
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy import stats
from scipy.fftpack import dct
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

def extract_sr_only_features(tile):
    """
    仅基于SR tile提取质量相关特征
    根据之前分析的失败模式设计特征
    """
    # 确保tile是浮点数格式 [0,1]
    if tile.max() > 1.0:
        tile = tile.astype(np.float32) / 255.0
    
    features = {}
    
    # =============================================================================
    # 1. 亮度相关特征（最重要的失败指标）
    # =============================================================================
    gray = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    # 基础亮度统计
    features['brightness_mean'] = np.mean(gray)
    features['brightness_std'] = np.std(gray)
    features['brightness_max'] = np.max(gray)
    features['brightness_min'] = np.min(gray)
    features['brightness_range'] = features['brightness_max'] - features['brightness_min']
    
    # 高亮度像素比例（SR容易在亮区域失败）
    features['bright_pixel_ratio'] = np.sum(gray > 0.7) / gray.size
    features['very_bright_pixel_ratio'] = np.sum(gray > 0.85) / gray.size
    
    # 亮度分布特征
    hist, _ = np.histogram(gray, bins=10, range=(0, 1))
    hist = hist / np.sum(hist)  # 归一化
    features['brightness_skewness'] = stats.skew(gray.flatten())
    features['brightness_kurtosis'] = stats.kurtosis(gray.flatten())
    
    # =============================================================================
    # 2. 边缘和细节特征（第二重要）
    # =============================================================================
    # Sobel边缘检测
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    features['edge_density'] = np.mean(edge_magnitude)
    features['edge_max'] = np.max(edge_magnitude)
    features['edge_std'] = np.std(edge_magnitude)
    
    # 强边缘像素比例
    features['strong_edge_ratio'] = np.sum(edge_magnitude > 0.1) / edge_magnitude.size
    
    # Canny边缘检测
    gray_uint8 = (gray * 255).astype(np.uint8)
    canny = cv2.Canny(gray_uint8, 50, 150)
    features['canny_edge_ratio'] = np.sum(canny > 0) / canny.size
    
    # =============================================================================
    # 3. 频域特征（高频成分）
    # =============================================================================
    # DCT频域分析
    dct_coeff = dct(dct(gray.T).T)
    h, w = dct_coeff.shape
    
    # 高频区域（右下角）
    high_freq_region = dct_coeff[h//2:, w//2:]
    features['high_freq_energy'] = np.sum(high_freq_region**2) / np.sum(dct_coeff**2)
    
    # FFT频域分析
    fft = np.fft.fft2(gray)
    fft_magnitude = np.abs(fft)
    
    # 计算高频比例
    h, w = fft_magnitude.shape
    center_h, center_w = h // 2, w // 2
    radius = min(h, w) // 4
    
    # 创建高频掩码
    y, x = np.ogrid[:h, :w]
    mask = (x - center_w)**2 + (y - center_h)**2 > radius**2
    
    high_freq_energy = np.sum(fft_magnitude[mask])
    total_energy = np.sum(fft_magnitude)
    features['fft_high_freq_ratio'] = high_freq_energy / total_energy if total_energy > 0 else 0
    
    # =============================================================================
    # 4. 纹理复杂度特征
    # =============================================================================
    # 局部方差（纹理复杂度）
    kernel = np.ones((3, 3), np.float32) / 9
    local_mean = cv2.filter2D(gray, -1, kernel)
    local_variance = cv2.filter2D(gray**2, -1, kernel) - local_mean**2
    
    features['texture_complexity'] = np.mean(local_variance)
    features['texture_max'] = np.max(local_variance)
    features['texture_std'] = np.std(local_variance)
    
    # GLCM纹理特征（简化版）
    def compute_glcm_features(image, distance=1):
        """计算灰度共生矩阵特征"""
        # 量化到16级灰度
        quantized = (image * 15).astype(np.uint8)
        h, w = quantized.shape
        
        # 计算水平方向的GLCM
        glcm = np.zeros((16, 16))
        for i in range(h):
            for j in range(w - distance):
                glcm[quantized[i, j], quantized[i, j + distance]] += 1
        
        # 归一化
        glcm = glcm / np.sum(glcm) if np.sum(glcm) > 0 else glcm
        
        # 计算对比度和同质性
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
    
    # =============================================================================
    # 5. 颜色特征
    # =============================================================================
    # RGB通道统计
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
    
    # 颜色变异性
    features['color_variance'] = np.var(tile.reshape(-1, 3), axis=0).mean()
    
    # =============================================================================
    # 6. 失真检测特征（基于SR常见失误）
    # =============================================================================
    # 过平滑检测（SR容易过度平滑）
    gray_uint8 = (gray * 255).astype(np.uint8)
    laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
    features['laplacian_variance'] = np.var(laplacian)
    features['sharpness'] = np.var(laplacian)  # 清晰度指标
    
    # 块效应检测（SR可能产生块效应）
    def detect_blocking_artifacts(image):
        """检测块效应"""
        h, w = image.shape
        block_size = 8
        
        # 水平块边界
        h_diff = 0
        count_h = 0
        for i in range(block_size, h, block_size):
            if i < h:
                diff = np.mean(np.abs(image[i-1, :] - image[i, :]))
                h_diff += diff
                count_h += 1
        
        # 垂直块边界
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
        """检测振铃效应"""
        # 使用高通滤波器
        kernel = np.array([[-1, -1, -1], 
                          [-1,  8, -1], 
                          [-1, -1, -1]])
        filtered = cv2.filter2D(image, -1, kernel)
        return np.std(filtered)
    
    features['ringing_artifacts'] = detect_ringing_artifacts(gray)
    
    # =============================================================================
    # 7. 局部一致性特征
    # =============================================================================
    # 梯度一致性
    grad_x = np.gradient(gray, axis=1)
    grad_y = np.gradient(gray, axis=0)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    features['gradient_variance'] = np.var(grad_magnitude)
    features['gradient_mean'] = np.mean(grad_magnitude)
    
    # 局部标准差的变化
    local_std_map = cv2.filter2D(gray**2, -1, kernel) - (cv2.filter2D(gray, -1, kernel))**2
    features['local_std_variance'] = np.var(local_std_map)
    
    return features

def load_sr_image_and_extract_tiles(sr_path, tile_size=32):
    """加载SR图像并提取tiles"""
    sr_img = np.array(Image.open(sr_path))
    if sr_img.max() > 1.0:
        sr_img = sr_img.astype(np.float32) / 255.0
    
    h, w = sr_img.shape[:2]
    tiles = []
    positions = []
    
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            if i + tile_size <= h and j + tile_size <= w:
                tile = sr_img[i:i+tile_size, j:j+tile_size]
                if tile.shape[:2] == (tile_size, tile_size):
                    tiles.append(tile)
                    positions.append((i//tile_size, j//tile_size))
    
    return tiles, positions

def build_quality_predictor(sr_tiles, true_psnr_losses):
    """
    构建质量预测模型
    
    Args:
        sr_tiles: SR tiles列表
        true_psnr_losses: 真实的PSNR损失（用于训练）
    """
    print("提取SR特征...")
    features_list = []
    
    for i, tile in enumerate(sr_tiles):
        if i % 100 == 0:
            print(f"  已处理 {i}/{len(sr_tiles)} 个tiles")
        
        features = extract_sr_only_features(tile)
        features_list.append(list(features.values()))
    
    # 转换为numpy数组
    X = np.array(features_list)
    y = np.array(true_psnr_losses)
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"目标向量形状: {y.shape}")
    
    # 划分训练和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练随机森林模型
    print("训练质量预测模型...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # 预测和评估
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n=== 模型性能 ===")
    print(f"训练集 MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
    print(f"测试集 MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
    
    # 特征重要性分析
    feature_names = list(extract_sr_only_features(sr_tiles[0]).keys())
    feature_importance = rf_model.feature_importances_
    
    # 排序特征重要性
    importance_indices = np.argsort(feature_importance)[::-1]
    
    print(f"\n=== 特征重要性 Top 15 ===")
    for i in range(min(15, len(feature_names))):
        idx = importance_indices[i]
        print(f"{feature_names[idx]:25}: {feature_importance[idx]:.4f}")
    
    return rf_model, feature_names, (X_test, y_test, y_pred_test)

def test_quality_prediction(model, feature_names, sr_tiles, true_psnr_losses, top_k=100):
    """
    测试质量预测的准确性
    重点关注是否能正确识别最差的K个tiles
    """
    print(f"\n=== 测试质量预测准确性 ===")
    
    # 提取所有SR tiles的特征
    print("提取所有tiles特征...")
    features_list = []
    for i, tile in enumerate(sr_tiles):
        if i % 100 == 0:
            print(f"  已处理 {i}/{len(sr_tiles)} 个tiles")
        features = extract_sr_only_features(tile)
        features_list.append(list(features.values()))
    
    X = np.array(features_list)
    
    # 预测质量
    print("预测tiles质量...")
    predicted_losses = model.predict(X)
    
    # 获取真实最差的K个tiles
    true_worst_indices = np.argsort(true_psnr_losses)[-top_k:]
    
    # 获取预测最差的K个tiles
    pred_worst_indices = np.argsort(predicted_losses)[-top_k:]
    
    # 计算重叠率
    overlap = len(set(true_worst_indices) & set(pred_worst_indices))
    overlap_rate = overlap / top_k
    
    print(f"\n=== Top {top_k} 最差tiles预测结果 ===")
    print(f"真实最差tiles数量: {len(true_worst_indices)}")
    print(f"预测最差tiles数量: {len(pred_worst_indices)}")
    print(f"重叠数量: {overlap}")
    print(f"重叠率: {overlap_rate:.3f} ({overlap_rate*100:.1f}%)")
    
    # 分析预测和真实的相关性
    correlation = np.corrcoef(true_psnr_losses, predicted_losses)[0, 1]
    print(f"整体PSNR损失相关性: {correlation:.4f}")
    
    # 计算不同阈值下的精确率和召回率
    print(f"\n=== 不同K值下的预测准确性 ===")
    k_values = [10, 30, 50, 100, 200, 300]
    
    results = {}
    for k in k_values:
        if k <= len(sr_tiles):
            true_worst_k = set(np.argsort(true_psnr_losses)[-k:])
            pred_worst_k = set(np.argsort(predicted_losses)[-k:])
            
            overlap_k = len(true_worst_k & pred_worst_k)
            precision = overlap_k / k
            recall = overlap_k / k  # 因为真实和预测的集合大小相同
            
            results[k] = {
                'overlap': overlap_k,
                'precision': precision,
                'recall': recall
            }
            
            print(f"K={k:3d}: 重叠={overlap_k:3d}, 精确率={precision:.3f}, 召回率={recall:.3f}")
    
    return predicted_losses, results

def visualize_prediction_results(sr_tiles, positions, true_losses, predicted_losses, 
                                tile_size=32, save_path="../results/images/quality_prediction_comparison.png"):
    """可视化预测结果"""
    # 重建损失热力图
    max_row = max(pos[0] for pos in positions) + 1
    max_col = max(pos[1] for pos in positions) + 1
    
    true_heatmap = np.zeros((max_row, max_col))
    pred_heatmap = np.zeros((max_row, max_col))
    
    for i, (row, col) in enumerate(positions):
        true_heatmap[row, col] = true_losses[i]
        pred_heatmap[row, col] = predicted_losses[i]
    
    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 真实PSNR损失
    im1 = axes[0].imshow(true_heatmap, cmap='viridis')
    axes[0].set_title('True PSNR Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Tile Column')
    axes[0].set_ylabel('Tile Row')
    plt.colorbar(im1, ax=axes[0])
    
    # 预测PSNR损失
    im2 = axes[1].imshow(pred_heatmap, cmap='viridis')
    axes[1].set_title('Predicted PSNR Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Tile Column')
    axes[1].set_ylabel('Tile Row')
    plt.colorbar(im2, ax=axes[1])
    
    # 散点图对比
    axes[2].scatter(true_losses, predicted_losses, alpha=0.6, s=20)
    axes[2].plot([min(true_losses), max(true_losses)], 
                [min(true_losses), max(true_losses)], 'r--', alpha=0.8)
    axes[2].set_xlabel('True PSNR Loss (dB)')
    axes[2].set_ylabel('Predicted PSNR Loss (dB)')
    axes[2].set_title('True vs Predicted PSNR Loss')
    axes[2].grid(True, alpha=0.3)
    
    # 添加相关系数
    correlation = np.corrcoef(true_losses, predicted_losses)[0, 1]
    axes[2].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=axes[2].transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"预测结果对比图保存到: {save_path}")
    plt.show()

def main():
    # 文件路径
    sr_path = "../data/no2_pred_sr.png"
    gt_path = "../data/no2_gt.png"
    
    if not os.path.exists(sr_path) or not os.path.exists(gt_path):
        print(f"错误：找不到{sr_path}或{gt_path}文件")
        print("请确保在正确的目录下运行此脚本")
        return
    
    print("=== SR质量预测器测试 ===")
    
    # 1. 加载图像并提取tiles
    print("加载SR图像并提取tiles...")
    sr_tiles, positions = load_sr_image_and_extract_tiles(sr_path)
    print(f"提取到 {len(sr_tiles)} 个tiles")
    
    # 2. 计算真实PSNR损失（仅用于验证模型）
    print("计算真实PSNR损失（用于验证）...")
    gt_img = np.array(Image.open(gt_path))
    if gt_img.max() > 1.0:
        gt_img = gt_img.astype(np.float32) / 255.0
    
    gt_tiles, _ = load_sr_image_and_extract_tiles(gt_path)
    
    # 计算每个tile的PSNR损失
    true_psnr_losses = []
    for sr_tile, gt_tile in zip(sr_tiles, gt_tiles):
        mse = np.mean((sr_tile - gt_tile) ** 2)
        if mse == 0:
            psnr_loss = 0
        else:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            # 假设完美PSNR为50dB，计算损失
            psnr_loss = max(0, 50 - psnr)
        true_psnr_losses.append(psnr_loss)
    
    true_psnr_losses = np.array(true_psnr_losses)
    print(f"PSNR损失范围: {true_psnr_losses.min():.2f} - {true_psnr_losses.max():.2f} dB")
    
    # 3. 构建和训练质量预测模型
    model, feature_names, test_data = build_quality_predictor(sr_tiles, true_psnr_losses)
    
    # 4. 测试预测准确性
    predicted_losses, results = test_quality_prediction(model, feature_names, sr_tiles, true_psnr_losses)
    
    # 5. 可视化结果
    visualize_prediction_results(sr_tiles, positions, true_psnr_losses, predicted_losses)
    
    # 6. 保存模型
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'results': results
    }
    
    with open('../models/sr_quality_predictor.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("模型保存到: ../models/sr_quality_predictor.pkl")
    
    # 7. 生成预测报告
    with open('../results/reports/quality_prediction_report.txt', 'w') as f:
        f.write("=== SR质量预测报告 ===\n\n")
        f.write(f"总tiles数量: {len(sr_tiles)}\n")
        f.write(f"真实PSNR损失范围: {true_psnr_losses.min():.2f} - {true_psnr_losses.max():.2f} dB\n")
        f.write(f"预测PSNR损失范围: {predicted_losses.min():.2f} - {predicted_losses.max():.2f} dB\n")
        f.write(f"整体相关性: {np.corrcoef(true_psnr_losses, predicted_losses)[0, 1]:.4f}\n\n")
        
        f.write("不同K值下的预测准确性:\n")
        for k, result in results.items():
            f.write(f"K={k}: 重叠={result['overlap']}, 精确率={result['precision']:.3f}\n")
    
    print("预测报告保存到: quality_prediction_report.txt")

if __name__ == "__main__":
    main()

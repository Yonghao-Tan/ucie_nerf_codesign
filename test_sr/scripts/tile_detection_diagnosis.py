#!/usr/bin/env python3
"""
Tile检测方法诊断工具
分析损失最大的tiles，看看各个方法是否能正确识别出这些需要替换的tiles

使用方法:
python tile_detection_diagnosis.py
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path

# 添加当前脚本目录到路径
sys.path.append('/home/ytanaz/access/IBRNet/test_sr/scripts')

# 设置matplotlib使用英文，避免中文字体问题
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_image(img_path):
    """加载图片并转换为numpy数组，返回 (H, W, 3) 的float32数组，值范围[0, 1]"""
    img = Image.open(img_path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    return img

def calculate_psnr(pred_img, gt_img):
    """计算PSNR"""
    if pred_img.shape != gt_img.shape:
        raise ValueError(f"图片尺寸不匹配: {pred_img.shape} vs {gt_img.shape}")
    
    mse = np.mean((pred_img - gt_img) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_tile_wise_psnr(pred_img, gt_img, tile_size=32):
    """计算每个tile的PSNR"""
    H, W, C = pred_img.shape
    
    h_tiles = (H + tile_size - 1) // tile_size
    w_tiles = (W + tile_size - 1) // tile_size
    
    tile_psnr_map = np.zeros((h_tiles, w_tiles))
    tile_coords = []
    
    for i in range(h_tiles):
        for j in range(w_tiles):
            h_start = i * tile_size
            h_end = min(h_start + tile_size, H)
            w_start = j * tile_size
            w_end = min(w_start + tile_size, W)
            
            pred_tile = pred_img[h_start:h_end, w_start:w_end, :]
            gt_tile = gt_img[h_start:h_end, w_start:w_end, :]
            
            tile_psnr = calculate_psnr(pred_tile, gt_tile)
            tile_psnr_map[i, j] = tile_psnr
            
            tile_coords.append({
                'tile_id': (i, j),
                'coords': (h_start, h_end, w_start, w_end),
                'psnr': tile_psnr
            })
    
    return tile_psnr_map, tile_coords

def extract_canny_high_res(fine_img, threshold=0.160):
    """Canny高分辨率方法 - 在fine分辨率图像上提取边缘"""
    fine_gray = cv2.cvtColor((fine_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(fine_gray, 50, 150)
    edge_density = edges.sum() / (edges.size * 255)
    return edge_density

def extract_canny_low_res(sr_img, threshold=0.250):
    """Canny低分辨率方法 - 在SR分辨率图像上提取边缘"""
    sr_gray = cv2.cvtColor((sr_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(sr_gray, 50, 150)
    edge_density = edges.sum() / (edges.size * 255)
    return edge_density

def extract_volume_rendering_confidence(render_weights_path):
    """体渲染置信度方法"""
    weights = torch.load(render_weights_path, map_location='cpu')
    
    dcm_score, need_full_render = density_complexity_metric(weights.numpy(), 
                    sigma_threshold=0.5,
                    grad_threshold=0.3,
                    curvature_threshold=0.2)
    if len(weights.shape) == 3:
    #     H, W, n_samples = weights.shape
    #     # 计算每个像素的渲染置信度 - 修复NaN问题
    #     weights_safe = torch.clamp(weights, min=1e-8)  # 避免log(0)
    #     confidence_map = torch.sum(weights_safe * torch.log(weights_safe + 1e-8), dim=-1)
    #     confidence_map = -confidence_map  # 熵越小，置信度越高
        
    #     # 安全归一化
    #     conf_min = confidence_map.min()
    #     conf_max = confidence_map.max()
    #     if conf_max > conf_min:
    #         confidence_map = (confidence_map - conf_min) / (conf_max - conf_min + 1e-8)
    #     else:
    #         confidence_map = torch.zeros_like(confidence_map)
        
    #     return confidence_map.numpy()
        print(dcm_score)
        return dcm_score
    else:
        print(f"警告：体渲染权重维度不匹配：{weights.shape}")
        return None

def extract_depth_discontinuity_score(depth_map_path):
    """深度不连续性方法"""
    try:
        depth_map = torch.load(depth_map_path, map_location='cpu')
        if len(depth_map.shape) == 2:
            # 安全归一化深度图
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            if depth_max > depth_min:
                depth_normalized = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
            else:
                depth_normalized = torch.zeros_like(depth_map)
            
            # 计算深度梯度
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sobel_y = sobel_x.T
            
            grad_x = F.conv2d(depth_normalized[None, None], sobel_x[None, None], padding=1)[0, 0]
            grad_y = F.conv2d(depth_normalized[None, None], sobel_y[None, None], padding=1)[0, 0]
            
            gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
            return gradient_magnitude.numpy()
        else:
            print(f"警告：深度图维度不匹配：{depth_map.shape}")
            return None
    except Exception as e:
        print(f"警告：无法加载深度图：{e}")
        return None

def density_complexity_metric(tile_sigma, 
                             sigma_threshold=0.5,
                             grad_threshold=0.3,
                             curvature_threshold=0.2):
    """
    纯密度场复杂度指标（Density Complexity Metric, DCM）
    
    参数:
        tile_sigma: [tile_size, tile_size, num_samples] 
                   密度场采样值 (sigma)，沿光线的采样点
        sigma_threshold: 密度阈值 (默认0.5)，用于筛选表面区域
        grad_threshold: 梯度阈值 (默认0.3)，用于筛选高变化区域
        curvature_threshold: 曲率阈值 (默认0.2)
    
    返回:
        dcm_score: 密度复杂度得分 (0~1)
        need_full_render: 是否需要完整NeRF渲染 (True/False)
    """
    # 1. 提取表面区域（高密度区域）
    surface_mask = (tile_sigma > sigma_threshold)  # [H, W, N]
    
    # 2. 计算密度梯度（沿光线方向）
    d_sigma = np.diff(tile_sigma, axis=-1)  # [H, W, N-1]
    grad_mag = np.abs(d_sigma)  # 梯度幅值
    
    # 3. 计算表面梯度强度（仅考虑表面区域）
    surface_grad = grad_mag * surface_mask[..., :-1]  # [H, W, N-1]
    avg_surface_grad = np.mean(surface_grad)
    
    # 4. 计算表面曲率（二阶导数）
    d2_sigma = np.diff(d_sigma, axis=-1)  # [H, W, N-2]
    curvature = np.abs(d2_sigma) * surface_mask[..., 2:]  # 仅表面区域
    avg_curvature = np.mean(curvature)
    
    # 5. 计算多层表面指标（防止薄物体漏检）
    layer_count = np.sum(surface_mask, axis=-1)  # 每像素的表面层数
    multi_layer_ratio = np.mean(layer_count >= 2)  # 多层表面占比
    
    # 6. 融合指标 → DCM得分
    dcm_score = (
        0.5 * np.clip(avg_surface_grad / grad_threshold, 0, 1) +
        0.3 * np.clip(avg_curvature / curvature_threshold, 0, 1) +
        0.2 * multi_layer_ratio
    )
    
    # 7. 决策（阈值可调）
    need_full_render = (dcm_score > 0.6)
    
    return dcm_score, need_full_render

def compute_gacm(tile_color, tile_depth, alpha=0.4, beta=0.3, gamma=0.3, tau_D_ratio=0.1, threshold=0.25):
    """
    计算几何感知复杂度指标 (GACM)
    
    参数:
        tile_color: [32, 32, 3] 降采样颜色图像 (torch.Tensor, 0~1)
        tile_depth: [32, 32] 降采样深度图 (torch.Tensor)
        alpha, beta, gamma: DD, SC, GAE 的权重 (默认 0.4, 0.3, 0.3)
        tau_D_ratio: 深度梯度阈值比例 (默认 0.1 → tau_D = 0.1 * max(G_D))
        threshold: 最终决策阈值 (默认 0.25)
    
    返回:
        gacm_score: 几何复杂度得分 (0~1)
        decision: bool (True=需完整NeRF, False=可超分)
    """
    # === Step 0: 深度图预处理 (关键！避免噪声干扰) ===
    depth = tile_depth.clone()
    
    # 检查深度图是否为空或无效
    if depth.numel() == 0:
        return 0.0, False
    
    # 归一化深度到 [0,1] (适配不同场景尺度)
    depth_min, depth_max = depth.min(), depth.max()
    if depth_max > depth_min:
        depth = (depth - depth_min) / (depth_max - depth_min + 1e-5)
    else:
        depth = torch.zeros_like(depth)
    
    # === Step 1: 计算深度驱动的几何复杂度 ===
    # 1.1 深度不连续性 (Depth Discontinuity, DD)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=depth.device)
    sobel_y = sobel_x.T
    
    if depth.dim() == 2 and depth.shape[0] > 2 and depth.shape[1] > 2:
        grad_x = torch.nn.functional.conv2d(depth[None, None], sobel_x[None, None], padding=1)[0,0]
        grad_y = torch.nn.functional.conv2d(depth[None, None], sobel_y[None, None], padding=1)[0,0]
        G_D = torch.sqrt(grad_x**2 + grad_y**2 + 1e-5)  # 避免除零
        DD_tile = G_D.mean().item()
    else:
        DD_tile = 0.0
        G_D = torch.zeros_like(depth)
    
    # 1.2 表面曲率 (Surface Curvature, SC)
    if depth.dim() == 2 and depth.shape[0] > 2 and depth.shape[1] > 2:
        laplacian = torch.nn.functional.conv2d(
            depth[None, None], 
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=depth.device, dtype=torch.float32)[None, None],
            padding=1
        )[0,0]
        SC_tile = torch.abs(laplacian).mean().item()
    else:
        SC_tile = 0.0
    
    # === Step 2: 计算几何对齐边缘 (Geometry-Aligned Edges, GAE) ===
    try:
        # 2.1 用Canny检测颜色边缘 (CPU操作，因OpenCV不支持GPU)
        color_np = (tile_color.cpu().numpy() * 255).astype('uint8')
        if color_np.shape[0] > 0 and color_np.shape[1] > 0:
            gray = cv2.cvtColor(color_np, cv2.COLOR_RGB2GRAY)
            edges_canny = cv2.Canny(gray, 100, 200)  # 标准Canny参数
            
            # 2.2 仅保留与深度不连续对齐的边缘 (核心创新!)
            if G_D.numel() > 0:
                tau_D = tau_D_ratio * G_D.max().item()  # 深度梯度阈值
                G_D_np = G_D.cpu().numpy()
                
                # 确保尺寸匹配
                if edges_canny.shape == G_D_np.shape:
                    GAE = edges_canny * (G_D_np > tau_D)  # 关键：过滤非几何边缘
                    GAE_density = GAE.sum() / (edges_canny.shape[0] * edges_canny.shape[1] * 255)  # 归一化到 [0,1]
                else:
                    GAE_density = 0.0
            else:
                GAE_density = 0.0
        else:
            GAE_density = 0.0
    except Exception as e:
        GAE_density = 0.0
    
    # === Step 3: 融合指标 → GACM ===
    gacm_score = alpha * DD_tile + beta * SC_tile + gamma * GAE_density
    
    # === Step 4: 决策 ===
    decision = (gacm_score > threshold)
    
    return gacm_score, decision

def diagnose_single_image(scene_name, image_idx=0):
    """诊断单张图片的tile检测方法"""
    print(f"\n{'='*80}")
    print(f"诊断场景: {scene_name}, 图片: {image_idx}")
    print(f"{'='*80}")
    
    # 图片路径 - trex场景
    base_path = "/home/ytanaz/access/IBRNet/eval/llff_test"
    gt_path = f"{base_path}/eval_llff_sr/{scene_name}/{image_idx}_gt_rgb_hr.png"
    fine_path = f"{base_path}/eval_llff/{scene_name}/{image_idx}_pred_fine.png"
    sr_path = f"{base_path}/eval_llff_sr/{scene_name}/{image_idx}_pred_sr.png"
    
    # 渲染数据路径 - 使用数字索引
    render_info_path = "/home/ytanaz/access/IBRNet/memory/render_infos"
    weights_path = f"{render_info_path}/render_weights_{image_idx}_n48.pt"
    depth_path = f"{render_info_path}/render_depth_map_{image_idx}_n48.pt"
    
    # 检查文件存在性
    for path, name in [(gt_path, "GT"), (fine_path, "Fine"), (sr_path, "SR")]:
        if not os.path.exists(path):
            print(f"❌ 缺少文件: {name} - {path}")
            return None
    
    print(f"✅ 找到所有必要文件")
    
    # 加载图片
    gt_img = load_image(gt_path)
    fine_img = load_image(fine_path)
    sr_img = load_image(sr_path)
    
    print(f"图片尺寸: GT={gt_img.shape}, Fine={fine_img.shape}, SR={sr_img.shape}")
    
    # 计算tile-wise损失
    fine_tile_psnr, fine_tile_coords = calculate_tile_wise_psnr(fine_img, gt_img, tile_size=32)
    sr_tile_psnr, sr_tile_coords = calculate_tile_wise_psnr(sr_img, gt_img, tile_size=32)
    
    # 计算每个tile的损失 (Fine PSNR - SR PSNR)
    tile_loss_info = []
    for fine_info, sr_info in zip(fine_tile_coords, sr_tile_coords):
        loss = fine_info['psnr'] - sr_info['psnr']
        tile_loss_info.append({
            'tile_id': fine_info['tile_id'],
            'coords': fine_info['coords'],
            'fine_psnr': fine_info['psnr'],
            'sr_psnr': sr_info['psnr'],
            'loss': loss
        })
    
    # 按损失排序，找出最差的tiles
    tile_loss_info.sort(key=lambda x: x['loss'], reverse=True)
    
    print(f"\n📊 Tile损失统计:")
    print(f"平均损失: {np.mean([t['loss'] for t in tile_loss_info]):.2f} dB")
    print(f"最大损失: {tile_loss_info[0]['loss']:.2f} dB")
    print(f"最小损失: {tile_loss_info[-1]['loss']:.2f} dB")
    
    # 分析最差的20个tiles
    worst_tiles = tile_loss_info[:30]
    print(f"\n🔍 分析最差的20个tiles:")
    
    # 加载辅助数据
    volume_confidence_map = extract_volume_rendering_confidence(weights_path)
    depth_discontinuity_map = extract_depth_discontinuity_score(depth_path)
    
    # 检测结果
    detection_results = []
    
    for i, tile_info in enumerate(worst_tiles):
        tile_id = tile_info['tile_id']
        coords = tile_info['coords']
        loss = tile_info['loss']
        h_start, h_end, w_start, w_end = coords
        
        print(f"\n--- Tile {i+1}: {tile_id} (损失: {loss:.2f} dB) ---")
        
        # 提取tile区域
        fine_tile = fine_img[h_start:h_end, w_start:w_end, :]
        sr_tile = sr_img[h_start:h_end, w_start:w_end, :]
        gt_tile = gt_img[h_start:h_end, w_start:w_end, :]
        
        result = {
            'tile_id': tile_id,
            'loss': loss,
            'fine_psnr': tile_info['fine_psnr'],
            'sr_psnr': tile_info['sr_psnr']
        }
        
        # 1. Canny高分辨率方法
        canny_high_score = extract_canny_high_res(fine_tile)
        canny_high_decision = canny_high_score > 0.160
        result['canny_high_score'] = canny_high_score
        result['canny_high_decision'] = canny_high_decision
        print(f"Canny高分辨率: 得分={canny_high_score:.3f}, 决策={'替换' if canny_high_decision else '不替换'}")
        
        # 2. Canny低分辨率方法
        canny_low_score = extract_canny_low_res(sr_tile)
        canny_low_decision = canny_low_score > 0.250
        result['canny_low_score'] = canny_low_score
        result['canny_low_decision'] = canny_low_decision
        print(f"Canny低分辨率: 得分={canny_low_score:.3f}, 决策={'替换' if canny_low_decision else '不替换'}")
        
        # 3. 体渲染置信度方法
        if volume_confidence_map is not None:
            print(volume_confidence_map.shape)
            confidence_tile = volume_confidence_map[h_start:h_end, w_start:w_end]
            volume_score = np.mean(confidence_tile)
            # 检查NaN值
            if np.isnan(volume_score):
                volume_score = 0.0
                volume_decision = False
                print(f"体渲染置信度: 得分=NaN(设为0.0), 决策=不替换")
            else:
                volume_decision = volume_score < 0.5  # 置信度低时需要替换
                print(f"体渲染置信度: 得分={volume_score:.3f}, 决策={'替换' if volume_decision else '不替换'}")
            result['volume_score'] = volume_score
            result['volume_decision'] = volume_decision
        else:
            result['volume_score'] = None
            result['volume_decision'] = None
            print(f"体渲染置信度: 数据缺失")
        
        # 4. 深度不连续性方法
        if depth_discontinuity_map is not None:
            depth_tile = depth_discontinuity_map[h_start:h_end, w_start:w_end]
            depth_score = np.mean(depth_tile)
            # 检查NaN值
            if np.isnan(depth_score):
                depth_score = 0.0
                depth_decision = False
                print(f"深度不连续性: 得分=NaN(设为0.0), 决策=不替换")
            else:
                depth_decision = depth_score > 0.1  # 深度变化大时需要替换
                print(f"深度不连续性: 得分={depth_score:.3f}, 决策={'替换' if depth_decision else '不替换'}")
            result['depth_score'] = depth_score
            result['depth_decision'] = depth_decision
        else:
            result['depth_score'] = None
            result['depth_decision'] = None
            print(f"深度不连续性: 数据缺失")
        
        # 5. GACM方法
        if depth_discontinuity_map is not None:
            try:
                # 准备GACM输入
                tile_color = torch.from_numpy(sr_tile).float()
                
                # 从完整深度图中提取对应tile
                depth_map = torch.load(depth_path, map_location='cpu')
                tile_depth = depth_map[h_start:h_end, w_start:w_end]
                
                gacm_score, gacm_decision = compute_gacm(tile_color, tile_depth, alpha=0.4, beta=0.3, gamma=0.3, tau_D_ratio=0.1, threshold=0.01)
                result['gacm_score'] = gacm_score
                result['gacm_decision'] = gacm_decision
                print(f"GACM方法: 得分={gacm_score:.3f}, 决策={'替换' if gacm_decision else '不替换'}")
            except Exception as e:
                result['gacm_score'] = None
                result['gacm_decision'] = None
                print(f"GACM方法: 计算错误 - {e}")
        else:
            result['gacm_score'] = None
            result['gacm_decision'] = None
            print(f"GACM方法: 深度数据缺失")
        
        detection_results.append(result)
    
    return detection_results, worst_tiles

def analyze_detection_accuracy(detection_results):
    """分析各方法的检测准确性"""
    print(f"\n{'='*80}")
    print(f"各方法检测准确性分析")
    print(f"{'='*80}")
    
    methods = [
        ('canny_high', 'Canny高分辨率'),
        ('canny_low', 'Canny低分辨率'),
        ('volume', '体渲染置信度'),
        ('depth', '深度不连续性'),
        ('gacm', 'GACM方法')
    ]
    
    # 这里我们假设损失最大的tiles应该被替换（即ground truth是需要替换）
    # 因为这些是SR表现最差的tiles
    total_tiles = len(detection_results)
    
    print(f"分析对象: 损失最大的{total_tiles}个tiles (理论上都应该被替换)")
    print(f"\n{'方法':<15} {'检测率':<10} {'缺失率':<10} {'可用性':<10}")
    print("-" * 50)
    
    for method_key, method_name in methods:
        detected_count = 0
        available_count = 0
        
        for result in detection_results:
            decision_key = f"{method_key}_decision"
            if result[decision_key] is not None:
                available_count += 1
                if result[decision_key]:  # 如果方法决定替换
                    detected_count += 1
        
        if available_count > 0:
            detection_rate = detected_count / available_count
            miss_rate = 1 - detection_rate
            availability = available_count / total_tiles
            
            print(f"{method_name:<15} {detection_rate:<9.1%} {miss_rate:<9.1%} {availability:<9.1%}")
        else:
            print(f"{method_name:<15} {'N/A':<9} {'N/A':<9} {'0.0%':<9}")
    
    # 详细分析每个方法的得分分布
    # print(f"\n📊 各方法得分分布:")
    
    # for method_key, method_name in methods:
    #     score_key = f"{method_key}_score"
    #     scores = [r[score_key] for r in detection_results if r[score_key] is not None]
        
    #     if scores:
    #         print(f"\n{method_name}:")
    #         print(f"  得分范围: {min(scores):.3f} - {max(scores):.3f}")
    #         print(f"  平均得分: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    #         print(f"  中位数: {np.median(scores):.3f}")
    #     else:
    #         print(f"\n{method_name}: 无可用数据")

def save_detection_report(detection_results, scene_name, image_idx):
    """保存详细的检测报告"""
    report_path = f"../results/reports/tile_detection_diagnosis_{scene_name}_{image_idx}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"=== Tile检测方法诊断报告 ===\n")
        f.write(f"场景: {scene_name}\n")
        f.write(f"图片: {image_idx}\n")
        f.write(f"分析时间: {pd.Timestamp.now()}\n\n")
        
        f.write("=== 最差20个Tiles详细检测结果 ===\n")
        f.write(f"{'Rank':<4} {'TileID':<8} {'损失':<8} {'Canny高':<8} {'Canny低':<8} {'体渲染':<8} {'深度':<8} {'GACM':<8}\n")
        f.write("-" * 70 + "\n")
        
        for i, result in enumerate(detection_results):
            tile_id = result['tile_id']
            loss = result['loss']
            
            canny_high = "✓" if result['canny_high_decision'] else "✗" if result['canny_high_decision'] is not None else "N/A"
            canny_low = "✓" if result['canny_low_decision'] else "✗" if result['canny_low_decision'] is not None else "N/A"
            volume = "✓" if result['volume_decision'] else "✗" if result['volume_decision'] is not None else "N/A"
            depth = "✓" if result['depth_decision'] else "✗" if result['depth_decision'] is not None else "N/A"
            gacm = "✓" if result['gacm_decision'] else "✗" if result['gacm_decision'] is not None else "N/A"
            
            f.write(f"{i+1:<4} {str(tile_id):<8} {loss:<8.2f} {canny_high:<8} {canny_low:<8} {volume:<8} {depth:<8} {gacm:<8}\n")
        
        f.write("\n符号说明: ✓=需要替换, ✗=不需要替换, N/A=数据不可用\n")
    
    print(f"检测报告保存到: {report_path}")

def main():
    """主函数"""
    print("🔍 Tile检测方法诊断工具")
    print("="*80)
    
    # 测试场景 - trex的多个图片
    test_cases = [
        ("trex", 0),
        ("trex", 1),
        ("trex", 2),
    ]
    
    all_results = []
    
    for scene_name, image_idx in test_cases:
        print(f"\n{'='*20} 处理 {scene_name} 图片 {image_idx} {'='*20}")
        
        # 诊断单张图片
        result = diagnose_single_image(scene_name, image_idx)
        
        if result is not None:
            detection_results, worst_tiles = result
            # 分析检测准确性
            analyze_detection_accuracy(detection_results)
            
            # 创建可视化
            # create_detection_visualization(detection_results, scene_name, image_idx)
            
            # 保存报告
            save_detection_report(detection_results, scene_name, image_idx)
            
            all_results.extend(detection_results)
        else:
            print(f"❌ 跳过 {scene_name} 图片 {image_idx}（数据不完整）")
    
    # 总体分析
    if all_results:
        print(f"\n{'='*80}")
        print(f"总体分析 (共{len(all_results)}个最差tiles)")
        print(f"{'='*80}")
        analyze_detection_accuracy(all_results)
    
    print(f"\n✅ 诊断完成！")
    print(f"详细结果保存在 ../results/reports/ 和 ../results/images/ 目录下")

if __name__ == "__main__":
    main()

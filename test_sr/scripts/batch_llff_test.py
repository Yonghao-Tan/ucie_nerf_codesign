#!/usr/bin/env python3
import os
import sys
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# 添加当前脚本目录到路径，以便导入我们的预测器
sys.path.append('/home/ytanaz/access/IBRNet/test_sr/scripts')

# 设置matplotlib支持中文
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False
# 设置matplotlib使用英文，避免中文字体问题
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class BatchTileReplacementTester:
    """批量tile替换测试器"""

    def __init__(self, llff_test_root="/home/ytanaz/access/IBRNet/eval/llff_test"):
        self.llff_test_root = Path(llff_test_root)
        self.eval_llff_path = self.llff_test_root / "eval_llff_golden" # TODO
        self.eval_llff_sr_path = self.llff_test_root / "eval_llff_sr"
        self.tile_size = 32
        self.tile_size = 20
        self.fine_tile_size = self.tile_size // 2  # 在fine分辨率上的tile大小
        
        # 预设阈值
        self.canny_threshold = 0.160  # 高分辨率方法的阈值
        self.canny_threshold_lowres = 0.250  # 低分辨率校正后的阈值
        
        print(f"🔍 批量测试器初始化")
        print(f"LLFF测试根目录: {self.llff_test_root}")
        print(f"Canny高分辨率阈值: {self.canny_threshold}")
        print(f"Canny低分辨率阈值: {self.canny_threshold_lowres}")

    def get_scenes(self, special_scene=None):
        """获取所有测试场景"""
        scenes = []
        print(special_scene)
        if self.eval_llff_path.exists():
            for scene_dir in self.eval_llff_path.iterdir():
                if special_scene is None:
                    if scene_dir.is_dir() and not scene_dir.name.startswith('.'):
                        scenes.append(scene_dir.name)
                elif scene_dir.is_dir() and not scene_dir.name.startswith('.') and special_scene in scene_dir.name:
                    scenes.append(scene_dir.name)
        return sorted(scenes)
    
    def get_image_indices(self, scene_path):
        """获取场景中所有图像的索引"""
        indices = []
        for file_path in scene_path.glob("*_gt_rgb.png"):
            index = int(file_path.stem.split('_')[0])
            indices.append(index)
        return sorted(indices)
    

    def extract_edge_score_canny(self, tile):
        """提取Canny边缘得分"""
        if tile.max() > 1.0:
            tile = tile.astype(np.float32) / 255.0
            
        # 转换为灰度图
        if len(tile.shape) == 3:
            gray = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (tile * 255).astype(np.uint8)
            
        # Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)
        # edges = manual_canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        # print('a',edge_ratio)
        # edges = self_canny(gray, 5, 15)
        # edge_ratio = float(edges) / gray.size
        # print('b',edge_ratio)
        return edge_ratio
    
    def density_complexity_metric(self, tile_sigma, 
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

    def extract_volume_rendering_confidence(self, tile_weights, tile):
        """
        基于体渲染权重的置信度评估
        
        Args:
            tile_weights: [tile_h, tile_w, n_samples] 的权重张量
            
        Returns:
            confidence_score: 置信度得分，越高表示渲染越不确定，越需要高质量渲染
        """
        if tile_weights is None:
            return 0.0
        
        # 确保是numpy数组
        if hasattr(tile_weights, 'cpu'):
            weights = tile_weights.cpu().numpy()
        else:
            weights = tile_weights
        # print(weights.shape, tile.shape)
        dcm_score, need_full_render = self.density_complexity_metric(weights, 
                    sigma_threshold=0.5,
                    grad_threshold=0.3,
                    curvature_threshold=0.2)
        
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
        # print(dcm_score, edge_ratio)
        threshold = 0.37
        return dcm_score * 0.4 + edge_ratio, threshold
        # # 1. 权重分布熵 - 衡量采样点权重的不确定性
        # weights_norm = weights / (weights.sum(axis=-1, keepdims=True) + 1e-8)
        # # 避免log(0)
        # weights_safe = weights_norm + 1e-8
        # weights_entropy = -np.sum(weights_norm * np.log(weights_safe), axis=-1)
        # mean_entropy = np.mean(weights_entropy)
        
        # # 2. 权重集中度 - 最大权重值的平均
        # max_weights = np.max(weights, axis=-1)
        # weight_concentration = np.mean(max_weights)
        
        # # 3. 有效采样点比例 - 权重显著的采样点数量
        # threshold = 0.05 * np.max(weights, axis=-1, keepdims=True)
        # effective_samples = np.sum(weights > threshold, axis=-1)
        # effective_ratio = np.mean(effective_samples) / weights.shape[-1]
        
        # # 综合置信度得分（归一化到0-1范围）
        # # 高熵、低集中度、分散的有效采样点 = 高不确定性 = 需要精细渲染
        # max_entropy = np.log(weights.shape[-1])  # 最大可能熵
        # normalized_entropy = mean_entropy / max_entropy
        
        # confidence_score = (
        #     0.5 * normalized_entropy +              # 权重分布不确定性
        #     0.3 * (1.0 - weight_concentration) +    # 权重分散度  
        #     0.2 * effective_ratio                   # 有效采样点分散度
        # )
        
        # return float(np.clip(confidence_score, 0, 1))
    
    # def extract_depth_discontinuity_score(self, tile_depth, percentile=90):
    def extract_depth_discontinuity_score(self, tile_depth, 
                                          tile,
                           depth_var_threshold=0.15,
                           grad_threshold=0.05,
                           curvature_threshold=0.1):
        """
        纯深度图复杂度指标（Depth Complexity Metric, DCM）
        
        参数:
            tile_depth: [tile_size, tile_size] 深度图 (0~1)
            depth_var_threshold: 深度方差阈值 (用于低纹理区域检测)
            grad_threshold: 深度梯度阈值 (用于边界检测)
            curvature_threshold: 曲率阈值 (用于尖锐边缘检测)
        
        返回:
            dcm_score: 深度复杂度得分 (0~1)
            need_full_render: 是否需要完整NeRF渲染
        """
        # === Step 1: 深度图预处理 ===
        # 归一化到[0,1]（如果尚未归一化）
        tile_depth = tile_depth.numpy()
        if tile_depth.max() > 1.0:
            tile_depth = (tile_depth - tile_depth.min()) / (tile_depth.max() - tile_depth.min() + 1e-5)
        
        # === Step 2: 计算几何关键指标 ===
        # 2.1 深度方差（整体复杂度）
        depth_var = np.var(tile_depth)
        
        # 2.2 深度梯度（边界强度）
        grad_x = cv2.Sobel(tile_depth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(tile_depth, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + 1e-5)
        edge_ratio = np.mean(grad_mag > grad_threshold)  # 深度边界占比
        
        # 2.3 表面曲率（弯曲度）
        laplacian = cv2.Laplacian(tile_depth, cv2.CV_32F)
        curvature_ratio = np.mean(np.abs(laplacian) > curvature_threshold)
        
        # === Step 3: 融合指标 → DCM得分 ===
        # 权重设计：梯度最重要（直接对应边界），曲率次之
        dcm_score = (
            0.5 * np.clip(depth_var / depth_var_threshold, 0, 1) +
            0.3 * edge_ratio +
            0.2 * curvature_ratio
        )
        
        # === Step 4: 自适应决策 ===
        # 低纹理区域特殊处理（深度图可能不可靠）
        if depth_var < 0.01:
            # 用曲率作为主要指标（低纹理区域曲率仍可靠）
            dcm_score = 0.7 * curvature_ratio + 0.3 * edge_ratio
        
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
            
        threshold = 0.42
        dcm_score = dcm_score * 0.3 + edge_ratio
        
        return dcm_score, threshold
    
    def compute_gacm(self, tile_color, tile_depth, 
                     alpha=0.4, beta=0.3, gamma=0.3, 
                     tau_D_ratio=0.1, threshold=0.25):
        """
        计算几何感知复杂度指标 (GACM) - 第三种创新NeRF感知方法
        
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
        # 归一化深度到 [0,1] (适配不同场景尺度)
        depth_min, depth_max = depth.min(), depth.max()
        depth = (depth - depth_min) / (depth_max - depth_min + 1e-5)
        
        # === Step 1: 计算深度驱动的几何复杂度 ===
        # 1.1 深度不连续性 (Depth Discontinuity, DD)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=depth.device)
        sobel_y = sobel_x.T
        grad_x = torch.nn.functional.conv2d(depth[None, None], sobel_x[None, None], padding=1)[0,0]
        grad_y = torch.nn.functional.conv2d(depth[None, None], sobel_y[None, None], padding=1)[0,0]
        G_D = torch.sqrt(grad_x**2 + grad_y**2 + 1e-5)  # 避免除零
        DD_tile = G_D.mean().item()
        
        # 1.2 表面曲率 (Surface Curvature, SC)
        laplacian = torch.nn.functional.conv2d(
            depth[None, None], 
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=depth.device)[None, None],
            padding=1
        )[0,0]
        SC_tile = torch.abs(laplacian).mean().item()
        
        # === Step 2: 计算几何对齐边缘 (Geometry-Aligned Edges, GAE) ===
        # 2.1 用Canny检测颜色边缘 (CPU操作，因OpenCV不支持GPU)
        color_np = (tile_color.cpu().numpy() * 255).astype('uint8')
        gray = cv2.cvtColor(color_np, cv2.COLOR_RGB2GRAY)
        edges_canny = cv2.Canny(gray, 100, 200)  # 标准Canny参数
        
        # 2.2 仅保留与深度不连续对齐的边缘 (核心创新!)
        tau_D = tau_D_ratio * G_D.max().item()  # 深度梯度阈值
        G_D_np = G_D.cpu().numpy()
        GAE = edges_canny * (G_D_np > tau_D)  # 关键：过滤非几何边缘
        GAE_density = GAE.sum() / (32*32*255)  # 归一化到 [0,1]
        
        # === Step 3: 融合指标 → GACM ===
        gacm_score = alpha * DD_tile + beta * SC_tile + gamma * GAE_density
        
        # === Step 4: 决策 ===
        decision = (gacm_score > threshold)
        
        return gacm_score, decision
    
    def depth_discontinuity_method(self, sr_img, fine_img, org_img, render_depth_path, threshold=0.7):
        """
        基于深度不连续性的tile替换方法
        
        工作流程:
        1. fine图像 (378x504) 的深度图 → 分析几何复杂度
        2. SR图像 (756x1008) 的32x32 tile ← fine图像16x16区域 (2倍对应)
        3. 如果fine区域几何复杂，用org_img的对应tile替换
        
        Args:
            sr_img: 超分辨率图像 [756, 1008, 3]
            org_img: 原始高质量图像 [756, 1008, 3]
            render_depth_path: 渲染深度图文件路径 (fine分辨率)
            threshold: 不连续性阈值
        """
        import torch
        
        # 加载深度图 (fine分辨率: 378x504)
        try:
            fine_depth = torch.load(render_depth_path, map_location='cpu')
        except Exception as e:
            print(f"❌ 加载深度图失败: {e}")
            return sr_img, 0
        
        # SR图像尺寸
        sr_h, sr_w = sr_img.shape[:2]
        sr_tile_h, sr_tile_w = sr_h // self.tile_size, sr_w // self.tile_size
        
        # Fine图像尺寸 (应该是SR的一半)
        fine_h, fine_w = fine_depth.shape[:2]
        fine_tile_size = self.fine_tile_size  # 16x16
        fine_tile_h, fine_tile_w = fine_h // fine_tile_size, fine_w // fine_tile_size
        
        # 确保tile数量匹配
        assert sr_tile_h == fine_tile_h and sr_tile_w == fine_tile_w, \
            f"Tile数量不匹配: SR {sr_tile_h}x{sr_tile_w} vs Fine {fine_tile_h}x{fine_tile_w}"
        
        hybrid_img = sr_img.copy()
        replaced_tiles = 0
        tile_scores = []
        
        # 逐tile分析 (同时处理fine和SR对应的tiles)
        for i in range(fine_tile_h):
            for j in range(fine_tile_w):
                # Fine深度图中的tile坐标 (16x16)
                fine_h_start = i * fine_tile_size
                fine_h_end = (i + 1) * fine_tile_size
                fine_w_start = j * fine_tile_size  
                fine_w_end = (j + 1) * fine_tile_size
                
                # SR图像中对应的tile坐标 (32x32)
                sr_h_start = i * self.tile_size
                sr_h_end = (i + 1) * self.tile_size
                sr_w_start = j * self.tile_size
                sr_w_end = (j + 1) * self.tile_size
                
                # 从fine深度图中提取对应的tile
                fine_depth_tile = fine_depth[fine_h_start:fine_h_end, fine_w_start:fine_w_end]
                
                fine_tile = fine_img[fine_h_start:fine_h_end, fine_w_start:fine_w_end]
                
                # 计算深度不连续性得分 (基于fine深度)
                discontinuity_score, threshold = self.extract_depth_discontinuity_score(fine_depth_tile, fine_tile)
                tile_scores.append(discontinuity_score)
                
                # 替换决策: 如果不连续性高(几何复杂)，则替换SR tile
                if discontinuity_score > threshold:
                    hybrid_img[sr_h_start:sr_h_end, sr_w_start:sr_w_end] = \
                        org_img[sr_h_start:sr_h_end, sr_w_start:sr_w_end]
                    replaced_tiles += 1
        
        replacement_ratio = replaced_tiles / (fine_tile_h * fine_tile_w)
        
        return hybrid_img, replaced_tiles
    
    def gacm_method(self, sr_img, org_img, render_depth_path, threshold=0.65, 
                    alpha=0.4, beta=0.3, gamma=0.3, tau_D_ratio=0.1):
        """
        GACM (几何感知复杂度指标) 方法：融合深度+曲率+几何对齐边缘
        
        工作流程:
        1. fine图像 (378x504) 的深度图 → 计算DD和SC
        2. SR图像 (756x1008) 的32x32 tile → 计算GAE
        3. 融合三个指标得到GACM得分，超过阈值则用org_img替换
        
        Args:
            sr_img: 超分辨率图像 [756, 1008, 3]
            org_img: 原始高质量图像 [756, 1008, 3]
            render_depth_path: 渲染深度图文件路径 (fine分辨率)
            threshold: GACM决策阈值
            alpha, beta, gamma: DD, SC, GAE的权重
            tau_D_ratio: 深度梯度阈值比例
        """
        import torch
        import cv2
        
        # 加载深度图 (fine分辨率: 378x504)
        try:
            fine_depth = torch.load(render_depth_path, map_location='cpu')
        except Exception as e:
            print(f"❌ 加载深度图失败: {e}")
            return sr_img, 0
        
        # SR图像尺寸
        sr_h, sr_w = sr_img.shape[:2]
        sr_tile_h, sr_tile_w = sr_h // self.tile_size, sr_w // self.tile_size
        
        # Fine图像尺寸 (应该是SR的一半)
        fine_h, fine_w = fine_depth.shape[:2]
        fine_tile_size = self.fine_tile_size  # 16x16
        fine_tile_h, fine_tile_w = fine_h // fine_tile_size, fine_w // fine_tile_size
        
        # 确保tile数量匹配
        assert sr_tile_h == fine_tile_h and sr_tile_w == fine_tile_w, \
            f"Tile数量不匹配: SR {sr_tile_h}x{sr_tile_w} vs Fine {fine_tile_h}x{fine_tile_w}"
        
        hybrid_img = sr_img.copy()
        replaced_tiles = 0
        gacm_scores = []
        
        # 逐tile分析 (同时处理fine和SR对应的tiles)
        for i in range(fine_tile_h):
            for j in range(fine_tile_w):
                # Fine深度图中的tile坐标 (16x16)
                fine_h_start = i * fine_tile_size
                fine_h_end = (i + 1) * fine_tile_size
                fine_w_start = j * fine_tile_size  
                fine_w_end = (j + 1) * fine_tile_size
                
                # SR图像中对应的tile坐标 (32x32)
                sr_h_start = i * self.tile_size
                sr_h_end = (i + 1) * self.tile_size
                sr_w_start = j * self.tile_size
                sr_w_end = (j + 1) * self.tile_size
                
                # 从fine深度图中提取对应的tile并升采样到32x32
                fine_depth_tile = fine_depth[fine_h_start:fine_h_end, fine_w_start:fine_w_end]
                tile_depth_32 = fine_depth_tile
                # tile_depth_32 = torch.nn.functional.interpolate(
                #     fine_depth_tile[None, None], 
                #     size=(32, 32), 
                #     mode='bilinear'
                # )[0,0]
                
                # 从SR图像中提取对应的颜色tile
                sr_tile = sr_img[fine_h_start:fine_h_end, fine_w_start:fine_w_end]
                tile_color = torch.tensor(sr_tile / 255.0, dtype=torch.float32)
                
                # 计算GACM得分和决策
                gacm_score, should_replace = self.compute_gacm(
                    tile_color, tile_depth_32, 
                    alpha=alpha, beta=beta, gamma=gamma,
                    tau_D_ratio=tau_D_ratio, threshold=threshold
                )
                gacm_scores.append(gacm_score)
                
                # 替换决策: 如果GACM决策为需要完整NeRF，则替换SR tile
                if should_replace:
                    hybrid_img[sr_h_start:sr_h_end, sr_w_start:sr_w_end] = \
                        org_img[sr_h_start:sr_h_end, sr_w_start:sr_w_end]
                    replaced_tiles += 1
        
        replacement_ratio = replaced_tiles / (fine_tile_h * fine_tile_w)
        
        return hybrid_img, replaced_tiles
    
    def volume_rendering_confidence_method(self, sr_img, fine_img, org_img, render_weights_path, threshold=0.35):
        """
        基于体渲染置信度的tile替换方法
        
        工作流程:
        1. fine图像 (378x504) 的权重 → 判断质量
        2. SR图像 (756x1008) 的32x32 tile ← fine图像16x16区域 (2倍对应)
        3. 如果fine区域质量不够，用org_img的对应tile替换
        
        Args:
            sr_img: 超分辨率图像 [756, 1008, 3]
            org_img: 原始高质量图像 [756, 1008, 3]
            render_weights_path: 渲染权重文件路径 (fine分辨率)
            threshold: 置信度阈值
        """
        import torch
        
        # 加载渲染权重 (fine分辨率: 378x504)
        try:
            fine_weights = torch.load(render_weights_path, map_location='cpu')
        except Exception as e:
            print(f"❌ 加载权重失败: {e}")
            return sr_img, 0
        
        # SR图像尺寸
        sr_h, sr_w = sr_img.shape[:2]
        sr_tile_h, sr_tile_w = sr_h // self.tile_size, sr_w // self.tile_size
        
        # Fine图像尺寸 (应该是SR的一半)
        fine_h, fine_w = fine_weights.shape[:2]
        fine_tile_size = self.fine_tile_size  # 16x16
        fine_tile_h, fine_tile_w = fine_h // fine_tile_size, fine_w // fine_tile_size
        
        # 确保tile数量匹配 (应该相等，因为2倍上采样)
        assert sr_tile_h == fine_tile_h and sr_tile_w == fine_tile_w, \
            f"Tile数量不匹配: SR {sr_tile_h}x{sr_tile_w} vs Fine {fine_tile_h}x{fine_tile_w}"
        
        hybrid_img = sr_img.copy()
        replaced_tiles = 0
        tile_scores = []
        
        # 逐tile分析 (同时处理fine和SR对应的tiles)
        for i in range(fine_tile_h):
            for j in range(fine_tile_w):
                # Fine图像中的tile坐标 (16x16)
                fine_h_start = i * fine_tile_size
                fine_h_end = (i + 1) * fine_tile_size
                fine_w_start = j * fine_tile_size  
                fine_w_end = (j + 1) * fine_tile_size
                
                # SR图像中对应的tile坐标 (32x32)
                sr_h_start = i * self.tile_size
                sr_h_end = (i + 1) * self.tile_size
                sr_w_start = j * self.tile_size
                sr_w_end = (j + 1) * self.tile_size
                
                # 从fine权重中提取对应的tile
                fine_tile_weights = fine_weights[fine_h_start:fine_h_end, fine_w_start:fine_w_end, :]
                
                
                h_start = i * self.fine_tile_size
                h_end = (i + 1) * self.fine_tile_size
                w_start = j * self.fine_tile_size
                w_end = (j + 1) * self.fine_tile_size
                
                fine_tile = fine_img[h_start:h_end, w_start:w_end]
                confidence_score, threshold = self.extract_volume_rendering_confidence(fine_tile_weights, fine_tile)
                tile_scores.append(confidence_score)
                
                # 替换决策: 如果置信度高(不确定性大)，则替换SR tile
                if confidence_score > threshold:
                    hybrid_img[sr_h_start:sr_h_end, sr_w_start:sr_w_end] = \
                        org_img[sr_h_start:sr_h_end, sr_w_start:sr_w_end]
                    replaced_tiles += 1
        
        replacement_ratio = replaced_tiles / (fine_tile_h * fine_tile_w)
        
        return hybrid_img, replaced_tiles
    
    def random_replacement_method(self, sr_img, org_img, num_tiles_to_replace):
        """随机替换指定数量的tiles"""
        h, w = sr_img.shape[:2]
        tile_h, tile_w = h // self.tile_size, w // self.tile_size
        total_tiles = tile_h * tile_w
        
        # 随机选择要替换的tiles
        replace_indices = np.random.choice(total_tiles, 
                                         min(num_tiles_to_replace, total_tiles), 
                                         replace=False)
        
        hybrid_img = sr_img.copy()
        replaced_tiles = 0
        
        for idx in replace_indices:
            i = idx // tile_w
            j = idx % tile_w
            h_start, h_end = i * self.tile_size, (i + 1) * self.tile_size
            w_start, w_end = j * self.tile_size, (j + 1) * self.tile_size
            
            hybrid_img[h_start:h_end, w_start:w_end] = org_img[h_start:h_end, w_start:w_end]
            replaced_tiles += 1
        
        return hybrid_img, replaced_tiles
    
    def canny_highres_method(self, sr_img, org_img, threshold=None):
        """高分辨率Canny边缘检测方法"""
        if threshold is None:
            threshold = self.canny_threshold
            
        h, w = sr_img.shape[:2]
        tile_h, tile_w = h // self.tile_size, w // self.tile_size
        
        hybrid_img = sr_img.copy()
        replaced_tiles = 0
        
        for i in range(tile_h):
            for j in range(tile_w):
                h_start, h_end = i * self.tile_size, (i + 1) * self.tile_size
                w_start, w_end = j * self.tile_size, (j + 1) * self.tile_size
                
                sr_tile = sr_img[h_start:h_end, w_start:w_end]
                edge_score = self.extract_edge_score_canny(sr_tile)
                
                if edge_score > threshold:
                    hybrid_img[h_start:h_end, w_start:w_end] = org_img[h_start:h_end, w_start:w_end]
                    replaced_tiles += 1
        
        return hybrid_img, replaced_tiles
    
    def canny_lowres_method(self, fine_img, sr_img, org_img, threshold=None):
        """低分辨率Canny边缘检测方法"""
        if threshold is None:
            threshold = self.canny_threshold_lowres
            
        # 在fine分辨率上进行分析
        h_fine, w_fine = fine_img.shape[:2]
        tile_h, tile_w = h_fine // self.fine_tile_size, w_fine // self.fine_tile_size
        
        # 创建fine分辨率的掩码
        fine_mask = np.zeros((h_fine, w_fine), dtype=bool)
        replaced_tiles = 0
        
        for i in range(tile_h):
            for j in range(tile_w):
                h_start = i * self.fine_tile_size
                h_end = (i + 1) * self.fine_tile_size
                w_start = j * self.fine_tile_size
                w_end = (j + 1) * self.fine_tile_size
                
                fine_tile = fine_img[h_start:h_end, w_start:w_end]
                edge_score = self.extract_edge_score_canny(fine_tile)
                
                if edge_score > threshold:
                    fine_mask[h_start:h_end, w_start:w_end] = True
                    replaced_tiles += 1
        
        # 将掩码上采样到SR分辨率
        sr_mask = cv2.resize(
            fine_mask.astype(np.uint8), 
            (sr_img.shape[1], sr_img.shape[0]), 
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        
        # 在SR图像上应用掩码
        hybrid_img = sr_img.copy()
        hybrid_img[sr_mask] = org_img[sr_mask]
        
        return hybrid_img, replaced_tiles
    
    def test_single_image(self, scene, img_idx):
        """测试单张图像的所有方法"""
        results = {
            'scene': scene,
            'img_idx': img_idx,
            'sr_psnr': 0,
            'total_tiles': 0,
            'random_max_psnr': 0,
            'random_max_improvement': 0,
            'random_max_tiles': 0,
            'canny_highres_psnr': 0,
            'canny_highres_improvement': 0,
            'canny_highres_tiles': 0,
            'canny_lowres_psnr': 0,
            'canny_lowres_improvement': 0,
            'canny_lowres_tiles': 0,
            'volume_confidence_psnr': 0,
            'volume_confidence_improvement': 0,
            'volume_confidence_tiles': 0,
            'depth_discontinuity_psnr': 0,
            'depth_discontinuity_improvement': 0,
            'depth_discontinuity_tiles': 0,
        }
        
        # 加载图像
        org_path = self.eval_llff_path / scene / f"{img_idx}_pred_fine.png"
        sr_path = self.eval_llff_sr_path / scene / f"{img_idx}_pred_sr.png"
        fine_path = self.eval_llff_sr_path / scene / f"{img_idx}_pred_fine.png"
        gt_path = self.eval_llff_sr_path / scene / f"{img_idx}_gt_rgb_hr.png" # important
        
        # 检查文件是否存在
        if not all(p.exists() for p in [org_path, sr_path, fine_path, gt_path]):
            return None
        
        org_img = np.array(Image.open(org_path))
        sr_img = np.array(Image.open(sr_path))
        fine_img = np.array(Image.open(fine_path))
        gt_img = np.array(Image.open(gt_path))
        
        # 调整GT图像尺寸以匹配SR输出
        if gt_img.shape != sr_img.shape:
            gt_img = cv2.resize(gt_img, (sr_img.shape[1], sr_img.shape[0]))
        
        # 计算总tile数
        h, w = sr_img.shape[:2]
        tile_h, tile_w = h // self.tile_size, w // self.tile_size
        total_tiles = tile_h * tile_w
        results['total_tiles'] = total_tiles
        
        # 计算原始SR PSNR
        sr_psnr = psnr(gt_img, sr_img)
        results['sr_psnr'] = sr_psnr
        
        # 2. 高分辨率Canny方法
        canny_hr_img, canny_hr_tiles = self.canny_highres_method(sr_img, org_img)
        canny_hr_psnr = psnr(gt_img, canny_hr_img)
        results['canny_highres_psnr'] = canny_hr_psnr
        results['canny_highres_improvement'] = canny_hr_psnr - sr_psnr
        results['canny_highres_tiles'] = canny_hr_tiles
        
        # 3. 低分辨率Canny方法
        canny_lr_img, canny_lr_tiles = self.canny_lowres_method(fine_img, sr_img, org_img)
        canny_lr_psnr = psnr(gt_img, canny_lr_img)
        results['canny_lowres_psnr'] = canny_lr_psnr
        results['canny_lowres_improvement'] = canny_lr_psnr - sr_psnr
        results['canny_lowres_tiles'] = canny_lr_tiles
        
        # 4. 体渲染置信度方法
        weights_path = Path(f"/home/ytanaz/access/IBRNet/test_sr/data/datasets/{scene}") / f"render_weights_{img_idx}_n48.pt"
        weights_path = Path(f"/home/ytanaz/access/IBRNet/test_sr/data/datasets/{scene}") / f"render_sigmas_{img_idx}_n48.pt"
        depth_path = Path(f"/home/ytanaz/access/IBRNet/test_sr/data/datasets/{scene}") / f"render_depths_{img_idx}_n48.pt"
        if weights_path.exists():
            volume_img, volume_tiles = self.volume_rendering_confidence_method(sr_img, fine_img, org_img, str(weights_path))
            volume_psnr = psnr(gt_img, volume_img)
            results['volume_confidence_psnr'] = volume_psnr
            results['volume_confidence_improvement'] = volume_psnr - sr_psnr
            results['volume_confidence_tiles'] = volume_tiles
        else:
            print(f"⚠️ 未找到权重文件: {weights_path}")
            results['volume_confidence_psnr'] = sr_psnr
            results['volume_confidence_improvement'] = 0.0
            results['volume_confidence_tiles'] = 0
        
        # 5. 深度不连续性方法
        if depth_path.exists():
            depth_img, depth_tiles = self.depth_discontinuity_method(sr_img, fine_img, org_img, str(depth_path))
            depth_psnr = psnr(gt_img, depth_img)
            results['depth_discontinuity_psnr'] = depth_psnr
            results['depth_discontinuity_improvement'] = depth_psnr - sr_psnr
            results['depth_discontinuity_tiles'] = depth_tiles
        else:
            print(f"⚠️ 未找到深度文件: {depth_path}")
            results['depth_discontinuity_psnr'] = sr_psnr
            results['depth_discontinuity_improvement'] = 0.0
            results['depth_discontinuity_tiles'] = 0
        
        # 6. GACM (几何感知复杂度指标) 方法
        if depth_path.exists():
            gacm_img, gacm_tiles = self.gacm_method(sr_img, org_img, str(depth_path))
            gacm_psnr = psnr(gt_img, gacm_img)
            results['gacm_psnr'] = gacm_psnr
            results['gacm_improvement'] = gacm_psnr - sr_psnr
            results['gacm_tiles'] = gacm_tiles
        else:
            print(f"⚠️ 未找到深度文件: {depth_path}")
            results['gacm_psnr'] = sr_psnr
            results['gacm_improvement'] = 0.0
            results['gacm_tiles'] = 0

        # 1. 随机替换（使用所有方法中的最大tile数作为基线）
        max_tiles = max(canny_hr_tiles, canny_lr_tiles, 
                        results['volume_confidence_tiles'], 
                        results['depth_discontinuity_tiles'],
                        results['gacm_tiles'])
        random_img, _ = self.random_replacement_method(sr_img, org_img, max_tiles)
        random_psnr = psnr(gt_img, random_img)
        results['random_max_psnr'] = random_psnr
        results['random_max_improvement'] = random_psnr - sr_psnr
        results['random_max_tiles'] = max_tiles
            
        # except Exception as e:
        #     print(f"错误处理 {scene}/{img_idx}: {e}")
        #     return None
        
        return results
    
    def test_all_images(self):
        """测试所有图像"""
        print("🚀 开始批量测试所有图像...")
        
        # scenes = self.get_scenes(special_scene='horns')
        scenes = self.get_scenes()
        print(f"发现场景: {scenes}")
        
        all_results = []
        total_images = 0
        
        # 统计总图像数
        for scene in scenes:
            scene_path = self.eval_llff_sr_path / scene
            if scene_path.exists():
                indices = self.get_image_indices(scene_path)
                total_images += len(indices)
        
        print(f"总共需要测试 {total_images} 张图像")
        
        # 使用进度条测试所有图像
        with tqdm(total=total_images, desc="测试进度") as pbar:
            for scene in scenes:
                scene_path = self.eval_llff_sr_path / scene
                if not scene_path.exists():
                    continue
                
                indices = self.get_image_indices(scene_path)
                print(f"\n正在测试场景 {scene}: {len(indices)} 张图像")
                
                for img_idx in indices:
                    result = self.test_single_image(scene, img_idx)
                    if result is not None:
                        all_results.append(result)
                    pbar.update(1)
        
        return all_results
    
    def analyze_results(self, results):
        """分析测试结果"""
        df = pd.DataFrame(results)
        
        print("\n📊 批量测试结果分析")
        print("="*70)
        
        # 总体统计
        print(f"测试图像总数: {len(df)}")
        print(f"测试场景: {df['scene'].unique()}")
        
        # 计算平均值
        avg_results = {
            'SR原始PSNR': df['sr_psnr'].mean(),
            '随机最大tiles': df['random_max_improvement'].mean(),
            'Canny高分辨率': df['canny_highres_improvement'].mean(),
            'Canny低分辨率': df['canny_lowres_improvement'].mean(),
            '体渲染置信度': df['volume_confidence_improvement'].mean(),
            '深度不连续性': df['depth_discontinuity_improvement'].mean(),
            'GACM': df['gacm_improvement'].mean()
        }
        
        print(f"\n📈 平均PSNR提升 (dB):")
        print(f"{'方法':<20} {'PSNR提升':<10} {'标准差':<10}")
        print("-" * 45)
        print(f"{'随机最大tiles':<20} {avg_results['随机最大tiles']:<9.3f} {df['random_max_improvement'].std():<9.3f}")
        print(f"{'Canny高分辨率':<20} {avg_results['Canny高分辨率']:<9.3f} {df['canny_highres_improvement'].std():<9.3f}")
        print(f"{'Canny低分辨率':<20} {avg_results['Canny低分辨率']:<9.3f} {df['canny_lowres_improvement'].std():<9.3f}")
        print(f"{'体渲染置信度':<20} {avg_results['体渲染置信度']:<9.3f} {df['volume_confidence_improvement'].std():<9.3f}")
        print(f"{'深度不连续性':<20} {avg_results['深度不连续性']:<9.3f} {df['depth_discontinuity_improvement'].std():<9.3f}")
        print(f"{'GACM':<20} {avg_results['GACM']:<9.3f} {df['gacm_improvement'].std():<9.3f}")
        
        # 替换tiles统计
        print(f"\n🔧 平均替换Tiles数:")
        avg_total_tiles = df['total_tiles'].mean()
        print(f"平均总tiles数: {avg_total_tiles:.0f} tiles")
        print(f"随机替换: {df['random_max_tiles'].mean():.1f} tiles (基线)")
        print(f"Canny高分辨率: {df['canny_highres_tiles'].mean():.1f} tiles")
        print(f"Canny低分辨率: {df['canny_lowres_tiles'].mean():.1f} tiles")
        print(f"体渲染置信度: {df['volume_confidence_tiles'].mean():.1f} tiles")
        print(f"深度不连续性: {df['depth_discontinuity_tiles'].mean():.1f} tiles")
        print(f"GACM: {df['gacm_tiles'].mean():.1f} tiles")
        
        # 替换比例统计
        print(f"\n📏 替换比例(替换tiles/总tiles):")
        print(f"随机替换: {df['random_max_tiles'].mean()/avg_total_tiles:.1%}")
        print(f"Canny高分辨率: {df['canny_highres_tiles'].mean()/avg_total_tiles:.1%}")
        print(f"Canny低分辨率: {df['canny_lowres_tiles'].mean()/avg_total_tiles:.1%}")
        print(f"体渲染置信度: {df['volume_confidence_tiles'].mean()/avg_total_tiles:.1%}")
        print(f"深度不连续性: {df['depth_discontinuity_tiles'].mean()/avg_total_tiles:.1%}")
        print(f"GACM: {df['gacm_tiles'].mean()/avg_total_tiles:.1%}")
        
        # 按场景分析
        print(f"\n🎬 按场景分析:")
        scene_columns = {
            'sr_psnr': 'mean',
            'total_tiles': 'mean',
            'random_max_improvement': 'mean',
            'random_max_tiles': 'mean',
            'canny_highres_improvement': 'mean',
            'canny_lowres_improvement': 'mean',
            'canny_highres_tiles': 'mean',
            'canny_lowres_tiles': 'mean',
            'volume_confidence_improvement': 'mean',
            'volume_confidence_tiles': 'mean',
            'depth_discontinuity_improvement': 'mean',
            'depth_discontinuity_tiles': 'mean',
        }
        
        scene_stats = df.groupby('scene').agg(scene_columns)
        
        for scene in scene_stats.index:
            stats = scene_stats.loc[scene]
            total_tiles = stats['total_tiles']
            print(f"\n{scene}:")
            print(f"  平均SR PSNR: {stats['sr_psnr']:.2f} dB")
            print(f"  总tiles数: {total_tiles:.0f} tiles")
            print(f"  随机最大tiles: +{stats['random_max_improvement']:.3f} dB ({stats['random_max_tiles']:.1f} tiles, {stats['random_max_tiles']/total_tiles:.1%})")
            print(f"  Canny高分辨率: +{stats['canny_highres_improvement']:.3f} dB ({stats['canny_highres_tiles']:.1f} tiles, {stats['canny_highres_tiles']/total_tiles:.1%})")
            print(f"  Canny低分辨率: +{stats['canny_lowres_improvement']:.3f} dB ({stats['canny_lowres_tiles']:.1f} tiles, {stats['canny_lowres_tiles']/total_tiles:.1%})")
            print(f"  体渲染置信度: +{stats['volume_confidence_improvement']:.3f} dB ({stats['volume_confidence_tiles']:.1f} tiles, {stats['volume_confidence_tiles']/total_tiles:.1%})")
            print(f"  深度不连续性: +{stats['depth_discontinuity_improvement']:.3f} dB ({stats['depth_discontinuity_tiles']:.1f} tiles, {stats['depth_discontinuity_tiles']/total_tiles:.1%})")
        
        return df, avg_results
    
    def save_results(self, df, avg_results):
        """保存结果"""
        output_dir = Path("/home/ytanaz/access/IBRNet/test_sr/results/batch_test")
        output_dir.mkdir(exist_ok=True)
        
        # 保存详细CSV
        csv_path = output_dir / "batch_test_detailed_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n💾 详细结果保存到: {csv_path}")
        
        # 保存总结报告
        report_path = output_dir / "batch_test_summary_report.txt"
        with open(report_path, 'w') as f:
            f.write("=== LLFF数据集批量测试总结报告 ===\n\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试图像总数: {len(df)}\n")
            f.write(f"测试场景: {', '.join(df['scene'].unique())}\n\n")
            
            f.write("平均PSNR提升结果:\n")
            f.write(f"随机最大tiles替换: +{avg_results['随机最大tiles']:.3f} dB\n")
            f.write(f"Canny高分辨率方法: +{avg_results['Canny高分辨率']:.3f} dB\n")
            f.write(f"Canny低分辨率方法: +{avg_results['Canny低分辨率']:.3f} dB\n")
            
            # 添加替换比例信息
            avg_total_tiles = df['total_tiles'].mean()
            f.write(f"\n替换比例统计(平均总tiles: {avg_total_tiles:.0f}):\n")
            f.write(f"随机替换: {df['random_max_tiles'].mean()/avg_total_tiles:.1%}\n")
            f.write(f"Canny高分辨率: {df['canny_highres_tiles'].mean()/avg_total_tiles:.1%}\n")
            f.write(f"Canny低分辨率: {df['canny_lowres_tiles'].mean()/avg_total_tiles:.1%}\n")
            
            f.write("\n关键结论:\n")
            canny_hr_vs_random = avg_results['Canny高分辨率'] / avg_results['随机最大tiles']
            canny_lr_vs_random = avg_results['Canny低分辨率'] / avg_results['随机最大tiles']
            f.write(f"Canny高分辨率 vs 随机: {canny_hr_vs_random:.2f}倍效果\n")
            f.write(f"Canny低分辨率 vs 随机: {canny_lr_vs_random:.2f}倍效果\n")
            
            f.write(f"低分辨率方法计算量减少: 75%\n")
        
        print(f"📋 总结报告保存到: {report_path}")
        
        # 生成可视化图表
        self.create_visualization(df, output_dir)
    
    def create_visualization(self, df, output_dir):
        """Create visualization charts"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. PSNR improvement comparison
        methods = ['Random Max Tiles', 'Canny High Resolution', 'Canny Low Resolution']
        improvements = [
            df['random_max_improvement'].mean(),
            df['canny_highres_improvement'].mean(),
            df['canny_lowres_improvement'].mean()
        ]
        
        colors = ['gray', 'blue', 'red', 'green', 'orange'][:len(methods)]
        ax1.bar(methods, improvements, color=colors)
        ax1.set_ylabel('Average PSNR Improvement (dB)')
        ax1.set_title('PSNR Improvement Comparison of Different Methods')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. PSNR improvement by scene
        scene_columns = {
            'canny_highres_improvement': 'mean',
            'canny_lowres_improvement': 'mean'
        }
        
        scene_stats = df.groupby('scene').agg(scene_columns)
        
        x = np.arange(len(scene_stats.index))
        width = 0.15 if len(scene_columns) > 2 else 0.35
        
        bars = []
        labels = ['Canny High Res', 'Canny Low Res']
        colors = ['blue', 'red']
        
        bars.append(ax2.bar(x - width, scene_stats['canny_highres_improvement'], width, 
                label='Canny High Resolution', color='blue', alpha=0.7))
        bars.append(ax2.bar(x, scene_stats['canny_lowres_improvement'], width,
                label='Canny Low Resolution', color='red', alpha=0.7))
        
        ax2.set_xlabel('Scene')
        ax2.set_ylabel('Average PSNR Improvement (dB)')
        ax2.set_title('PSNR Improvement Comparison by Scene')
        ax2.set_xticks(x)
        ax2.set_xticklabels(scene_stats.index, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Number of replaced tiles comparison
        tile_methods = ['Canny High Res', 'Canny Low Res']
        avg_tiles = [df['canny_highres_tiles'].mean(), df['canny_lowres_tiles'].mean()]
        colors = ['blue', 'red']
        
        ax3.bar(tile_methods, avg_tiles, color=colors, alpha=0.7)
        ax3.set_ylabel('Average Number of Replaced Tiles')
        ax3.set_title('Comparison of Average Number of Replaced Tiles')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. PSNR improvement distribution
        ax4.hist(df['canny_highres_improvement'], bins=20, alpha=0.7, 
                label='Canny High Resolution', color='blue')
        ax4.hist(df['canny_lowres_improvement'], bins=20, alpha=0.7,
                label='Canny Low Resolution', color='red')
        ax4.set_xlabel('PSNR Improvement (dB)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('PSNR Improvement Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the chart
        chart_path = output_dir / "batch_test_analysis_charts.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """主函数"""
    print("🔍 LLFF数据集批量测试开始")
    print("="*70)
    
    # 创建测试器
    tester = BatchTileReplacementTester()
    
    # 设置随机种子确保可重复性
    np.random.seed(42)
    
    # 执行批量测试
    start_time = time.time()
    results = tester.test_all_images()
    end_time = time.time()
    
    if not results:
        print("❌ 没有找到有效的测试结果")
        return
    
    print(f"\n⏱️ 测试完成，耗时: {end_time - start_time:.1f} 秒")
    
    # 分析结果
    df, avg_results = tester.analyze_results(results)
    
    # 保存结果
    tester.save_results(df, avg_results)

if __name__ == "__main__":
    main()

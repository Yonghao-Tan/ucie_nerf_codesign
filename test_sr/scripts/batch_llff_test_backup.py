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

    def __init__(self, llff_test_root="/home/ytanaz/access/IBRNet/eval/llff_test", use_nn=False):
        self.llff_test_root = Path(llff_test_root)
        self.eval_llff_path = self.llff_test_root / "eval_llff_golden" # TODO
        self.eval_llff_sr_path = self.llff_test_root / "eval_llff_sr"
        self.tile_size = 32
        self.tile_size = 20
        self.fine_tile_size = self.tile_size // 2  # 在fine分辨率上的tile大小
        self.use_nn = use_nn
        
        # 预设阈值
        self.canny_threshold = 0.160  # 高分辨率方法的阈值
        self.canny_threshold_lowres = 0.250  # 低分辨率校正后的阈值
        
        # 初始化MobileNetV2预测器
        if self.use_nn:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.mobilenet_predictor = self._init_mobilenet_predictor()
            self.mobilenet_threshold = 1.80  # MobileNetV2高分辨率阈值（调整以控制替换率~100）
            self.mobilenet_threshold_lowres = 2.20  # MobileNetV2低分辨率阈值（调整以控制替换率~100）
        else:
            self.mobilenet_predictor = None
        
        print(f"🔍 批量测试器初始化")
        print(f"LLFF测试根目录: {self.llff_test_root}")
        print(f"Canny高分辨率阈值: {self.canny_threshold}")
        print(f"Canny低分辨率阈值: {self.canny_threshold_lowres}")
        if self.use_nn:
            print(f"MobileNet高分辨率阈值: {self.mobilenet_threshold}")
            print(f"MobileNet低分辨率阈值: {self.mobilenet_threshold_lowres}")
            print(f"计算设备: {self.device}")
    
    def _init_mobilenet_predictor(self):
        """初始化MobileNetV2复杂度预测器"""
        try:
            # 加载MobileNetV2并提取前几层
            mobilenet = models.mobilenet_v2(pretrained=True)
            feature_extractor = nn.Sequential(
                *list(mobilenet.features.children())[:4]  # 前4层，轻量级
            ).to(self.device)
            
            # 冻结预训练权重
            feature_extractor.eval()
            for param in feature_extractor.parameters():
                param.requires_grad = False
            
            # ImageNet标准化
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            
            return {
                'feature_extractor': feature_extractor,
                'normalize': normalize
            }
        except Exception as e:
            print(f"⚠️ MobileNetV2初始化失败: {e}")
            print("将跳过MobileNetV2测试")
            return None
    
    def get_scenes(self):
        """获取所有测试场景"""
        scenes = []
        if self.eval_llff_path.exists():
            for scene_dir in self.eval_llff_path.iterdir():
                if scene_dir.is_dir() and not scene_dir.name.startswith('.') and 'trex' in scene_dir.name:
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
        edge_ratio = np.sum(edges > 0) / edges.size
        return edge_ratio
    
    def extract_mobilenet_score(self, tile):
        """提取MobileNetV2复杂度得分"""
        if self.mobilenet_predictor is None:
            return 0.0  # 如果MobileNet不可用，返回默认值
            
        try:
            # 预处理tile
            if tile.max() > 1.0:
                tile = tile.astype(np.float32) / 255.0
            
            # 转换为torch tensor
            if len(tile.shape) == 3:  # RGB
                tensor = torch.from_numpy(tile).permute(2, 0, 1)  # HWC -> CHW
            else:  # 灰度图
                tensor = torch.from_numpy(tile).unsqueeze(0)  # 添加通道维度
                tensor = tensor.repeat(3, 1, 1)  # 转换为3通道
            
            # 标准化
            tensor = self.mobilenet_predictor['normalize'](tensor)
            
            # 添加batch维度
            tensor = tensor.unsqueeze(0).to(self.device)  # [1, 3, H, W]
            
            with torch.no_grad():
                # 特征提取
                features = self.mobilenet_predictor['feature_extractor'](tensor)
                
                # 复杂度指标计算
                spatial_variance = torch.var(features, dim=[2, 3]).mean()
                channel_means = torch.mean(features, dim=[2, 3])
                channel_diversity = torch.std(channel_means)
                edge_response = torch.norm(features, dim=1).mean()
                activation_sparsity = torch.mean(torch.abs(features))
                
                # 梯度幅度
                grad_x = torch.abs(features[:, :, 1:, :] - features[:, :, :-1, :]).mean()
                grad_y = torch.abs(features[:, :, :, 1:] - features[:, :, :, :-1]).mean()
                gradient_magnitude = (grad_x + grad_y) / 2
                
                # 综合复杂度得分
                complexity_score = (
                    0.30 * spatial_variance.item() +
                    0.25 * channel_diversity.item() + 
                    0.20 * edge_response.item() +
                    0.15 * activation_sparsity.item() +
                    0.10 * gradient_magnitude.item()
                )
                
                return complexity_score
                
        except Exception as e:
            print(f"MobileNet预测错误: {e}")
            return 0.0
    
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
    
    def mobilenet_highres_method(self, sr_img, org_img, threshold=None):
        """高分辨率MobileNetV2方法"""
        if threshold is None:
            threshold = self.mobilenet_threshold
            
        if self.mobilenet_predictor is None:
            print("⚠️ MobileNetV2不可用，跳过此方法")
            return sr_img.copy(), 0
            
        h, w = sr_img.shape[:2]
        tile_h, tile_w = h // self.tile_size, w // self.tile_size
        
        hybrid_img = sr_img.copy()
        replaced_tiles = 0
        
        for i in range(tile_h):
            for j in range(tile_w):
                h_start, h_end = i * self.tile_size, (i + 1) * self.tile_size
                w_start, w_end = j * self.tile_size, (j + 1) * self.tile_size
                
                sr_tile = sr_img[h_start:h_end, w_start:w_end]
                complexity_score = self.extract_mobilenet_score(sr_tile)
                
                if complexity_score > threshold:
                    hybrid_img[h_start:h_end, w_start:w_end] = org_img[h_start:h_end, w_start:w_end]
                    replaced_tiles += 1
        
        return hybrid_img, replaced_tiles
    
    def mobilenet_lowres_method(self, fine_img, sr_img, org_img, threshold=None):
        """低分辨率MobileNetV2方法"""
        if threshold is None:
            threshold = self.mobilenet_threshold_lowres  # 使用校准后的低分辨率阈值
            
        if self.mobilenet_predictor is None:
            print("⚠️ MobileNetV2不可用，跳过此方法")
            return sr_img.copy(), 0
            
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
                complexity_score = self.extract_mobilenet_score(fine_tile)
                
                if complexity_score > threshold:
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
            'mobilenet_highres_psnr': 0,
            'mobilenet_highres_improvement': 0,
            'mobilenet_highres_tiles': 0,
            'mobilenet_lowres_psnr': 0,
            'mobilenet_lowres_improvement': 0,
            'mobilenet_lowres_tiles': 0
        }
        
        try:
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
            
            # 4. 高分辨率MobileNetV2方法
            mobilenet_hr_tiles = 0
            mobilenet_lr_tiles = 0
            if self.mobilenet_predictor is not None:
                mobilenet_hr_img, mobilenet_hr_tiles = self.mobilenet_highres_method(sr_img, org_img)
                mobilenet_hr_psnr = psnr(gt_img, mobilenet_hr_img)
                results['mobilenet_highres_psnr'] = mobilenet_hr_psnr
                results['mobilenet_highres_improvement'] = mobilenet_hr_psnr - sr_psnr
                results['mobilenet_highres_tiles'] = mobilenet_hr_tiles
                
                # 5. 低分辨率MobileNetV2方法
                mobilenet_lr_img, mobilenet_lr_tiles = self.mobilenet_lowres_method(fine_img, sr_img, org_img)
                mobilenet_lr_psnr = psnr(gt_img, mobilenet_lr_img)
                results['mobilenet_lowres_psnr'] = mobilenet_lr_psnr
                results['mobilenet_lowres_improvement'] = mobilenet_lr_psnr - sr_psnr
                results['mobilenet_lowres_tiles'] = mobilenet_lr_tiles
            
            # 1. 随机替换（使用所有方法中的最大tile数作为基线）
            max_tiles = max(canny_hr_tiles, canny_lr_tiles, mobilenet_hr_tiles, mobilenet_lr_tiles)
            random_img, _ = self.random_replacement_method(sr_img, org_img, max_tiles)
            random_psnr = psnr(gt_img, random_img)
            results['random_max_psnr'] = random_psnr
            results['random_max_improvement'] = random_psnr - sr_psnr
            results['random_max_tiles'] = max_tiles
            
        except Exception as e:
            print(f"错误处理 {scene}/{img_idx}: {e}")
            return None
        
        return results
    
    def test_all_images(self):
        """测试所有图像"""
        print("🚀 开始批量测试所有图像...")
        
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
            'Canny低分辨率': df['canny_lowres_improvement'].mean()
        }
        
        # 如果有MobileNetV2结果，添加到平均值中
        if self.mobilenet_predictor is not None and 'mobilenet_highres_improvement' in df.columns:
            avg_results['MobileNet高分辨率'] = df['mobilenet_highres_improvement'].mean()
            avg_results['MobileNet低分辨率'] = df['mobilenet_lowres_improvement'].mean()
        
        print(f"\n📈 平均PSNR提升 (dB):")
        print(f"{'方法':<20} {'PSNR提升':<10} {'标准差':<10}")
        print("-" * 45)
        print(f"{'随机最大tiles':<20} {avg_results['随机最大tiles']:<9.3f} {df['random_max_improvement'].std():<9.3f}")
        print(f"{'Canny高分辨率':<20} {avg_results['Canny高分辨率']:<9.3f} {df['canny_highres_improvement'].std():<9.3f}")
        print(f"{'Canny低分辨率':<20} {avg_results['Canny低分辨率']:<9.3f} {df['canny_lowres_improvement'].std():<9.3f}")
        
        if 'MobileNet高分辨率' in avg_results:
            print(f"{'MobileNet高分辨率':<20} {avg_results['MobileNet高分辨率']:<9.3f} {df['mobilenet_highres_improvement'].std():<9.3f}")
            print(f"{'MobileNet低分辨率':<20} {avg_results['MobileNet低分辨率']:<9.3f} {df['mobilenet_lowres_improvement'].std():<9.3f}")
        
        # 替换tiles统计
        print(f"\n🔧 平均替换Tiles数:")
        avg_total_tiles = df['total_tiles'].mean()
        print(f"平均总tiles数: {avg_total_tiles:.0f} tiles")
        print(f"随机替换: {df['random_max_tiles'].mean():.1f} tiles (基线)")
        print(f"Canny高分辨率: {df['canny_highres_tiles'].mean():.1f} tiles")
        print(f"Canny低分辨率: {df['canny_lowres_tiles'].mean():.1f} tiles")
        
        if 'mobilenet_highres_tiles' in df.columns and self.mobilenet_predictor is not None:
            print(f"MobileNet高分辨率: {df['mobilenet_highres_tiles'].mean():.1f} tiles")
            print(f"MobileNet低分辨率: {df['mobilenet_lowres_tiles'].mean():.1f} tiles")
        
        # 替换比例统计
        print(f"\n📏 替换比例(替换tiles/总tiles):")
        print(f"随机替换: {df['random_max_tiles'].mean()/avg_total_tiles:.1%}")
        print(f"Canny高分辨率: {df['canny_highres_tiles'].mean()/avg_total_tiles:.1%}")
        print(f"Canny低分辨率: {df['canny_lowres_tiles'].mean()/avg_total_tiles:.1%}")
        
        if 'mobilenet_highres_tiles' in df.columns and self.mobilenet_predictor is not None:
            print(f"MobileNet高分辨率: {df['mobilenet_highres_tiles'].mean()/avg_total_tiles:.1%}")
            print(f"MobileNet低分辨率: {df['mobilenet_lowres_tiles'].mean()/avg_total_tiles:.1%}")
        
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
            'canny_lowres_tiles': 'mean'
        }
        
        # 如果有MobileNetV2结果，添加到分析中
        if self.mobilenet_predictor is not None and 'mobilenet_highres_improvement' in df.columns:
            scene_columns.update({
                'mobilenet_highres_improvement': 'mean',
                'mobilenet_lowres_improvement': 'mean',
                'mobilenet_highres_tiles': 'mean',
                'mobilenet_lowres_tiles': 'mean'
            })
        
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
            
            if 'mobilenet_highres_improvement' in stats:
                print(f"  MobileNet高分辨率: +{stats['mobilenet_highres_improvement']:.3f} dB ({stats['mobilenet_highres_tiles']:.1f} tiles, {stats['mobilenet_highres_tiles']/total_tiles:.1%})")
                print(f"  MobileNet低分辨率: +{stats['mobilenet_lowres_improvement']:.3f} dB ({stats['mobilenet_lowres_tiles']:.1f} tiles, {stats['mobilenet_lowres_tiles']/total_tiles:.1%})")
        
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
            
            if 'MobileNet高分辨率' in avg_results:
                f.write(f"MobileNet高分辨率方法: +{avg_results['MobileNet高分辨率']:.3f} dB\n")
                f.write(f"MobileNet低分辨率方法: +{avg_results['MobileNet低分辨率']:.3f} dB\n")
            
            # 添加替换比例信息
            avg_total_tiles = df['total_tiles'].mean()
            f.write(f"\n替换比例统计(平均总tiles: {avg_total_tiles:.0f}):\n")
            f.write(f"随机替换: {df['random_max_tiles'].mean()/avg_total_tiles:.1%}\n")
            f.write(f"Canny高分辨率: {df['canny_highres_tiles'].mean()/avg_total_tiles:.1%}\n")
            f.write(f"Canny低分辨率: {df['canny_lowres_tiles'].mean()/avg_total_tiles:.1%}\n")
            
            if 'mobilenet_highres_tiles' in df.columns:
                f.write(f"MobileNet高分辨率: {df['mobilenet_highres_tiles'].mean()/avg_total_tiles:.1%}\n")
                f.write(f"MobileNet低分辨率: {df['mobilenet_lowres_tiles'].mean()/avg_total_tiles:.1%}\n")
            
            f.write("\n关键结论:\n")
            canny_hr_vs_random = avg_results['Canny高分辨率'] / avg_results['随机最大tiles']
            canny_lr_vs_random = avg_results['Canny低分辨率'] / avg_results['随机最大tiles']
            f.write(f"Canny高分辨率 vs 随机: {canny_hr_vs_random:.2f}倍效果\n")
            f.write(f"Canny低分辨率 vs 随机: {canny_lr_vs_random:.2f}倍效果\n")
            
            if 'MobileNet高分辨率' in avg_results:
                mobilenet_hr_vs_random = avg_results['MobileNet高分辨率'] / avg_results['随机最大tiles']
                mobilenet_lr_vs_random = avg_results['MobileNet低分辨率'] / avg_results['随机最大tiles']
                f.write(f"MobileNet高分辨率 vs 随机: {mobilenet_hr_vs_random:.2f}倍效果\n")
                f.write(f"MobileNet低分辨率 vs 随机: {mobilenet_lr_vs_random:.2f}倍效果\n")
                
                mobilenet_hr_vs_canny_hr = avg_results['MobileNet高分辨率'] / avg_results['Canny高分辨率']
                mobilenet_lr_vs_canny_lr = avg_results['MobileNet低分辨率'] / avg_results['Canny低分辨率']
                f.write(f"MobileNet高分辨率 vs Canny高分辨率: {mobilenet_hr_vs_canny_hr:.2f}倍效果\n")
                f.write(f"MobileNet低分辨率 vs Canny低分辨率: {mobilenet_lr_vs_canny_lr:.2f}倍效果\n")
            
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
        
        # 如果有MobileNetV2结果，添加到对比中
        if self.mobilenet_predictor is not None and 'mobilenet_highres_improvement' in df.columns:
            methods.extend(['MobileNet High Resolution', 'MobileNet Low Resolution'])
            improvements.extend([
                df['mobilenet_highres_improvement'].mean(),
                df['mobilenet_lowres_improvement'].mean()
            ])
        
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
        
        if self.mobilenet_predictor is not None and 'mobilenet_highres_improvement' in df.columns:
            scene_columns.update({
                'mobilenet_highres_improvement': 'mean',
                'mobilenet_lowres_improvement': 'mean'
            })
        
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
        
        if 'mobilenet_highres_improvement' in scene_stats.columns:
            bars.append(ax2.bar(x + width, scene_stats['mobilenet_highres_improvement'], width,
                    label='MobileNet High Resolution', color='green', alpha=0.7))
            bars.append(ax2.bar(x + 2*width, scene_stats['mobilenet_lowres_improvement'], width,
                    label='MobileNet Low Resolution', color='orange', alpha=0.7))
        
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
        
        if self.mobilenet_predictor is not None and 'mobilenet_highres_tiles' in df.columns:
            tile_methods.extend(['MobileNet High Res', 'MobileNet Low Res'])
            avg_tiles.extend([df['mobilenet_highres_tiles'].mean(), df['mobilenet_lowres_tiles'].mean()])
            colors.extend(['green', 'orange'])
        
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
        
        if self.mobilenet_predictor is not None and 'mobilenet_highres_improvement' in df.columns:
            ax4.hist(df['mobilenet_highres_improvement'], bins=20, alpha=0.7,
                    label='MobileNet High Resolution', color='green')
            ax4.hist(df['mobilenet_lowres_improvement'], bins=20, alpha=0.7,
                    label='MobileNet Low Resolution', color='orange')
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
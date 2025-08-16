#!/usr/bin/env python3
"""
MobileNetV2阈值校准脚本
基于现有LLFF数据校准MobileNetV2的最优阈值，使其与Canny方法的替换率接近
"""

import sys
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
import matplotlib.pyplot as plt

# 添加当前脚本目录到路径
sys.path.append('/home/ytanaz/access/IBRNet/test_sr/scripts')

class MobileNetV2ThresholdCalibrator:
    """MobileNetV2阈值校准器"""
    
    def __init__(self, llff_test_root="/home/ytanaz/access/IBRNet/eval/llff_test"):
        self.llff_test_root = Path(llff_test_root)
        self.eval_llff_path = self.llff_test_root / "eval_llff"
        self.eval_llff_sr_path = self.llff_test_root / "eval_llff_sr"
        self.tile_size = 32
        self.fine_tile_size = 16
        
        # Canny基准阈值
        self.canny_threshold = 0.120
        self.canny_threshold_lowres = 0.200
        
        # 初始化MobileNetV2
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mobilenet_predictor = self._init_mobilenet_predictor()
        
        print(f"🎯 MobileNetV2阈值校准器初始化")
        print(f"计算设备: {self.device}")
    
    def _init_mobilenet_predictor(self):
        """初始化MobileNetV2预测器"""
        try:
            mobilenet = models.mobilenet_v2(pretrained=True)
            feature_extractor = nn.Sequential(
                *list(mobilenet.features.children())[:4]
            ).to(self.device)
            
            feature_extractor.eval()
            for param in feature_extractor.parameters():
                param.requires_grad = False
            
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            
            return {
                'feature_extractor': feature_extractor,
                'normalize': normalize
            }
        except Exception as e:
            print(f"❌ MobileNetV2初始化失败: {e}")
            return None
    
    def extract_mobilenet_score(self, tile):
        """提取MobileNetV2复杂度得分"""
        if self.mobilenet_predictor is None:
            return 0.0
            
        try:
            # 预处理tile
            if tile.max() > 1.0:
                tile = tile.astype(np.float32) / 255.0
            
            # 转换为torch tensor
            if len(tile.shape) == 3:
                tensor = torch.from_numpy(tile).permute(2, 0, 1)
            else:
                tensor = torch.from_numpy(tile).unsqueeze(0)
                tensor = tensor.repeat(3, 1, 1)
            
            # 标准化
            tensor = self.mobilenet_predictor['normalize'](tensor)
            tensor = tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
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
    
    def extract_canny_score(self, tile):
        """提取Canny边缘得分"""
        if tile.max() > 1.0:
            tile = tile.astype(np.float32) / 255.0
            
        if len(tile.shape) == 3:
            gray = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (tile * 255).astype(np.uint8)
            
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        return edge_ratio
    
    def collect_sample_tiles(self, max_tiles=500):
        """从LLFF数据集收集样本tiles"""
        print("📥 收集样本tiles...")
        
        sr_tiles = []
        fine_tiles = []
        canny_highres_labels = []
        canny_lowres_labels = []
        mobilenet_highres_scores = []
        mobilenet_lowres_scores = []
        
        scenes = ["fortress", "horns", "room"]  # 根据您的结果选择这些场景
        
        for scene in scenes:
            scene_sr_path = self.eval_llff_sr_path / scene
            scene_org_path = self.eval_llff_path / scene
            
            if not scene_sr_path.exists() or not scene_org_path.exists():
                continue
                
            print(f"  处理场景: {scene}")
            
            # 获取图像文件
            for img_file in scene_sr_path.glob("*_pred_sr.png"):
                if len(sr_tiles) >= max_tiles:
                    break
                    
                img_idx = img_file.stem.split('_')[0]
                
                # 加载SR和fine图像
                sr_path = scene_sr_path / f"{img_idx}_pred_sr.png"
                fine_path = scene_sr_path / f"{img_idx}_pred_fine.png"
                
                if not sr_path.exists() or not fine_path.exists():
                    continue
                
                try:
                    sr_img = np.array(Image.open(sr_path))
                    fine_img = np.array(Image.open(fine_path))
                    
                    # 提取tiles
                    h, w = sr_img.shape[:2]
                    tile_h, tile_w = h // self.tile_size, w // self.tile_size
                    
                    for i in range(tile_h):
                        for j in range(tile_w):
                            if len(sr_tiles) >= max_tiles:
                                break
                                
                            # SR tiles (32x32)
                            h_start, h_end = i * self.tile_size, (i + 1) * self.tile_size
                            w_start, w_end = j * self.tile_size, (j + 1) * self.tile_size
                            sr_tile = sr_img[h_start:h_end, w_start:w_end]
                            
                            # Fine tiles (16x16)
                            h_fine, w_fine = fine_img.shape[:2]
                            fine_tile_h, fine_tile_w = h_fine // self.fine_tile_size, w_fine // self.fine_tile_size
                            
                            if i < fine_tile_h and j < fine_tile_w:
                                fine_h_start = i * self.fine_tile_size
                                fine_h_end = (i + 1) * self.fine_tile_size
                                fine_w_start = j * self.fine_tile_size
                                fine_w_end = (j + 1) * self.fine_tile_size
                                fine_tile = fine_img[fine_h_start:fine_h_end, fine_w_start:fine_w_end]
                                
                                # 计算Canny标签
                                canny_hr_score = self.extract_canny_score(sr_tile)
                                canny_lr_score = self.extract_canny_score(fine_tile)
                                
                                canny_hr_label = canny_hr_score > self.canny_threshold
                                canny_lr_label = canny_lr_score > self.canny_threshold_lowres
                                
                                # 计算MobileNet得分
                                mobilenet_hr_score = self.extract_mobilenet_score(sr_tile)
                                mobilenet_lr_score = self.extract_mobilenet_score(fine_tile)
                                
                                # 保存数据
                                sr_tiles.append(sr_tile)
                                fine_tiles.append(fine_tile)
                                canny_highres_labels.append(canny_hr_label)
                                canny_lowres_labels.append(canny_lr_label)
                                mobilenet_highres_scores.append(mobilenet_hr_score)
                                mobilenet_lowres_scores.append(mobilenet_lr_score)
                    
                except Exception as e:
                    print(f"    跳过图像 {img_idx}: {e}")
                    continue
        
        print(f"✅ 收集完成，共 {len(sr_tiles)} 个tiles")
        
        return {
            'sr_tiles': sr_tiles,
            'fine_tiles': fine_tiles,
            'canny_highres_labels': np.array(canny_highres_labels),
            'canny_lowres_labels': np.array(canny_lowres_labels),
            'mobilenet_highres_scores': np.array(mobilenet_highres_scores),
            'mobilenet_lowres_scores': np.array(mobilenet_lowres_scores)
        }
    
    def find_optimal_thresholds(self, data):
        """寻找最优阈值"""
        print("\n🎯 寻找最优阈值...")
        
        # 计算Canny的阳性率作为目标
        canny_hr_positive_rate = np.mean(data['canny_highres_labels'])
        canny_lr_positive_rate = np.mean(data['canny_lowres_labels'])
        
        print(f"Canny高分辨率阳性率: {canny_hr_positive_rate:.3f}")
        print(f"Canny低分辨率阳性率: {canny_lr_positive_rate:.3f}")
        
        # 为MobileNet高分辨率寻找阈值
        mobilenet_hr_scores = data['mobilenet_highres_scores']
        mobilenet_lr_scores = data['mobilenet_lowres_scores']
        
        # 尝试不同阈值
        thresholds = np.arange(0.1, 5.0, 0.1)
        
        best_hr_threshold = None
        best_hr_diff = float('inf')
        best_lr_threshold = None
        best_lr_diff = float('inf')
        
        hr_results = []
        lr_results = []
        
        for threshold in thresholds:
            # 高分辨率
            hr_predictions = mobilenet_hr_scores > threshold
            hr_positive_rate = np.mean(hr_predictions)
            hr_diff = abs(hr_positive_rate - canny_hr_positive_rate)
            
            if hr_diff < best_hr_diff:
                best_hr_diff = hr_diff
                best_hr_threshold = threshold
            
            hr_results.append({
                'threshold': threshold,
                'positive_rate': hr_positive_rate,
                'diff': hr_diff,
                'agreement': np.mean(hr_predictions == data['canny_highres_labels'])
            })
            
            # 低分辨率
            lr_predictions = mobilenet_lr_scores > threshold
            lr_positive_rate = np.mean(lr_predictions)
            lr_diff = abs(lr_positive_rate - canny_lr_positive_rate)
            
            if lr_diff < best_lr_diff:
                best_lr_diff = lr_diff
                best_lr_threshold = threshold
            
            lr_results.append({
                'threshold': threshold,
                'positive_rate': lr_positive_rate,
                'diff': lr_diff,
                'agreement': np.mean(lr_predictions == data['canny_lowres_labels'])
            })
        
        print(f"\n✅ 最优阈值:")
        print(f"MobileNet高分辨率: {best_hr_threshold:.2f} (阳性率差异: {best_hr_diff:.3f})")
        print(f"MobileNet低分辨率: {best_lr_threshold:.2f} (阳性率差异: {best_lr_diff:.3f})")
        
        return {
            'best_hr_threshold': best_hr_threshold,
            'best_lr_threshold': best_lr_threshold,
            'hr_results': hr_results,
            'lr_results': lr_results
        }
    
    def create_calibration_visualization(self, data, threshold_results):
        """创建校准可视化图表"""
        print("\n📊 生成校准可视化...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 得分分布对比
        ax1.hist(data['mobilenet_highres_scores'], bins=50, alpha=0.7, 
                label='MobileNet High Res', color='blue')
        ax1.axvline(threshold_results['best_hr_threshold'], color='blue', 
                   linestyle='--', label=f'Optimal Threshold: {threshold_results["best_hr_threshold"]:.2f}')
        ax1.set_xlabel('Complexity Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('MobileNet High Resolution Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 低分辨率得分分布
        ax2.hist(data['mobilenet_lowres_scores'], bins=50, alpha=0.7, 
                label='MobileNet Low Res', color='red')
        ax2.axvline(threshold_results['best_lr_threshold'], color='red', 
                   linestyle='--', label=f'Optimal Threshold: {threshold_results["best_lr_threshold"]:.2f}')
        ax2.set_xlabel('Complexity Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('MobileNet Low Resolution Score Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 阈值vs阳性率曲线
        hr_thresholds = [r['threshold'] for r in threshold_results['hr_results']]
        hr_positive_rates = [r['positive_rate'] for r in threshold_results['hr_results']]
        
        ax3.plot(hr_thresholds, hr_positive_rates, 'b-', label='MobileNet High Res')
        ax3.axhline(np.mean(data['canny_highres_labels']), color='blue', 
                   linestyle='--', label='Canny High Res Target')
        ax3.axvline(threshold_results['best_hr_threshold'], color='blue', 
                   linestyle=':', alpha=0.7)
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Positive Rate')
        ax3.set_title('Threshold vs Positive Rate (High Resolution)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 一致性分析
        hr_agreements = [r['agreement'] for r in threshold_results['hr_results']]
        lr_agreements = [r['agreement'] for r in threshold_results['lr_results']]
        
        ax4.plot(hr_thresholds, hr_agreements, 'b-', label='High Resolution')
        ax4.plot(hr_thresholds, lr_agreements, 'r-', label='Low Resolution')
        ax4.axvline(threshold_results['best_hr_threshold'], color='blue', 
                   linestyle=':', alpha=0.7)
        ax4.axvline(threshold_results['best_lr_threshold'], color='red', 
                   linestyle=':', alpha=0.7)
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Agreement with Canny')
        ax4.set_title('Agreement with Canny vs Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = "/home/ytanaz/access/IBRNet/test_sr/results/mobilenet_threshold_calibration.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 校准图表保存到: {output_path}")
        
        return output_path

def main():
    """主函数"""
    print("🎯 MobileNetV2阈值校准")
    print("="*50)
    
    calibrator = MobileNetV2ThresholdCalibrator()
    
    if calibrator.mobilenet_predictor is None:
        print("❌ MobileNetV2不可用，退出校准")
        return
    
    # 收集样本数据
    data = calibrator.collect_sample_tiles(max_tiles=500)
    
    if len(data['sr_tiles']) == 0:
        print("❌ 没有收集到有效数据，退出校准")
        return
    
    # 寻找最优阈值
    threshold_results = calibrator.find_optimal_thresholds(data)
    
    # 创建可视化
    calibrator.create_calibration_visualization(data, threshold_results)
    
    # 输出建议的配置
    print("\n" + "="*50)
    print("🎯 校准完成！建议配置:")
    print("="*50)
    print(f"self.mobilenet_threshold = {threshold_results['best_hr_threshold']:.2f}  # 高分辨率阈值")
    print(f"self.mobilenet_threshold_lowres = {threshold_results['best_lr_threshold']:.2f}  # 低分辨率阈值")
    
    print(f"\n在batch_llff_test.py中更新:")
    print(f"- 将 self.mobilenet_threshold = 2.0 改为 {threshold_results['best_hr_threshold']:.2f}")
    print(f"- 添加 self.mobilenet_threshold_lowres = {threshold_results['best_lr_threshold']:.2f}")
    print(f"- 在mobilenet_lowres_method中使用 self.mobilenet_threshold_lowres")

if __name__ == "__main__":
    main()

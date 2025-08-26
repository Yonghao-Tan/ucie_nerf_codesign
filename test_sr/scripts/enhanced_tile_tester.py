#!/usr/bin/env python3
"""
集成NeRF-aware质量预测器到现有测试框架
扩展batch_llff_test.py以支持新的创新方法
"""
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

# 导入我们的新预测器
from nerf_hybrid_predictor import NeRFAwareTileReplacementSystem, TileRenderingData
from manual_canny import manual_canny

class EnhancedTileReplacementTester:
    """增强的Tile替换测试器 - 集成多种创新方法"""

    def __init__(self, llff_test_root="/home/ytanaz/access/IBRNet/eval/llff_test"):
        self.llff_test_root = Path(llff_test_root)
        self.eval_llff_path = self.llff_test_root / "eval_llff_golden"
        self.eval_llff_sr_path = self.llff_test_root / "eval_llff_sr"
        self.tile_size = 32
        self.fine_tile_size = 16
        
        # 创建不同的预测器
        self.nerf_system = NeRFAwareTileReplacementSystem(self.tile_size)
        
        # 方法配置
        self.methods = {
            'canny_highres': {
                'name': 'Canny高分辨率',
                'threshold': 0.160,
                'function': self.extract_edge_score_canny
            },
            'canny_lowres': {
                'name': 'Canny低分辨率',
                'threshold': 0.250,
                'function': self.extract_edge_score_canny_lowres
            },
            'nerf_informed': {
                'name': 'NeRF感知方法',
                'threshold': 0.5,  # 将动态调整
                'function': self.extract_nerf_informed_score
            },
            'hybrid_adaptive': {
                'name': '混合自适应方法',
                'threshold': 0.5,  # 将动态调整
                'function': self.extract_hybrid_adaptive_score
            }
        }
        
        print(f"🚀 增强测试器初始化完成")
        print(f"支持方法: {list(self.methods.keys())}")
    
    def get_scenes(self):
        """获取所有测试场景"""
        scenes = []
        if self.eval_llff_path.exists():
            for scene_dir in self.eval_llff_path.iterdir():
                if scene_dir.is_dir() and not scene_dir.name.startswith('.'):
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
        """标准Canny边缘检测"""
        if tile.max() > 1.0:
            tile = tile.astype(np.float32) / 255.0
            
        gray = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return float(np.sum(edges > 0) / edges.size)
    
    def extract_edge_score_canny_lowres(self, tile):
        """低分辨率Canny边缘检测"""
        # 模拟低分辨率分析过程
        h, w = tile.shape[:2]
        
        # 下采样到fine分辨率
        fine_tile = cv2.resize(tile, (self.fine_tile_size, self.fine_tile_size))
        
        # 在低分辨率上做边缘检测
        if fine_tile.max() > 1.0:
            fine_tile = fine_tile.astype(np.float32) / 255.0
        
        gray = cv2.cvtColor((fine_tile * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        return float(np.sum(edges > 0) / edges.size)
    
    def extract_nerf_informed_score(self, tile_rgb, render_data=None):
        """NeRF感知得分提取"""
        if render_data is None:
            # 如果没有真实的NeRF数据，使用模拟数据进行演示
            render_data = self._simulate_render_data(tile_rgb.shape[:2])
        
        # 构建tile数据
        tile_data = TileRenderingData(
            rgb=tile_rgb,
            depth=render_data.get('depth'),
            weights=render_data.get('weights'),
            alpha=render_data.get('alpha'),
            mask=render_data.get('mask')
        )
        
        # 使用NeRF预测器计算得分
        scores = self.nerf_system.predictor.compute_tile_scores(tile_data)
        return scores['final_score']
    
    def extract_hybrid_adaptive_score(self, tile_rgb, render_data=None, global_stats=None):
        """混合自适应得分提取"""
        if render_data is None:
            render_data = self._simulate_render_data(tile_rgb.shape[:2])
        
        # 1. NeRF特征
        nerf_score = self.extract_nerf_informed_score(tile_rgb, render_data)
        
        # 2. 传统特征
        canny_score = self.extract_edge_score_canny(tile_rgb)
        
        # 3. 纹理复杂度
        gray = cv2.cvtColor((tile_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        texture_score = np.std(gray) / 255.0
        
        # 4. 高频内容
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        high_freq_score = np.var(laplacian) / (255.0**2)
        
        # 加权组合
        hybrid_score = (
            0.4 * nerf_score +
            0.3 * canny_score +
            0.2 * texture_score +
            0.1 * high_freq_score
        )
        
        return float(hybrid_score)
    
    def _simulate_render_data(self, tile_shape):
        """模拟NeRF渲染数据（用于演示）"""
        h, w = tile_shape
        n_samples = 64
        
        # 模拟深度图 - 添加一些几何复杂度
        x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        depth = 2.0 + 0.5 * np.sin(4 * np.pi * x) * np.cos(4 * np.pi * y)
        depth += 0.1 * np.random.randn(h, w)  # 添加噪声
        
        # 模拟权重 - 基于深度变化
        weights = np.random.exponential(1.0, (h, w, n_samples))
        
        # 让权重在深度变化大的地方更分散（不确定性更高）
        depth_grad = np.gradient(depth)
        complexity = np.sqrt(depth_grad[0]**2 + depth_grad[1]**2)
        
        for i in range(n_samples):
            weights[:, :, i] *= (1.0 + 2.0 * complexity)  # 复杂区域权重更分散
        
        # 归一化权重
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-8)
        
        # 模拟alpha值
        alpha = np.random.exponential(0.5, (h, w, n_samples))
        
        # 模拟mask - 大部分区域可见
        mask = (np.random.rand(h, w) > 0.1).astype(float)
        
        return {
            'depth': depth,
            'weights': weights,
            'alpha': alpha,
            'mask': mask
        }
    
    def apply_tile_replacement_method(self, sr_img, org_img, method_name, render_data=None):
        """应用指定的tile替换方法"""
        h, w = sr_img.shape[:2]
        tile_h, tile_w = h // self.tile_size, w // self.tile_size
        
        hybrid_img = sr_img.copy()
        replaced_tiles = 0
        all_scores = []
        
        # 获取方法配置
        method_config = self.methods[method_name]
        threshold = method_config['threshold']
        extract_func = method_config['function']
        
        # 如果是NeRF方法，分析全局统计信息
        global_stats = None
        if 'nerf' in method_name or 'hybrid' in method_name:
            if render_data is None:
                render_data = self._simulate_render_data((h, w))
            global_stats = self._analyze_global_scene(render_data)
            
            # 动态调整阈值
            if 'adaptive' in method_name:
                threshold = self._get_adaptive_threshold(global_stats)
        
        # 逐tile处理
        for i in range(tile_h):
            for j in range(tile_w):
                y_start, y_end = i * self.tile_size, (i + 1) * self.tile_size
                x_start, x_end = j * self.tile_size, (j + 1) * self.tile_size
                
                sr_tile = sr_img[y_start:y_end, x_start:x_end]
                
                # 提取tile级别的渲染数据
                tile_render_data = None
                if render_data is not None:
                    tile_render_data = {}
                    for key, value in render_data.items():
                        if value is not None:
                            if len(value.shape) == 2:
                                tile_render_data[key] = value[y_start:y_end, x_start:x_end]
                            elif len(value.shape) == 3:
                                tile_render_data[key] = value[y_start:y_end, x_start:x_end, :]
                
                # 计算得分
                if method_name in ['nerf_informed', 'hybrid_adaptive']:
                    score = extract_func(sr_tile, tile_render_data, global_stats)
                else:
                    score = extract_func(sr_tile)
                
                all_scores.append(score)
                
                # 替换决策
                if score > threshold:
                    hybrid_img[y_start:y_end, x_start:x_end] = org_img[y_start:y_end, x_start:x_end]
                    replaced_tiles += 1
        
        replacement_ratio = replaced_tiles / (tile_h * tile_w)
        
        return {
            'hybrid_img': hybrid_img,
            'replaced_tiles': replaced_tiles,
            'total_tiles': tile_h * tile_w,
            'replacement_ratio': replacement_ratio,
            'all_scores': all_scores,
            'threshold_used': threshold
        }
    
    def _analyze_global_scene(self, render_data):
        """分析全局场景统计信息"""
        stats = {}
        
        if 'depth' in render_data:
            depth = render_data['depth']
            stats['depth_range'] = np.ptp(depth)
            stats['depth_std'] = np.std(depth)
            stats['mean_depth'] = np.mean(depth)
        
        if 'weights' in render_data:
            weights = render_data['weights']
            weights_norm = weights / (weights.sum(axis=-1, keepdims=True) + 1e-8)
            entropy = -np.sum(weights_norm * np.log(weights_norm + 1e-8), axis=-1)
            stats['global_uncertainty'] = np.mean(entropy)
        
        # 计算复杂度
        complexity_factors = []
        if 'depth_range' in stats and stats['mean_depth'] > 0:
            complexity_factors.append(stats['depth_range'] / stats['mean_depth'])
        if 'global_uncertainty' in stats:
            complexity_factors.append(stats['global_uncertainty'])
        
        stats['complexity'] = np.mean(complexity_factors) if complexity_factors else 0.5
        
        return stats
    
    def _get_adaptive_threshold(self, global_stats):
        """获取自适应阈值"""
        base_threshold = 0.5
        complexity = global_stats.get('complexity', 0.5)
        
        # 复杂场景使用更低阈值
        adaptive_threshold = base_threshold - 0.2 * complexity
        return np.clip(adaptive_threshold, 0.3, 0.7)
    
    def test_single_image_all_methods(self, scene_name, image_index, save_results=True):
        """测试单张图像的所有方法"""
        # 加载图像
        scene_path_golden = self.eval_llff_path / scene_name
        scene_path_sr = self.eval_llff_sr_path / scene_name
        
        gt_path = scene_path_golden / f"{image_index:03d}_gt_rgb.png"
        sr_path = scene_path_sr / f"{image_index:03d}_pred_fine.png"
        
        if not gt_path.exists() or not sr_path.exists():
            return None
        
        # 读取图像
        gt_img = np.array(Image.open(gt_path)).astype(np.float32) / 255.0
        sr_img = np.array(Image.open(sr_path)).astype(np.float32) / 255.0
        
        # 确保尺寸匹配
        if gt_img.shape != sr_img.shape:
            sr_img = cv2.resize(sr_img, (gt_img.shape[1], gt_img.shape[0]))
        
        # 生成渲染数据（实际使用时应该从NeRF渲染器获取）
        render_data = self._simulate_render_data(gt_img.shape[:2])
        
        results = {}
        
        # 测试所有方法
        for method_name in self.methods.keys():
            print(f"  测试方法: {self.methods[method_name]['name']}")
            
            method_result = self.apply_tile_replacement_method(
                sr_img, gt_img, method_name, render_data
            )
            
            # 计算PSNR
            hybrid_psnr = psnr(gt_img, method_result['hybrid_img'])
            sr_psnr = psnr(gt_img, sr_img)
            improvement = hybrid_psnr - sr_psnr
            
            results[method_name] = {
                'hybrid_psnr': hybrid_psnr,
                'sr_psnr': sr_psnr,
                'improvement': improvement,
                'replacement_ratio': method_result['replacement_ratio'],
                'replaced_tiles': method_result['replaced_tiles'],
                'threshold_used': method_result['threshold_used'],
                'hybrid_img': method_result['hybrid_img']
            }
            
            print(f"    PSNR提升: {improvement:.3f}dB, 替换率: {method_result['replacement_ratio']:.1%}")
        
        # 保存结果
        if save_results:
            self._save_comparison_results(scene_name, image_index, results, gt_img, sr_img)
        
        return results
    
    def _save_comparison_results(self, scene_name, image_index, results, gt_img, sr_img):
        """保存对比结果"""
        output_dir = Path(f"enhanced_test_results/{scene_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建对比图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始图像
        axes[0, 0].imshow(gt_img)
        axes[0, 0].set_title('Ground Truth')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(sr_img)
        axes[0, 1].set_title(f'SR (PSNR: {results["canny_highres"]["sr_psnr"]:.2f}dB)')
        axes[0, 1].axis('off')
        
        # 显示最好的方法
        best_method = max(results.keys(), key=lambda k: results[k]['improvement'])
        best_result = results[best_method]
        
        axes[0, 2].imshow(best_result['hybrid_img'])
        axes[0, 2].set_title(f'Best: {self.methods[best_method]["name"]}\n'
                            f'PSNR: {best_result["hybrid_psnr"]:.2f}dB '
                            f'(+{best_result["improvement"]:.2f})')
        axes[0, 2].axis('off')
        
        # 显示其他方法
        method_names = list(results.keys())
        for i, method_name in enumerate(method_names[:3]):
            if i < 3:
                result = results[method_name]
                axes[1, i].imshow(result['hybrid_img'])
                axes[1, i].set_title(f'{self.methods[method_name]["name"]}\n'
                                   f'+{result["improvement"]:.2f}dB, '
                                   f'{result["replacement_ratio"]:.1%} replaced')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{image_index:03d}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存数值结果
        summary = {
            'scene': scene_name,
            'image_index': image_index,
            'gt_psnr_baseline': results[list(results.keys())[0]]['sr_psnr']
        }
        
        for method_name, result in results.items():
            prefix = method_name
            summary[f'{prefix}_psnr'] = result['hybrid_psnr']
            summary[f'{prefix}_improvement'] = result['improvement']
            summary[f'{prefix}_replacement_ratio'] = result['replacement_ratio']
            summary[f'{prefix}_threshold'] = result['threshold_used']
        
        # 保存到CSV
        csv_path = output_dir / "results_summary.csv"
        df = pd.DataFrame([summary])
        
        if csv_path.exists():
            existing_df = pd.read_csv(csv_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(csv_path, index=False)
    
    def test_all_scenes(self, max_images_per_scene=5):
        """测试所有场景"""
        scenes = self.get_scenes()
        all_results = []
        
        print(f"🧪 开始测试，共{len(scenes)}个场景")
        
        for scene_name in scenes:
            print(f"\n📁 测试场景: {scene_name}")
            
            scene_path = self.eval_llff_path / scene_name
            image_indices = self.get_image_indices(scene_path)
            
            # 限制每个场景的图像数量
            test_indices = image_indices[:max_images_per_scene]
            
            for image_index in test_indices:
                print(f"  🖼️  测试图像: {image_index:03d}")
                
                results = self.test_single_image_all_methods(scene_name, image_index)
                
                if results:
                    # 添加场景和图像信息
                    for method_name, result in results.items():
                        result['scene'] = scene_name
                        result['image_index'] = image_index
                        result['method'] = method_name
                    
                    all_results.extend(results.values())
        
        # 生成总结报告
        self._generate_summary_report(all_results)
        
        return all_results
    
    def _generate_summary_report(self, all_results):
        """生成总结报告"""
        if not all_results:
            return
        
        df = pd.DataFrame(all_results)
        
        # 按方法分组统计
        method_stats = df.groupby('method').agg({
            'improvement': ['mean', 'std', 'count'],
            'replacement_ratio': ['mean', 'std'],
            'hybrid_psnr': 'mean'
        }).round(4)
        
        print(f"\n📊 测试总结报告")
        print("=" * 60)
        print(method_stats)
        
        # 保存详细结果
        output_dir = Path("enhanced_test_results")
        output_dir.mkdir(exist_ok=True)
        
        df.to_csv(output_dir / "complete_results.csv", index=False)
        method_stats.to_csv(output_dir / "method_summary.csv")
        
        # 创建对比图表
        self._create_summary_plots(df, output_dir)
        
        print(f"\n✅ 结果已保存到: {output_dir}")
    
    def _create_summary_plots(self, df, output_dir):
        """创建总结图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. PSNR改进对比
        method_improvements = df.groupby('method')['improvement'].mean()
        axes[0, 0].bar(range(len(method_improvements)), method_improvements.values)
        axes[0, 0].set_xticks(range(len(method_improvements)))
        axes[0, 0].set_xticklabels([self.methods[m]['name'] for m in method_improvements.index], 
                                  rotation=45, ha='right')
        axes[0, 0].set_ylabel('Average PSNR Improvement (dB)')
        axes[0, 0].set_title('PSNR Improvement by Method')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 替换率对比
        method_ratios = df.groupby('method')['replacement_ratio'].mean()
        axes[0, 1].bar(range(len(method_ratios)), method_ratios.values)
        axes[0, 1].set_xticks(range(len(method_ratios)))
        axes[0, 1].set_xticklabels([self.methods[m]['name'] for m in method_ratios.index], 
                                  rotation=45, ha='right')
        axes[0, 1].set_ylabel('Average Replacement Ratio')
        axes[0, 1].set_title('Tile Replacement Ratio by Method')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 改进分布
        for method in df['method'].unique():
            method_data = df[df['method'] == method]['improvement']
            axes[1, 0].hist(method_data, alpha=0.7, 
                           label=self.methods[method]['name'], bins=20)
        axes[1, 0].set_xlabel('PSNR Improvement (dB)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of PSNR Improvements')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 效果 vs 替换率散点图
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            axes[1, 1].scatter(method_data['replacement_ratio'], 
                             method_data['improvement'],
                             alpha=0.6, label=self.methods[method]['name'])
        axes[1, 1].set_xlabel('Replacement Ratio')
        axes[1, 1].set_ylabel('PSNR Improvement (dB)')
        axes[1, 1].set_title('Quality vs Replacement Efficiency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "summary_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """主函数 - 运行增强测试"""
    print("🚀 启动增强的NeRF-SR Tile替换测试")
    
    # 创建测试器
    tester = EnhancedTileReplacementTester()
    
    # 测试单个图像（快速验证）
    scenes = tester.get_scenes()
    if scenes:
        print(f"\n🔬 快速测试第一个场景的第一张图像...")
        first_scene = scenes[0]
        scene_path = tester.eval_llff_path / first_scene
        indices = tester.get_image_indices(scene_path)
        
        if indices:
            results = tester.test_single_image_all_methods(first_scene, indices[0])
            if results:
                print(f"\n✅ 快速测试完成！")
                print("各方法PSNR改进:")
                for method, result in results.items():
                    print(f"  {tester.methods[method]['name']}: "
                          f"+{result['improvement']:.3f}dB "
                          f"(替换率: {result['replacement_ratio']:.1%})")
        
        # 询问是否继续完整测试
        response = input(f"\n是否继续完整测试所有场景? [y/N]: ").strip().lower()
        if response == 'y':
            print(f"\n🧪 开始完整测试...")
            all_results = tester.test_all_scenes(max_images_per_scene=3)
            print(f"\n🎉 完整测试完成！共处理{len(all_results)}个结果")
    else:
        print("❌ 未找到测试场景，请检查路径设置")

if __name__ == "__main__":
    main()

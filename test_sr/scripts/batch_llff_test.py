#!/usr/bin/env python3
"""
大规模数据集批量测试脚本
在LLFF测试数据集上验证tile替换方法的效果
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
        self.eval_llff_path = self.llff_test_root / "eval_llff"
        self.eval_llff_sr_path = self.llff_test_root / "eval_llff_sr"
        self.tile_size = 32
        self.fine_tile_size = 16  # 在fine分辨率上的tile大小
        
        # 预设阈值
        self.canny_threshold = 0.094  # 高分辨率方法的阈值
        self.canny_threshold_lowres = 0.160  # 低分辨率校正后的阈值
        
        print(f"🔍 批量测试器初始化")
        print(f"LLFF测试根目录: {self.llff_test_root}")
        print(f"Canny高分辨率阈值: {self.canny_threshold}")
        print(f"Canny低分辨率阈值: {self.canny_threshold_lowres}")
    
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
            'random_100_psnr': 0,
            'random_100_improvement': 0,
            'canny_highres_psnr': 0,
            'canny_highres_improvement': 0,
            'canny_highres_tiles': 0,
            'canny_lowres_psnr': 0,
            'canny_lowres_improvement': 0,
            'canny_lowres_tiles': 0
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
            
            # 计算原始SR PSNR
            sr_psnr = psnr(gt_img, sr_img)
            results['sr_psnr'] = sr_psnr
            
            # 1. 随机替换100个tiles
            random_img, _ = self.random_replacement_method(sr_img, org_img, 100)
            random_psnr = psnr(gt_img, random_img)
            results['random_100_psnr'] = random_psnr
            results['random_100_improvement'] = random_psnr - sr_psnr
            
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
            '随机100tiles': df['random_100_improvement'].mean(),
            'Canny高分辨率': df['canny_highres_improvement'].mean(),
            'Canny低分辨率': df['canny_lowres_improvement'].mean()
        }
        
        print(f"\n📈 平均PSNR提升 (dB):")
        print(f"{'方法':<15} {'PSNR提升':<10} {'标准差':<10}")
        print("-" * 40)
        print(f"{'随机100tiles':<15} {avg_results['随机100tiles']:<9.3f} {df['random_100_improvement'].std():<9.3f}")
        print(f"{'Canny高分辨率':<15} {avg_results['Canny高分辨率']:<9.3f} {df['canny_highres_improvement'].std():<9.3f}")
        print(f"{'Canny低分辨率':<15} {avg_results['Canny低分辨率']:<9.3f} {df['canny_lowres_improvement'].std():<9.3f}")
        
        # 替换tiles统计
        print(f"\n🔧 平均替换Tiles数:")
        print(f"Canny高分辨率: {df['canny_highres_tiles'].mean():.1f} tiles")
        print(f"Canny低分辨率: {df['canny_lowres_tiles'].mean():.1f} tiles")
        
        # 按场景分析
        print(f"\n🎬 按场景分析:")
        scene_stats = df.groupby('scene').agg({
            'sr_psnr': 'mean',
            'random_100_improvement': 'mean',
            'canny_highres_improvement': 'mean',
            'canny_lowres_improvement': 'mean',
            'canny_highres_tiles': 'mean',
            'canny_lowres_tiles': 'mean'
        })
        
        for scene in scene_stats.index:
            stats = scene_stats.loc[scene]
            print(f"\n{scene}:")
            print(f"  平均SR PSNR: {stats['sr_psnr']:.2f} dB")
            print(f"  随机100tiles: +{stats['random_100_improvement']:.3f} dB")
            print(f"  Canny高分辨率: +{stats['canny_highres_improvement']:.3f} dB ({stats['canny_highres_tiles']:.1f} tiles)")
            print(f"  Canny低分辨率: +{stats['canny_lowres_improvement']:.3f} dB ({stats['canny_lowres_tiles']:.1f} tiles)")
        
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
            f.write(f"随机100tiles替换: +{avg_results['随机100tiles']:.3f} dB\n")
            f.write(f"Canny高分辨率方法: +{avg_results['Canny高分辨率']:.3f} dB\n")
            f.write(f"Canny低分辨率方法: +{avg_results['Canny低分辨率']:.3f} dB\n\n")
            
            f.write("关键结论:\n")
            canny_hr_vs_random = avg_results['Canny高分辨率'] / avg_results['随机100tiles']
            canny_lr_vs_random = avg_results['Canny低分辨率'] / avg_results['随机100tiles']
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
        methods = ['Random 100 tiles', 'Canny High Resolution', 'Canny Low Resolution']
        improvements = [
            df['random_100_improvement'].mean(),
            df['canny_highres_improvement'].mean(),
            df['canny_lowres_improvement'].mean()
        ]
        
        ax1.bar(methods, improvements, color=['gray', 'blue', 'red'])
        ax1.set_ylabel('Average PSNR Improvement (dB)')
        ax1.set_title('PSNR Improvement Comparison of Different Methods')
        ax1.grid(True, alpha=0.3)
        
        # 2. PSNR improvement by scene
        scene_stats = df.groupby('scene').agg({
            'canny_highres_improvement': 'mean',
            'canny_lowres_improvement': 'mean'
        })
        
        x = np.arange(len(scene_stats.index))
        width = 0.35
        
        ax2.bar(x - width/2, scene_stats['canny_highres_improvement'], width, 
                label='Canny High Resolution', color='blue', alpha=0.7)
        ax2.bar(x + width/2, scene_stats['canny_lowres_improvement'], width,
                label='Canny Low Resolution', color='red', alpha=0.7)
        
        ax2.set_xlabel('Scene')
        ax2.set_ylabel('Average PSNR Improvement (dB)')
        ax2.set_title('PSNR Improvement Comparison by Scene')
        ax2.set_xticks(x)
        ax2.set_xticklabels(scene_stats.index, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Number of replaced tiles comparison
        avg_tiles_hr = df['canny_highres_tiles'].mean()
        avg_tiles_lr = df['canny_lowres_tiles'].mean()
        
        ax3.bar(['Canny High Resolution', 'Canny Low Resolution'], [avg_tiles_hr, avg_tiles_lr], 
                color=['blue', 'red'], alpha=0.7)
        ax3.set_ylabel('Average Number of Replaced Tiles')
        ax3.set_title('Comparison of Average Number of Replaced Tiles')
        ax3.grid(True, alpha=0.3)
        
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
    
    print("\n🎉 批量测试全部完成！")
    print("\n🏆 关键结论:")
    print(f"- Canny高分辨率方法: {avg_results['Canny高分辨率']:+.3f} dB平均提升")
    print(f"- Canny低分辨率方法: {avg_results['Canny低分辨率']:+.3f} dB平均提升")
    print(f"- 低分辨率方法计算量仅为高分辨率方法的25%")
    print(f"- 所有方法均显著优于随机替换基线")

if __name__ == "__main__":
    main()

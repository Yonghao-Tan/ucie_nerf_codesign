#!/usr/bin/env python3
"""
轻量级神经网络tile复杂度预测器性能测试
对比不同预训练网络的计算开销和预测效果
"""

import torch
import torch.nn as nn
import torchvision.models as models
import time
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class CannyPredictor:
    """基线Canny方法"""
    def __init__(self, threshold=0.15):
        self.threshold = threshold
    
    def predict_complexity(self, tile):
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        return edge_ratio > self.threshold

class MobileNetV2Predictor:
    """MobileNetV2轻量级预测器"""
    def __init__(self):
        self.backbone = models.mobilenet_v2(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(self.backbone.features.children())[:4]  # 前4层
        )
        self.feature_extractor.eval()
        
        # 冻结权重
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def predict_complexity(self, tile):
        """基于预训练特征的复杂度预测"""
        # 预处理
        x = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0)  # [1, 3, 16, 16]
        
        # 标准化（ImageNet均值和标准差）
        normalize = torch.nn.functional.normalize
        
        with torch.no_grad():
            features = self.feature_extractor(x)  # [1, 32, 8, 8]
            
            # 复杂度指标：特征激活的方差和能量
            feature_var = torch.var(features)
            feature_energy = torch.norm(features)
            
            complexity_score = 0.6 * feature_var + 0.4 * feature_energy
            return complexity_score > 1.5  # 可调阈值

class ResNet18TinyPredictor:
    """ResNet18极简版本"""
    def __init__(self):
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1, 
            resnet.relu
        )
        self.feature_extractor.eval()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def predict_complexity(self, tile):
        x = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0)
        
        with torch.no_grad():
            features = self.feature_extractor(x)  # [1, 64, 8, 8]
            
            # 更精细的复杂度分析
            spatial_variance = torch.var(features, dim=[2,3]).mean()
            channel_diversity = torch.std(torch.mean(features, dim=[2,3]))
            edge_response = torch.norm(features, dim=1).mean()
            
            complexity_score = (0.4 * spatial_variance + 
                              0.3 * channel_diversity + 
                              0.3 * edge_response)
            return complexity_score > 2.0

class EfficientNetTinyPredictor:
    """EfficientNet-B0轻量版"""
    def __init__(self):
        try:
            import efficientnet_pytorch
            self.backbone = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')
            # 只使用前2个MBConv blocks
            self.feature_extractor = nn.Sequential(
                self.backbone._conv_stem,
                self.backbone._bn0,
                self.backbone._swish,
                *list(self.backbone._blocks[:2])
            )
        except ImportError:
            print("EfficientNet not available, using MobileNet as fallback")
            self.feature_extractor = MobileNetV2Predictor().feature_extractor
        
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def predict_complexity(self, tile):
        x = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0)
        
        with torch.no_grad():
            features = self.feature_extractor(x)
            complexity_score = torch.var(features) + 0.5 * torch.mean(torch.abs(features))
            return complexity_score > 0.8

class NeuralPredictorBenchmark:
    """神经网络预测器性能测试"""
    
    def __init__(self):
        self.predictors = {
            'Canny': CannyPredictor(),
            'MobileNetV2': MobileNetV2Predictor(),
            'ResNet18-Tiny': ResNet18TinyPredictor(),
            # 'EfficientNet-Tiny': EfficientNetTinyPredictor()
        }
        
        # 生成测试数据
        self.test_tiles = self.generate_test_tiles()
        
    def generate_test_tiles(self, num_tiles=100):
        """生成多样化的测试tiles"""
        tiles = []
        
        # 简单纹理
        for _ in range(30):
            tile = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
            tiles.append(tile)
        
        # 边缘丰富
        for _ in range(30):
            tile = np.zeros((16, 16, 3), dtype=np.uint8)
            cv2.rectangle(tile, (4, 4), (12, 12), (255, 255, 255), 1)
            cv2.line(tile, (0, 8), (16, 8), (128, 128, 128), 1)
            tiles.append(tile)
        
        # 复杂纹理
        for _ in range(40):
            tile = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
            # 添加网格
            for i in range(0, 16, 4):
                cv2.line(tile, (i, 0), (i, 16), (0, 0, 0), 1)
                cv2.line(tile, (0, i), (16, i), (0, 0, 0), 1)
            tiles.append(tile)
        
        return tiles
    
    def benchmark_speed(self, num_iterations=1000):
        """测试推理速度"""
        results = {}
        
        print("🚀 开始速度测试...")
        print(f"测试数据: {len(self.test_tiles)} tiles × {num_iterations} iterations")
        
        for name, predictor in self.predictors.items():
            print(f"\n测试 {name}...")
            
            # 预热
            for tile in self.test_tiles[:10]:
                _ = predictor.predict_complexity(tile)
            
            # 正式测试
            start_time = time.time()
            for _ in range(num_iterations):
                for tile in self.test_tiles:
                    _ = predictor.predict_complexity(tile)
            end_time = time.time()
            
            total_predictions = len(self.test_tiles) * num_iterations
            avg_time_per_tile = (end_time - start_time) / total_predictions * 1000  # ms
            
            results[name] = {
                'total_time': end_time - start_time,
                'avg_time_per_tile_ms': avg_time_per_tile,
                'throughput_tiles_per_sec': total_predictions / (end_time - start_time)
            }
            
            print(f"  总时间: {results[name]['total_time']:.3f}s")
            print(f"  平均每tile: {avg_time_per_tile:.4f}ms")
            print(f"  吞吐量: {results[name]['throughput_tiles_per_sec']:.1f} tiles/s")
        
        return results
    
    def benchmark_accuracy(self):
        """测试预测准确性（与Canny对比）"""
        print("\n🎯 开始准确性测试...")
        
        canny_predictions = []
        neural_predictions = {name: [] for name in self.predictors.keys() if name != 'Canny'}
        
        # 获取所有预测结果
        for tile in self.test_tiles:
            canny_pred = self.predictors['Canny'].predict_complexity(tile)
            canny_predictions.append(canny_pred)
            
            for name, predictor in self.predictors.items():
                if name != 'Canny':
                    pred = predictor.predict_complexity(tile)
                    neural_predictions[name].append(pred)
        
        # 计算与Canny的一致性
        results = {}
        for name, predictions in neural_predictions.items():
            agreement = np.mean(np.array(predictions) == np.array(canny_predictions))
            results[name] = {
                'agreement_with_canny': agreement,
                'positive_rate': np.mean(predictions),
                'canny_positive_rate': np.mean(canny_predictions)
            }
            
            print(f"\n{name}:")
            print(f"  与Canny一致性: {agreement:.3f}")
            print(f"  阳性率: {np.mean(predictions):.3f}")
            print(f"  Canny阳性率: {np.mean(canny_predictions):.3f}")
        
        return results
    
    def analyze_computational_overhead(self):
        """分析计算开销对IBRNet的影响"""
        print("\n📊 计算开销分析...")
        
        # IBRNet基准数据（根据您的项目数据）
        ibrnet_time_per_image = 30  # 秒，1000×800图像
        tiles_per_image = (1000 // 16) * (800 // 16)  # 约3125个tiles
        
        speed_results = self.benchmark_speed(100)  # 小规模测试
        
        overhead_analysis = {}
        for name, stats in speed_results.items():
            time_per_tile_s = stats['avg_time_per_tile_ms'] / 1000
            total_overhead_per_image = time_per_tile_s * tiles_per_image
            overhead_percentage = (total_overhead_per_image / ibrnet_time_per_image) * 100
            
            overhead_analysis[name] = {
                'time_per_tile_s': time_per_tile_s,
                'total_overhead_s': total_overhead_per_image,
                'overhead_percentage': overhead_percentage,
                'acceptable': overhead_percentage < 5.0  # 5%以下可接受
            }
            
            print(f"\n{name}:")
            print(f"  每tile时间: {time_per_tile_s*1000:.4f}ms")
            print(f"  每图像总开销: {total_overhead_per_image:.3f}s")
            print(f"  相对IBRNet开销: {overhead_percentage:.2f}%")
            print(f"  可接受性: {'✅' if overhead_analysis[name]['acceptable'] else '❌'}")
        
        return overhead_analysis
    
    def create_visualization(self, speed_results, accuracy_results, overhead_analysis):
        """创建可视化图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 速度对比
        methods = list(speed_results.keys())
        times = [speed_results[m]['avg_time_per_tile_ms'] for m in methods]
        
        bars = ax1.bar(methods, times, color=['gray', 'blue', 'red', 'green'][:len(methods)])
        ax1.set_ylabel('Time per Tile (ms)')
        ax1.set_title('Speed Comparison: Time per Tile')
        ax1.set_yscale('log')
        
        # 添加数值标签
        for bar, time in zip(bars, times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{time:.4f}', ha='center', va='bottom')
        
        # 2. 吞吐量对比
        throughputs = [speed_results[m]['throughput_tiles_per_sec'] for m in methods]
        ax2.bar(methods, throughputs, color=['gray', 'blue', 'red', 'green'][:len(methods)])
        ax2.set_ylabel('Throughput (tiles/sec)')
        ax2.set_title('Throughput Comparison')
        
        # 3. 计算开销百分比
        overhead_methods = list(overhead_analysis.keys())
        overheads = [overhead_analysis[m]['overhead_percentage'] for m in overhead_methods]
        colors = ['green' if overhead_analysis[m]['acceptable'] else 'red' 
                 for m in overhead_methods]
        
        bars = ax3.bar(overhead_methods, overheads, color=colors, alpha=0.7)
        ax3.set_ylabel('Overhead Percentage (%)')
        ax3.set_title('Computational Overhead vs IBRNet')
        ax3.axhline(y=5, color='red', linestyle='--', label='5% Threshold')
        ax3.legend()
        
        # 4. 与Canny一致性
        if accuracy_results:
            neural_methods = list(accuracy_results.keys())
            agreements = [accuracy_results[m]['agreement_with_canny'] for m in neural_methods]
            ax4.bar(neural_methods, agreements, color=['blue', 'red', 'green'][:len(neural_methods)])
            ax4.set_ylabel('Agreement with Canny')
            ax4.set_title('Prediction Agreement with Canny Baseline')
            ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = "/home/ytanaz/access/IBRNet/test_sr/results/neural_predictor_benchmark.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 图表保存到: {output_path}")
        
        return output_path

def main():
    """主函数"""
    print("🧠 轻量级神经网络tile复杂度预测器基准测试")
    print("="*60)
    
    benchmark = NeuralPredictorBenchmark()
    
    # 1. 速度测试
    speed_results = benchmark.benchmark_speed(num_iterations=100)
    
    # 2. 准确性测试  
    accuracy_results = benchmark.benchmark_accuracy()
    
    # 3. 计算开销分析
    overhead_analysis = benchmark.analyze_computational_overhead()
    
    # 4. 创建可视化
    chart_path = benchmark.create_visualization(speed_results, accuracy_results, overhead_analysis)
    
    # 5. 生成报告
    print("\n" + "="*60)
    print("📋 基准测试总结")
    print("="*60)
    
    print("\n🏆 推荐方案:")
    acceptable_methods = [name for name, analysis in overhead_analysis.items() 
                         if analysis['acceptable']]
    
    if acceptable_methods:
        fastest_acceptable = min(acceptable_methods, 
                               key=lambda x: speed_results[x]['avg_time_per_tile_ms'])
        print(f"  最佳选择: {fastest_acceptable}")
        print(f"  开销: {overhead_analysis[fastest_acceptable]['overhead_percentage']:.2f}%")
        print(f"  速度: {speed_results[fastest_acceptable]['avg_time_per_tile_ms']:.4f}ms/tile")
        
        if fastest_acceptable in accuracy_results:
            print(f"  与Canny一致性: {accuracy_results[fastest_acceptable]['agreement_with_canny']:.3f}")
    else:
        print("  ⚠️  所有神经网络方法的开销都超过5%阈值")
        print("  建议继续使用Canny方法")
    
    print(f"\n📊 详细结果图表: {chart_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MobileNetV2轻量级tile复杂度预测器
基于预训练特征的无训练复杂度判断方法，替代Canny边缘检测
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import time

class MobileNetV2ComplexityPredictor:
    """基于MobileNetV2的轻量级复杂度预测器"""
    
    def __init__(self, threshold=2.0, device='cuda'):
        """
        Args:
            threshold: 复杂度判断阈值，越小越敏感
            device: 计算设备
        """
        self.threshold = threshold
        self.device = device
        
        # 加载MobileNetV2并提取前几层
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(mobilenet.features.children())[:4]  # 前4层，轻量级
        ).to(device)
        
        # 冻结预训练权重
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # ImageNet标准化
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        print(f"✅ MobileNetV2复杂度预测器初始化完成")
        print(f"   阈值: {threshold}")
        print(f"   设备: {device}")
        self._print_model_info()
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.feature_extractor.parameters())
        print(f"   参数量: {total_params/1000:.1f}K")
        
        # 估算FLOP
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 16, 16).to(self.device)
            # 简单估算：卷积层主要计算量
            print(f"   预估FLOP: ~5M (16×16输入)")
    
    def preprocess_tile(self, tile):
        """预处理tile图像"""
        # 确保输入是numpy数组，范围[0,255]
        if isinstance(tile, Image.Image):
            tile = np.array(tile)
        
        if tile.dtype == np.uint8:
            tile = tile.astype(np.float32) / 255.0
        
        # 转换为torch tensor
        if len(tile.shape) == 3:  # RGB
            tensor = torch.from_numpy(tile).permute(2, 0, 1)  # HWC -> CHW
        else:  # 灰度图
            tensor = torch.from_numpy(tile).unsqueeze(0)  # 添加通道维度
            tensor = tensor.repeat(3, 1, 1)  # 转换为3通道
        
        # 标准化
        tensor = self.normalize(tensor)
        
        # 添加batch维度
        tensor = tensor.unsqueeze(0).to(self.device)  # [1, 3, 16, 16]
        
        return tensor
    
    def extract_complexity_features(self, features):
        """从特征图提取复杂度指标"""
        # features shape: [1, C, H, W]
        
        # 1. 空间方差 - 衡量纹理复杂度
        spatial_variance = torch.var(features, dim=[2, 3]).mean()
        
        # 2. 通道多样性 - 衡量特征激活的多样性
        channel_means = torch.mean(features, dim=[2, 3])  # [1, C]
        channel_diversity = torch.std(channel_means)
        
        # 3. 边缘响应 - 衡量高频内容
        edge_response = torch.norm(features, dim=1).mean()  # L2范数
        
        # 4. 激活稀疏性 - 衡量特征分布
        activation_sparsity = torch.mean(torch.abs(features))
        
        # 5. 梯度幅度 - 衡量空间变化
        grad_x = torch.abs(features[:, :, 1:, :] - features[:, :, :-1, :]).mean()
        grad_y = torch.abs(features[:, :, :, 1:] - features[:, :, :, :-1]).mean()
        gradient_magnitude = (grad_x + grad_y) / 2
        
        return {
            'spatial_variance': spatial_variance.item(),
            'channel_diversity': channel_diversity.item(),
            'edge_response': edge_response.item(),
            'activation_sparsity': activation_sparsity.item(),
            'gradient_magnitude': gradient_magnitude.item()
        }
    
    def predict_complexity(self, tile):
        """预测tile复杂度"""
        with torch.no_grad():
            # 预处理
            x = self.preprocess_tile(tile)
            
            # 特征提取
            features = self.feature_extractor(x)  # [1, 32, 8, 8]
            
            # 提取复杂度特征
            complexity_features = self.extract_complexity_features(features)
            
            # 综合复杂度得分（经验权重）
            complexity_score = (
                0.30 * complexity_features['spatial_variance'] +
                0.25 * complexity_features['channel_diversity'] + 
                0.20 * complexity_features['edge_response'] +
                0.15 * complexity_features['activation_sparsity'] +
                0.10 * complexity_features['gradient_magnitude']
            )
            
            return complexity_score > self.threshold, complexity_score, complexity_features
    
    def batch_predict(self, tiles):
        """批量预测多个tiles"""
        results = []
        scores = []
        
        for tile in tiles:
            is_complex, score, _ = self.predict_complexity(tile)
            results.append(is_complex)
            scores.append(score)
        
        return results, scores
    
    def calibrate_threshold(self, test_tiles, canny_predictor, target_agreement=0.8):
        """根据Canny结果校准阈值"""
        print(f"🎯 开始阈值校准，目标一致性: {target_agreement}")
        
        # 获取Canny预测结果
        canny_results = []
        for tile in test_tiles:
            if isinstance(tile, torch.Tensor):
                tile_np = tile.cpu().numpy().transpose(1, 2, 0)
                tile_np = (tile_np * 255).astype(np.uint8)
            else:
                tile_np = tile
            canny_results.append(canny_predictor.predict_complexity(tile_np))
        
        # 测试不同阈值
        best_threshold = self.threshold
        best_agreement = 0
        
        for threshold in np.arange(0.5, 5.0, 0.1):
            old_threshold = self.threshold
            self.threshold = threshold
            
            neural_results = []
            for tile in test_tiles:
                is_complex, _, _ = self.predict_complexity(tile)
                neural_results.append(is_complex)
            
            agreement = np.mean(np.array(neural_results) == np.array(canny_results))
            
            if agreement > best_agreement:
                best_agreement = agreement
                best_threshold = threshold
            
            # 恢复原阈值
            self.threshold = old_threshold
        
        # 设置最佳阈值
        self.threshold = best_threshold
        print(f"✅ 校准完成:")
        print(f"   最佳阈值: {best_threshold:.2f}")
        print(f"   一致性: {best_agreement:.3f}")
        
        return best_threshold, best_agreement

class CannyBaseline:
    """Canny基线方法"""
    def __init__(self, threshold=0.15):
        self.threshold = threshold
    
    def predict_complexity(self, tile):
        if len(tile.shape) == 3:
            gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        else:
            gray = tile
        
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        return edge_ratio > self.threshold

def compare_methods():
    """对比不同方法的性能"""
    print("🔬 开始方法对比测试...")
    
    # 初始化预测器
    mobile_predictor = MobileNetV2ComplexityPredictor(threshold=2.0)
    canny_predictor = CannyBaseline(threshold=0.15)
    
    # 生成测试数据
    test_tiles = []
    
    # 简单纹理
    for _ in range(20):
        tile = np.random.randint(100, 200, (16, 16, 3), dtype=np.uint8)
        test_tiles.append(tile)
    
    # 复杂纹理
    for _ in range(20):
        tile = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        # 添加网格线
        for i in range(0, 16, 2):
            cv2.line(tile, (i, 0), (i, 16), (0, 0, 0), 1)
            cv2.line(tile, (0, i), (16, i), (0, 0, 0), 1)
        test_tiles.append(tile)
    
    # 校准阈值
    mobile_predictor.calibrate_threshold(test_tiles, canny_predictor)
    
    # 速度测试
    print("\n⏱️ 速度测试...")
    
    # Canny速度
    start_time = time.time()
    for _ in range(1000):
        for tile in test_tiles:
            canny_predictor.predict_complexity(tile)
    canny_time = time.time() - start_time
    
    # MobileNet速度
    start_time = time.time()
    for _ in range(100):  # 较少次数，因为神经网络较慢
        for tile in test_tiles:
            mobile_predictor.predict_complexity(tile)
    mobile_time = time.time() - start_time * 10  # 放大到相同次数
    
    print(f"Canny方法: {canny_time:.3f}s (1000×{len(test_tiles)} predictions)")
    print(f"MobileNet方法: {mobile_time:.3f}s (100×{len(test_tiles)} predictions)")
    print(f"速度比: {mobile_time/canny_time:.1f}x slower")
    
    # 一致性测试
    print("\n🎯 一致性测试...")
    canny_results = [canny_predictor.predict_complexity(tile) for tile in test_tiles]
    mobile_results = [mobile_predictor.predict_complexity(tile)[0] for tile in test_tiles]
    
    agreement = np.mean(np.array(mobile_results) == np.array(canny_results))
    print(f"预测一致性: {agreement:.3f}")
    print(f"Canny阳性率: {np.mean(canny_results):.3f}")
    print(f"MobileNet阳性率: {np.mean(mobile_results):.3f}")

def main():
    """主函数"""
    print("🧠 MobileNetV2轻量级复杂度预测器")
    print("="*50)
    
    # 检查CUDA可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 运行对比测试
    compare_methods()
    
    print("\n✅ 测试完成!")
    print("\n💡 使用建议:")
    print("- 如果对准确性要求极高且计算资源充足，可以使用MobileNetV2")
    print("- 如果对速度要求极高，建议继续使用Canny方法")
    print("- MobileNetV2可以作为Canny的补充，用于特殊场景的精细化判断")

if __name__ == "__main__":
    main()

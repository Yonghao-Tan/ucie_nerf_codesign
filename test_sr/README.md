# SR Quality Analysis Project

## 📁 文件结构说明

### `data/` - 测试数据文件
- `no2_gt.png` - Ground Truth图像 (原始高质量参考)
- `no2_pred_org.png` - Original渲染结果 (IBRNet原始输出)
- `no2_pred_sr.png` - Super Resolution结果 (经过SR增强的输出)

### `scripts/` - Python脚本文件
- `analyze_psnr.py` - 主要的PSNR分析脚本，包含tile分析和特征提取
- `sr_quality_predictor.py` - 机器学习模型训练脚本 (43维特征，随机森林)
- `ml_tile_replacement_test.py` - 完整的ML方法测试脚本
- `practical_sr_predictor.py` - 实际部署用的质量预测器
- `simple_sr_predictor.py` - 简单启发式质量预测器
- `test_sr.py` - 基础SR测试脚本

### `models/` - 训练好的机器学习模型
- `sr_quality_predictor.pkl` - 训练好的随机森林回归模型 (43特征→质量预测)

### `results/` - 实验结果
#### `results/images/` - 生成的图片和可视化
- **对比图片**: `comparison_sr_hybrid_org.png`, `hybrid_sr_org.png`
- **ML替换结果**: `ml_hybrid_k10.png`, `ml_hybrid_k30.png`, `ml_hybrid_k50.png`, `ml_hybrid_k100.png`
- **PSNR分析**: `sr_tile_psnr.png`, `org_tile_psnr.png`, `psnr_diff.png`
- **特征分析**: `tile_feature_comparison_top100.png`, `tile_feature_correlation_top100.png`
- **质量预测**: `ml_quality_prediction_map.png`, `quality_prediction_comparison.png`
- **性能分析**: `k_value_performance_analysis.png`, `heuristic_prediction_validation.png`
- **Tile分析**: `tile_analysis/` (包含top10最差tiles的详细对比)

#### `results/reports/` - 文本报告和数据
- **PSNR数据**: `sr_tile_psnr.txt`, `org_tile_psnr.txt`, `psnr_diff.txt`
- **特征分析**: `tile_feature_analysis_top100.txt`
- **实验报告**: `ml_tile_replacement_report.txt`, `quality_prediction_report.txt`
- **性能分析**: `k_value_test_results.txt`, `hybrid_performance_report.txt`
- **预测验证**: `simple_quality_prediction_report.txt`

#### `results/data/` - 原始数据文件 (预留)

### `docs/` - 文档和分析报告
- `COMPLETE_ML_ANALYSIS.md` - 完整的机器学习方法分析报告
- `ML_METHOD_ANALYSIS.md` - ML方法原理和实验结果总结

## 🚀 主要功能

### 1. Tile-wise质量分析
- 32x32像素块级别的PSNR分析
- 最差tiles识别和可视化
- 特征工程和失败模式分析

### 2. 机器学习质量预测
- 43维特征提取 (亮度、边缘、纹理、频域等)
- 随机森林回归模型 (R²=0.751, 相关性=0.947)
- 无参考质量评估 (仅需SR图像)

### 3. 智能Tile替换
- 基于ML预测的动态替换策略
- K值可调节的质量-效率权衡
- 实时PSNR改善验证

### 4. 性能验证和分析
- 与真实PSNR方法对比
- 多种K值的性能测试
- 预测准确性验证 (重叠率分析)

## 📊 主要实验结果

| 方法 | K=100 | PSNR提升 | 像素比例 | 重叠率 |
|------|-------|----------|----------|--------|
| 真实PSNR | 参考标准 | 0.95dB | 3.4% | 100% |
| **机器学习** | **推荐** | **1.00dB** | **13.0%** | **76%** |

**最佳实践**: K=30提供理想的质量-效率平衡 (0.58dB提升, 3.9%像素)

## 🔧 使用方法

**注意**: 所有脚本都需要从scripts文件夹中运行，以确保相对路径正确

```bash
cd test_sr/scripts

# 1. 运行完整PSNR分析和特征提取
python analyze_psnr.py

# 2. 训练机器学习质量预测模型  
python sr_quality_predictor.py

# 3. 测试ML方法的tile替换效果
python ml_tile_replacement_test.py

# 4. 简单启发式质量预测测试
python simple_sr_predictor.py

# 5. 实际部署用的质量预测器
python practical_sr_predictor.py
```

**输出文件**:
- 图片结果保存到: `../results/images/`
- 文本报告保存到: `../results/reports/`  
- 训练模型保存到: `../models/`

## 📈 核心价值

✅ **无参考评估**: 仅需SR图像，无需GT  
✅ **高预测准确性**: 76%重叠率，0.947相关性  
✅ **实际PSNR改善**: 最高1.00dB提升  
✅ **生产就绪**: 可直接集成到IBRNet流程

# 定义标题列表
titles = [
    "78mJ/Frame 373fps 3D GS Processor Based on Shape-Aware Hybrid Architecture Using Earlier Computation Skipping and Gaussian Cache Scheduler",
    "IRIS: A 8.55mJ/frame Spatial Computing SoC for Interactable Rendering and Surface-Aware Modeling with 3D Gaussian Splatting",
    "A 16nm 216kb, 188.4TOPS/W and 133.5TFLOPS/W Microscaling Multi-Mode Gain-Cell CIM Macro Edge-AI Devices",
    "A 51.6TFLOPs/W Full-Datapath CIM Macro Approaching Sparsity Bound and <2-30 Loss for Compound AI",
    "RNGD: A 5nm Tensor-Contraction Processor for Power-Efficient Inference on Large Language Models",
    "An On-Device Generative AI Focused Neural Processing Unit in 4nm Flagship Mobile SoC with Fan-Out Wafer-Level Package",
    "SambaNova SN40L: A 5nm 2.5D Dataflow Accelerator with Three Memory Tiers for Trillion Parameter AI",
    "T-REX: A 68-to-567μs/Token 0.41-to-3.95μJ/Token Transformer Accelerator with Reduced External Memory Access and Enhanced Hardware Utilization in 16nm FinFET",
    "28nm 0.22μJ/Token Memory-Compute-Intensity-Aware CNN-Transformer Accelerator with Hybrid-Attention-Based Layer-Fusion and Cascaded Pruning for Semantic-Segmentation",
    "EdgeDiff: 418.4mJ/Inference Multi-Modal Few-Step Diffusion Model Accelerator with Mixed-Precision and Reordered Group Quantization",
    "Nebula: A 28nm 109.8TOPS/W 3D PNN Accelerator Featuring Adaptive Partition, Multi-Skipping, and Block-Wise Aggregation",
    "MAE: A 3nm 0.168mm2 576MAC Mini AutoEncoder with Line-based Depth-First Scheduling for Generative AI in Vision on Edge Devices",
    "MEGA.mini: A Universal Generative AI Processor with a New Big/Little Core Architecture for NPU",
    "BROCA: A 52.4-to-559.2mW Mobile Social Agent System-on-Chip with Adaptive Bit-Truncate Unit and Acoustic-Cluster Bit Grouping",
    "An 88.36TOPS/W Bit-Level-Weight-Compressed Large-Language-Model Accelerator with Cluster-Aligned INT-FP-GEMM and Bi-Dimensional Workflow Reformulation",
    "Slim-Llama: A 4.69mW Large-Language-Model Processor with Binary/Ternary Weights for Billion-Parameter Llama Model",
    "HuMoniX: A 57.3fps 12.8TFLOPS/W Text-to-Motion Processor with Inter-Iteration Output Sparsity and Inter-Frame Joint Similarity",
    "A 22nm 60.81TFLOPS/W Diffusion Accelerator with Bandwidth-Aware Memory Partition and BL-Segmented Compute-in-Memory for Efficient Multi-Task Content Generation"
]

# 统计词数并筛选出词数超过14的标题
long_titles = []
for title in titles:
    word_count = len(title.split())  # 空格数加1
    if word_count > 14:
        long_titles.append((title, word_count))  # 存储标题和词数

# 输出结果
print("词数超过14的标题：")
for title, word_count in long_titles:
    print(f"词数: {word_count} -> 标题: {title}")

print(f"\n共有 {len(long_titles)} 个标题的词数超过了14。")
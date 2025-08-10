import torch
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
from tqdm import tqdm

# 配置参数
H_t = 378  # 图像高度
W_t = 504   # 图像宽度
H_s = 756
W_s = 1008
window_size = 128  # 窗口大小，可调节的变量
fps = 64

# 加速选项
GENERATE_PLOTS = True  # 是否生成图片（设为False可大幅加速）
PLOT_DPI = 300  # 降低DPI加速（原来400）
PLOT_EVERY_N = 1  # 每N个窗口生成一张图（设为更大值可加速）

# 创建输出目录（先清空再创建）
output_dir = "pixel_distribution_plots"
if GENERATE_PLOTS:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Cleared existing directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

# 加载张量
pixel_locations = torch.load("../eval/outputs/pixel_locations.pt")
print("Loaded tensor shape:", pixel_locations.shape)

# Reshape 并提取切片
pixel_locations_2d = pixel_locations.reshape(8, H_t, W_t, *pixel_locations.shape[2:])
print("pixel_locations_2d shape:", pixel_locations_2d.shape)

total_mem = 0.

# 遍历所有source views
for source_view_idx in range(pixel_locations_2d.shape[0]):
    print(f"Processing source view {source_view_idx}...")
    source_view_mem = 0.  # 为每个source view单独统计内存
    
    # 获取当前source view的切片，转换为numpy以加速计算
    pixel_locations_slice = pixel_locations_2d[source_view_idx].cpu().numpy()
    # print(f"pixel_locations_slice shape: {pixel_locations_slice.shape}")

    # 预计算所有窗口的边界
    window_coords = []
    for i in range(0, pixel_locations_slice.shape[0], window_size):
        for j in range(0, pixel_locations_slice.shape[1], window_size):
            end_i = min(i + window_size, pixel_locations_slice.shape[0])
            end_j = min(j + window_size, pixel_locations_slice.shape[1])
            window_coords.append((i, j, end_i, end_j))

    # 创建进度条
    pbar = tqdm(total=len(window_coords), desc=f"Processing s{source_view_idx}")
    cnt = 0

    # 遍历每个窗口
    for window_idx, (i, j, end_i, end_j) in enumerate(window_coords):
        cnt += 1
        
        # 提取窗口数据（numpy操作更快）
        window_data = pixel_locations_slice[i:end_i, j:end_j]
        window_flat = window_data.reshape(-1, 2)
        
        # 使用numpy进行掩码操作（比torch更快）
        valid_mask = (window_flat[:, 0] >= 0) & (window_flat[:, 0] < W_s) & \
                     (window_flat[:, 1] >= 0) & (window_flat[:, 1] < H_s)
        valid_coordinates = window_flat[valid_mask]
        
        # 如果没有有效坐标，跳过
        if valid_coordinates.shape[0] == 0:
            pbar.update(1)
            continue
            
        # 计算统计信息（numpy操作）
        x_coords = valid_coordinates[:, 0]
        y_coords = valid_coordinates[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # 计算矩形面积
        rect_width = x_max - x_min
        rect_height = y_max - y_min
        rect_area = np.ceil(rect_width) * np.ceil(rect_height)
        mem = rect_area / 1024 # KB
        mem = mem * (1. + 0.6667)
        source_view_mem += (mem * 1024)  # 累加到当前source view的内存
        total_mem += (mem * 1024)  # 累加到总内存
        
        # 只在需要时生成图片
        should_plot = GENERATE_PLOTS and (window_idx % PLOT_EVERY_N == 0)
        
        if should_plot:
            # 创建散点图
            plt.figure(figsize=(8, 6))  # 稍微减小图片尺寸
            plt.scatter(y_coords, x_coords, s=0.5, c='blue', alpha=0.7)
            
            # 绘制最小矩形边界
            rect_x = [y_min, y_max, y_max, y_min, y_min]
            rect_y = [x_min, x_min, x_max, x_max, x_min]
            plt.plot(rect_x, rect_y, 'r-', linewidth=2, alpha=0.8, label=f'Bounding Box (Area: {rect_area:.1f})')
            
            # 设置图表标题和轴标签
            plt.title(f"Source View {source_view_idx} - Window [{i}:{end_i}, {j}:{end_j}]\nArea: {rect_area:.1f} px², Mem: {mem:.1f}KB", fontsize=14)
            plt.xlabel("Width (W)", fontsize=12)
            plt.ylabel("Height (H)", fontsize=12)
            
            # 添加图例
            plt.legend(loc='upper right')
            
            # 设置坐标轴范围
            plt.xlim(0, W_s)
            plt.ylim(0, H_s)
            plt.grid(True, alpha=0.3)
            
            # 生成简洁的文件名（添加source view前缀）
            filename = f"s{source_view_idx}_window_{i:03d}_{j:03d}.png"
            filepath = os.path.join(output_dir, filename)
            
            # 保存图表到文件（降低DPI）
            plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()  # 关闭图表以释放内存
            plot_status = f"Saved: {filename}"
        else:
            plot_status = "Skipped plot"
        
        # 更新进度条并显示信息
        progress_info = f"{plot_status} (Valid: {valid_coordinates.shape[0]}/{window_flat.shape[0]}, Area: {rect_area:.0f}, Mem: {mem:.1f}KB)"
        pbar.set_postfix_str(progress_info)
        pbar.update(1)

    # 关闭进度条
    pbar.close()
    
    # 打印当前source view的内存统计
    print(f"Source view {source_view_idx} memory: {source_view_mem / (1024 * 1024):.1f}MB")

print(f"\nTotal memory (all source views): {total_mem / (1024 * 1024):.1f}MB")
print(f"Average memory per source view: {(total_mem / 8) / (1024 * 1024):.1f}MB")
print(f"Estimated for 1008x756 resolution with {fps} FPS: {fps * total_mem / (1024 * 1024 * 1024):.1f}GB")
print(f"Estimated for 800x800 resolution with {fps} FPS: {fps * total_mem * (400*400/(378*504)) / (1024 * 1024 * 1024):.1f}GB")
print(f"All charts saved in '{output_dir}' directory")
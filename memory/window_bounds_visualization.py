import torch
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.patches as patches

# 配置参数
H_t = 756  # 图像高度
W_t = 1008   # 图像宽度
H_s = 756
W_s = 1008
window_size = 16  # 窗口大小，可调节的变量

# 输出设置
GENERATE_PLOTS = True
PLOT_DPI = 300
output_dir = "./plots/window_bounds_plots"

# 创建输出目录（先清空再创建）
if GENERATE_PLOTS:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Cleared existing directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

# 加载张量
pixel_locations = torch.load("./locations/pixel_locations_0_n48.pt")
print("Loaded tensor shape:", pixel_locations.shape)

# Reshape 并提取切片
pixel_locations_2d = pixel_locations.reshape(8, H_t, W_t, *pixel_locations.shape[2:])
print("pixel_locations_2d shape:", pixel_locations_2d.shape)

# 遍历所有source views
for source_view_idx in range(pixel_locations_2d.shape[0]):
    print(f"Processing source view {source_view_idx}...")
    
    # 获取当前source view的切片，转换为numpy以加速计算
    pixel_locations_slice = pixel_locations_2d[source_view_idx].cpu().numpy()
    
    # 创建图表
    plt.figure(figsize=(12, 10))
    
    # 计算所有窗口的边界
    window_bounds = []
    valid_windows = 0
    total_area = 0
    
    # 遍历每个窗口
    for i in range(0, pixel_locations_slice.shape[0], window_size):
        for j in range(0, pixel_locations_slice.shape[1], window_size):
            end_i = min(i + window_size, pixel_locations_slice.shape[0])
            end_j = min(j + window_size, pixel_locations_slice.shape[1])
            
            # 提取窗口数据
            window_data = pixel_locations_slice[i:end_i, j:end_j]
            window_flat = window_data.reshape(-1, 2)
            
            # 筛选有效坐标
            valid_mask = (window_flat[:, 0] >= 0) & (window_flat[:, 0] < W_s) & \
                         (window_flat[:, 1] >= 0) & (window_flat[:, 1] < H_s)
            valid_coordinates = window_flat[valid_mask]
            
            # 如果有有效坐标，计算边界
            if valid_coordinates.shape[0] > 0:
                x_coords = valid_coordinates[:, 0]
                y_coords = valid_coordinates[:, 1]
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                
                # 计算矩形面积
                rect_width = x_max - x_min
                rect_height = y_max - y_min
                rect_area = rect_width * rect_height
                
                window_bounds.append({
                    'window_pos': (i, j, end_i, end_j),
                    'bounds': (x_min, y_min, x_max, y_max),
                    'area': rect_area,
                    'valid_points': valid_coordinates.shape[0]
                })
                
                valid_windows += 1
                total_area += rect_area
    
    print(f"Source view {source_view_idx}: Found {valid_windows} valid windows")
    
    # 绘制所有窗口的边界矩形
    for idx, window_info in enumerate(window_bounds):
        x_min, y_min, x_max, y_max = window_info['bounds']
        rect_width = x_max - x_min
        rect_height = y_max - y_min
        
        # 创建矩形框（只有边框，无填充）
        rect = patches.Rectangle((y_min, x_min), rect_height, rect_width,
                               linewidth=1., edgecolor='red', 
                               facecolor='none')  # 无填充
        plt.gca().add_patch(rect)
        
        # 在矩形中心添加数字标签
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        plt.text(center_y, center_x, f'{idx}', 
                fontsize=6, ha='center', va='center', 
                color='blue', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # 设置图表属性
    plt.xlim(0, W_s)
    plt.ylim(0, H_s)
    plt.xlabel("Width (W)", fontsize=14)
    plt.ylabel("Height (H)", fontsize=14)
    plt.title(f"Source View {source_view_idx} - Window Boundaries\n"
              f"Window Size: {window_size}x{window_size}, Valid Windows: {valid_windows}, "
              f"Total Area: {total_area:.0f} px²", fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    filename = f"source_view_{source_view_idx}_window_bounds.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")

print(f"\nAll window boundary plots saved in '{output_dir}' directory")
print(f"Window size used: {window_size}x{window_size}")

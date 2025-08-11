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
window_size = 8  # 窗口大小，可调节的变量
fps = 64

# 加速选项
GENERATE_PLOTS = True  # 是否生成图片（设为False可大幅加速）
PLOT_DPI = 300  # 降低DPI加速（原来400）
PLOT_EVERY_N = 1  # 每N个窗口生成一张图（设为更大值可加速）
RAY_DIFF = True  # 是否用不同颜色表示不同的samples
DEBUG = False

# 创建输出目录（先清空再创建）
output_dir = "plots/pixel_distribution_plots"
if GENERATE_PLOTS:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Cleared existing directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

# 加载张量
pixel_locations = torch.load("../eval/outputs/pixel_locations_1.pt")
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
        
        if RAY_DIFF:
            # 保持原始shape以便区分不同的pixels
            original_shape = window_data.shape  # [window_h, window_w, n_samples, 2]
            window_flat = window_data.reshape(-1, 2)
            
            # 创建pixel标签，用于区分窗口内的不同pixels
            window_h, window_w = original_shape[0], original_shape[1]
            n_samples = original_shape[2]
            
            # 为每个pixel创建唯一标签 (h_idx * window_w + w_idx)
            pixel_labels = []
            for h_idx in range(window_h):
                for w_idx in range(window_w):
                    pixel_id = h_idx * window_w + w_idx
                    # 每个pixel有n_samples个points，都标记为同一个pixel_id
                    pixel_labels.extend([pixel_id] * n_samples)
            pixel_labels = np.array(pixel_labels)
        else:
            window_flat = window_data.reshape(-1, 2)
            pixel_labels = None
        
        # 使用numpy进行掩码操作（比torch更快）
        valid_mask = (window_flat[:, 0] >= 0) & (window_flat[:, 0] < W_s) & \
                     (window_flat[:, 1] >= 0) & (window_flat[:, 1] < H_s)
        valid_coordinates = window_flat[valid_mask]
        
        if RAY_DIFF and pixel_labels is not None:
            valid_pixel_labels = pixel_labels[valid_mask]
        else:
            valid_pixel_labels = None
        
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
            
            if RAY_DIFF and valid_pixel_labels is not None:
                # 随机选择16个pixels用不同颜色，其余用灰色
                import matplotlib.cm as cm
                unique_pixels = np.unique(valid_pixel_labels)
                if DEBUG: print(f"Debug: Found {len(unique_pixels)} unique pixels in window [{i}:{end_i}, {j}:{end_j}]")
                
                # 随机选择16个pixels
                np.random.seed(42)  # 固定随机种子以保证结果可重现
                num_colored_pixels = min(16, len(unique_pixels))
                selected_pixels = np.random.choice(unique_pixels, size=num_colored_pixels, replace=False)
                
                # 扩展颜色列表到16种
                bright_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan',
                               'lime', 'magenta', 'yellow', 'navy', 'maroon', 'olive', 'teal', 'silver']
                
                # 先画灰色背景点（最低优先级）
                if len(unique_pixels) > num_colored_pixels:
                    other_pixels = np.setdiff1d(unique_pixels, selected_pixels)
                    other_pixels_mask = np.isin(valid_pixel_labels, other_pixels)
                    other_coords = valid_coordinates[other_pixels_mask]
                    if len(other_coords) > 0:
                        plt.scatter(other_coords[:, 1], other_coords[:, 0], 
                                  s=0.2, c='gray', alpha=0.8, 
                                  marker='o', edgecolors='none',  # 实心圆点
                                  zorder=100,  # 低优先级
                                  label=f'Other Pixels ({len(other_pixels)})')
                        if DEBUG: print(f"Debug: Plotted {len(other_coords)} points for other pixels in gray")
                
                # 再画选中的16个pixel，每个用不同颜色（高优先级）
                for idx, pixel_idx in enumerate(selected_pixels):
                    mask = valid_pixel_labels == pixel_idx
                    pixel_coords = valid_coordinates[mask]
                    if len(pixel_coords) > 0:
                        color = bright_colors[idx % len(bright_colors)]
                        zorder_value = 1000 - idx  # 递减优先级
                        plt.scatter(pixel_coords[:, 1], pixel_coords[:, 0], 
                                  s=0.5, c=color, alpha=0.9,
                                  marker='o', edgecolors='none',  # 实心圆点
                                  zorder=zorder_value,  # 高优先级，递减
                                  label=f'Pixel {pixel_idx}')
                        if DEBUG: print(f"Debug: Plotted {len(pixel_coords)} points for Pixel {pixel_idx} in {color} with zorder {zorder_value}")
            else:
                # 单一颜色显示所有点
                plt.scatter(y_coords, x_coords, s=0.05, c='blue', alpha=0.7, 
                          marker='o', edgecolors='none')  # 实心圆点
            
            # 绘制最小矩形边界
            rect_x = [y_min, y_max, y_max, y_min, y_min]
            rect_y = [x_min, x_min, x_max, x_max, x_min]
            plt.plot(rect_x, rect_y, 'r-', linewidth=0.75, alpha=0.9, label=f'Bounding Box (Area: {rect_area:.1f})')
            
            # 设置图表标题和轴标签
            title_suffix = " (Pixel Diff)" if RAY_DIFF else ""
            plt.title(f"Source View {source_view_idx} - Window [{i}:{end_i}, {j}:{end_j}]{title_suffix}\nArea: {rect_area:.1f} px², Mem: {mem:.1f}KB", fontsize=14)
            plt.xlabel("Width (W)", fontsize=12)
            plt.ylabel("Height (H)", fontsize=12)
            
            plt.legend(loc='upper right', fontsize=6)
            
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
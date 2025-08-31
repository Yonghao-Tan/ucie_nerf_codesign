import torch
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.patches as patches
import pandas as pd

# 配置参数
H_t = 756  # 图像高度
W_t = 1008   # 图像宽度
H_s = 756
W_s = 1008

H_t = 800  # 图像高度
W_t = 800   # 图像宽度
H_s = 800
W_s = 800
# 实验参数
window_sizes = [[40, 40], [80, 80]]
gs_values = [8, 16, 48]
window_sizes = [[5, 5], [20, 20], [40, 40], [80, 80], [120, 120], [160, 160]]
gs_values = [8, 16, 48]


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
pixel_locations = torch.load("./nerf_synthetic/locations_hr/pixel_locations_0_n48.pt")
print("Loaded tensor shape:", pixel_locations.shape)

# Reshape 并提取切片
pixel_locations_2d = pixel_locations.reshape(8, H_t, W_t, *pixel_locations.shape[2:])
print("pixel_locations_2d shape:", pixel_locations_2d.shape)

import numpy as np

def group_based_dynamic_balancing_v2(tensor_2d):
    # 1. 初始化4个最终寄存器的累加和 (使用纯Python列表)
    final_register_sums = [0.0] * 4

    # 2. 遍历每一个8元素的group
    for i, group in enumerate(tensor_2d):
        if len(group) != 8:
            raise ValueError(f"第 {i+1} 组的元素数量不是8，请检查输入数据。")
        
        # a. 将当前group的8个数字从大到小排序
        sorted_group_nums = sorted(group, reverse=True)
        
        # b. 为当前group重置“接收计数器” (使用纯Python列表)
        current_group_counts = [0] * 4
        
        # c. 逐个分配这8个数字
        for number_to_assign in sorted_group_nums:
            
            # --- 在“可用”寄存器中寻找最佳目标 ---
            best_target_idx = -1
            min_sum_found = float('inf')
            
            for reg_idx in range(4):
                # 检查两个条件：
                # 1. 这个寄存器在本轮还没拿满2个
                # 2. 它的当前和是所有可用寄存器中最小的
                if current_group_counts[reg_idx] < 2 and final_register_sums[reg_idx] < min_sum_found:
                    min_sum_found = final_register_sums[reg_idx]
                    best_target_idx = reg_idx
            
            # --- 分配并更新状态 ---
            if best_target_idx != -1:
                # print(f"  -> 数字 {number_to_assign} 分配给寄存器 {best_target_idx} (原和: {final_register_sums[best_target_idx]:.1f})")
                final_register_sums[best_target_idx] += number_to_assign
                current_group_counts[best_target_idx] += 1
            else:
                # 理论上不会发生，因为总有位置可放
                print(f"警告：数字 {number_to_assign} 没有找到可分配的位置！")


    # 3. 所有group处理完毕，返回最大和 (使用纯Python的max函数)
    max_final_sum = max(final_register_sums)
    
    # 计算总和 (使用纯Python的sum和列表推导)
    # total_sum = sum(sum(row) for row in tensor_2d)

    # print("\n--- 所有组处理完毕 ---")
    # print(f"最终4个寄存器的和为: {[f'{s:.2f}' for s in final_register_sums]}")
    # print(f"所有元素的总和为: {total_sum:.2f}")
    
    return max_final_sum

def area_agg(flat_tensor):
    """
    将一个元素数量为4的倍数的列表，均分成4组，并使各组之和尽可能均衡。

    算法步骤:
    1. 检查输入列表长度是否为4的倍数。
    2. 将所有元素从大到小排序。
    3. 创建4个空组。
    4. 使用“蛇形”分配法，轮流将元素分配到4个组中，以保证每组元素数量相等且和均衡。
    5. 计算所有组的和，并返回其中的最大值。

    参数:
    flat_tensor (list or np.array): 一个扁平化的列表或Numpy数组，长度是4的倍数。

    返回:
    float: 4个组中最大的那个和。
    """
    num_elements = len(flat_tensor)

    # 1. 检查输入合法性
    if num_elements == 0:
        return 0.0
    if num_elements % 4 != 0:
        return 0.0
        # raise ValueError(f"输入元素数量必须是4的倍数，但现在是 {num_elements}个。")

    # 2. 从大到小排序
    sorted_numbers = sorted(flat_tensor, reverse=True)
    
    # 3. 初始化4个组
    groups = [[] for _ in range(4)]
    
    # 4. “蛇形”分配
    # print("开始进行蛇形分配...")
    for i, number in enumerate(sorted_numbers):
        # 计算当前属于第几轮（每4个为一轮）
        round_index = i // 4
        
        if round_index % 2 == 0:
            # 偶数轮 (0, 2, 4...)：正向分配 0 -> 1 -> 2 -> 3
            group_index = i % 4
        else:
            # 奇数轮 (1, 3, 5...)：反向分配 3 -> 2 -> 1 -> 0
            group_index = 3 - (i % 4)
            
        groups[group_index].append(number)

    # 5. 计算各组的和
    group_sums = [sum(g) for g in groups]
    
    # 6. 找到并返回最大的和
    max_sum = max(group_sums)
    
    # --- 打印详细信息 ---
    # print("\n分配完成！")
    # print("-" * 30)
    # print(f"每组有 {len(groups[0])} 个元素。")
    # # 为了方便查看，可以打印出每个组的具体内容
    # # for i, g in enumerate(groups):
    # #     print(f"第 {i+1} 组 (和={group_sums[i]}): {g}")
    # print(f"最终4个组的和分别为: {group_sums}")
    # print(f"各组和的平均值为: {np.mean(group_sums)}")
    # print(f"所有元素的总和为: {sum(flat_tensor)}")
    # print("-" * 30)
    
    return max_sum
    

def run_single_experiment(window_size_h, window_size_w, gs):
    """运行单次实验，返回结果指标"""
    
    window_size_h = np.minimum(window_size_h, H_s)
    window_size_w = np.minimum(window_size_w, W_s)
    
    total_area = 0.
    total_valid_windows = 0
    # 遍历所有source views
    all_windows = []
    for source_view_idx in range(pixel_locations_2d.shape[0]):
        
        # 获取当前source view的切片，转换为numpy以加速计算
        pixel_locations_slice = pixel_locations_2d[source_view_idx].cpu().numpy()
        
        valid_windows = 0
        source_view_windows = []
        
        # 遍历每个窗口
        for i in range(0, pixel_locations_slice.shape[0], window_size_h):
            patch_windows_h = []
            for j in range(0, pixel_locations_slice.shape[1], window_size_w):
                patch_windows_w = []
                for k in range(0, pixel_locations_slice.shape[2], gs):
                    end_i = min(i + window_size_h, pixel_locations_slice.shape[0])
                    end_j = min(j + window_size_w, pixel_locations_slice.shape[1])
                    end_k = min(k + gs, pixel_locations_slice.shape[2])

                    # 提取窗口数据
                    window_data = pixel_locations_slice[i:end_i, j:end_j, k:end_k]
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
                        rect_area = rect_width * rect_height * 3 * 1.67
                        
                        valid_windows += 1
                        total_valid_windows += 1
                        total_area += rect_area # TODO
                    else:
                        rect_area = 0
                    patch_windows_w.append(torch.tensor(rect_area, dtype=torch.float32))
                patch_windows_h.append(torch.stack(patch_windows_w))
            source_view_windows.append(torch.stack(patch_windows_h))
        all_windows.append(torch.stack(source_view_windows))
    
    all_windows = torch.stack(all_windows)
    all_windows = all_windows.permute(1, 2, 3, 0)
    
    # 计算最大内存
    max_area = 0.
    max_area_agg = 0.
    max_area_agg2 = 0.
    for i in range(all_windows.shape[0]):
        for j in range(all_windows.shape[1]):
            patch = all_windows[i, j]
            tmp_area_agg2 = group_based_dynamic_balancing_v2(patch)
            tmp_area_agg = area_agg(patch.flatten())
            patch = patch.sum(dim=-2) # 8
            # unified but not agg
            patch = patch.reshape(-1, 2)
            patch_area = patch.sum(dim=-1)
            tmp_area = patch_area.max()
            # not unified, not agg
            # tmp_area = patch.sum()
            if tmp_area > max_area:
                max_area = tmp_area
            if tmp_area_agg > max_area_agg:
                max_area_agg = tmp_area_agg
            if tmp_area_agg2 > max_area_agg2:
                max_area_agg2 = tmp_area_agg2
    # max_area = max_area / (all_windows.shape[0] * all_windows.shape[1])
    # 计算指标
    total_mem_mb = total_area / (1024**2)
    avg_mem_kb = total_area / (1024**1 * total_valid_windows) if total_valid_windows > 0 else 0
    
    effective_pixel = total_area / (total_valid_windows * window_size_h * window_size_w * gs) if total_valid_windows > 0 else 0
    
    max_mem_mb = max_area / (1024 * 1024)
    max_area_agg_mb = max_area_agg / (1024 * 1024)
    max_area_agg2_mb = max_area_agg2 / (1024 * 1024)

    return {
        'total_mem_mb': total_mem_mb,
        'avg_mem_kb': avg_mem_kb,
        'effective_pixel': effective_pixel,
        'max_mem_mb': max_mem_mb,
        'max_area_agg_mb': max_area_agg_mb,
        'max_area_agg2_mb': max_area_agg2_mb,
        'total_valid_windows': total_valid_windows
    }

# 运行所有实验
results = []

print("Starting experiments...")
print("=" * 80)

for window_size in window_sizes:
    window_size_h, window_size_w = window_size
    print(f"\nWindow Size: {window_size_h}x{window_size_w}")
    print("-" * 40)
    
    for gs in gs_values:
        print(f"  Running gs={gs}...", end=" ")
        
        # 运行实验
        result = run_single_experiment(window_size_h, window_size_w, gs)
        
        # 保存结果
        result_entry = {
            'Window_Size': f"{window_size_h}x{window_size_w}",
            'GS': gs,
            'Total_Mem_MB': result['total_mem_mb'],
            'Avg_Mem_KB': result['avg_mem_kb'],
            'Effective_Pixel': result['effective_pixel'],
            'Max_Mem_MB': result['max_mem_mb'],
            'Max_Area_Agg_MB': result['max_area_agg_mb'],
            'Max_Area_Agg2_MB': result['max_area_agg2_mb'],
            'Total_Valid_Windows': result['total_valid_windows']
        }
        results.append(result_entry)

        print(f"Total: {result['total_mem_mb']:.2f}MB, Avg: {result['avg_mem_kb']:.3f}KB, Eff: {result['effective_pixel']:.2f}, Max: {result['max_mem_mb']:.2f}MB, Max_Area_Agg: {result['max_area_agg_mb']:.2f}MB, Max_Area_Agg2: {result['max_area_agg2_mb']:.2f}MB")

print("\n" + "=" * 80)
print("All experiments completed!")

# 创建DataFrame并保存结果
df = pd.DataFrame(results)

# 显示完整表格
print("\nComplete Results Table:")
print("=" * 120)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(df.to_string(index=False))

# 保存到CSV文件
csv_filename = "experimental_results.csv"
df.to_csv(csv_filename, index=False)
print(f"\nResults saved to: {csv_filename}")

# 创建按window_size分组的汇总表
print("\nResults by Window Size:")
print("=" * 120)
for window_size in window_sizes:
    window_size_str = f"{window_size[0]}x{window_size[1]}"
    subset = df[df['Window_Size'] == window_size_str]
    print(f"\nWindow Size: {window_size_str}")
    print(subset[['GS', 'Total_Mem_MB', 'Avg_Mem_KB', 'Effective_Pixel', 'Max_Mem_MB', 'Max_Area_Agg_MB', 'Max_Area_Agg2_MB']].to_string(index=False))

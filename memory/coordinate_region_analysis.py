import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# 配置参数
H_t = 378  # 图像高度
W_t = 504   # 图像宽度
H_s = 756
W_s = 1008
window_size = 8  # 窗口大小，可调节的变量
small_region_size = 4  # 小区域大小，可调节（例如：1表示1x1，4表示4x4）
threshold_points = 4  # 小区域内点数阈值
area_ratio_threshold = 4.  # 矩形面积比例阈值，用于判断密集度
area_ratio_threshold = 0.25 * (small_region_size ** 2)
# area_ratio_threshold = 0.25 * (small_region_size ** 2.5)
sample_group_size = 16  # 每组采样点数量，用于分组计算面积比例
K = 1

def get_small_region(x, y):
    """
    将坐标(x, y)映射到对应的小区域
    例如: small_region_size=1时，(1.2, 1.3) -> (1, 1), (1.8, 1.6) -> (1, 1), (2.1, 1.5) -> (2, 1)
    例如: small_region_size=4时，(1.2, 1.3) -> (0, 0), (4.8, 5.6) -> (1, 1)
    """
    region_x = int(np.floor(x / small_region_size))
    region_y = int(np.floor(y / small_region_size))
    return (region_x, region_y)

def calculate_bounding_box_area(coordinates):
    """
    计算坐标集合的外接矩形面积
    """
    if len(coordinates) == 0:
        return 0
    
    min_x = np.min(coordinates[:, 0])
    max_x = np.max(coordinates[:, 0])
    min_y = np.min(coordinates[:, 1])
    max_y = np.max(coordinates[:, 1])
    
    width = max_x - min_x
    height = max_y - min_y
    area = width * height
    
    return area, width, height, (min_x, min_y, max_x, max_y)

def calculate_grouped_area_ratio(pixel_locations_slice, i, j, end_i, end_j, sample_group_size, num_sample_points):
    """
    计算分组的面积比例，避免极端点聚集影响
    """
    # 提取窗口数据的原始形状
    window_data = pixel_locations_slice[i:end_i, j:end_j]  # [height, width, num_samples, 2]
    window_height, window_width, total_samples = window_data.shape[:3]
    
    # 计算需要多少组
    num_groups = total_samples // sample_group_size
    if num_groups == 0:
        return 0, []  # 如果样本数不足一组，返回0
    
    group_area_ratios = []
    window_pixel_count = window_height * window_width
    
    for group_idx in range(num_groups):
        start_sample = group_idx * sample_group_size
        end_sample = start_sample + sample_group_size
        
        # 提取当前组的数据
        group_data = window_data[:, :, start_sample:end_sample, :]
        group_flat = group_data.reshape(-1, 2)
        
        # 筛选有效坐标
        valid_mask = (group_flat[:, 0] >= 0) & (group_flat[:, 0] < W_s) & \
                     (group_flat[:, 1] >= 0) & (group_flat[:, 1] < H_s)
        valid_coordinates = group_flat[valid_mask]
        
        if len(valid_coordinates) > 0:
            # 计算这组的外接矩形面积
            bbox_area, _, _, _ = calculate_bounding_box_area(valid_coordinates)
            # 计算面积比例（相对于这组的容量）
            group_capacity = window_pixel_count * sample_group_size
            area_ratio = bbox_area / group_capacity if group_capacity > 0 else 0
            group_area_ratios.append(area_ratio)
        else:
            group_area_ratios.append(0)
    
    # 返回平均面积比例和所有组的比例列表
    avg_area_ratio = np.mean(group_area_ratios) if group_area_ratios else 0
    return avg_area_ratio, group_area_ratios

def analyze_window_coordinate_distribution(pixel_locations_slice, window_coords, num_sample_points):
    """
    分析每个窗口内坐标的小区域分布
    pixel_locations_slice: 当前source view的坐标数据
    window_coords: 窗口坐标列表
    num_sample_points: 每个像素的采样点数量
    """
    window_stats = []
    
    # 初始化统计计数器
    small_region_dense_count = 0
    small_region_count_dense_count = 0  # 新增：基于计数方法的密集窗口数
    area_ratio_dense_count = 0
    area_ratio_grouped_dense_count = 0  # 新增：基于分组面积方法的密集窗口数
    correct_predictions = 0
    correct_predictions_count = 0  # 新增：计数方法的正确预测数
    correct_predictions_grouped = 0  # 新增：分组面积方法的正确预测数
    total_processed = 0
    
    # 使用tqdm显示进度和实时统计
    pbar = tqdm(window_coords, desc="Processing")
    
    for window_idx, (i, j, end_i, end_j) in enumerate(pbar):
        # 提取窗口数据
        window_data = pixel_locations_slice[i:end_i, j:end_j]
        window_flat = window_data.reshape(-1, 2)
        
        # 筛选有效坐标
        valid_mask = (window_flat[:, 0] >= 0) & (window_flat[:, 0] < W_s) & \
                     (window_flat[:, 1] >= 0) & (window_flat[:, 1] < H_s)
        valid_coordinates = window_flat[valid_mask]
        
        if valid_coordinates.shape[0] == 0:
            continue
            
        # 统计每个小区域的坐标数量
        region_count = defaultdict(int)
        
        for coord in valid_coordinates:
            x, y = coord[0], coord[1]
            region = get_small_region(x, y)
            region_count[region] += 1
        # 计算统计信息
        total_points = len(valid_coordinates)
        total_regions = len(region_count)
        points_per_region = list(region_count.values())
        
        # # 去掉最大和最小的5%数据
        # if len(points_per_region) > 0:
        #     # 排序并计算需要去掉的数量
        #     sorted_points = sorted(points_per_region)
        #     n = len(sorted_points)
        #     remove_count = max(1, int(n * 0.05))  # 至少去掉1个，如果数量足够的话
            
        #     # 去掉最小和最大的5%
        #     if n > 2 * remove_count:  # 确保去掉后还有数据剩余
        #         trimmed_points = sorted_points[remove_count:-remove_count]
        #     else:
        #         trimmed_points = sorted_points  # 如果数据太少，不进行trimming
        # else:
        #     trimmed_points = points_per_region
        trimmed_points = points_per_region
        
        # 统计达到阈值的区域数量（使用原始数据）
        regions_above_threshold = sum(1 for count in points_per_region if count >= threshold_points)
        regions_below_threshold = sum(1 for count in points_per_region if count < threshold_points)
        
        # 计算矩形面积相关统计
        window_pixel_count = (end_i - i) * (end_j - j)  # 窗口像素总数
        window_total_capacity = window_pixel_count * num_sample_points  # 窗口总容量（像素数 × 每像素采样点数）
        bbox_area, bbox_width, bbox_height, bbox_coords = calculate_bounding_box_area(valid_coordinates)
        area_ratio = bbox_area / window_total_capacity if window_total_capacity > 0 else 0
        
        # 方法对照：小区域密度 vs 矩形面积比例
        # 小区域密度 = 平均每个小区域的点数 / 小区域面积
        # small_region_area = small_region_size * small_region_size
        small_region_area = 1
        region_density = np.mean(trimmed_points) / small_region_area if trimmed_points else 0
        
        # 密集度判断
        # 方法1: 小区域方法 (Ground Truth) - 基于平均每个小区域的点数（使用trimmed数据）
        is_dense_small_region = np.mean(trimmed_points) > threshold_points if trimmed_points else False
        
        # 方法1.5: 小区域计数方法 (新增) - 基于密集区域数量是否多于稀疏区域
        dense_regions_count = sum(1 for count in trimmed_points if count >= threshold_points)
        sparse_regions_count = sum(1 for count in trimmed_points if count < threshold_points)
        is_dense_small_region_count = dense_regions_count > sparse_regions_count if trimmed_points else False
        
        # 方法2: 矩形面积方法 - 基于面积比例
        is_dense_area_ratio = area_ratio <= area_ratio_threshold
        
        # 方法2.5: 分组矩形面积方法 (新增) - 基于分组计算的平均面积比例
        grouped_area_ratio, group_ratios = calculate_grouped_area_ratio(
            pixel_locations_slice, i, j, end_i, end_j, sample_group_size, num_sample_points
        )
        is_dense_area_ratio_grouped = grouped_area_ratio <= area_ratio_threshold
        
        print(np.mean(points_per_region), area_ratio, grouped_area_ratio)
        
        # 判断两种方法是否一致
        methods_consistent = (is_dense_small_region == is_dense_area_ratio)
        methods_consistent_count = (is_dense_small_region_count == is_dense_area_ratio)  # 新增：计数方法与面积方法的一致性
        methods_consistent_grouped = (is_dense_small_region == is_dense_area_ratio_grouped)  # 新增：平均值方法与分组面积方法的一致性
        
        # 更新统计计数器
        total_processed += 1
        if is_dense_small_region:
            small_region_dense_count += 1
        if is_dense_small_region_count:  # 新增
            small_region_count_dense_count += 1
        if is_dense_area_ratio:
            area_ratio_dense_count += 1
        if is_dense_area_ratio_grouped:  # 新增
            area_ratio_grouped_dense_count += 1
        if methods_consistent:
            correct_predictions += 1
        if methods_consistent_count:  # 新增
            correct_predictions_count += 1
        if methods_consistent_grouped:  # 新增
            correct_predictions_grouped += 1
        
        # 计算当前准确率
        accuracy = correct_predictions / total_processed if total_processed > 0 else 0
        accuracy_count = correct_predictions_count / total_processed if total_processed > 0 else 0  # 新增
        accuracy_grouped = correct_predictions_grouped / total_processed if total_processed > 0 else 0  # 新增
        
        # 更新进度条显示
        pbar.set_postfix({
            'Small_D': small_region_dense_count,
            'Count_D': small_region_count_dense_count,  # 新增
            'Area_D': area_ratio_dense_count,
            'Group_D': area_ratio_grouped_dense_count,  # 新增
            'Acc_Avg': f'{accuracy:.3f}',
            'Acc_Grp': f'{accuracy_grouped:.3f}'  # 新增
        })
        
        window_stats.append({
            'window_idx': window_idx,
            'window_pos': (i, j, end_i, end_j),
            'total_points': total_points,
            'total_regions': total_regions,
            'points_per_region': points_per_region,
            'trimmed_points_per_region': trimmed_points,
            'avg_points_per_region': np.mean(points_per_region),
            'avg_trimmed_points_per_region': np.mean(trimmed_points) if trimmed_points else 0,
            'max_points_in_region': np.max(points_per_region),
            'min_points_in_region': np.min(points_per_region),
            'regions_above_threshold': regions_above_threshold,
            'regions_below_threshold': regions_below_threshold,
            'region_distribution': dict(region_count),
            # 新增矩形面积相关统计
            'window_pixel_count': window_pixel_count,
            'window_total_capacity': window_total_capacity,
            'bbox_area': bbox_area,
            'bbox_width': bbox_width,
            'bbox_height': bbox_height,
            'bbox_coords': bbox_coords,
            'area_ratio': area_ratio,
            'grouped_area_ratio': grouped_area_ratio,  # 新增
            'group_ratios': group_ratios,  # 新增：各组的面积比例列表
            'region_density': region_density,
            # 新增密集度判断
            'is_dense_small_region': is_dense_small_region,
            'is_dense_small_region_count': is_dense_small_region_count,  # 新增
            'dense_regions_count': dense_regions_count,  # 新增
            'sparse_regions_count': sparse_regions_count,  # 新增
            'is_dense_area_ratio': is_dense_area_ratio,
            'is_dense_area_ratio_grouped': is_dense_area_ratio_grouped,  # 新增
            'methods_consistent': methods_consistent,
            'methods_consistent_count': methods_consistent_count,  # 新增
            'methods_consistent_grouped': methods_consistent_grouped  # 新增
        })
    
    return window_stats

# 加载张量
pixel_locations = torch.load("../eval/outputs/pixel_locations_0_n48.pt")
print("Loaded tensor shape:", pixel_locations.shape)

# Reshape 并提取切片
pixel_locations_2d = pixel_locations.reshape(8, H_t, W_t, *pixel_locations.shape[2:])
print("pixel_locations_2d shape:", pixel_locations_2d.shape)

# 计算每个像素的采样点数量
num_sample_points = pixel_locations_2d.shape[-2]  # 倒数第二个维度是采样点数
print(f"Number of sample points per pixel: {num_sample_points}")

# 存储所有source view的统计结果
all_source_stats = []

# 遍历所有source views
for source_view_idx in range(K, pixel_locations_2d.shape[0]):
    print(f"\nProcessing source view {source_view_idx}...")
    
    # 获取当前source view的切片，转换为numpy以加速计算
    pixel_locations_slice = pixel_locations_2d[source_view_idx].cpu().numpy()
    
    # 预计算所有窗口的边界
    window_coords = []
    for i in range(0, pixel_locations_slice.shape[0], window_size):
        for j in range(0, pixel_locations_slice.shape[1], window_size):
            end_i = min(i + window_size, pixel_locations_slice.shape[0])
            end_j = min(j + window_size, pixel_locations_slice.shape[1])
            window_coords.append((i, j, end_i, end_j))
    
    # 分析窗口坐标分布
    window_stats = analyze_window_coordinate_distribution(pixel_locations_slice, window_coords, num_sample_points)
    
    # 计算source view级别的统计
    if window_stats:
        avg_points_per_region_list = [stat['avg_points_per_region'] for stat in window_stats]
        total_regions_list = [stat['total_regions'] for stat in window_stats]
        total_points_list = [stat['total_points'] for stat in window_stats]
        
        # 矩形面积相关统计
        area_ratio_list = [stat['area_ratio'] for stat in window_stats]
        region_density_list = [stat['region_density'] for stat in window_stats]
        bbox_area_list = [stat['bbox_area'] for stat in window_stats]
        
        # 密集度判断一致性统计
        small_region_dense_list = [stat['is_dense_small_region'] for stat in window_stats]
        area_ratio_dense_list = [stat['is_dense_area_ratio'] for stat in window_stats]
        consistency_list = [stat['methods_consistent'] for stat in window_stats]
        
        # 计算一致性统计
        total_windows_analyzed = len(window_stats)
        consistent_windows = sum(consistency_list)
        consistency_rate = consistent_windows / total_windows_analyzed if total_windows_analyzed > 0 else 0
        
        # 分类统计
        both_dense = sum(1 for i in range(len(window_stats)) 
                        if window_stats[i]['is_dense_small_region'] and window_stats[i]['is_dense_area_ratio'])
        both_sparse = sum(1 for i in range(len(window_stats)) 
                         if not window_stats[i]['is_dense_small_region'] and not window_stats[i]['is_dense_area_ratio'])
        small_dense_area_sparse = sum(1 for i in range(len(window_stats)) 
                                     if window_stats[i]['is_dense_small_region'] and not window_stats[i]['is_dense_area_ratio'])
        small_sparse_area_dense = sum(1 for i in range(len(window_stats)) 
                                     if not window_stats[i]['is_dense_small_region'] and window_stats[i]['is_dense_area_ratio'])
        
        # 统计所有窗口中达到阈值的区域总数
        total_regions_above_threshold = sum(stat['regions_above_threshold'] for stat in window_stats)
        total_regions_below_threshold = sum(stat['regions_below_threshold'] for stat in window_stats)
        
        source_stats = {
            'source_view_idx': source_view_idx,
            'total_windows': len(window_stats),
            'avg_points_per_region_overall': np.mean(avg_points_per_region_list),
            'std_points_per_region_overall': np.std(avg_points_per_region_list),
            'avg_total_regions_per_window': np.mean(total_regions_list),
            'avg_total_points_per_window': np.mean(total_points_list),
            'total_regions_above_threshold': total_regions_above_threshold,
            'total_regions_below_threshold': total_regions_below_threshold,
            # 矩形面积相关统计
            'avg_area_ratio': np.mean(area_ratio_list),
            'std_area_ratio': np.std(area_ratio_list),
            'avg_region_density': np.mean(region_density_list),
            'std_region_density': np.std(region_density_list),
            'avg_bbox_area': np.mean(bbox_area_list),
            # 新增一致性统计
            'consistency_rate': consistency_rate,
            'both_dense': both_dense,
            'both_sparse': both_sparse,
            'small_dense_area_sparse': small_dense_area_sparse,
            'small_sparse_area_dense': small_sparse_area_dense,
            'window_details': window_stats
        }
        
        all_source_stats.append(source_stats)
        
        # 打印source view统计
        print(f"Source View {source_view_idx} Statistics:")
        print(f"  Total valid windows: {len(window_stats)}")
        print(f"  Average points per small region: {np.mean(avg_points_per_region_list):.2f} ± {np.std(avg_points_per_region_list):.2f}")
        print(f"  Average total regions per window: {np.mean(total_regions_list):.1f}")
        print(f"  Average total points per window: {np.mean(total_points_list):.1f}")
        print(f"  Small regions with >= {threshold_points} points: {total_regions_above_threshold}")
        print(f"  Small regions with < {threshold_points} points: {total_regions_below_threshold}")
        print(f"  Percentage of regions >= {threshold_points} points: {total_regions_above_threshold/(total_regions_above_threshold+total_regions_below_threshold)*100:.1f}%")
        # 新增矩形面积统计输出
        print(f"  Average bounding box area ratio: {np.mean(area_ratio_list):.4f} ± {np.std(area_ratio_list):.4f}")
        print(f"  Average region density: {np.mean(region_density_list):.4f} ± {np.std(region_density_list):.4f}")
        print(f"  Average bounding box area: {np.mean(bbox_area_list):.1f} ± {np.std(bbox_area_list):.1f}")
        print(f"  Window capacity (pixels × sample points): {window_size}×{window_size}×{num_sample_points} = {window_size*window_size*num_sample_points}")
        
        # 新增密集度判断对比统计
        print(f"\n  --- Density Classification Summary ---")
        print(f"  Windows classified as DENSE by small region method: {both_dense + small_dense_area_sparse}")
        print(f"  Windows classified as DENSE by area ratio method: {both_dense + small_sparse_area_dense}")
        print(f"  Area ratio method accuracy: {consistency_rate:.3f} ({consistency_rate*100:.1f}%)")

# 计算全局统计
if all_source_stats:
    print(f"\n{'='*60}")
    print("GLOBAL STATISTICS")
    print(f"{'='*60}")
    
    # 收集所有source view的数据
    all_avg_points_per_region = []
    all_total_regions = []
    all_total_points = []
    all_area_ratios = []
    all_region_densities = []
    all_bbox_areas = []
    global_regions_above_threshold = 0
    global_regions_below_threshold = 0
    
    # 密集度判断统计
    global_both_dense = 0
    global_both_sparse = 0
    global_small_dense_area_sparse = 0
    global_small_sparse_area_dense = 0
    global_total_windows = 0
    
    for source_stat in all_source_stats:
        global_regions_above_threshold += source_stat['total_regions_above_threshold']
        global_regions_below_threshold += source_stat['total_regions_below_threshold']
        global_both_dense += source_stat['both_dense']
        global_both_sparse += source_stat['both_sparse']
        global_small_dense_area_sparse += source_stat['small_dense_area_sparse']
        global_small_sparse_area_dense += source_stat['small_sparse_area_dense']
        global_total_windows += source_stat['total_windows']
        
        for window_stat in source_stat['window_details']:
            all_avg_points_per_region.append(window_stat['avg_points_per_region'])
            all_total_regions.append(window_stat['total_regions'])
            all_total_points.append(window_stat['total_points'])
            all_area_ratios.append(window_stat['area_ratio'])
            all_region_densities.append(window_stat['region_density'])
            all_bbox_areas.append(window_stat['bbox_area'])
    
    print(f"Total windows analyzed: {len(all_avg_points_per_region)}")
    print(f"Window size: {window_size}x{window_size}")
    print(f"Small region size: {small_region_size}x{small_region_size}")
    print(f"Sample points per pixel: {num_sample_points}")
    print(f"Window total capacity: {window_size*window_size*num_sample_points}")
    print(f"Points threshold: {threshold_points}")
    print(f"Area ratio threshold: {area_ratio_threshold}")
    print(f"Average points per small region (global): {np.mean(all_avg_points_per_region):.2f} ± {np.std(all_avg_points_per_region):.2f}")
    print(f"Median points per small region: {np.median(all_avg_points_per_region):.2f}")
    print(f"Average total regions per window: {np.mean(all_total_regions):.1f} ± {np.std(all_total_regions):.1f}")
    print(f"Average total points per window: {np.mean(all_total_points):.1f} ± {np.std(all_total_points):.1f}")
    print(f"Global small regions with >= {threshold_points} points: {global_regions_above_threshold}")
    print(f"Global small regions with < {threshold_points} points: {global_regions_below_threshold}")
    total_global_regions = global_regions_above_threshold + global_regions_below_threshold
    print(f"Global percentage of regions >= {threshold_points} points: {global_regions_above_threshold/total_global_regions*100:.1f}%")
    
    # 矩形面积方法的全局统计
    print(f"\n--- Bounding Box Area Method Statistics ---")
    print(f"Average bounding box area ratio: {np.mean(all_area_ratios):.4f} ± {np.std(all_area_ratios):.4f}")
    print(f"Median bounding box area ratio: {np.median(all_area_ratios):.4f}")
    print(f"Average region density: {np.mean(all_region_densities):.4f} ± {np.std(all_region_densities):.4f}")
    print(f"Average bounding box area: {np.mean(all_bbox_areas):.1f} ± {np.std(all_bbox_areas):.1f}")
    
    # 全局密集度判断对比分析
    global_consistency_rate = (global_both_dense + global_both_sparse) / global_total_windows if global_total_windows > 0 else 0
    total_small_dense = global_both_dense + global_small_dense_area_sparse
    total_area_dense = global_both_dense + global_small_sparse_area_dense
    
    print(f"\n--- Global Density Classification Summary ---")
    print(f"Windows classified as DENSE by small region method: {total_small_dense}")
    print(f"Windows classified as DENSE by area ratio method: {total_area_dense}")
    print(f"Area ratio method accuracy: {global_consistency_rate:.3f} ({global_consistency_rate*100:.1f}%)")
    
    # 创建直方图
    plt.figure(figsize=(20, 12))
    
    # 子图1: 每个小区域的平均点数分布
    plt.subplot(2, 4, 1)
    plt.hist(all_avg_points_per_region, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Average Points per Small Region')
    plt.ylabel('Number of Windows')
    plt.title('Distribution of Average Points\nper Small Region')
    plt.grid(True, alpha=0.3)
    
    # 子图2: 每个窗口的小区域总数分布
    plt.subplot(2, 4, 2)
    plt.hist(all_total_regions, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Total Small Regions per Window')
    plt.ylabel('Number of Windows')
    plt.title('Distribution of Total Small Regions\nper Window')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 每个窗口的总点数分布
    plt.subplot(2, 4, 3)
    plt.hist(all_total_points, bins=30, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel('Total Points per Window')
    plt.ylabel('Number of Windows')
    plt.title('Distribution of Total Points\nper Window')
    plt.grid(True, alpha=0.3)
    
    # 子图4: 矩形面积比例分布
    plt.subplot(2, 4, 4)
    plt.hist(all_area_ratios, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Bounding Box Area Ratio')
    plt.ylabel('Number of Windows')
    plt.title('Distribution of Bounding Box\nArea Ratios')
    plt.grid(True, alpha=0.3)
    
    # 子图5: 区域密度分布
    plt.subplot(2, 4, 5)
    plt.hist(all_region_densities, bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Region Density')
    plt.ylabel('Number of Windows')
    plt.title('Distribution of Region\nDensities')
    plt.grid(True, alpha=0.3)
    
    # 子图6: 相关性散点图 - 区域密度 vs 平均点数
    plt.subplot(2, 4, 6)
    plt.scatter(all_region_densities, all_avg_points_per_region, alpha=0.6, s=10)
    plt.xlabel('Region Density')
    plt.ylabel('Avg Points per Region')
    plt.title('Region Density vs Avg Points')
    plt.grid(True, alpha=0.3)
    
    # 子图7: 相关性散点图 - 面积比例 vs 区域总数
    plt.subplot(2, 4, 7)
    plt.scatter(all_area_ratios, all_total_regions, alpha=0.6, s=10)
    plt.xlabel('Area Ratio')
    plt.ylabel('Total Regions')
    plt.title('Area Ratio vs Total Regions')
    plt.grid(True, alpha=0.3)
    
    # 子图8: 方法一致性可视化
    plt.subplot(2, 4, 8)
    consistency_data = []
    labels = ['Both Dense', 'Both Sparse', 'Inconsistent']
    colors = ['green', 'blue', 'red']
    inconsistent_count = global_small_dense_area_sparse + global_small_sparse_area_dense
    values = [global_both_dense, global_both_sparse, inconsistent_count]
    
    plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f'Method Consistency\n(Overall: {global_consistency_rate*100:.1f}%)')
    
    plt.tight_layout()
    plt.savefig('coordinate_region_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison analysis plot saved as 'coordinate_region_comparison_analysis.png'")
    
    # 详细分析：查看一些典型窗口的分布
    print(f"\n{'='*60}")
    print("DETAILED EXAMPLES")
    print(f"{'='*60}")
    
    # 找出一些有代表性的窗口
    sorted_windows = sorted([(i, stat) for i, source_stat in enumerate(all_source_stats) 
                           for stat in source_stat['window_details']], 
                          key=lambda x: x[1]['avg_points_per_region'])
    
    # 显示最少、中等、最多的几个例子
    example_indices = [0, len(sorted_windows)//4, len(sorted_windows)//2, 
                      3*len(sorted_windows)//4, len(sorted_windows)-1]
    
    for i, idx in enumerate(example_indices):
        if idx < len(sorted_windows):
            source_idx, window_stat = sorted_windows[idx]
            print(f"\nExample {i+1} (Source View {source_idx}):")
            print(f"  Window position: {window_stat['window_pos']}")
            print(f"  Total points: {window_stat['total_points']}")
            print(f"  Total small regions: {window_stat['total_regions']}")
            print(f"  Average points per region: {window_stat['avg_points_per_region']:.2f}")
            print(f"  Max points in single region: {window_stat['max_points_in_region']}")
            print(f"  Min points in single region: {window_stat['min_points_in_region']}")
            # 新增矩形面积方法的信息
            print(f"  Bounding box area: {window_stat['bbox_area']:.1f}")
            print(f"  Bounding box area ratio: {window_stat['area_ratio']:.4f}")
            print(f"  Region density: {window_stat['region_density']:.4f}")
            print(f"  Bounding box size: {window_stat['bbox_width']:.1f} x {window_stat['bbox_height']:.1f}")
            
            # 密集度判断对照
            small_region_result = "DENSE" if window_stat['is_dense_small_region'] else "SPARSE"
            area_ratio_result = "DENSE" if window_stat['is_dense_area_ratio'] else "SPARSE"
            consistency = "✓ CONSISTENT" if window_stat['methods_consistent'] else "✗ INCONSISTENT"
            print(f"  Classification: Small Region={small_region_result}, Area Ratio={area_ratio_result} - {consistency}")

print(f"\nAnalysis complete!")

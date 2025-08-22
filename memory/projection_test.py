import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_camera_matrix(fx, fy, cx, cy):
    """Create camera intrinsic matrix"""
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K

def create_pose_matrix(rotation, translation):
    """Create camera extrinsic matrix [R|t]"""
    pose = np.zeros((3, 4))
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose

def pixel_to_ray(pixels, K_inv, pose):
    """Convert pixel coordinates to rays in world coordinate system
    Args:
        pixels: (N, 2) pixel coordinates
        K_inv: (3, 3) inverse camera intrinsic matrix
        pose: (3, 4) camera extrinsic matrix [R|t]
    Returns:
        origins: (N, 3) ray origins (camera center in world coordinates)
        directions: (N, 3) ray directions (world coordinates)
    """
    # Convert pixel coordinates to homogeneous coordinates
    pixels_homo = np.column_stack([pixels, np.ones(len(pixels))])
    
    # Convert to camera coordinate directions
    ray_dirs_cam = (K_inv @ pixels_homo.T).T
    
    # Convert to world coordinates
    R = pose[:3, :3]
    t = pose[:3, 3]
    
    # Camera center in world coordinates
    camera_center = -R.T @ t
    origins = np.tile(camera_center, (len(pixels), 1))
    
    # Convert ray directions to world coordinates
    directions = (R.T @ ray_dirs_cam.T).T
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    
    return origins, directions

def sample_points_on_rays(origins, directions, depths):
    """在射线上采样点
    Args:
        origins: (N, 3) 射线起点
        directions: (N, 3) 射线方向
        depths: (M,) 深度值列表
    Returns:
        points_3d: (N*M, 3) 3D采样点
        ray_indices: (N*M,) 每个点对应的射线索引
        depth_values: (N*M,) 每个点的深度值
    """
    points_3d = []
    ray_indices = []
    depth_values = []
    
    for i, (origin, direction) in enumerate(zip(origins, directions)):
        for depth in depths:
            point = origin + depth * direction
            points_3d.append(point)
            ray_indices.append(i)
            depth_values.append(depth)
    
    return np.array(points_3d), np.array(ray_indices), np.array(depth_values)

def project_points_to_camera(points_3d, K, pose):
    """将3D点投影到相机像素坐标
    Args:
        points_3d: (N, 3) 3D点
        K: (3, 3) 相机内参
        pose: (3, 4) 相机外参 [R|t]
    Returns:
        pixels: (N, 2) 投影后的像素坐标
        depths: (N,) 在相机坐标系下的深度
    """
    R = pose[:3, :3]
    t = pose[:3, 3]
    
    # 转为相机坐标系
    points_cam = (R @ points_3d.T + t.reshape(-1, 1)).T
    
    # 投影到像素坐标
    pixels_homo = (K @ points_cam.T).T
    pixels = pixels_homo[:, :2] / pixels_homo[:, 2:3]
    depths = points_cam[:, 2]
    
    return pixels, depths

def calculate_bounding_rect_area(pixels):
    """计算投影点的外接矩形面积"""
    if len(pixels) == 0:
        return 0
    
    x_min, x_max = pixels[:, 0].min(), pixels[:, 0].max()
    y_min, y_max = pixels[:, 1].min(), pixels[:, 1].max()
    
    width = x_max - x_min
    height = y_max - y_min
    area = width * height
    
    return area, (x_min, y_min, x_max, y_max)

def test_projection_bounding_rect():
    """测试投影外接矩形计算"""
    print("=== 投影外接矩形测试 ===\n")
    
    # 相机内参
    fx, fy = 500, 500
    cx, cy = 320, 240
    K = create_camera_matrix(fx, fy, cx, cy)
    K_inv = np.linalg.inv(K)
    
    # 相机A的位姿（单位矩阵，原点）
    R_A = np.eye(3)
    t_A = np.array([0, 0, 0])
    pose_A = create_pose_matrix(R_A, t_A)
    
    # 相机B的位姿（稍微旋转和平移）
    angle = np.pi / 6  # 30度
    R_B = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    t_B = np.array([2, 1, 0.5])
    pose_B = create_pose_matrix(R_B, t_B)
    
    # 在相机A的像素平面上选择4个正方形顶点
    square_size = 100  # 像素
    center_x, center_y = cx, cy
    square_pixels = np.array([
        [center_x - square_size/2, center_y - square_size/2],  # 左下
        [center_x + square_size/2, center_y - square_size/2],  # 右下
        [center_x + square_size/2, center_y + square_size/2],  # 右上
        [center_x - square_size/2, center_y + square_size/2],  # 左上
    ])
    
    print(f"相机A像素正方形顶点:")
    for i, pixel in enumerate(square_pixels):
        print(f"  点{i+1}: ({pixel[0]:.1f}, {pixel[1]:.1f})")
    print()
    
    # 计算射线
    origins, directions = pixel_to_ray(square_pixels, K_inv, pose_A)
    
    # 在射线上采样不同深度的点
    depths = np.array([1, 2, 3, 4, 5, 8, 10, 15, 20])
    points_3d, ray_indices, depth_values = sample_points_on_rays(origins, directions, depths)
    
    print(f"在{len(depths)}个深度值上采样，总共{len(points_3d)}个3D点")
    print(f"深度范围: {depths.min():.1f} - {depths.max():.1f}")
    print()
    
    # 投影到相机B
    projected_pixels, projected_depths = project_points_to_camera(points_3d, K, pose_B)
    
    # 方法1: 使用所有投影点计算外接矩形
    area_all, bbox_all = calculate_bounding_rect_area(projected_pixels)
    
    # 方法2: 只使用最大和最小深度的点
    min_depth_idx = np.argmin(depth_values)
    max_depth_idx = np.argmax(depth_values)
    extreme_pixels = projected_pixels[[min_depth_idx, max_depth_idx]]
    area_extreme, bbox_extreme = calculate_bounding_rect_area(extreme_pixels)
    
    print("=== 结果比较 ===")
    print(f"方法1 - 使用所有投影点:")
    print(f"  外接矩形: ({bbox_all[0]:.1f}, {bbox_all[1]:.1f}) - ({bbox_all[2]:.1f}, {bbox_all[3]:.1f})")
    print(f"  面积: {area_all:.1f} 像素²")
    print()
    
    print(f"方法2 - 只使用最大最小深度点:")
    print(f"  外接矩形: ({bbox_extreme[0]:.1f}, {bbox_extreme[1]:.1f}) - ({bbox_extreme[2]:.1f}, {bbox_extreme[3]:.1f})")
    print(f"  面积: {area_extreme:.1f} 像素²")
    print()
    
    error_percentage = abs(area_all - area_extreme) / area_all * 100
    print(f"面积误差: {abs(area_all - area_extreme):.1f} 像素² ({error_percentage:.1f}%)")
    
    if error_percentage > 5:  # 如果误差超过5%
        print("❌ 结论: 不能只用最大最小深度点来计算外接矩形面积！")
    else:
        print("✅ 在这种情况下，使用最大最小深度点的近似结果还算可以")
    
    # 可视化
    visualize_projection_test(square_pixels, projected_pixels, depth_values, ray_indices, 
                            bbox_all, bbox_extreme, depths)

def visualize_projection_test(original_pixels, projected_pixels, depth_values, ray_indices, 
                            bbox_all, bbox_extreme, depths):
    """Visualize projection test results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Original pixel square
    ax1.plot(original_pixels[[0,1,2,3,0], 0], original_pixels[[0,1,2,3,0], 1], 'r-o', linewidth=2, markersize=8)
    ax1.set_title('Camera A - Original Pixel Square')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.grid(True)
    ax1.set_aspect('equal')
    
    # 2. All projected points (colored by depth)
    scatter = ax2.scatter(projected_pixels[:, 0], projected_pixels[:, 1], 
                         c=depth_values, cmap='viridis', s=30, alpha=0.7)
    
    # Draw true bounding rectangle
    rect_all = plt.Rectangle((bbox_all[0], bbox_all[1]), 
                           bbox_all[2]-bbox_all[0], bbox_all[3]-bbox_all[1], 
                           fill=False, edgecolor='red', linewidth=2, label='All points bounding box')
    ax2.add_patch(rect_all)
    
    # Draw extreme points bounding rectangle
    rect_extreme = plt.Rectangle((bbox_extreme[0], bbox_extreme[1]), 
                               bbox_extreme[2]-bbox_extreme[0], bbox_extreme[3]-bbox_extreme[1], 
                               fill=False, edgecolor='blue', linewidth=2, linestyle='--', label='Extreme points bounding box')
    ax2.add_patch(rect_extreme)
    
    ax2.set_title('Camera B - Projected Points and Bounding Rectangle Comparison')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.legend()
    ax2.grid(True)
    plt.colorbar(scatter, ax=ax2, label='Depth value')
    
    # 3. Display by ray grouping
    colors = ['red', 'green', 'blue', 'orange']
    for ray_idx in range(4):
        mask = ray_indices == ray_idx
        points = projected_pixels[mask]
        depths_ray = depth_values[mask]
        ax3.plot(depths_ray, points[:, 0], 'o-', color=colors[ray_idx], 
                label=f'Ray{ray_idx+1} - X coordinate', alpha=0.7)
    
    ax3.set_title('X Coordinates of Projected Points vs Depth for Each Ray')
    ax3.set_xlabel('Depth value')
    ax3.set_ylabel('Projected X coordinate (pixels)')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Y coordinate variation
    for ray_idx in range(4):
        mask = ray_indices == ray_idx
        points = projected_pixels[mask]
        depths_ray = depth_values[mask]
        ax4.plot(depths_ray, points[:, 1], 's-', color=colors[ray_idx], 
                label=f'Ray{ray_idx+1} - Y coordinate', alpha=0.7)
    
    ax4.set_title('Y Coordinates of Projected Points vs Depth for Each Ray')
    ax4.set_xlabel('Depth value')
    ax4.set_ylabel('Projected Y coordinate (pixels)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/ytanaz/access/IBRNet/memory/projection_test_result.png', dpi=150, bbox_inches='tight')
    plt.show()

def test_different_camera_poses():
    """Test different camera poses"""
    print("\n=== Testing Different Camera Poses ===\n")
    
    # 相机内参
    fx, fy = 500, 500
    cx, cy = 320, 240
    K = create_camera_matrix(fx, fy, cx, cy)
    K_inv = np.linalg.inv(K)
    
    # 相机A
    R_A = np.eye(3)
    t_A = np.array([0, 0, 0])
    pose_A = create_pose_matrix(R_A, t_A)
    
    # 像素正方形
    square_size = 80
    center_x, center_y = cx, cy
    square_pixels = np.array([
        [center_x - square_size/2, center_y - square_size/2],
        [center_x + square_size/2, center_y - square_size/2],
        [center_x + square_size/2, center_y + square_size/2],
        [center_x - square_size/2, center_y + square_size/2],
    ])
    
    origins, directions = pixel_to_ray(square_pixels, K_inv, pose_A)
    depths = np.linspace(1, 20, 10)
    points_3d, ray_indices, depth_values = sample_points_on_rays(origins, directions, depths)
    
    # 测试不同的相机B位姿
    test_cases = [
        ("小角度旋转", np.pi/12, [1, 0, 0]),
        ("大角度旋转", np.pi/3, [2, 1, 0]),
        ("垂直移动", 0, [0, 0, 3]),
        ("复杂位姿", np.pi/4, [3, 2, 1])
    ]
    
    for name, angle, translation in test_cases:
        R_B = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        t_B = np.array(translation)
        pose_B = create_pose_matrix(R_B, t_B)
        
        projected_pixels, _ = project_points_to_camera(points_3d, K, pose_B)
        
        area_all, _ = calculate_bounding_rect_area(projected_pixels)
        
        min_depth_idx = np.argmin(depth_values)
        max_depth_idx = np.argmax(depth_values)
        extreme_pixels = projected_pixels[[min_depth_idx, max_depth_idx]]
        area_extreme, _ = calculate_bounding_rect_area(extreme_pixels)
        
        error_percentage = abs(area_all - area_extreme) / area_all * 100
        
        print(f"{name}:")
        print(f"  完整计算面积: {area_all:.1f}")
        print(f"  极值点面积: {area_extreme:.1f}")
        print(f"  误差: {error_percentage:.1f}%")
        print()

if __name__ == "__main__":
    test_projection_bounding_rect()
    test_different_camera_poses()

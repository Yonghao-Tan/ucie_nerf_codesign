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

def project_points_to_camera(points_3d, K, pose):
    """Project 3D points to camera pixel coordinates
    Args:
        points_3d: (N, 3) 3D points in world coordinates
        K: (3, 3) camera intrinsic matrix
        pose: (3, 4) camera extrinsic matrix [R|t]
    Returns:
        pixels: (N, 2) projected pixel coordinates
    """
    R = pose[:3, :3]
    t = pose[:3, 3]
    
    # Transform to camera coordinates
    points_cam = (R @ points_3d.T + t.reshape(-1, 1)).T
    
    # Project to image plane
    pixels_homo = (K @ points_cam.T).T
    pixels = pixels_homo[:, :2] / pixels_homo[:, 2:3]
    
    return pixels

def main_test():
    """Main test function: does only using min/max depth points give correct bounding box?"""
    print("=== Testing: Can we calculate bounding rectangle area using only min/max depth points? ===\n")
    
    # Camera parameters
    fx, fy = 500, 500  # focal length
    cx, cy = 320, 240  # principal point
    image_width, image_height = 640, 480
    
    K = create_camera_matrix(fx, fy, cx, cy)
    K_inv = np.linalg.inv(K)
    
    # Camera A (reference camera)
    R_A = np.eye(3)
    t_A = np.array([0, 0, 0])
    pose_A = create_pose_matrix(R_A, t_A)
    
    # Camera B (target camera) - slightly rotated and translated
    angle = np.radians(15)  # 15 degree rotation
    R_B = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    t_B = np.array([2, 1, 0])  # translation
    pose_B = create_pose_matrix(R_B, t_B)
    
    # Create square pixels on camera A
    center_x, center_y = cx, cy
    square_size = 40  # pixel units
    half_size = square_size // 2
    
    square_pixels = np.array([
        [center_x - half_size, center_y - half_size],  # top-left
        [center_x + half_size, center_y - half_size],  # top-right
        [center_x + half_size, center_y + half_size],  # bottom-right
        [center_x - half_size, center_y + half_size],  # bottom-left
    ])
    
    # Convert to rays in world coordinates
    ray_origins, ray_directions = pixel_to_ray(square_pixels, K_inv, pose_A)
    
    # Sample points along rays at different depths
    depth_min, depth_max = 2.0, 10.0
    num_samples_per_ray = 20
    
    all_points_3d = []
    all_depths = []
    all_ray_indices = []
    
    for ray_idx in range(4):  # 4 rays for 4 corner pixels
        depths = np.linspace(depth_min, depth_max, num_samples_per_ray)
        origin = ray_origins[ray_idx]
        direction = ray_directions[ray_idx]
        
        # Points along this ray
        points_3d = origin + depths.reshape(-1, 1) * direction
        
        all_points_3d.append(points_3d)
        all_depths.extend(depths)
        all_ray_indices.extend([ray_idx] * num_samples_per_ray)
    
    # Combine all points
    all_points_3d = np.vstack(all_points_3d)
    all_depths = np.array(all_depths)
    all_ray_indices = np.array(all_ray_indices)
    
    # Project all points to camera B
    projected_pixels = project_points_to_camera(all_points_3d, K, pose_B)
    
    # Calculate TRUE bounding rectangle (using all projected points)
    x_min_all = np.min(projected_pixels[:, 0])
    x_max_all = np.max(projected_pixels[:, 0])
    y_min_all = np.min(projected_pixels[:, 1])
    y_max_all = np.max(projected_pixels[:, 1])
    area_all = (x_max_all - x_min_all) * (y_max_all - y_min_all)
    
    # Calculate WRONG bounding rectangle (using only min/max depth points)
    min_depth_idx = np.argmin(all_depths)
    max_depth_idx = np.argmax(all_depths)
    
    extreme_points = projected_pixels[[min_depth_idx, max_depth_idx]]
    x_min_extreme = np.min(extreme_points[:, 0])
    x_max_extreme = np.max(extreme_points[:, 0])
    y_min_extreme = np.min(extreme_points[:, 1])
    y_max_extreme = np.max(extreme_points[:, 1])
    area_extreme = (x_max_extreme - x_min_extreme) * (y_max_extreme - y_min_extreme)
    
    # Results
    print(f"True bounding rectangle area (all points): {area_all:.2f}")
    print(f"Wrong bounding rectangle area (only min/max depth): {area_extreme:.2f}")
    print(f"Area difference: {abs(area_all - area_extreme):.2f}")
    print(f"Relative error: {abs(area_all - area_extreme) / area_all * 100:.1f}%")
    print()
    
    if abs(area_all - area_extreme) / area_all > 0.01:  # > 1% error
        print("❌ CONCLUSION: You CANNOT use only min/max depth points!")
        print("   The perspective projection is non-linear, and points with intermediate")
        print("   depths may have more extreme projection coordinates.")
    else:
        print("✅ In this specific case, min/max depth points happen to work")
        print("   But this is not guaranteed in general!")
    
    # Visualization
    visualize_projection_test(square_pixels, all_points_3d, projected_pixels, 
                            all_depths, all_ray_indices, 
                            [x_min_all, y_min_all, x_max_all, y_max_all],
                            [x_min_extreme, y_min_extreme, x_max_extreme, y_max_extreme])
    
    return area_all, area_extreme

def visualize_projection_test(original_pixels, points_3d, projected_pixels, 
                            depth_values, ray_indices, bbox_all, bbox_extreme):
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
    plt.savefig('/home/ytanaz/access/IBRNet/memory/projection_test_result_en.png', dpi=150, bbox_inches='tight')
    plt.show()

def test_different_camera_poses():
    """Test different camera poses"""
    print("\n=== Testing Different Camera Poses ===\n")
    
    # Camera parameters
    fx, fy = 500, 500
    cx, cy = 320, 240
    K = create_camera_matrix(fx, fy, cx, cy)
    K_inv = np.linalg.inv(K)
    
    # Camera A
    R_A = np.eye(3)
    t_A = np.array([0, 0, 0])
    pose_A = create_pose_matrix(R_A, t_A)
    
    # Test different camera B poses
    test_configs = [
        {"name": "Small rotation", "angle": 5, "translation": [0.5, 0, 0]},
        {"name": "Medium rotation", "angle": 30, "translation": [2, 1, 0]},
        {"name": "Large rotation", "angle": 60, "translation": [3, 2, 1]},
    ]
    
    square_pixels = np.array([
        [cx-50, cy-50], [cx+50, cy-50], 
        [cx+50, cy+50], [cx-50, cy+50]
    ])
    
    ray_origins, ray_directions = pixel_to_ray(square_pixels, K_inv, pose_A)
    
    for config in test_configs:
        print(f"Testing: {config['name']}")
        
        # Create camera B pose
        angle = np.radians(config["angle"])
        R_B = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        t_B = np.array(config["translation"])
        pose_B = create_pose_matrix(R_B, t_B)
        
        # Sample points and project
        depths = np.linspace(1.0, 8.0, 15)
        all_points = []
        all_depths_list = []
        
        for ray_idx in range(4):
            origin = ray_origins[ray_idx]
            direction = ray_directions[ray_idx]
            points_3d = origin + depths.reshape(-1, 1) * direction
            all_points.append(points_3d)
            all_depths_list.extend(depths)
        
        all_points = np.vstack(all_points)
        all_depths_array = np.array(all_depths_list)
        
        projected_pixels = project_points_to_camera(all_points, K, pose_B)
        
        # True vs wrong area calculation
        area_true = (np.max(projected_pixels[:, 0]) - np.min(projected_pixels[:, 0])) * \
                   (np.max(projected_pixels[:, 1]) - np.min(projected_pixels[:, 1]))
        
        min_depth_idx = np.argmin(all_depths_array)
        max_depth_idx = np.argmax(all_depths_array)
        extreme_pixels = projected_pixels[[min_depth_idx, max_depth_idx]]
        area_wrong = (np.max(extreme_pixels[:, 0]) - np.min(extreme_pixels[:, 0])) * \
                    (np.max(extreme_pixels[:, 1]) - np.min(extreme_pixels[:, 1]))
        
        error_percent = abs(area_true - area_wrong) / area_true * 100
        
        print(f"  True area: {area_true:.2f}, Wrong area: {area_wrong:.2f}")
        print(f"  Relative error: {error_percent:.1f}%")
        print()

def test_aligned_depth_layers():
    """Test if we can use only min/max depth layers when sampling points are depth-aligned"""
    print("\n=== Testing Aligned Depth Layers ===\n")
    
    # Camera parameters
    fx, fy = 500, 500
    cx, cy = 320, 240
    K = create_camera_matrix(fx, fy, cx, cy)
    K_inv = np.linalg.inv(K)
    
    # Camera A (reference)
    R_A = np.eye(3)
    t_A = np.array([0, 0, 0])
    pose_A = create_pose_matrix(R_A, t_A)
    
    # Camera B (different pose)
    angle = np.pi / 3  # 60 degrees
    R_B = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    t_B = np.array([2, 1, 0.5])
    pose_B = create_pose_matrix(R_B, t_B)
    
    # Original square in camera A pixel plane
    square_size = 100
    square_pixels = np.array([
        [cx-square_size/2, cy-square_size/2],  # bottom-left
        [cx+square_size/2, cy-square_size/2],  # bottom-right
        [cx+square_size/2, cy+square_size/2],  # top-right
        [cx-square_size/2, cy+square_size/2],  # top-left
    ])
    
    print("Testing scenario: Multiple sampling points on each ray")
    print("- All rays have same depth values at each sampling layer")
    print("- Question: Can we use only min/max depth layers?")
    print()
    
    # Get rays from camera A
    ray_origins, ray_directions = pixel_to_ray(square_pixels, K_inv, pose_A)
    
    # Create aligned depth layers - all rays sample at same depths
    depth_layers = np.array([1.0, 2.5, 4.0, 6.0, 8.5, 12.0, 16.0])
    print(f"Depth layers: {depth_layers}")
    print(f"Min depth: {depth_layers.min()}, Max depth: {depth_layers.max()}")
    print()
    
    # Method 1: Sample all points on all rays at all depths
    all_points_3d = []
    all_ray_indices = []
    all_depths = []
    
    for ray_idx in range(4):
        origin = ray_origins[ray_idx]
        direction = ray_directions[ray_idx]
        
        for depth in depth_layers:
            point_3d = origin + depth * direction
            all_points_3d.append(point_3d)
            all_ray_indices.append(ray_idx)
            all_depths.append(depth)
    
    all_points_3d = np.array(all_points_3d)
    all_depths = np.array(all_depths)
    
    # Project all points to camera B
    all_projected = project_points_to_camera(all_points_3d, K, pose_B)
    
    # Calculate true bounding rectangle
    x_min_true = all_projected[:, 0].min()
    x_max_true = all_projected[:, 0].max()
    y_min_true = all_projected[:, 1].min()
    y_max_true = all_projected[:, 1].max()
    area_true = (x_max_true - x_min_true) * (y_max_true - y_min_true)
    
    print("Method 1 - Using ALL sampling points:")
    print(f"  Bounding box: ({x_min_true:.2f}, {y_min_true:.2f}) to ({x_max_true:.2f}, {y_max_true:.2f})")
    print(f"  Area: {area_true:.2f}")
    print()
    
    # Method 2: Use only min and max depth layers (8 points total)
    min_depth = depth_layers.min()
    max_depth = depth_layers.max()
    
    min_depth_points = []
    max_depth_points = []
    
    for ray_idx in range(4):
        origin = ray_origins[ray_idx]
        direction = ray_directions[ray_idx]
        
        min_point = origin + min_depth * direction
        max_point = origin + max_depth * direction
        
        min_depth_points.append(min_point)
        max_depth_points.append(max_point)
    
    extreme_points_3d = np.vstack([min_depth_points, max_depth_points])
    extreme_projected = project_points_to_camera(extreme_points_3d, K, pose_B)
    
    x_min_extreme = extreme_projected[:, 0].min()
    x_max_extreme = extreme_projected[:, 0].max()
    y_min_extreme = extreme_projected[:, 1].min()
    y_max_extreme = extreme_projected[:, 1].max()
    area_extreme = (x_max_extreme - x_min_extreme) * (y_max_extreme - y_min_extreme)
    
    print("Method 2 - Using only MIN/MAX depth layers:")
    print(f"  Bounding box: ({x_min_extreme:.2f}, {y_min_extreme:.2f}) to ({x_max_extreme:.2f}, {y_max_extreme:.2f})")
    print(f"  Area: {area_extreme:.2f}")
    print()
    
    # Method 3: Use only the 2 most extreme points (one min, one max)
    min_depth_idx = np.argmin(all_depths)
    max_depth_idx = np.argmax(all_depths)
    two_extreme_projected = all_projected[[min_depth_idx, max_depth_idx]]
    
    x_min_two = two_extreme_projected[:, 0].min()
    x_max_two = two_extreme_projected[:, 0].max()
    y_min_two = two_extreme_projected[:, 1].min()
    y_max_two = two_extreme_projected[:, 1].max()
    area_two = (x_max_two - x_min_two) * (y_max_two - y_min_two)
    
    print("Method 3 - Using only 2 most extreme depth points:")
    print(f"  Bounding box: ({x_min_two:.2f}, {y_min_two:.2f}) to ({x_max_two:.2f}, {y_max_two:.2f})")
    print(f"  Area: {area_two:.2f}")
    print()
    
    # Compare errors
    error_extreme = abs(area_true - area_extreme) / area_true * 100
    error_two = abs(area_true - area_two) / area_true * 100
    
    print("=== COMPARISON ===")
    print(f"True area (all points): {area_true:.2f}")
    print(f"Min/max layers area: {area_extreme:.2f} (error: {error_extreme:.2f}%)")
    print(f"Two extreme points area: {area_two:.2f} (error: {error_two:.2f}%)")
    print()
    
    if error_extreme < 1.0:
        print("✅ Using min/max depth LAYERS is accurate!")
    else:
        print("❌ Using min/max depth LAYERS still has significant error!")
        
    if error_two < 1.0:
        print("✅ Using just 2 extreme points is accurate!")
    else:
        print("❌ Using just 2 extreme points has significant error!")

def test_same_depth_optimization():
    """Test if we can use only corner points when all points have the same depth"""
    print("\n=== Testing Same Depth Optimization ===\n")
    
    # Camera parameters
    fx, fy = 500, 500
    cx, cy = 320, 240
    K = create_camera_matrix(fx, fy, cx, cy)
    K_inv = np.linalg.inv(K)
    
    # Camera A (reference)
    R_A = np.eye(3)
    t_A = np.array([0, 0, 0])
    pose_A = create_pose_matrix(R_A, t_A)
    
    # Camera B (different pose)
    angle = np.pi / 4  # 45 degrees
    R_B = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    t_B = np.array([3, 2, 1])
    pose_B = create_pose_matrix(R_B, t_B)
    
    # Original square in camera A pixel plane
    square_size = 100
    square_pixels = np.array([
        [cx-square_size/2, cy-square_size/2],  # bottom-left
        [cx+square_size/2, cy-square_size/2],  # bottom-right
        [cx+square_size/2, cy+square_size/2],  # top-right
        [cx-square_size/2, cy+square_size/2],  # top-left
    ])
    
    print("Original square corners in Camera A:")
    for i, pixel in enumerate(square_pixels):
        print(f"  Corner {i+1}: ({pixel[0]:.1f}, {pixel[1]:.1f})")
    print()
    
    # Get rays from camera A
    ray_origins, ray_directions = pixel_to_ray(square_pixels, K_inv, pose_A)
    
    # Test different depths
    test_depths = [2.0, 5.0, 10.0, 15.0]
    
    for depth in test_depths:
        print(f"Testing at depth = {depth}")
        
        # Method 1: Use only the 4 corner points at this depth
        corner_points_3d = ray_origins + depth * ray_directions
        corner_projected = project_points_to_camera(corner_points_3d, K, pose_B)
        
        # Calculate bounding rectangle from corners only
        x_min_corner = corner_projected[:, 0].min()
        x_max_corner = corner_projected[:, 0].max()
        y_min_corner = corner_projected[:, 1].min()
        y_max_corner = corner_projected[:, 1].max()
        area_corner = (x_max_corner - x_min_corner) * (y_max_corner - y_min_corner)
        
        # Method 2: Sample many points on the square boundary and interior
        # Create dense sampling within the original square
        n_samples = 20
        x_samples = np.linspace(cx-square_size/2, cx+square_size/2, n_samples)
        y_samples = np.linspace(cy-square_size/2, cy+square_size/2, n_samples)
        xx, yy = np.meshgrid(x_samples, y_samples)
        dense_pixels = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Get rays for all dense pixels
        dense_origins, dense_directions = pixel_to_ray(dense_pixels, K_inv, pose_A)
        
        # Sample at the same depth
        dense_points_3d = dense_origins + depth * dense_directions
        dense_projected = project_points_to_camera(dense_points_3d, K, pose_B)
        
        # Calculate bounding rectangle from all dense points
        x_min_dense = dense_projected[:, 0].min()
        x_max_dense = dense_projected[:, 0].max()
        y_min_dense = dense_projected[:, 1].min()
        y_max_dense = dense_projected[:, 1].max()
        area_dense = (x_max_dense - x_min_dense) * (y_max_dense - y_min_dense)
        
        # Compare results
        error_abs = abs(area_corner - area_dense)
        error_rel = error_abs / area_dense * 100
        
        print(f"  Corner-only method area: {area_corner:.2f}")
        print(f"  Dense sampling area: {area_dense:.2f}")
        print(f"  Absolute error: {error_abs:.4f}")
        print(f"  Relative error: {error_rel:.4f}%")
        
        if error_rel < 0.1:  # Less than 0.1% error
            print("  ✅ Corner-only method is ACCURATE!")
        else:
            print("  ❌ Corner-only method has significant error!")
        print()

def test_corner_optimization_theory():
    """Theoretical verification of corner-only method for same depth"""
    print("\n=== Theoretical Verification ===\n")
    
    print("When all points have the same depth:")
    print("1. All 3D points lie on a plane parallel to the original camera plane")
    print("2. The projection from this plane to camera B is an affine transformation")
    print("3. Affine transformations preserve:")
    print("   - Parallel lines remain parallel")
    print("   - Ratios of parallel line segments")
    print("   - The convex hull structure")
    print("4. Therefore, the bounding rectangle of the projection equals")
    print("   the bounding rectangle of the projected corner points!")
    print()
    
    # Demonstrate with a simple mathematical example
    print("Mathematical proof:")
    print("- Original square corners: (x₁,y₁), (x₂,y₁), (x₂,y₂), (x₁,y₂)")
    print("- After affine transformation T: T(x₁,y₁), T(x₂,y₁), T(x₂,y₂), T(x₁,y₂)")
    print("- Any point inside original square: αT(x₁,y₁) + βT(x₂,y₁) + γT(x₂,y₂) + δT(x₁,y₂)")
    print("  where α+β+γ+δ=1 and α,β,γ,δ≥0")
    print("- This convex combination stays within the bounding box of corner points")
    print("- Therefore: min/max coordinates are achieved at corner points only!")

if __name__ == "__main__":
    # Run main test
    main_test()
    
    # Test aligned depth layers (your specific question)
    test_aligned_depth_layers()
    
    # Run additional tests
    test_different_camera_poses()
    
    # Test same depth optimization
    test_same_depth_optimization()
    
    # Theoretical explanation
    test_corner_optimization_theory()
    
    print("\n" + "="*60)
    print("CONCLUSIONS:")
    print("1. CANNOT use min/max depth points for varying depths")
    print("2. CAN use only corner points when depth is constant!")
    print("3. Same-depth case: projection is affine transformation")
    print("4. Aligned depth layers: depends on camera geometry")
    print("="*60)

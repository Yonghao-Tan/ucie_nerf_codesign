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
    """Convert pixels to rays in world coordinates"""
    pixels_homo = np.column_stack([pixels, np.ones(len(pixels))])
    ray_dirs_cam = (K_inv @ pixels_homo.T).T
    
    R = pose[:3, :3]
    t = pose[:3, 3]
    
    camera_center = -R.T @ t
    origins = np.tile(camera_center, (len(pixels), 1))
    directions = (R.T @ ray_dirs_cam.T).T
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    
    return origins, directions

def project_points_to_camera(points_3d, K, pose):
    """Project 3D points to camera pixels"""
    R = pose[:3, :3]
    t = pose[:3, 3]
    
    points_cam = (R @ points_3d.T + t.reshape(-1, 1)).T
    pixels_homo = (K @ points_cam.T).T
    pixels = pixels_homo[:, :2] / pixels_homo[:, 2:3]
    depths = points_cam[:, 2]
    
    return pixels, depths

def create_frustum_from_rays(ray_origins, ray_directions, near_depth, far_depth):
    """Create a frustum from 4 rays and near/far depths"""
    # Near plane points (4 corners)
    near_points = ray_origins + near_depth * ray_directions
    
    # Far plane points (4 corners)
    far_points = ray_origins + far_depth * ray_directions
    
    # All 8 vertices of the frustum
    frustum_vertices = np.vstack([near_points, far_points])
    
    return frustum_vertices, near_points, far_points

def sample_frustum_edges(ray_origins, ray_directions, near_depth, far_depth, n_samples=50):
    """Sample points along the edges of the frustum"""
    edge_points = []
    
    # Sample along each ray (4 edges connecting near and far)
    for i in range(4):
        depths = np.linspace(near_depth, far_depth, n_samples)
        for depth in depths:
            point = ray_origins[i] + depth * ray_directions[i]
            edge_points.append(point)
    
    # Sample along near plane edges (4 edges)
    near_points = ray_origins + near_depth * ray_directions
    for i in range(4):
        next_i = (i + 1) % 4
        for t in np.linspace(0, 1, n_samples):
            point = (1 - t) * near_points[i] + t * near_points[next_i]
            edge_points.append(point)
    
    # Sample along far plane edges (4 edges)
    far_points = ray_origins + far_depth * ray_directions
    for i in range(4):
        next_i = (i + 1) % 4
        for t in np.linspace(0, 1, n_samples):
            point = (1 - t) * far_points[i] + t * far_points[next_i]
            edge_points.append(point)
    
    return np.array(edge_points)

def test_frustum_projection():
    """Test if frustum vertices alone are sufficient for bounding rectangle"""
    print("=== Frustum Projection Test ===\n")
    
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
    
    # Original square in camera A
    square_size = 100
    square_pixels = np.array([
        [cx-square_size/2, cy-square_size/2],  # bottom-left
        [cx+square_size/2, cy-square_size/2],  # bottom-right
        [cx+square_size/2, cy+square_size/2],  # top-right
        [cx-square_size/2, cy+square_size/2],  # top-left
    ])
    
    print("Original square corners:")
    for i, pixel in enumerate(square_pixels):
        print(f"  Corner {i+1}: ({pixel[0]:.1f}, {pixel[1]:.1f})")
    print()
    
    # Get rays
    ray_origins, ray_directions = pixel_to_ray(square_pixels, K_inv, pose_A)
    
    # Define near and far depths
    near_depth = 2.0
    far_depth = 10.0
    
    print(f"Frustum depths: near = {near_depth}, far = {far_depth}")
    print()
    
    # Create frustum
    frustum_vertices, near_points, far_points = create_frustum_from_rays(
        ray_origins, ray_directions, near_depth, far_depth)
    
    print("Frustum structure:")
    print(f"  Near plane (depth {near_depth}):")
    for i, pt in enumerate(near_points):
        print(f"    Vertex {i+1}: ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})")
    print(f"  Far plane (depth {far_depth}):")
    for i, pt in enumerate(far_points):
        print(f"    Vertex {i+5}: ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})")
    print()
    
    # Method 1: Use only the 8 vertices
    vertices_projected, _ = project_points_to_camera(frustum_vertices, K, pose_B)
    
    x_min_vertices = vertices_projected[:, 0].min()
    x_max_vertices = vertices_projected[:, 0].max()
    y_min_vertices = vertices_projected[:, 1].min()
    y_max_vertices = vertices_projected[:, 1].max()
    area_vertices = (x_max_vertices - x_min_vertices) * (y_max_vertices - y_min_vertices)
    
    # Method 2: Sample points along all edges
    edge_points = sample_frustum_edges(ray_origins, ray_directions, near_depth, far_depth, n_samples=20)
    edge_projected, _ = project_points_to_camera(edge_points, K, pose_B)
    
    x_min_edges = edge_projected[:, 0].min()
    x_max_edges = edge_projected[:, 0].max()
    y_min_edges = edge_projected[:, 1].min()
    y_max_edges = edge_projected[:, 1].max()
    area_edges = (x_max_edges - x_min_edges) * (y_max_edges - y_min_edges)
    
    # Method 3: Dense sampling inside the frustum
    dense_points = []
    depth_samples = np.linspace(near_depth, far_depth, 10)
    
    for depth in depth_samples:
        # Sample within the square at this depth
        points_at_depth = ray_origins + depth * ray_directions
        
        # Create grid within the quadrilateral at this depth
        for u in np.linspace(0, 1, 8):
            for v in np.linspace(0, 1, 8):
                # Bilinear interpolation within the quad
                p1 = (1-u) * (1-v) * points_at_depth[0]  # bottom-left
                p2 = u * (1-v) * points_at_depth[1]      # bottom-right
                p3 = u * v * points_at_depth[2]          # top-right
                p4 = (1-u) * v * points_at_depth[3]      # top-left
                
                point = p1 + p2 + p3 + p4
                dense_points.append(point)
    
    dense_points = np.array(dense_points)
    dense_projected, _ = project_points_to_camera(dense_points, K, pose_B)
    
    x_min_dense = dense_projected[:, 0].min()
    x_max_dense = dense_projected[:, 0].max()
    y_min_dense = dense_projected[:, 1].min()
    y_max_dense = dense_projected[:, 1].max()
    area_dense = (x_max_dense - x_min_dense) * (y_max_dense - y_min_dense)
    
    # Results
    print("=== Projection Results ===")
    print(f"Method 1 - Only 8 vertices:")
    print(f"  Bounding box: ({x_min_vertices:.2f}, {y_min_vertices:.2f}) to ({x_max_vertices:.2f}, {y_max_vertices:.2f})")
    print(f"  Area: {area_vertices:.2f}")
    print()
    
    print(f"Method 2 - Edge sampling:")
    print(f"  Bounding box: ({x_min_edges:.2f}, {y_min_edges:.2f}) to ({x_max_edges:.2f}, {y_max_edges:.2f})")
    print(f"  Area: {area_edges:.2f}")
    print()
    
    print(f"Method 3 - Dense sampling:")
    print(f"  Bounding box: ({x_min_dense:.2f}, {y_min_dense:.2f}) to ({x_max_dense:.2f}, {y_max_dense:.2f})")
    print(f"  Area: {area_dense:.2f}")
    print()
    
    # Compare errors
    error_vertices = abs(area_dense - area_vertices) / area_dense * 100
    error_edges = abs(area_dense - area_edges) / area_dense * 100
    
    print("=== Error Analysis ===")
    print(f"Vertices-only error: {error_vertices:.2f}%")
    print(f"Edge sampling error: {error_edges:.2f}%")
    print()
    
    if error_vertices < 1.0:
        print("✅ Using only frustum vertices is ACCURATE!")
        print("   The 8 corner points are sufficient for bounding rectangle calculation.")
    else:
        print("❌ Using only frustum vertices has significant error!")
        print("   Need to consider edge points or interior points.")
    
    if error_edges < 0.1:
        print("✅ Edge sampling captures the true bounding rectangle.")
    
    # Visualize
    visualize_frustum_projection(frustum_vertices, edge_points, dense_points, 
                               vertices_projected, edge_projected, dense_projected)

def visualize_frustum_projection(frustum_vertices, edge_points, dense_points, 
                               vertices_proj, edge_proj, dense_proj):
    """Visualize the frustum and its projection"""
    fig = plt.figure(figsize=(15, 10))
    
    # 3D view of frustum
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Plot frustum vertices
    near_verts = frustum_vertices[:4]
    far_verts = frustum_vertices[4:]
    
    # Plot near and far plane squares
    near_square = np.vstack([near_verts, near_verts[0]])
    far_square = np.vstack([far_verts, far_verts[0]])
    
    ax1.plot(near_square[:, 0], near_square[:, 1], near_square[:, 2], 'r-', linewidth=2, label='Near plane')
    ax1.plot(far_square[:, 0], far_square[:, 1], far_square[:, 2], 'b-', linewidth=2, label='Far plane')
    
    # Connect corresponding vertices
    for i in range(4):
        ax1.plot([near_verts[i, 0], far_verts[i, 0]], 
                [near_verts[i, 1], far_verts[i, 1]], 
                [near_verts[i, 2], far_verts[i, 2]], 'g-', alpha=0.7)
    
    ax1.scatter(frustum_vertices[:, 0], frustum_vertices[:, 1], frustum_vertices[:, 2], 
               c='red', s=50, label='Vertices')
    ax1.set_title('3D Frustum')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # Projected vertices only
    ax2 = fig.add_subplot(222)
    ax2.scatter(vertices_proj[:, 0], vertices_proj[:, 1], c='red', s=50, label='8 Vertices', alpha=0.8)
    
    # Draw bounding rectangle
    x_min, x_max = vertices_proj[:, 0].min(), vertices_proj[:, 0].max()
    y_min, y_max = vertices_proj[:, 1].min(), vertices_proj[:, 1].max()
    rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                        fill=False, edgecolor='red', linewidth=2)
    ax2.add_patch(rect)
    
    ax2.set_title('Projection: Vertices Only')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.grid(True)
    ax2.legend()
    
    # Projected edge points
    ax3 = fig.add_subplot(223)
    ax3.scatter(edge_proj[:, 0], edge_proj[:, 1], c='blue', s=10, alpha=0.6, label='Edge points')
    ax3.scatter(vertices_proj[:, 0], vertices_proj[:, 1], c='red', s=50, label='Vertices')
    
    # Draw bounding rectangles
    x_min_edge, x_max_edge = edge_proj[:, 0].min(), edge_proj[:, 0].max()
    y_min_edge, y_max_edge = edge_proj[:, 1].min(), edge_proj[:, 1].max()
    rect_edge = plt.Rectangle((x_min_edge, y_min_edge), x_max_edge-x_min_edge, y_max_edge-y_min_edge, 
                             fill=False, edgecolor='blue', linewidth=2, label='Edge bbox')
    ax3.add_patch(rect_edge)
    
    rect_vert = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                             fill=False, edgecolor='red', linewidth=2, linestyle='--', label='Vertex bbox')
    ax3.add_patch(rect_vert)
    
    ax3.set_title('Projection: Vertices vs Edge Points')
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    ax3.grid(True)
    ax3.legend()
    
    # All points comparison
    ax4 = fig.add_subplot(224)
    ax4.scatter(dense_proj[:, 0], dense_proj[:, 1], c='lightblue', s=5, alpha=0.4, label='Dense sampling')
    ax4.scatter(edge_proj[:, 0], edge_proj[:, 1], c='blue', s=10, alpha=0.6, label='Edge points')
    ax4.scatter(vertices_proj[:, 0], vertices_proj[:, 1], c='red', s=50, label='Vertices')
    
    ax4.set_title('Projection: All Methods Comparison')
    ax4.set_xlabel('X (pixels)')
    ax4.set_ylabel('Y (pixels)')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('/home/ytanaz/access/IBRNet/memory/frustum_projection_test.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    test_frustum_projection()
    
    print("\n" + "="*60)
    print("FRUSTUM ANALYSIS CONCLUSION:")
    print("Yes, the 4 rays form a frustum when truncated by near/far planes.")
    print("However, the frustum's 8 vertices may not be sufficient")
    print("for accurate bounding rectangle calculation after projection.")
    print("Edge points or surface sampling may be needed.")
    print("="*60)

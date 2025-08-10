import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('/home/ytanaz/access/IBRNet')
from ibrnet.sample_ray import RaySamplerSingleImage

def validate_block_sampling(H=200, W=200, sr=False, use_moe=False, num_tests=5):
    """
    验证block_single模式下的select_inds是否真的形成了连续的矩形块
    
    Args:
        H: 图像高度
        W: 图像宽度  
        sr: 是否为超分辨率模式
        use_moe: 是否使用MOE模式
        num_tests: 测试次数
    """
    
    # 创建模拟数据 - 需要正确的相机参数格式
    # 根据parse_camera函数，相机参数应该是: [H, W, intrinsics(16), c2w(16)]
    intrinsics = torch.eye(4).flatten()  # 4x4单位矩阵
    c2w = torch.eye(4).flatten()  # 4x4单位矩阵
    camera_params = torch.cat([
        torch.tensor([H, W], dtype=torch.float32),  # H, W
        intrinsics,  # 16个参数
        c2w  # 16个参数
    ]).unsqueeze(0)  # 添加batch维度
    
    data = {
        'rgb': torch.randn(1, H, W, 3),
        'camera': camera_params,  # 正确的相机参数格式
        'rgb_path': 'test.jpg',
        'depth_range': torch.tensor([[1.0, 10.0]])
    }
    
    # 创建RaySampler实例
    ray_sampler = RaySamplerSingleImage(data, device='cpu', sr=sr, use_moe=use_moe)
    
    print(f"Testing block sampling with H={H}, W={W}, sr={sr}, use_moe={use_moe}")
    print("=" * 60)
    
    for test_idx in range(num_tests):
        print(f"\nTest {test_idx + 1}:")
        
        # 调用sample_random_pixel方法获取select_inds
        if sr:
            select_inds, select_inds_rgb = ray_sampler.sample_random_pixel(0, 'block_single')
            print(f"Generated {len(select_inds)} indices for regular block")
            print(f"Generated {len(select_inds_rgb)} indices for RGB block")
            
            # 验证regular block
            is_valid_block, block_info = check_if_block(select_inds, W, H)
            print(f"Regular block validation: {is_valid_block}")
            if is_valid_block:
                print(f"  Block info: {block_info}")
            
            # 验证RGB block  
            is_valid_rgb_block, rgb_block_info = check_if_block(select_inds_rgb, W*2, H*2)
            print(f"RGB block validation: {is_valid_rgb_block}")
            if is_valid_rgb_block:
                print(f"  RGB block info: {rgb_block_info}")
                
        else:
            select_inds = ray_sampler.sample_random_pixel(0, 'block_single')
            print(f"Generated {len(select_inds)} indices")
            
            # 验证是否为连续的矩形块
            is_valid_block, block_info = check_if_block(select_inds, W, H)
            print(f"Block validation: {is_valid_block}")
            if is_valid_block:
                print(f"  Block info: {block_info}")

def check_if_block(select_inds, W, H):
    """
    检查给定的一维索引是否构成连续的矩形块
    
    Args:
        select_inds: 一维索引列表
        W: 图像宽度
        H: 图像高度
        
    Returns:
        is_block: 是否为连续矩形块
        block_info: 块的信息字典
    """
    if len(select_inds) == 0:
        return False, {}
    
    # 将一维索引转换为二维坐标 (row, col)
    rows = [idx // W for idx in select_inds]
    cols = [idx % W for idx in select_inds]
    
    # 获取边界
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    
    # 计算预期的块尺寸
    expected_height = max_row - min_row + 1
    expected_width = max_col - min_col + 1
    expected_size = expected_height * expected_width
    
    # 检查是否所有预期的像素都存在
    expected_indices = set()
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            expected_indices.add(r * W + c)
    
    actual_indices = set(select_inds)
    
    is_block = (expected_indices == actual_indices) and (len(select_inds) == expected_size)
    
    block_info = {
        'start_row': min_row,
        'start_col': min_col,
        'height': expected_height,
        'width': expected_width,
        'total_pixels': len(select_inds),
        'expected_pixels': expected_size,
        'is_rectangular': len(select_inds) == expected_size,
        'all_pixels_present': expected_indices == actual_indices
    }
    
    return is_block, block_info

def visualize_block_sampling(H=100, W=100, sr=False, use_moe=False):
    """
    可视化block采样的结果
    """
    # 创建模拟数据 - 需要正确的相机参数格式
    intrinsics = torch.eye(4).flatten()
    c2w = torch.eye(4).flatten()
    camera_params = torch.cat([
        torch.tensor([H, W], dtype=torch.float32),
        intrinsics,
        c2w
    ]).unsqueeze(0)
    
    data = {
        'rgb': torch.randn(1, H, W, 3),
        'camera': camera_params,
        'rgb_path': 'test.jpg', 
        'depth_range': torch.tensor([[1.0, 10.0]])
    }
    
    ray_sampler = RaySamplerSingleImage(data, device='cpu', sr=sr, use_moe=use_moe)
    
    if sr:
        select_inds, select_inds_rgb = ray_sampler.sample_random_pixel(0, 'block_single')
        
        # 可视化regular block
        img1 = np.zeros((H, W))
        for idx in select_inds:
            row, col = idx // W, idx % W
            img1[row, col] = 1
            
        # 可视化RGB block  
        img2 = np.zeros((H*2, W*2))
        for idx in select_inds_rgb:
            row, col = idx // (W*2), idx % (W*2)
            img2[row, col] = 1
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(img1, cmap='Blues')
        ax1.set_title('Regular Block')
        ax1.grid(True, alpha=0.3)
        
        ax2.imshow(img2, cmap='Reds')
        ax2.set_title('RGB Block (2x size)')
        ax2.grid(True, alpha=0.3)
        
    else:
        select_inds = ray_sampler.sample_random_pixel(0, 'block_single')
        
        # 创建可视化图像
        img = np.zeros((H, W))
        for idx in select_inds:
            row, col = idx // W, idx % W
            img[row, col] = 1
            
        plt.figure(figsize=(8, 6))
        plt.imshow(img, cmap='Blues')
        plt.title(f'Block Sampling Visualization (use_moe={use_moe})')
        plt.colorbar()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'/home/ytanaz/access/IBRNet/block_sampling_sr{sr}_moe{use_moe}.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Block Sampling Validation")
    print("=" * 60)
    
    # 测试不同模式
    print("\n1. Testing normal mode (sr=False, use_moe=False):")
    validate_block_sampling(H=200, W=200, sr=False, use_moe=False, num_tests=3)
    
    print("\n2. Testing MOE mode (sr=False, use_moe=True):")
    validate_block_sampling(H=200, W=200, sr=False, use_moe=True, num_tests=3)
    
    print("\n3. Testing SR mode (sr=True, use_moe=False):")
    validate_block_sampling(H=200, W=200, sr=True, use_moe=False, num_tests=3)
    
    # 生成可视化
    print("\nGenerating visualizations...")
    visualize_block_sampling(H=100, W=100, sr=False, use_moe=False)
    visualize_block_sampling(H=100, W=100, sr=False, use_moe=True)
    visualize_block_sampling(H=100, W=100, sr=True, use_moe=False)
    
    print("\nValidation complete! Check the generated PNG files for visual confirmation.")

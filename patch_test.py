import numpy as np


def get_block_mask_optimized(A, sv_idx, h, w, s, block_size=8):
    """
    高效获取同时满足存储块和坐标块的双重掩码
    
    参数:
        A: 输入数组 [N_sv, H, W, N_s, 2]
        sv_idx: 当前源视图索引
        h/w/s: 参考点在H/W/N_s维度的索引
        block_size: 存储块尺寸（默认8x8）
    
    返回:
        full_mask: 完整维度的布尔掩码 [H, W, N_s]
        block_info: (x_min, x_max, y_min, y_max)
    """
    # 1. 确定存储块范围（H/W维度）
    h_block = h // block_size
    w_block = w // block_size
    h_start = h_block * block_size
    h_end = h_start + block_size
    w_start = w_block * block_size
    w_end = w_start + block_size

    # 2. 获取存储块内的坐标数据
    storage_block = A[sv_idx, h_start:h_end, w_start:w_end, :, :]  # [8,8,N_s,2]
    
    # 3. 确定坐标空间块范围
    ref_x, ref_y = A[sv_idx, h, w, s]
    x_block = ref_x // block_size
    y_block = ref_y // block_size
    x_min = x_block * block_size
    x_max = x_min + block_size
    y_min = y_block * block_size
    y_max = y_min + block_size
    
    # 4. 生成坐标掩码（向量化比较）
    x_mask = (storage_block[..., 0] >= x_min) & (storage_block[..., 0] < x_max)
    y_mask = (storage_block[..., 1] >= y_min) & (storage_block[..., 1] < y_max)
    coord_mask = x_mask & y_mask  # [8,8,N_s]
    
    # 5. 构建完整掩码
    full_mask = np.zeros(A.shape[1:4], dtype=bool)  # [H, W, N_s]
    full_mask[h_start:h_end, w_start:w_end, :] = coord_mask
    
    return full_mask, (x_min, x_max, y_min, y_max)

if __name__ == "__main__":
    # 示例使用
    pixel_locations_np = np.load('./eval/pixel_locations.npy')
    n_samples = pixel_locations_np.shape[2]
    A = np.reshape(pixel_locations_np, (10, 32, 504, n_samples, 2)) 

    # 选择参考点
    sv_idx = 0
    h_ref, w_ref, s_ref = 0, 0, 0
    ref_point = A[sv_idx, h_ref, w_ref, s_ref]
    x_ref, y_ref = ref_point[0], ref_point[1]

    # 获取优化后的掩码
    mask_optimized, block_info = get_block_mask_optimized(
        A, sv_idx, h_ref, w_ref, s_ref
    )

    print("参考点坐标:", (x_ref, y_ref))
    print("坐标块范围: x∈[%d,%d), y∈[%d,%d)" % block_info)
    print("存储块范围: H[%d:%d), W[%d:%d)" % (
        h_ref//8*8, (h_ref//8*8)+8,
        w_ref//8*8, (w_ref//8*8)+8
    ))
    print("掩码中True的数量:", np.sum(mask_optimized))
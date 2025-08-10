import numpy as np
from collections import defaultdict

class PriorityRayProcessor:
    def __init__(self, A, block_size=8, patch_size=16):
        self.A = A
        self.N_sv, self.H, self.W, self.N_s, _ = A.shape
        self.block_size = block_size
        self.patch_size = patch_size
        self.state = np.zeros((self.H, self.W, self.N_s), dtype=np.uint64)
        self.full_mask = (1 << self.N_sv) - 1
        self.processed = np.zeros((self.H, self.W, self.N_s), dtype=bool)
        self.processed_blocks = defaultdict(set)  # 记录已处理的块

        # 预计算位计数掩码
        self.bit_count_mask = [
            np.uint64(0x5555555555555555),
            np.uint64(0x3333333333333333),
            np.uint64(0x0F0F0F0F0F0F0F0F),
            np.uint64(0x00FF00FF00FF00FF),
            np.uint64(0x0000FFFF0000FFFF),
            np.uint64(0x00000000FFFFFFFF)
        ]
        self.shifts = [1, 2, 4, 8, 16, 32]

    def bit_count(self, arr):
        count = arr.astype(np.uint64)
        for mask, shift in zip(self.bit_count_mask, self.shifts):
            count = (count & mask) + ((count >> shift) & mask)
        return count

    def get_completion_counts(self):
        return self.bit_count(self.state).reshape(self.H, self.W, self.N_s)

    def select_priority_sample(self, patch_mask):
        # 优先级策略：先选完成sample points最多的ray，再选该ray中完成度最高的sample point
        completed_samples = self.get_completed_samples()
        # 排除已完成和已处理的sample points
        valid_mask = patch_mask & ~self.processed & ~completed_samples
        if not np.any(valid_mask):
            return None
        
        # 计算每个ray的有效sample points数量
        ray_completion = np.sum(valid_mask, axis=2)  # [H, W]
        
        # 选择完成度最高的ray
        max_ray_comp = np.max(ray_completion)
        if max_ray_comp == 0:
            return None
        
        ray_indices = np.argwhere(ray_completion == max_ray_comp)
        h_ray, w_ray = ray_indices[0]
        
        # 在选定的ray中选择最优sample point
        sample_mask = valid_mask[h_ray, w_ray, :]
        sample_completion = self.get_completion_counts()[h_ray, w_ray, :]
        sample_completion = np.where(sample_mask, sample_completion, -1)
        
        if np.max(sample_completion) == -1:
            return None
        
        max_sample_comp = np.max(sample_completion)
        sample_indices = np.argwhere(sample_completion == max_sample_comp)
        s_sample = sample_indices[0][0]
        
        return (h_ray, w_ray, s_sample)

    def get_missing_views(self, h, w, s):
        mask = self.state[h, w, s]
        missing = []
        for sv in range(self.N_sv):
            if not (mask & np.uint64(1 << sv)):
                missing.append(sv)
        return missing

    def process_patch(self, ph_start, pw_start):
        ph_end = ph_start + self.patch_size
        pw_end = pw_start + self.patch_size
        patch_mask = np.zeros((self.H, self.W, self.N_s), dtype=bool)
        patch_mask[ph_start:ph_end, pw_start:pw_end, :] = True
        
        iteration = 0
        rd = 0
        rd2 = 0
        while True:
            priority_idx = self.select_priority_sample(patch_mask)
            if priority_idx is None:
                break
            
            h, w, s = priority_idx
            missing_views = self.get_missing_views(h, w, s)
            
            # 统计有效块处理
            valid_sv_count = 0
            total_valid_samples = 0
            sv_masks = []
            
            for sv_idx in missing_views:
                h_block = h // self.block_size
                w_block = w // self.block_size
                block_key = (h_block, w_block)
                
                # 跳过已处理的块
                if block_key in self.processed_blocks[sv_idx]:
                    continue
                
                # 生成块掩码
                mask, _ = get_block_mask_optimized(
                    self.A, sv_idx, h, w, s, self.block_size
                )
                valid_count = np.sum(mask)
                if valid_count > 1:
                    sv_masks.append((sv_idx, mask))
                    total_valid_samples += valid_count
                    valid_sv_count += 1
                else:
                    # 标记低效块为已处理
                    self.processed_blocks[sv_idx].add(block_key)
                    self.processed[h, w, s] = True
            
            # 计算平均效率
            if valid_sv_count > 0:
                avg_valid = total_valid_samples / valid_sv_count
                if avg_valid <= 1:
                    # 标记所有相关块为已处理
                    for sv_idx, _ in sv_masks:
                        h_block = h // self.block_size
                        w_block = w // self.block_size
                        self.processed_blocks[sv_idx].add((h_block, w_block))
                        self.processed[h, w, s] = True
                    continue  # 跳过低效处理
            
            # 执行有效块更新
            for sv_idx, mask in sv_masks:
                h_block = h // self.block_size
                w_block = w // self.block_size
                bit_mask = mask.astype(np.uint64) << sv_idx
                self.state |= bit_mask
                self.processed |= mask
                self.processed_blocks[sv_idx].add((h_block, w_block))
            
            if iteration % 100 == 0:
                completed_rays = processor.get_completed_rays()
                print(f"iter {iteration}: 已完成rays数量: {np.sum(completed_rays)}; 读取量: {rd / (1024*1024):.2f}MB")
            iteration += 1

    def process_all_patches(self):
        for ph_start in range(0, self.H, self.patch_size):
            for pw_start in range(0, self.W, self.patch_size):
                self.process_patch(ph_start, pw_start)

    def get_completed_samples(self):
        return (self.state == self.full_mask)

    def get_completed_rays(self):
        completed_samples = self.get_completed_samples()
        return np.all(completed_samples, axis=2)

def get_block_mask_optimized(A, sv_idx, h, w, s, block_size=8):
    h_block = h // block_size
    w_block = w // block_size
    h_start = h_block * block_size
    h_end = h_start + block_size
    w_start = w_block * block_size
    w_end = w_start + block_size

    storage_block = A[sv_idx, h_start:h_end, w_start:w_end, :, :]
    
    ref_x, ref_y = A[sv_idx, h, w, s]
    x_block = ref_x // block_size
    y_block = ref_y // block_size
    x_min = x_block * block_size
    x_max = x_min + block_size
    y_min = y_block * block_size
    y_max = y_min + block_size
    
    x_mask = (storage_block[..., 0] >= x_min) & (storage_block[..., 0] < x_max)
    y_mask = (storage_block[..., 1] >= y_min) & (storage_block[..., 1] < y_max)
    coord_mask = x_mask & y_mask
    
    full_mask = np.zeros(A.shape[1:4], dtype=bool)
    full_mask[h_start:h_end, w_start:w_end, :] = coord_mask
    
    return full_mask, (x_min, x_max, y_min, y_max)

# 主程序
if __name__ == "__main__":
    # 加载数据
    pixel_locations_np = np.load('./eval/pixel_locations.npy')
    n_samples = pixel_locations_np.shape[2]
    A = np.reshape(pixel_locations_np, (10, 32, 504, n_samples, 2)) 

    # 初始化处理器（内部块8x8，外部分块16x16）
    processor = PriorityRayProcessor(A, block_size=16, patch_size=16)
    
    # 处理所有16x16 patch
    processor.process_all_patches()

    # 最终结果
    completed_rays = processor.get_completed_rays()
    print("\n最终结果:")
    print(f"完成rays数量: {np.sum(completed_rays)}")
    print(f"处理历史记录形状: {processor.processed.shape}")
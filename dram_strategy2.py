import numpy as np

def wr_instr(output_hex_file, output_bin_file, instr_group):
    # 定义表头字段
    headers = ["          Ro       ", "Ba ", "Ra   ", "Co", "    Ch"]
    # 打开两个输出文件
    with open(output_hex_file, "w") as f_hex, open(output_bin_file, "w") as f_bin:
        # 写入表头到二进制文件
        f_bin.write(" ".join(headers) + "\n")
        # 生成指令
        for i in range(len(instr_group)):
            # 写入十六进制文件
            hex_output = instr_group[i][0]
            f_hex.write(hex_output + "\n")
            
            # 写入二进制文件
            bin_output = instr_group[i][1]
            f_bin.write(bin_output + "\n")

    print(f"生成完成！结果已保存到以下文件：")
    print(f"- 十六进制文件: {output_hex_file}")
    print(f"- 二进制文件: {output_bin_file}")
    
def gen_instr(bank_addr, row_addr, col_addr):
    segment_sizes = [16, 3, 1, 10, 1]
    
    # 初始化所有段为全 0
    segments = ["0" * size for size in segment_sizes]
    
    # 修改 Bank 段为计算出的值
    bank_index = 1  # Row 是第 2 段（索引从 0 开始）
    bank_bits = segment_sizes[bank_index]
    bank_value = bin(bank_addr % (2**bank_bits))[2:].zfill(bank_bits)  # 确保不超过分配的位数
    segments[bank_index] = bank_value
    
    # 修改 Row 段为计算出的值
    row_index = 0  # Row 是第 1 段（索引从 0 开始）
    row_bits = segment_sizes[row_index]
    row_value = bin(row_addr % (2**row_bits))[2:].zfill(row_bits)  # 确保不超过分配的位数
    segments[row_index] = row_value
    
    # 修改 Column 段为计算出的值
    col_index = 3  # Column 是第 4 段（索引从 0 开始）
    col_bits = segment_sizes[col_index]
    col_value = bin(col_addr % (2**col_bits))[2:].zfill(col_bits)  # 确保不超过分配的位数
    segments[col_index] = col_value
    
    # 合并分段，生成完整的二进制字符串
    binary_value = "".join(segments)
    binary_value = "0" + binary_value  # 添加前导0以确保31位
    
    # 将二进制转换为十六进制，并转为大写
    hex_value = hex(int(binary_value, 2))[2:].zfill(8).upper()
    
    # 写入十六进制文件
    hex_output = f"LD 0x{hex_value}"
    
    # 写入二进制文件
    bin_output = f"LD {' '.join(segments)}"
    return hex_output, bin_output
            
def ddr_mapping(sv_idx, y_idx, x_idx):
    if y_idx < 0 or y_idx * 16 > 756 or x_idx < 0 or x_idx * 16 > 1008:
        return None
    bank_addr = sv_idx % 8
    last_2_row_offset = 8192
    # 10块16×16的是8chip ddr的一行
    total_idx = y_idx * 1008 // 16 + x_idx
    row_addr = total_idx // 10
    col_addr = total_idx % 10
    return bank_addr, row_addr, col_addr

class PriorityRayProcessor:
    def __init__(self, A, block_size=8, patch_size=16):
        self.A = A
        self.N_sv, self.H, self.W, self.N_s, _ = A.shape
        self.block_size = block_size
        self.patch_size = patch_size
        self.state = np.zeros((self.H, self.W, self.N_s), dtype=np.uint64)
        self.full_mask = (1 << self.N_sv) - 1
        self.processed = np.zeros((self.H, self.W, self.N_s), dtype=bool)

        self.pt = False
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

    def get_completed_samples(self):
        return (self.state == self.full_mask)

    def select_priority_sample(self, patch_mask):
        # 优先级策略：先选完成sample points最多的ray，再选该ray中完成度最高的sample point
        completed_samples = self.get_completed_samples()
        completed_samples = np.where(patch_mask, completed_samples, -1)  # 限制在当前patch
        ray_completion = np.sum(completed_samples, axis=2)  # [H, W]
        
        # 排除已完成所有sample points的rays
        valid_rays = (ray_completion < self.N_s)
        if not np.any(valid_rays):
            return None
        # 将无效rays的完成度设为-1
        ray_completion = np.where(valid_rays, ray_completion, -1)
        max_ray_completion = np.max(ray_completion)
        
        # 选择完成度最高的ray
        ray_indices = np.argwhere(ray_completion == max_ray_completion)
        h_ray, w_ray = ray_indices[0]
        
        # 在选定的ray中选择最优sample point
        sample_completion = self.get_completion_counts()[h_ray, w_ray, :]
        sample_completion = np.where(sample_completion == self.N_sv, -1, sample_completion)
        
        if np.max(sample_completion) == -1:
            return None
        
        max_sample_comp = np.max(sample_completion)
        sample_indices = np.argwhere(sample_completion == max_sample_comp)
        s_sample = sample_indices[0][0]
        
        return (h_ray, w_ray, s_sample)

    def is_sample_complete(self, h, w, s):
        return (self.state[h, w, s] == self.full_mask)

    def get_missing_views(self, h, w, s):
        mask = self.state[h, w, s]
        missing = []
        for sv in range(self.N_sv):
            if not (mask & np.uint64(1 << sv)):
                missing.append(sv)
        return missing

    def process_patch(self, ph_start, pw_start, rd):
        ph_end = ph_start + self.patch_size
        pw_end = pw_start + self.patch_size
        patch_mask = np.zeros((self.H, self.W, self.N_s), dtype=bool)
        patch_mask[ph_start:ph_end, pw_start:pw_end, :] = True
        
        iteration = 0
        rd2 = 0
        mapping_cnt = 0
        instr_group = []
        while True:
            priority_idx = self.select_priority_sample(patch_mask)
            if priority_idx is None:
                break
            
            h, w, s = priority_idx
            if not patch_mask[h, w, s]:
                continue  # 确保在当前patch范围内
            
            missing_views = self.get_missing_views(h, w, s)
            
            sv_masks = []
            for sv_idx in missing_views:
                mask, _, (y_block, x_block) = get_block_mask_optimized(
                    self.A, sv_idx, h, w, s, self.block_size
                )
                sv_masks.append((sv_idx, mask))
                map_tmp = ddr_mapping(sv_idx, y_block, x_block)
                if map_tmp is not None:
                    bank_addr, row_addr, col_addr = map_tmp
                    for b_idx in range(12):
                        instr = gen_instr(bank_addr, row_addr, col_addr+b_idx*8)
                        instr_group.append(instr)
                mapping_cnt += 1
                    
                # rd += self.block_size * self.block_size * 3
            rd2 += 10 * self.block_size * self.block_size * 3
            s = 0
            for sv_idx, mask in sv_masks:
                s += mask.sum()
            if s >= 0:
                for sv_idx, mask in sv_masks:
                    rd += self.block_size * self.block_size * 3
                
            for sv_idx, mask in sv_masks:
                bit_mask = mask.astype(np.uint64) << sv_idx
                self.state |= bit_mask
                self.processed |= mask
                
            if iteration % 100 == 0:
                completed_rays = processor.get_completed_rays()
                print(f"iter {iteration}: 已完成rays数量: {np.sum(completed_rays)}; 读取量: {rd / (1024*1024):.2f}MB")
            iteration += 1
        print(mapping_cnt / len(instr_group))
        
        hex_file = "/home/ytanaz/access/ramulator2/zz/nerf_hex.trace"  # 生成的十六进制指令文件路径
        bin_file = "/home/ytanaz/access/ramulator2/zz/nerf_bin.trace"  # 生成的二进制指令文件路径
        wr_instr(hex_file, bin_file, instr_group)
        exit()
        return rd, instr_group

    def process_all_patches(self):
        rd = 0
        for ph_start in range(0, self.H, self.patch_size):
            for pw_start in range(0, self.W, self.patch_size):
                print(f"处理Patch: H[{ph_start}:{ph_start+self.patch_size}), "
                      f"W[{pw_start}:{pw_start+self.patch_size})")
                rd = self.process_patch(ph_start, pw_start, rd)

    def get_completed_rays(self):
        completed_samples = (self.state == self.full_mask)
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
    # print(y_block, x_block)
    return full_mask, (x_min, x_max, y_min, y_max), (int(y_block), int(x_block))

# 主程序
if __name__ == "__main__":
    # 加载数据
    pixel_locations_np = np.load('./eval/pixel_locations_valid.npy')
    n_samples = pixel_locations_np.shape[2]
    A = np.reshape(pixel_locations_np, (10, 32, 504, n_samples, 2)) 
    # print(A[...,0].min(), A[...,0].max(), A[...,1].min(), A[...,1].max())
    
    # 初始化处理器（内部块8x8，外部分块16x16）
    processor = PriorityRayProcessor(A[:10], block_size=16, patch_size=16)
    
    # 处理所有16x16 patch
    processor.process_all_patches()

    # 最终结果
    completed_rays = processor.get_completed_rays()
    print("\n最终结果:")
    print(f"完成rays数量: {np.sum(completed_rays)}")
    print(f"处理历史记录形状: {processor.processed.shape}")
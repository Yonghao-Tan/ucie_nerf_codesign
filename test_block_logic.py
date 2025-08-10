import numpy as np
import torch
import sys
sys.path.append('/home/ytanaz/access/IBRNet')

# 简化版本的验证，直接复制相关逻辑
def test_block_logic():
    rng = np.random.RandomState(234)
    
    # 测试参数
    H, W = 200, 200
    
    print(f"Image size: H={H}, W={W}")
    
    # 测试normal mode
    print("\n=== Normal Mode ===")
    for i in range(5):
        block_height = rng.randint(16, 33)
        block_width = rng.randint(16, 33)
        
        max_row_start = H - block_height
        max_col_start = W - block_width
        
        min_row_start = block_height // 2
        min_col_start = block_width // 2
        
        print(f"Test {i+1}:")
        print(f"  block_height={block_height}, block_width={block_width}")
        print(f"  max_row_start={max_row_start}, max_col_start={max_col_start}")
        print(f"  min_row_start={min_row_start}, min_col_start={min_col_start}")
        print(f"  Valid range for row: [{min_row_start}, {max_row_start})")
        print(f"  Valid range for col: [{min_col_start}, {max_col_start})")
        
        if max_row_start <= min_row_start:
            print(f"  ERROR: max_row_start ({max_row_start}) <= min_row_start ({min_row_start})")
        if max_col_start <= min_col_start:
            print(f"  ERROR: max_col_start ({max_col_start}) <= min_col_start ({min_col_start})")
            
        if max_row_start > min_row_start and max_col_start > min_col_start:
            select_row_start = rng.randint(min_row_start, max_row_start)
            select_col_start = rng.randint(min_col_start, max_col_start)
            
            print(f"  Selected start: row={select_row_start}, col={select_col_start}")
            
            # 生成索引
            select_inds = []
            for row_offset in range(block_height):
                for col_offset in range(block_width):
                    idx = (select_row_start + row_offset) * W + (select_col_start + col_offset)
                    select_inds.append(idx)
            
            print(f"  Generated {len(select_inds)} indices")
            
            # 验证是否为连续块
            is_valid = verify_block(select_inds, W, H, block_height, block_width)
            print(f"  Is valid block: {is_valid}")
        print()

def verify_block(select_inds, W, H, expected_height, expected_width):
    """验证索引是否构成连续的矩形块"""
    # 转换为坐标
    coords = [(idx // W, idx % W) for idx in select_inds]
    
    # 获取边界
    rows = [coord[0] for coord in coords]
    cols = [coord[1] for coord in coords]
    
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    
    actual_height = max_row - min_row + 1
    actual_width = max_col - min_col + 1
    
    print(f"    Expected: {expected_height}x{expected_width}, Actual: {actual_height}x{actual_width}")
    print(f"    Block region: rows [{min_row}, {max_row}], cols [{min_col}, {max_col}]")
    
    # 检查尺寸是否匹配
    if actual_height != expected_height or actual_width != expected_width:
        return False
        
    # 检查是否所有像素都存在
    expected_coords = set()
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            expected_coords.add((r, c))
    
    actual_coords = set(coords)
    
    return expected_coords == actual_coords

if __name__ == "__main__":
    test_block_logic()

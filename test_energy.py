# 参数定义
def calculate_energy(N, C, ifmap_ratio, weight_ratio, ofmap_ratio, 
                     read_energy_per_bit, write_energy_per_bit, op_energy_per_op):
    """
    计算 SRAM 和 OP 的能耗、总能耗、SRAM 占比、GOPS/W。
    
    参数：
    N: 批量大小 (Batch Size)
    C: 通道数 (Channels)
    ifmap_ratio: Ifmap 数据复用比例（例如 4096 / 256）
    weight_ratio: Weight 数据复用比例（例如 1 / (1x1)）
    ofmap_ratio: Ofmap 数据比例（通常为 1）
    read_energy_per_bit: 每 bit 读取能耗 (pJ/bit)
    write_energy_per_bit: 每 bit 写入能耗 (pJ/bit)
    op_energy_per_op: 每操作能耗 (pJ/OP)
    """
    # 数据量计算
    ifmap_bytes = ifmap_ratio * N * C  # Ifmap 数据量 (Bytes)
    weight_bytes = weight_ratio * C * C  # Weight 数据量 (Bytes)
    ofmap_bytes = ofmap_ratio * N * C        # Ofmap 数据量 (Bytes)
    
    # OP 数量计算
    op_count = N * C * C * 2  # 操作数量 (OPs)

    # SRAM 能耗计算
    sram_read_energy = (ifmap_bytes + weight_bytes) * 8 * read_energy_per_bit  # Ifmap 和 Weight 的读能量 (pJ)
    sram_write_energy = ofmap_bytes * 8 * write_energy_per_bit                 # Ofmap 的写能量 (pJ)
    sram_energy = sram_read_energy + sram_write_energy                         # 总 SRAM 能耗 (pJ)

    # OP 能耗计算
    op_energy = op_count * op_energy_per_op  # 总 OP 能耗 (pJ)

    # 总能耗计算
    total_energy = sram_energy + op_energy   # 总能耗 (pJ)

    # SRAM 占比计算
    sram_percentage = (sram_energy / total_energy) * 100  # SRAM 占比 (%)

    # GOPS/W 计算
    gops_per_watt = (op_count / total_energy) * 1e-9      # GOPS/W

    # 返回结果
    return {
        "Ifmap 数据量 (Bytes)": ifmap_bytes,
        "Weight 数据量 (Bytes)": weight_bytes,
        "Ofmap 数据量 (Bytes)": ofmap_bytes,
        "操作数量 (OPs)": op_count,
        "SRAM 读能量 (pJ)": sram_read_energy,
        "SRAM 写能量 (pJ)": sram_write_energy,
        "SRAM 总能耗 (J)": sram_energy * 1e-9,
        "OP 总能耗 (J)": op_energy * 1e-9,
        "总能耗 (J)": total_energy * 1e-9,
        "SRAM 占比 (%)": sram_percentage,
        "GOPS/W": gops_per_watt
    }

# 示例参数
N = 32               # 批量大小
C = 4096             # 通道数
# N = 4096               # 批量大小
# C = 128             # 通道数
T_rc = 16
T_m = 16
ifmap_ratio = C / T_m    # Ifmap 复用比例 (4096 / 256)
weight_ratio = N / T_rc     # Weight 复用比例 (1 / (1x1))
ofmap_ratio = 1      # Ofmap 比例 (1)
read_energy_per_bit = 0.754  * 1e-12 # 每 bit 读取能耗 (pJ/bit)
write_energy_per_bit = 0.987 * 1e-12 # 每 bit 写入能耗 (pJ/bit)
op_energy_per_op = 0.6256    * 1e-12 # 每操作能耗 (pJ/OP)

read_energy_per_bit = 0.355  * 1e-12 # 每 bit 读取能耗 (pJ/bit)
write_energy_per_bit = 0.472 * 1e-12 # 每 bit 写入能耗 (pJ/bit)
op_energy_per_op = 0.3612    * 1e-12 # 每操作能耗 (pJ/OP)

# 调用函数
results = calculate_energy(N, C, ifmap_ratio, weight_ratio, ofmap_ratio,
                           read_energy_per_bit, write_energy_per_bit, op_energy_per_op)

# 打印结果
for key, value in results.items():
    print(f"{key}: {value}")
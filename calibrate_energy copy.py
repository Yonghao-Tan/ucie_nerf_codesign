def calibrate_op_energy(N, C, ifmap_ratio, weight_ratio, ofmap_ratio, 
                     read_energy_per_bit, write_energy_per_bit,  power, frequency, cycles):
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
    print(sram_energy)

    total_energy = power * (cycles / frequency)
    
    # OP 能耗计算
    op_energy = total_energy - sram_energy # 总 OP 能耗 (pJ)

    # SRAM 占比计算
    sram_percentage = (sram_energy / total_energy) * 100  # SRAM 占比 (%)

    # GOPS/W 计算
    gops_per_watt = (op_count / total_energy) * 1e-9      # GOPS/W
    print(f"SRAM 占比 (%): {sram_percentage}; GOPS/W: {gops_per_watt}")

    op_energy_per_op = op_energy / op_count
    # 返回结果
    return op_energy_per_op

def calibrate_sram_energy(frequency, cycles, sram_power, mac_power):
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
    # frequency = 800 * 1e6
    # cycles = 265
    
    R, C = 8, 16
    N, M = 16, 16
    K = 3
    
    op_a = R * C * M * N * K * K * 2
    ifmap_ratio = M / T_m    # Ifmap 复用比例 (4096 / 256)
    weight_ratio = R * C / T_rc     # Weight 复用比例 (1 / (1x1))
    ofmap_ratio = 1      # Ofmap 比例 (1)
    ifmap_bytes = ifmap_ratio * R * C * N
    weight_bytes = weight_ratio * M * N * K * K
    ofmap_bytes = ofmap_ratio * R * C * M

    sram_access_a = ifmap_bytes + weight_bytes + ofmap_bytes
    
    R, C = 8, 16
    N, M = 16, 16
    K = 1
    
    op_b = R * C * M * N * K * K * 2
    ifmap_ratio = M / T_m    # Ifmap 复用比例 (4096 / 256)
    weight_ratio = R * C / T_rc     # Weight 复用比例 (1 / (1x1))
    ofmap_ratio = 1      # Ofmap 比例 (1)
    ifmap_bytes = ifmap_ratio * R * C * N
    weight_bytes = weight_ratio * M * N * K * K
    ofmap_bytes = ofmap_ratio * R * C * M
    
    sram_access_b = ifmap_bytes + weight_bytes + ofmap_bytes
    
    op_c = R * C * M
    total_op = op_a + op_b + op_c
    sram_access_c = 3 * R * C * M
    
    total_sram_access = sram_access_a + sram_access_b + sram_access_c
    print(op_a, op_b, op_c, total_op)
    
    # sram_power = 127.5 * 1e-3
    sram_energy = sram_power * cycles / frequency
    sram_energy_per_byte = sram_energy / total_sram_access
    
    # mac_power = 209.0 * 1e-3
    mac_energy = mac_power * cycles / frequency
    mac_energy_per_op = mac_energy / total_op
    
    # 返回结果
    return mac_energy_per_op, sram_energy_per_byte

# 示例参数
N = 4096               # 批量大小
C = 128             # 通道数
T_rc = 16
T_m = 8
ifmap_ratio = C / T_m    # Ifmap 复用比例 (4096 / 256)
weight_ratio = N / T_rc     # Weight 复用比例 (1 / (1x1))
ofmap_ratio = 1      # Ofmap 比例 (1)
read_energy_per_bit = 0.754  * 1e-12 # 每 bit 读取能耗 (pJ/bit)
write_energy_per_bit = 0.987 * 1e-12 # 每 bit 写入能耗 (pJ/bit)
# op_energy_per_op = 0.6256    * 1e-12 # 每操作能耗 (pJ/OP)

power = 2.047
frequency = 242.5 * 1e6
cycles = 164344 * 0.1

# op_energy_per_op = calibrate_op_energy(N, C, ifmap_ratio, weight_ratio, ofmap_ratio,
#                            read_energy_per_bit, write_energy_per_bit, power, frequency, cycles)
# print(op_energy_per_op)

power = 209.0 * 1e-3
frequency = 800 * 1e6
cycles = 265
ops = 655360

# N = 1024               # 批量大小
# C = 48  
# ifmap_ratio = C / T_m    # Ifmap 复用比例 (4096 / 256)
# weight_ratio = N / T_rc     # Weight 复用比例 (1 / (1x1))
# ofmap_ratio = 1      # Ofmap 比例 (1)
# op_energy_per_op = (power * cycles / frequency) / ops
# case_energy = (power * cycles / frequency)
# print(f"op energy: {op_energy_per_op*1e12:.2f} pJ")
# print((power * cycles / frequency))

frequency = 1200 * 1e6
cycles = 265
sram_power = 518.3 * 1e-3
mac_power = 812.0 * 1e-3
mac_energy_per_op, sram_energy_per_byte = calibrate_sram_energy(frequency, cycles, sram_power, mac_power)
print(f"MAC OP Energy: {mac_energy_per_op*1e12:.3f}; SRAM R/W Avg Energy: {sram_energy_per_byte*1e12:.3f}")
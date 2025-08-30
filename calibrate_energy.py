frequency = 800 * 1e6
cycles = 265
sram_power = 127.5 * 1e-3
mac_power = 209.0 * 1e-3

frequency = 1200 * 1e6
sram_power = 518.3 * 1e-3
mac_power = 812.0 * 1e-3

p_token, p_weight = 16, 16

# td0005
# conv 1
R, C = 8, 16
N, M = 16, 16
K = 3

op_a = R * C * M * N * K * K * 2
ifmap_ratio = M / p_weight
weight_ratio = R * C / p_token
ofmap_ratio = 1
ifmap_bytes = ifmap_ratio * R * C * N
weight_bytes = weight_ratio * M * N * K * K
ofmap_bytes = ofmap_ratio * R * C * M
print(ifmap_ratio, weight_ratio, ofmap_ratio)
print(ifmap_bytes, weight_bytes, ofmap_bytes)
sram_access_a = ifmap_bytes + weight_bytes + ofmap_bytes

# conv 2
R, C = 8, 16
N, M = 16, 16
K = 1

op_b = R * C * M * N * K * K * 2
ifmap_ratio = M / p_weight
weight_ratio = R * C / p_token
ofmap_ratio = 1
ifmap_bytes = ifmap_ratio * R * C * N
weight_bytes = weight_ratio * M * N * K * K
ofmap_bytes = ofmap_ratio * R * C * M

print(ifmap_ratio, weight_ratio, ofmap_ratio)
print(ifmap_bytes, weight_bytes, ofmap_bytes)
sram_access_b = ifmap_bytes + weight_bytes + ofmap_bytes

# elementwise add
op_c = R * C * M
total_op = op_a + op_b + op_c
sram_access_c = 3 * R * C * M
print(sram_access_c)
total_sram_access = sram_access_a + sram_access_b + sram_access_c

sram_energy = sram_power * cycles / frequency
sram_energy_per_byte = sram_energy / total_sram_access

mac_energy = mac_power * cycles / frequency
mac_energy_per_op = mac_energy / total_op

print(f"MAC OP Energy: {mac_energy_per_op*1e12:.3f} pJ/OP; SRAM R/W Avg Energy: {sram_energy_per_byte*1e12:.3f} pJ/Byte")
print(total_sram_access)
sram_energy_per_byte_new = sram_energy / 47104
print(f"MAC OP Energy: {mac_energy_per_op*1e12:.3f} pJ/OP; SRAM R/W Avg Energy New: {sram_energy_per_byte_new*1e12:.3f} pJ/Byte")
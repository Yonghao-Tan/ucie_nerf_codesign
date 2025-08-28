
rays_per_tile = 25
org_before_d2d_ops = 7.109 * 1e6 * rays_per_tile
org_after_d2d_ops = 0.831 * 1e6 * rays_per_tile
org_rgb_fc_ops = 0.559 * 1e6 * rays_per_tile
org_attn_ops = org_after_d2d_ops - org_rgb_fc_ops

proposed_sv_prune_rate = 0.52
chn_sparsity_rate = 0.5
sr_chn_sparsity_rate = 0.5

lbuf_bandwidth = 2048
d2d_bandwidth = 256

ai_core_frequency = 1000 * 1e6
ucie_frequency = 2000 * 1e6

d2d_effective_bandwidth = d2d_bandwidth * ucie_frequency / ai_core_frequency

macs = 2048
chiplets = 4

H, W = 800, 800
patch_size = 10 * 10

def ucie_process(transfer_size, buffer_bandwidth, ucie_bandwidth):
    overhead = 100
    overhead = 15
    return 2 * transfer_size / buffer_bandwidth + transfer_size / ucie_bandwidth + overhead

def mac_compute(op, macs):
    efficiency = 0.85
    efficiency = 0.95
    return op / (macs * 2 * efficiency)

chn_pruned_before_d2d_ops = org_before_d2d_ops * (1 - chn_sparsity_rate)
chn_pruned_attn_ops = org_attn_ops * (1 - chn_sparsity_rate)
chn_pruned_rgb_fc_ops = org_rgb_fc_ops * (1 - chn_sparsity_rate)

sv_chn_pruned_before_d2d_ops = chn_pruned_before_d2d_ops * (1 - proposed_sv_prune_rate)
sv_chn_pruned_rgb_fc_ops = chn_pruned_rgb_fc_ops * (1 - proposed_sv_prune_rate)

sv_chn_pruned_before_d2d_cycles = mac_compute(sv_chn_pruned_before_d2d_ops*chiplets, macs*chiplets)
chn_pruned_attn_ops_cycles = mac_compute(chn_pruned_attn_ops*chiplets, macs)
sv_chn_pruned_rgb_fc_ops_cycles = mac_compute(sv_chn_pruned_rgb_fc_ops*chiplets, macs*chiplets)

sv_pruned_d2d_fmap_size = 25 * 48 * 16 * 8 + 25 * 48 * 8 * 8
sv_pruned_d2d_transfer_cycles = 2 * ucie_process(sv_pruned_d2d_fmap_size, lbuf_bandwidth, d2d_effective_bandwidth)
sv_chn_pruned_total_cycles = sv_chn_pruned_before_d2d_cycles + chn_pruned_attn_ops_cycles + sv_chn_pruned_rgb_fc_ops_cycles + sv_pruned_d2d_transfer_cycles

print(f"sv_chn_pruned_before_d2d_cycles: {sv_chn_pruned_before_d2d_cycles:.3f} cycles")
print(f"chn_pruned_attn_ops_cycles: {chn_pruned_attn_ops_cycles:.3f} cycles")
print(f"sv_chn_pruned_rgb_fc_ops_cycles: {sv_chn_pruned_rgb_fc_ops_cycles:.3f} cycles")
print(f"sv_pruned_d2d_transfer_cycles: {sv_pruned_d2d_transfer_cycles:.3f} cycles")
print(f"Total: {sv_chn_pruned_total_cycles:.3f} cycles")


chn_pruned_total_ops = chn_pruned_before_d2d_ops + chn_pruned_attn_ops + chn_pruned_rgb_fc_ops
chn_pruned_d2d_fmap_size = 25 * 48 * 3 * 8 + 25 * 48 * 8 * 8
chn_pruned_d2d_transfer_cycles = 2 * ucie_process(chn_pruned_d2d_fmap_size, lbuf_bandwidth, d2d_effective_bandwidth)
chn_pruned_total_cycles = mac_compute(chn_pruned_total_ops*chiplets, macs*chiplets) + chn_pruned_d2d_transfer_cycles
print(f"No SV prune: {chn_pruned_total_cycles:.3f} cycles")

basic_sv_prune_rate = 0.5
basic_sv_chn_pruned_before_d2d_ops = chn_pruned_before_d2d_ops * (1 - basic_sv_prune_rate)
basic_chn_pruned_rgb_fc_ops = chn_pruned_rgb_fc_ops * (1 - basic_sv_prune_rate)
basic_sv_chn_pruned_ops = basic_sv_chn_pruned_before_d2d_ops + chn_pruned_attn_ops + basic_chn_pruned_rgb_fc_ops
basic_sv_chn_pruned_d2d_fmap_size = 25 * 48 * 3 * 8 + 25 * 48 * 8 * 8
basic_sv_chn_pruned_d2d_transfer_cycles = 2 * ucie_process(basic_sv_chn_pruned_d2d_fmap_size, lbuf_bandwidth, d2d_effective_bandwidth)
basic_sv_chn_pruned_cycles = mac_compute((basic_sv_chn_pruned_ops)*chiplets, macs*chiplets) + basic_sv_chn_pruned_d2d_transfer_cycles
print(f"Basic SV prune: {basic_sv_chn_pruned_cycles:.3f} cycles")

speed_up_0 = chn_pruned_total_cycles / sv_chn_pruned_total_cycles
speed_up_1 = basic_sv_chn_pruned_cycles / sv_chn_pruned_total_cycles
print(f"Speed up 0: {speed_up_0:.3f}x")
print(f"Speed up 1: {speed_up_1:.3f}x")

sv_chn_pruned_total_cycles = basic_sv_chn_pruned_cycles # TODO
sv_chn_pruned_latency = sv_chn_pruned_total_cycles / ai_core_frequency
sv_chn_pruned_latency_frame = sv_chn_pruned_latency * (400*400) / patch_size

coarse_ops = 2.614 * 1e6 * rays_per_tile
coarse_share_ops = coarse_ops / rays_per_tile
coarse_share_cycles = mac_compute(coarse_share_ops*chiplets, macs*chiplets)
coarse_share_latency_frame = coarse_share_cycles / ai_core_frequency
nerf_latency = coarse_share_latency_frame + sv_chn_pruned_latency_frame

sr_patch_ops = 1e12 * 0.000382768128 * (10*10) / (16*16) * (1 - sr_chn_sparsity_rate)
sr_rate = 0.85

total_pixels = H * W
sr_pixels = total_pixels * sr_rate
total_patches_base = total_pixels / (2 * 2 * patch_size)
sr_total_ops = total_patches_base * sr_patch_ops

total_patches_hr = total_patches_base * (1 - sr_rate) * 2 * 2
total_patches_lr = total_patches_base * sr_rate
total_patches_nerf = total_patches_hr + total_patches_lr
total_nerf_cycles = total_patches_nerf * (coarse_share_cycles + sv_chn_pruned_total_cycles)
total_sr_cycles = mac_compute(sr_total_ops, macs*chiplets)
total_cycles_frame = total_nerf_cycles + total_sr_cycles
total_latency_frame = total_cycles_frame / ai_core_frequency
print(f"Total_latency_frame: {total_latency_frame:.3f}")
print(f"FPS: {1/total_latency_frame:.3f}")
a = total_cycles_frame * (macs * 2 * 0.95)
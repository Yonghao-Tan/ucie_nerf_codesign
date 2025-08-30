# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import onnx
from onnx import numpy_helper
import argparse


    
def train():
    coarse_path = './onnx/ibrnet_generalizable_16_simp_s8.onnx'
    fine_path = './onnx/ibrnet_generalizable_48_simp_s8.onnx'
    sr_path = './onnx/osr_simp.onnx'
    
    optimizations = {}
    hardware_configurations = {
        'Mode': 1, # 0: efficiency; 1: performance
        'MACs': 2048,
        'Tm': 16,
        'Trc': 16,
        'MAC Efficiency': 1., # 如果这个小了那功耗也应该变小? 好像没变应该是除的时候化过
        'Activation Buffer Bandwidth': 2048 / 8,
        'Weight Buffer Bandwidth': 1024 / 8,
        'Chiplets': 4,
        'DRAM Bandwidth': 25.6 * 1000 ** 3,
        # 'DRAM Energy': 112.54 * 1e-12,
        'DRAM Energy': 60.4 * 1e-12,
        'UCIE Bandwidth': 64 * 1000 ** 3,
        'UCIE Energy': 1.108 * 8 * 1e-12,
        'OP per Projection': 16, # [3, 4] @ [4, 1] (12), & 1 division (4)
        'OP per Interpolation': 13*2, # ?
        # 'UCIe Bandwidth':    
    }
    mac_scaling = 2.5
    sram_scaling = 1.25
    if hardware_configurations['Mode'] == 1:
        hardware_configurations['AI Core Frequency'] = 1200 * 1e6
        hardware_configurations['OP Energy'] = 0.273 * 1e-12 / mac_scaling
        hardware_configurations['SRAM Energy'] = 2.430 * 1e-12 / sram_scaling
    else:
        hardware_configurations['AI Core Frequency'] = 800 * 1e6
        hardware_configurations['OP Energy'] = 0.105 * 1e-12 / mac_scaling
        hardware_configurations['SRAM Energy'] = 0.897 * 1e-12 / sram_scaling
    
    nerf_config = {
        'H': 800,
        'W': 800,
        'Patch Size': 100
    }
    
    results = {
        'Baseline': {},
        'Solution 1': {
            
        },
        'Solution 2a': {
            'Ray Skipping': {},
            'SV Pruning': {},
            'Channel Pruning': {},
            'ALL': {},
        },
        'Solution 2b': {
            'SR': {},
            'Select SR': {},
        },
        'ALL': {
            
        },
    }

    ray_skip = 25
    sv_ratio = 0.5
    fine_sharing = 9/25
    nerf_sparsity = 0.5
    sr_sparsity = 0.25
    select_sr = 0.9
    
    # ----------------------------------- Baseline -----------------------------------
    print(f"------------- Baseline -------------")
    H, W = nerf_config['H'], nerf_config['W']
    graph_coarse = onnx.shape_inference.infer_shapes(onnx.load(coarse_path)).graph
    coarse_mac_ops, coarse_mac_latency, coarse_sram_access, coarse_sram_latency = calculate_flops_and_latency(graph_coarse, hardware_configurations, optimizations, G=True, debug=False)
    print(f"Ray: Coarse Network OPs: {coarse_mac_ops / 1e6:.3f}M")
    print(f"Ray: Coarse Network Latency: {coarse_mac_latency * 1e6:.3f}us")

    graph_fine = onnx.shape_inference.infer_shapes(onnx.load(fine_path)).graph
    fine_mac_ops, fine_mac_latency, fine_sram_access, fine_sram_latency = calculate_flops_and_latency(graph_fine, hardware_configurations, optimizations, G=True, debug=False)
    print(f"Ray: Fine Network OPs: {fine_mac_ops / 1e6:.3f}M")
    print(f"Ray: Fine Network Latency: {fine_mac_latency * 1e6:.3f}us")
    # print(f"Ray: Fine Network Latency: {fine_sram_latency * 1e6:.3f}us")
    nerf_total_mac_latency = (coarse_mac_latency + fine_mac_latency) * H * W
    nerf_total_mac_ops = (coarse_mac_ops + fine_mac_ops) * H * W
    nerf_total_mac_energy = nerf_total_mac_ops * hardware_configurations['OP Energy']
    nerf_total_sram_access = (coarse_sram_access + fine_sram_access) * H * W
    nerf_total_sram_energy = nerf_total_sram_access * hardware_configurations['SRAM Energy']
    nerf_total_energy = nerf_total_mac_energy + nerf_total_sram_energy
    print(f"Total OPs: {nerf_total_mac_ops/1e12:.3f}T")
    print(f"NeRF Energy: {nerf_total_energy:.3f}J, MAC Ratio: {nerf_total_mac_energy/nerf_total_energy:.3f}")
    results['Baseline']['OPs'] = nerf_total_mac_ops
    results['Baseline']['Chip Latency'] = nerf_total_mac_latency
    results['Baseline']['Chip Energy'] = nerf_total_energy
    
    baseline_dram_access_sv_coarse = 6027.055623 * (1024 ** 2) # assume 5x5 tile, no unified cache
    baseline_dram_access_sv_fine = 6014.637089 * (1024 ** 2) # assume 5x5 tile, no unified cache
    baseline_dram_access_weight = (0.04 + 0.75) * (1024 ** 2) * hardware_configurations['Chiplets']
    baseline_dram_access = baseline_dram_access_weight + baseline_dram_access_sv_coarse + baseline_dram_access_sv_fine
    baseline_dram_energy = baseline_dram_access * hardware_configurations['DRAM Energy']
    baseline_dram_latency = baseline_dram_access / hardware_configurations['DRAM Bandwidth']
    results['Baseline']['Off-Chip Latency'] = baseline_dram_latency
    results['Baseline']['Off-Chip Energy'] = baseline_dram_energy
    print(f"EMA Energy: {baseline_dram_energy:.3f}J, EMA Latency: {baseline_dram_latency:.3f}s")
    
    
    # ----------------------------------- Solution 1a -----------------------------------
    # rgb复用？
    print(f"------------- Solution 1 -------------")
    # solution1_dram_access_sv_coarse = 201.476916 * (1024 ** 2) # assume 5x5 tile, no unified cache
    # solution1_dram_access_sv_fine = 362.878799 * (1024 ** 2) # assume 5x5 tile, no unified cache
    solution1_dram_access_sv_coarse = 108.818 * (1024 ** 2) # assume 5x5 tile, no unified cache
    solution1_dram_access_sv_fine = 108.818 * (1024 ** 2) # assume 5x5 tile, no unified cache
    solution1_dram_access_weight = (0.04 + 0.75) * (1024 ** 2) * hardware_configurations['Chiplets']
    solution1_dram_access = solution1_dram_access_weight + solution1_dram_access_sv_coarse + solution1_dram_access_sv_fine
    solution1_dram_energy = solution1_dram_access * hardware_configurations['DRAM Energy']
    solution1_dram_latency = solution1_dram_access / hardware_configurations['DRAM Bandwidth']
    results['Solution 1']['Off-Chip Latency'] = solution1_dram_latency
    results['Solution 1']['Off-Chip Energy'] = solution1_dram_energy
    print(f"EMA Energy: {solution1_dram_energy:.3f}J, EMA Latency: {solution1_dram_latency:.3f}s")
    
    solution1_d2d_access_sv_coarse = 1500.193149 * (1024 ** 2) # assume 5x5 tile, no unified cache
    solution1_d2d_access_sv_fine = 1828.555128 * (1024 ** 2) # assume 5x5 tile, no unified cache
    solution1_d2d_access = solution1_d2d_access_sv_coarse + solution1_d2d_access_sv_fine
    solution1_d2d_energy = solution1_d2d_access * hardware_configurations['UCIE Energy'] # do we have sram energy to cal?
    solution1_d2d_latency = solution1_d2d_access / (hardware_configurations['Chiplets'] * hardware_configurations['UCIE Bandwidth'])
    results['Solution 1']['Inter-Chip Latency'] = solution1_d2d_latency
    results['Solution 1']['Inter-Chip Energy'] = solution1_d2d_energy
    print(f"D2D Energy: {solution1_d2d_energy:.3f}J, D2D Latency: {solution1_d2d_latency:.3f}s")
    
    # ----------------------------------- Solution 2a -----------------------------------
    print(f"------------- Solution 2a -------------")
    optimizations = {
        'Ray Skipping': ray_skip,
        'SV Pruning': sv_ratio,
        'Fine Sharing': fine_sharing,
        'Channel Pruning': nerf_sparsity,
    }
    H, W = nerf_config['H'], nerf_config['W']
    graph_coarse = onnx.shape_inference.infer_shapes(onnx.load(coarse_path)).graph
    coarse_mac_ops, coarse_mac_latency, coarse_sram_access, coarse_sram_latency = calculate_flops_and_latency(graph_coarse, hardware_configurations, optimizations, G=True, debug=False)

    graph_fine = onnx.shape_inference.infer_shapes(onnx.load(fine_path)).graph
    fine_mac_ops, fine_mac_latency, fine_sram_access, fine_sram_latency = calculate_flops_and_latency(graph_fine, hardware_configurations, optimizations, G=True, debug=False)
    ray_skipping_size = optimizations['Ray Skipping']
    fine_comp_ratio = 1 - optimizations['Fine Sharing']
    # print(f"Ray: Fine Network Latency: {fine_sram_latency * 1e6:.3f}us")
    nerf_total_mac_latency = (coarse_mac_latency / ray_skipping_size + fine_mac_latency * fine_comp_ratio) * H * W
    nerf_total_mac_ops = (coarse_mac_ops / ray_skipping_size + fine_mac_ops * fine_comp_ratio) * H * W
    nerf_total_mac_energy = nerf_total_mac_ops * hardware_configurations['OP Energy']
    nerf_total_sram_access = (coarse_sram_access / ray_skipping_size + fine_sram_access * fine_comp_ratio) * H * W
    nerf_total_sram_energy = nerf_total_sram_access * hardware_configurations['SRAM Energy']
    nerf_total_energy = nerf_total_mac_energy + nerf_total_sram_energy
    print(f"Total OPs: {nerf_total_mac_ops/1e12:.3f}T")
    print(f"NeRF Latency: {nerf_total_mac_latency:.3f}s")
    print(f"NeRF Energy: {nerf_total_energy:.3f}J")
    results['Solution 2a']['ALL']['OPs'] = nerf_total_mac_ops
    results['Solution 2a']['ALL']['Chip Latency'] = nerf_total_mac_latency
    results['Solution 2a']['ALL']['Chip Energy'] = nerf_total_energy
    
    # ----------------------------------- Solution 2b -----------------------------------
    print(f"------------- Solution 2b -------------")
    optimizations = {
        'SR': 1.,
        'Select SR': select_sr,
    }
    H, W = nerf_config['H'], nerf_config['W']
    patch_size = nerf_config['Patch Size']
    graph_coarse = onnx.shape_inference.infer_shapes(onnx.load(coarse_path)).graph
    coarse_mac_ops, coarse_mac_latency, coarse_sram_access, coarse_sram_latency = calculate_flops_and_latency(graph_coarse, hardware_configurations, optimizations, G=True, debug=False)

    graph_fine = onnx.shape_inference.infer_shapes(onnx.load(fine_path)).graph
    fine_mac_ops, fine_mac_latency, fine_sram_access, fine_sram_latency = calculate_flops_and_latency(graph_fine, hardware_configurations, optimizations, G=True, debug=False)
    
    graph_sr = onnx.shape_inference.infer_shapes(onnx.load(sr_path)).graph
    sr_mac_ops, sr_mac_latency, sr_sram_access, sr_sram_latency = calculate_flops_and_latency(graph_sr, hardware_configurations, optimizations, G=False, debug=False)
    print(f'Vanilla Full-SR OPs: {sr_mac_ops/1e12:.3f}T')
    K = patch_size / (368 * 504) * 0.2773 * 1e12 / sr_mac_ops # TODO
    sr_mac_ops, sr_mac_latency, sr_sram_access, sr_sram_latency = sr_mac_ops*K, sr_mac_latency*K, sr_sram_access*K, sr_sram_latency*K
    sel_sr_rate = optimizations['Select SR']
    total_pixels = H * W
    total_patches_base = total_pixels / (2 * 2 * patch_size)
    
    sr_patches = total_patches_base * sel_sr_rate
    sr_total_mac_ops = sr_patches * sr_mac_ops
    sr_total_mac_latency = sr_patches * sr_mac_latency
    sr_total_sram_access = sr_patches * sr_sram_access
    sr_total_mac_energy = sr_total_mac_ops * hardware_configurations['OP Energy']
    sr_total_sram_energy = sr_total_sram_access * hardware_configurations['SRAM Energy']
    sr_total_energy = sr_total_mac_energy + sr_total_sram_energy

    total_patches_hr = total_patches_base * (1 - sel_sr_rate) * 2 * 2
    total_patches_lr = total_patches_base * sel_sr_rate
    total_patches_nerf = total_patches_hr + total_patches_lr
    
    nerf_patch_mac_ops = (coarse_mac_ops + fine_mac_ops) * patch_size
    nerf_patch_mac_latency = (coarse_mac_latency + fine_mac_latency) * patch_size
    nerf_patch_sram_access = (coarse_sram_access + fine_sram_access) * patch_size
    
    nerf_total_mac_ops = total_patches_nerf * nerf_patch_mac_ops
    nerf_total_mac_latency = total_patches_nerf * nerf_patch_mac_latency
    nerf_total_sram_access = total_patches_nerf * nerf_patch_sram_access
    
    nerf_total_mac_energy = nerf_total_mac_ops * hardware_configurations['OP Energy']
    nerf_total_sram_energy = nerf_total_sram_access * hardware_configurations['SRAM Energy']
    nerf_total_energy = nerf_total_mac_energy + nerf_total_sram_energy
    
    total_mac_ops = nerf_total_mac_ops + sr_total_mac_ops
    total_mac_latency = nerf_total_mac_latency + sr_total_mac_latency
    total_energy = nerf_total_energy + sr_total_energy
    print(f"Total OPs: {total_mac_ops/1e12:.3f}T, NeRF OPs: {nerf_total_mac_ops/1e12:.3f}T, sr OPs: {sr_total_mac_ops/1e12:.3f}T")
    print(f"Total Latency: {total_mac_latency:.3f}s, NeRF Latency: {nerf_total_mac_latency:.3f}s, SR Latency: {sr_total_mac_latency:.3f}s")
    print(f"Total Energy: {total_energy:.3f}J, NeRF Energy: {nerf_total_energy:.3f}J, SR Energy: {sr_total_energy:.3f}J")
    results['Solution 2b']['Select SR']['OPs'] = total_mac_ops
    results['Solution 2b']['Select SR']['Chip Latency'] = total_mac_latency
    results['Solution 2b']['Select SR']['Chip Energy'] = total_energy
    
    # ----------------------------------- ALL -----------------------------------
    print(f"------------- ALL -------------")
    optimizations = {
        'Ray Skipping': ray_skip,
        'SV Pruning': sv_ratio,
        'Fine Sharing': fine_sharing,
        'Channel Pruning': nerf_sparsity,
        'SR': 1.,
        'Select SR': select_sr,
    }
    H, W = nerf_config['H'], nerf_config['W']
    patch_size = nerf_config['Patch Size']
    graph_coarse = onnx.shape_inference.infer_shapes(onnx.load(coarse_path)).graph
    coarse_mac_ops, coarse_mac_latency, coarse_sram_access, coarse_sram_latency = calculate_flops_and_latency(graph_coarse, hardware_configurations, optimizations, G=True, debug=False)

    graph_fine = onnx.shape_inference.infer_shapes(onnx.load(fine_path)).graph
    fine_mac_ops, fine_mac_latency, fine_sram_access, fine_sram_latency = calculate_flops_and_latency(graph_fine, hardware_configurations, optimizations, G=True, debug=False)
    
    optimizations = {
        'Ray Skipping': ray_skip,
        'SV Pruning': sv_ratio,
        'Fine Sharing': fine_sharing,
        'Channel Pruning': sr_sparsity,
        'SR': 1.,
        'Select SR': select_sr,
    }
    graph_sr = onnx.shape_inference.infer_shapes(onnx.load(sr_path)).graph
    sr_mac_ops, sr_mac_latency, sr_sram_access, sr_sram_latency = calculate_flops_and_latency(graph_sr, hardware_configurations, optimizations, G=False, debug=False)
    print(f'Vanilla Full-SR OPs: {sr_mac_ops/1e12:.3f}T')
    # K = patch_size / (368 * 504) * 0.2773 * 1e12 / sr_mac_ops # TODO 这里不能用了,要用原始的再弄
    sr_mac_ops, sr_mac_latency, sr_sram_access, sr_sram_latency = sr_mac_ops*K, sr_mac_latency*K, sr_sram_access*K, sr_sram_latency*K
    sel_sr_rate = optimizations['Select SR']
    total_pixels = H * W
    total_patches_base = total_pixels / (2 * 2 * patch_size)
    
    sr_patches = total_patches_base * sel_sr_rate
    sr_total_mac_ops = sr_patches * sr_mac_ops
    sr_total_mac_latency = sr_patches * sr_mac_latency
    sr_total_sram_access = sr_patches * sr_sram_access
    sr_total_mac_energy = sr_total_mac_ops * hardware_configurations['OP Energy']
    sr_total_sram_energy = sr_total_sram_access * hardware_configurations['SRAM Energy']
    sr_total_energy = sr_total_mac_energy + sr_total_sram_energy

    total_patches_hr = total_patches_base * (1 - sel_sr_rate) * 2 * 2
    total_patches_lr = total_patches_base * sel_sr_rate
    total_patches_nerf = total_patches_hr + total_patches_lr
    
    ray_skipping_size = optimizations['Ray Skipping']
    fine_comp_ratio = 1 - optimizations['Fine Sharing']
    nerf_patch_mac_ops = (coarse_mac_ops / ray_skipping_size + fine_mac_ops * fine_comp_ratio) * patch_size
    nerf_patch_mac_latency = (coarse_mac_latency / ray_skipping_size + fine_mac_latency * fine_comp_ratio) * patch_size
    nerf_patch_sram_access = (coarse_sram_access / ray_skipping_size + fine_sram_access * fine_comp_ratio) * patch_size
    
    nerf_total_mac_ops = total_patches_nerf * nerf_patch_mac_ops
    nerf_total_mac_latency = total_patches_nerf * nerf_patch_mac_latency
    nerf_total_sram_access = total_patches_nerf * nerf_patch_sram_access
    
    nerf_total_mac_energy = nerf_total_mac_ops * hardware_configurations['OP Energy']
    nerf_total_sram_energy = nerf_total_sram_access * hardware_configurations['SRAM Energy']
    nerf_total_energy = nerf_total_mac_energy + nerf_total_sram_energy
    
    total_mac_ops = nerf_total_mac_ops + sr_total_mac_ops
    total_mac_latency = nerf_total_mac_latency + sr_total_mac_latency
    total_energy = nerf_total_energy + sr_total_energy
    print(f"Total OPs: {total_mac_ops/1e12:.3f}T, NeRF OPs: {nerf_total_mac_ops/1e12:.3f}T, sr OPs: {sr_total_mac_ops/1e12:.3f}T")
    print(f"Total Latency: {total_mac_latency:.3f}s, NeRF Latency: {nerf_total_mac_latency:.3f}s, SR Latency: {sr_total_mac_latency:.3f}s")
    # print(f"MAC Energy: {}")
    print(f"Total Energy: {total_energy*1e3:.2f}mJ, NeRF Energy: {nerf_total_energy:.3f}J, SR Energy: {sr_total_energy:.3f}J")
    print(f"NeRF SRAM Ratio: {nerf_total_sram_energy/total_energy:.3f}, SR SRAM Ratio: {sr_total_sram_energy/total_energy:.3f}")
    results['ALL']['OPs'] = total_mac_ops
    results['ALL']['Chip Latency'] = total_mac_latency
    results['ALL']['Chip Energy'] = total_energy
    
    # DRAM应该是和solution1 完全一样的
    results['ALL']['Off-Chip Latency'] = solution1_dram_latency
    results['ALL']['Off-Chip Energy'] = solution1_dram_energy
    # print(f"EMA Energy: {solution1_dram_energy:.3f}J, EMA Latency: {solution1_dram_latency:.3f}s")
    
    solution1_d2d_access_sv_coarse_per_patch = solution1_d2d_access_sv_coarse / (2*2*total_patches_base) # 因为这是一个20*20的大patch(HR), 如果是LR的话只需要做其中1/4, 而后面total_patches_nerf就已经把需要做的10x10大小的patch全部考虑了
    solution1_d2d_access_sv_fine_per_patch = solution1_d2d_access_sv_fine / (2*2*total_patches_base)
    sr_solution1_d2d_access_sv_coarse = total_patches_nerf * solution1_d2d_access_sv_coarse_per_patch
    sr_solution1_d2d_access_sv_fine = total_patches_nerf * solution1_d2d_access_sv_fine_per_patch
    
    sr_solution1_d2d_access = sr_solution1_d2d_access_sv_coarse + sr_solution1_d2d_access_sv_fine
    sr_solution1_d2d_energy = sr_solution1_d2d_access * hardware_configurations['UCIE Energy'] # do we have sram energy to cal?
    sr_solution1_d2d_latency = sr_solution1_d2d_access / (hardware_configurations['Chiplets'] * hardware_configurations['UCIE Bandwidth'])
    results['ALL']['Inter-Chip Latency'] = sr_solution1_d2d_latency
    results['ALL']['Inter-Chip Energy'] = sr_solution1_d2d_energy
    print(f"SR D2D Access: {sr_solution1_d2d_access/1e9:.3f}GB; SR D2D Energy: {sr_solution1_d2d_energy:.3f}J, SR D2D Latency: {sr_solution1_d2d_latency:.3f}s")
    
    # Projection & Interpolation, SRAM & Compute Latency Hide? 还没考虑fine sharing情况，不过Fine只减少计算，投影都要做的？检查一下结构 会不会只需要rgb的话有些不用
    projection_sr_solution1_sram_access = patch_size * total_patches_nerf * (16. / ray_skipping_size + 48) * 3 * (1 - optimizations['SV Pruning']) # 3是3d pts大小
    projection_sr_solution1_proj_ops = hardware_configurations['OP per Projection'] * patch_size * total_patches_nerf * (16. / ray_skipping_size + 48) * (1 - optimizations['SV Pruning'])
    
    interpolation_sr_solution1_sram_access = sr_solution1_d2d_access + patch_size * total_patches_nerf * (16. / ray_skipping_size + 48) * (3 + 32) * (1 - optimizations['SV Pruning']) + projection_sr_solution1_sram_access
    interpolation_sr_solution1_sram_energy = interpolation_sr_solution1_sram_access * hardware_configurations['SRAM Energy']
    interpolation_sr_solution1_interpolation_ops = hardware_configurations['OP per Interpolation'] * patch_size * total_patches_nerf * (16. / ray_skipping_size + 48) * (3 + 32) * (1 - optimizations['SV Pruning']) + projection_sr_solution1_proj_ops # coarse没有SV Prune
    interpolation_sr_solution1_interpolation_energy = interpolation_sr_solution1_interpolation_ops * hardware_configurations['OP Energy']
    print(f"Interpolation SRAM Access: {interpolation_sr_solution1_sram_access/1e9:.3f}GB, OP: {interpolation_sr_solution1_interpolation_ops/1e9:.3f}G")
    print(f"Interpolation SRAM Energy: {interpolation_sr_solution1_sram_energy:.3f}J, OP Energy: {interpolation_sr_solution1_interpolation_energy:.6f}J")
    
    # results['ALL']['Chip Latency'] += total_mac_latency
    total_interpolation_energy = sr_solution1_d2d_energy + interpolation_sr_solution1_sram_energy + interpolation_sr_solution1_interpolation_energy
    print(f"{results['ALL']['Chip Energy']:.2f}J, {total_interpolation_energy:.2f}J, {results['ALL']['Inter-Chip Energy']:.2f}J")
    results['ALL']['Chip Energy'] += total_interpolation_energy
    
    org_mac_ops = results['Baseline']['OPs']
    print(f"TOPS/W: {org_mac_ops*1e-12/total_energy:.3f}")
    total_energy = results['ALL']['Chip Energy'] # + results['ALL']['Inter-Chip Energy'] 算过了
    print(f"Final FPS per Frame: {1/total_mac_latency:.2f}")
    print(f"Final Energy per Frame: {total_energy*1e3:.2f}mJ")
    print(f"Final Energy per Pixel: {total_energy*1e6/(H*W):.2f}μJ")
    print(f"TOPS/W Theoretical Scaling Number: {org_mac_ops/total_mac_ops:.2f}x")
    print(f"TOPS/W with Interpolation: {org_mac_ops*1e-12/total_energy:.3f}")
    
# 定义卷积层 FLOPs 计算
def calculate_conv_flops(output_shape, weight_shape):
    flops_per_position = weight_shape[2] * weight_shape[3] * weight_shape[1]
    total_flops = output_shape[1] * output_shape[2] * output_shape[3] * flops_per_position
    return total_flops * 2 # MAC->OP

def calculate_conv_sram_os(output_shape, weight_shape, Tm, Trc):
    factor_ifmap = output_shape[1] / Tm
    sram_ifmap = output_shape[1] * output_shape[2] * output_shape[3] * factor_ifmap
    factor_weight = (output_shape[2] * output_shape[3]) / Trc
    sram_weight = weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[3] * factor_weight
    factor_ofmap = 1.
    sram_ofmap = output_shape[1] * output_shape[2] * output_shape[3] * factor_ofmap
    return sram_ifmap, sram_weight, sram_ofmap

# 定义 Attention 中 MatMul 的 FLOPs 计算
def calculate_attention_matmul_flops(input_shape_1, input_shape_2):
    batch, head, mul_0, mul_1 = input_shape_1
    batch, head, mul_1, mul_2 = input_shape_2
    return batch * head * mul_0 * mul_1 * mul_2 * 2 # MAC->OP

# 定义 DepthToSpace FLOPs 计算
def calculate_depthtospace_flops(input_shape):
    batch, channels, height, width = input_shape
    return batch * channels * height * width * 2 # MAC->OP

# 定义 DepthToSpace 延时计算
def depthtospace_latency(input_shape, frequency):
    flops = calculate_depthtospace_flops(input_shape)
    return flops / (frequency * 1e6)

# 计算 ops
def calculate_ops(macs, frequency, num_chiplets, efficiency=0.95):
    return macs * frequency * efficiency * 2 * num_chiplets

def calculate_bandwidth(bandwidth, frequency, num_chiplets, efficiency=0.95):
    return bandwidth * frequency * efficiency * num_chiplets

# 获取权重或输入的形状
def get_shape_from_initializer_or_value(graph, name):
    # 检查是否在 initializer 中
    for initializer in graph.initializer:
        if initializer.name == name:
            weight = numpy_helper.to_array(initializer)
            return list(weight.shape)

    # 检查是否在 value_info 中
    for value_info in graph.value_info:
        if value_info.name == name:
            return [dim.dim_value if dim.dim_value > 0 else 1 for dim in value_info.type.tensor_type.shape.dim]

    # 检查是否在 graph.input 或 graph.output 中
    for input_info in graph.input:
        if input_info.name == name:
            return [dim.dim_value if dim.dim_value > 0 else 1 for dim in input_info.type.tensor_type.shape.dim]
    for output_info in graph.output:
        if output_info.name == name:
            return [dim.dim_value if dim.dim_value > 0 else 1 for dim in output_info.type.tensor_type.shape.dim]

    return None

# 计算总 FLOPs 和延时
def calculate_flops_and_latency(graph, hardware_configurations, optimizations, G, debug):
    macs = hardware_configurations['MACs']
    Tm, Trc = hardware_configurations['Tm'], hardware_configurations['Trc']
    macs_efficiency = hardware_configurations['MAC Efficiency']
    ai_core_frequency = hardware_configurations['AI Core Frequency']
    activation_buffer_bandwidth_cycle = hardware_configurations['Activation Buffer Bandwidth']
    weight_buffer_bandwidth_cycle = hardware_configurations['Weight Buffer Bandwidth']
    num_chiplets = hardware_configurations['Chiplets']
    ops = calculate_ops(macs, ai_core_frequency, num_chiplets, macs_efficiency)  # 计算 ops
    activation_buffer_bandwidth = calculate_bandwidth(activation_buffer_bandwidth_cycle, ai_core_frequency, macs_efficiency)
    weight_buffer_bandwidth = calculate_bandwidth(weight_buffer_bandwidth_cycle, ai_core_frequency, macs_efficiency)
    # print(f'{activation_buffer_bandwidth/1e9:.3f}, {ops/1e9:.3f}')
    
    total_mac_ops = 0
    total_mac_latency = 0
    total_sram_latency = 0
    total_sram_access = 0
    
    if 'Channel Pruning' in optimizations:
        channel_pruning_ratio = optimizations['Channel Pruning']
    else:
        channel_pruning_ratio = 0.
    
    for node in graph.node:
        node_name = node.name if node.name else "Unnamed Node"
        if 'fn/to_out' in node_name: continue
        if node.op_type == 'Conv':
            weight_shape = get_shape_from_initializer_or_value(graph, node.input[1])
            if weight_shape is None:
                print(f"Warning: Cannot find weight shape for Conv node: {node_name}")
                continue

            # 获取 stride 信息，默认 stride 为 [1, 1]
            stride = [1, 1]
            for attr in node.attribute:
                if attr.name == "strides":
                    stride = attr.ints

            output_shape = get_shape_from_initializer_or_value(graph, node.output[0])
            if output_shape is None:
                print(f"Warning: Cannot find output shape for Conv node: {node_name}")
                continue

            mac_ops = calculate_conv_flops(output_shape, weight_shape) * (1 - channel_pruning_ratio)
            mac_latency = mac_ops / ops

            # # 若 stride 不为 1，则调整延时 TODO
            # if stride[0] != 1 or stride[1] != 1:
            #     latency *= stride[0] * stride[1]
            sram_ifmap, sram_weight, sram_ofmap = calculate_conv_sram_os(output_shape, weight_shape, Tm, Trc)
            sram_weight = sram_weight * (1 - channel_pruning_ratio)

            sram_access_activation = (sram_ifmap + sram_ofmap)
            sram_access_weight = sram_weight
            
            sram_latency_activation = sram_access_activation / activation_buffer_bandwidth
            sram_latency_weight = sram_access_weight / weight_buffer_bandwidth
            # assert sram_latency_activation == sram_latency_weight
            sram_latency = sram_latency_activation
            
            total_mac_ops += mac_ops
            total_mac_latency += mac_latency
            total_sram_access += (sram_access_activation + sram_access_weight)
            total_sram_latency += sram_latency
            
            # 打印单个节点的 FLOPs 和延时
            if debug:
                if G: print(f"Node: {node_name} (Conv) - FLOPs: {mac_ops / 1e6:.3f} MFLOPs, Latency: {mac_latency * 1e6:.3f} us, SRAM access: {sram_access / 1024:.3f}KB")
                else: print(f"Node: {node_name} (Conv) - FLOPs: {mac_ops / 1e9:.3f} GFLOPs, Latency: {mac_latency * 1e3:.3f} ms, SRAM access: {sram_access / (1024**2):.3f}MB")
        elif node.op_type == 'MatMul':
            input_shape_1 = get_shape_from_initializer_or_value(graph, node.input[0])
            input_shape_2 = get_shape_from_initializer_or_value(graph, node.input[1])
            
            if input_shape_1 is None or input_shape_2 is None:
                print(f"Warning: Cannot find input shapes for {node.op_type} node: {node_name}")
                continue

            # 提取最后两个维度
            if len(input_shape_1) < 2 or len(input_shape_2) < 2:
                print(f"Warning: Input shapes are invalid for {node.op_type} node: {node_name}, input_shape_1: {input_shape_1}, input_shape_2: {input_shape_2}")
                continue

            last_dim_1 = input_shape_1[-2]  # 倒数第二个维度
            last_dim_2 = input_shape_1[-1]  # 最后一个维度
            last_dim_3 = input_shape_2[-1]  # input_shape_2 的最后一个维度

            # 确保形状匹配（input_shape_1 的最后一个维度 == input_shape_2 的倒数第二个维度）
            if last_dim_2 != input_shape_2[-2]:
                print(f"Error: Dimension mismatch for {node.op_type} node: {node_name}, input_shape_1: {input_shape_1}, input_shape_2: {input_shape_2}")
                continue
            
            assert input_shape_1[-1] == input_shape_2[-2]

            # 计算 FLOPs = 最后两个维度相乘 * input_shape_1 除最后两个维度外的乘积
            flops_inner = last_dim_1 * last_dim_2 * last_dim_3
            
            # SRAM access
            input_1_size = input_shape_1[-2] * input_shape_1[-1]
            input_2_size = input_shape_2[-2] * input_shape_2[-1]
            output_size = input_shape_1[-2] * input_shape_2[-1]
            factor_1 = input_shape_2[-1] / Tm
            factor_2 = input_shape_1[-2] / Trc
            factor_3 = 1.
            sram_ifmap = input_1_size * factor_1
            sram_weight = input_2_size * factor_2
            sram_weight = sram_weight * (1 - channel_pruning_ratio)
            sram_ofmap = output_size * factor_3
            
            outer_dims = 1
            if len(input_shape_1) > 2:
                for dim in input_shape_1[:-2]:  # 除最后两个维度外的其他维度
                    outer_dims *= dim
            
            mac_ops = flops_inner * outer_dims * 2 * (1 - channel_pruning_ratio)
            mac_latency = mac_ops / ops
            if 'SV Pruning' in optimizations: # reorder overhead?
                sv_prune_ratio = optimizations['SV Pruning']
                if not ('ray_att' in node_name or 'out_geo' in node_name): # 'rgb_fc' in node_name or 
                    mac_ops = mac_ops * (1 - sv_prune_ratio)
                    mac_latency = mac_latency * (1 - sv_prune_ratio)
                    sram_ifmap = sram_ifmap * (1 - sv_prune_ratio)
                    sram_ofmap = sram_ofmap * (1 - sv_prune_ratio)
                    
            # sram_access_activation = (sram_ifmap + sram_ofmap) * outer_dims
            sram_access_activation = (sram_ifmap) * outer_dims # TODO
            sram_access_weight = sram_weight * outer_dims
            
            sram_latency_activation = sram_access_activation / activation_buffer_bandwidth
            sram_latency_weight = sram_access_weight / weight_buffer_bandwidth
            # assert sram_latency_activation == sram_latency_weight
            sram_latency = sram_latency_activation
                    
            total_mac_ops += mac_ops
            total_mac_latency += mac_latency
            total_sram_access += (sram_access_activation + sram_access_weight)
            total_sram_latency += sram_latency
            
            # print(f'{outer_dims}, {mac_latency*1e6:.3f}μs, {sram_latency_activation*1e6:.3f}μs, {sram_latency_weight*1e6:.3f}μs, {sram_latency*1e6:.3f}μs, {activation_buffer_bandwidth/1e9:.3f}, {weight_buffer_bandwidth/1e9:.3f}')
            # 打印单个节点的 FLOPs 和延时
            if debug:
                if G: print(f"Node: {node_name} ({node.op_type}) - FLOPs: {mac_ops / 1e6:.3f} MFLOPs, Latency: {total_mac_latency * 1e6:.3f} us, SRAM access: {sram_access / 1024:.3f}KB")
                else: print(f"Node: {node_name} ({node.op_type}) - FLOPs: {mac_ops / 1e9:.3f} GFLOPs, Latency: {total_mac_latency * 1e3:.3f} ms, SRAM access: {sram_access / (1024**2):.3f}MB")
        elif node.op_type == 'DepthToSpace':
            continue
            input_shape = get_shape_from_initializer_or_value(graph, node.input[0])
            output_shape = get_shape_from_initializer_or_value(graph, node.output[0])

            if input_shape is None or output_shape is None:
                print(f"Warning: Cannot find input/output shape for DepthToSpace node: {node_name}")
                continue

            # 计算 DepthToSpace 节点的 FLOPs 和延时
            mac_ops = calculate_depthtospace_flops(input_shape)
            mac_latency = mac_ops / ops
            
            total_mac_ops += mac_ops
            total_mac_latency += mac_latency
            total_sram_latency += (sram_access / activation_buffer_bandwidth)

            # 打印单个节点的 FLOPs 和延时
            if debug:
                if G: print(f"Node: {node_name} (DepthToSpace) - FLOPs: {mac_ops / 1e6:.3f} MFLOPs, Latency: {mac_latency * 1e6:.3f} us")
                else: print(f"Node: {node_name} (DepthToSpace) - FLOPs: {mac_ops / 1e9:.3f} GFLOPs, Latency: {mac_latency * 1e3:.3f} ms")

    return total_mac_ops, total_mac_latency, total_sram_access, total_sram_latency


if __name__ == '__main__':

    train()

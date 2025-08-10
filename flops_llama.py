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


import os
import time
import numpy as np
import shutil
import torch
import torch.utils.data.distributed
import os




def train():
    parser = argparse.ArgumentParser(description="Calculate GFLOPs and latency for an ONNX model.")
    parser.add_argument("--model_path", type=str, default="./resnet50-v2-7.onnx",
                        help="Path to the ONNX model file.")
    # parser.add_argument("--model_path", type=str, default="/home/ytanaz/access/IBRNet/ibrnet_generalizable_fine_48_simp.onnx", help="Path to the ONNX model file.")

    parser.add_argument("--frequency", type=int, default=500, help="Frequency in MHz (default: 500 MHz).")
    args = parser.parse_args()

    G = True if 'ibr' in args.model_path else False
    # 加载模型
    model = onnx.load(args.model_path)
    model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph

    # 计算 GFLOPs 和延时
    total_flops, total_latency = calculate_flops_and_latency(graph, args.frequency, G=G)

    # 输出总结果：GFLOPs 和延时 (ms)
    print(f"Total FLOPs: {total_flops / 1e9:.3f} GFLOPs")


import onnx
from onnx import numpy_helper
import argparse

# 固定的 MAC 数量
MAC_units = 2048


# 定义卷积层 FLOPs 计算
def calculate_conv_flops(output_shape, weight_shape):
    print(output_shape, weight_shape)
    flops_per_position = weight_shape[2] * weight_shape[3] * weight_shape[1]
    total_flops = output_shape[1] * output_shape[2] * output_shape[3] * flops_per_position
    return total_flops * 2  # MAC->OP


# 定义 Attention 中 MatMul 的 FLOPs 计算
def calculate_attention_matmul_flops(input_shape_1, input_shape_2):
    batch, head, mul_0, mul_1 = input_shape_1
    batch, head, mul_1, mul_2 = input_shape_2
    return batch * head * mul_0 * mul_1 * mul_2 * 2  # MAC->OP


# 定义 DepthToSpace FLOPs 计算
def calculate_depthtospace_flops(input_shape):
    batch, channels, height, width = input_shape
    return batch * channels * height * width * 2  # MAC->OP


# 定义 DepthToSpace 延时计算
def depthtospace_latency(input_shape, frequency):
    flops = calculate_depthtospace_flops(input_shape)
    return flops / (frequency * 1e6)


# 计算 TMACs
def calculate_tmacs(frequency, efficiency=0.95):
    return MAC_units * frequency * 1e6 * efficiency  # TMACs based on frequency in Hz


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
def calculate_flops_and_latency(graph, frequency, G):
    tmacs = calculate_tmacs(frequency)  # 计算 TMACs
    total_flops = 0
    total_latency = 0

    for node in graph.node:
        node_name = node.name if node.name else "Unnamed Node"

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

            flops = calculate_conv_flops(output_shape, weight_shape)
            latency = flops / tmacs

            # 若 stride 不为 1，则调整延时
            if stride[0] != 1 or stride[1] != 1:
                latency *= stride[0] * stride[1]

            total_flops += flops
            total_latency += latency

            # 打印单个节点的 FLOPs 和延时
            if G:
                print(f"Node: {node_name} (Conv) - FLOPs: {flops / 1e6:.3f} MFLOPs, Latency: {latency * 1e6:.3f} us")
            else:
                print(f"Node: {node_name} (Conv) - FLOPs: {flops / 1e9:.3f} GFLOPs, Latency: {latency * 1e3:.3f} ms")
        elif node.op_type == 'DepthToSpace':
            block_size = 1
            for attr in node.attribute:
                if attr.name == "blocksize":
                    block_size = attr.i

            input_shape = get_shape_from_initializer_or_value(graph, node.input[0])
            output_shape = get_shape_from_initializer_or_value(graph, node.output[0])

            if input_shape is None or output_shape is None:
                print(f"Warning: Cannot find input/output shape for DepthToSpace node: {node_name}")
                continue

            # 计算 DepthToSpace 节点的 FLOPs 和延时
            flops = calculate_depthtospace_flops(input_shape)
            latency = flops / tmacs

            total_flops += flops
            total_latency += latency

            # 打印单个节点的 FLOPs 和延时
            if G:
                print(
                    f"Node: {node_name} (DepthToSpace) - FLOPs: {flops / 1e6:.3f} MFLOPs, Latency: {latency * 1e6:.3f} us")
            else:
                print(
                    f"Node: {node_name} (DepthToSpace) - FLOPs: {flops / 1e9:.3f} GFLOPs, Latency: {latency * 1e3:.3f} ms")
        elif node.op_type == 'MatMul':
            input_shape_1 = get_shape_from_initializer_or_value(graph, node.input[0])
            input_shape_2 = get_shape_from_initializer_or_value(graph, node.input[1])

            if input_shape_1 is None or input_shape_2 is None:
                print(f"Warning: Cannot find input shapes for {node.op_type} node: {node_name}")
                continue

            # 提取最后两个维度
            if len(input_shape_1) < 2 or len(input_shape_2) < 2:
                print(
                    f"Warning: Input shapes are invalid for {node.op_type} node: {node_name}, input_shape_1: {input_shape_1}, input_shape_2: {input_shape_2}")
                continue

            last_dim_1 = input_shape_1[-2]  # 倒数第二个维度
            last_dim_2 = input_shape_1[-1]  # 最后一个维度
            last_dim_3 = input_shape_2[-1]  # input_shape_2 的最后一个维度

            # 确保形状匹配（input_shape_1 的最后一个维度 == input_shape_2 的倒数第二个维度）
            if last_dim_2 != input_shape_2[-2]:
                print(
                    f"Error: Dimension mismatch for {node.op_type} node: {node_name}, input_shape_1: {input_shape_1}, input_shape_2: {input_shape_2}")
                continue

            # 计算 FLOPs = 最后两个维度相乘 * input_shape_1 除最后两个维度外的乘积
            flops_inner = last_dim_1 * last_dim_2 * last_dim_3
            outer_dims = 1
            if len(input_shape_1) > 2:
                for dim in input_shape_1[:-2]:  # 除最后两个维度外的其他维度
                    outer_dims *= dim
                    # print(last_dim_1, last_dim_2, last_dim_3, input_shape_1, outer_dims)

            flops = flops_inner * outer_dims

            # 计算延时
            latency = flops / tmacs * 2  # MAC->OP
            total_flops += flops * 2  # MAC->OP
            total_latency += latency * 2  # MAC->OP

            # 打印单个节点的 FLOPs 和延时
            if G:
                print(
                    f"Node: {node_name} ({node.op_type}) - FLOPs: {flops / 1e6:.3f} MFLOPs, Latency: {latency * 1e6:.3f} us")
            else:
                print(
                    f"Node: {node_name} ({node.op_type}) - FLOPs: {flops / 1e9:.3f} GFLOPs, Latency: {latency * 1e3:.3f} ms")

    return total_flops, total_latency


if __name__ == '__main__':
    train()

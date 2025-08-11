import torch

# 加载 .pth 文件
pth_path = "pretrained/epoch896_OmniSR.pth"  # 替换为你的 .pth 文件路径
state_dict = torch.load(pth_path, map_location=torch.device('cpu'))

# 统计所有权重张量的元素总数
total_params = sum(tensor.numel() for tensor in state_dict.values())

print(f"Total Parameters (Estimated): {total_params}")

import torch
from thop import profile

# 定义一个简单的模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x)

# 初始化模型和输入
model = SimpleModel()
input_tensor = torch.randn(1, 2)

# 使用 thop 计算 FLOPs
flops, params = profile(model, inputs=(input_tensor,))
print(f"FLOPs: {flops}")
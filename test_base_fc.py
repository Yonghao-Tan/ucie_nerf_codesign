import torch
import torch.nn as nn

n_rays, n_samples, n_views, n_feat, out_dim = 1, 2, 3, 4, 5

base_fc = nn.Linear(3 * n_feat, out_dim, bias=True)
globalfeat = torch.randn(n_rays, n_samples, 1, 2 * n_feat)
rgb_feat = torch.randn(n_rays, n_samples, n_views, n_feat)

# 原始做法
globalfeat_expand = globalfeat.expand(-1, -1, n_views, -1)
x = torch.cat([globalfeat_expand, rgb_feat], dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat]
out1 = base_fc(x)

# 拆权重做法
W_global = base_fc.weight[:, :2 * n_feat]     # [out_dim, 2*n_feat]
W_rgb = base_fc.weight[:, 2 * n_feat:]        # [out_dim, n_feat]
b = base_fc.bias                              # [out_dim]

out2_g = torch.matmul(globalfeat, W_global.T).expand(-1, -1, n_views, -1)
out2_r = torch.matmul(rgb_feat, W_rgb.T)
out2 = out2_g + out2_r + b
print(out1)
print(out2)
print(torch.allclose(out1, out2))  # True 表示等价
print('max abs diff:', (out1 - out2).abs().max())
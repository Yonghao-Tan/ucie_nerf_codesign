import torch
import torch.nn as nn
import torch.nn.functional as F

def reshape(x_in, H, W, block_length=5, out=False):  # block_size*cols, n_samples, n_views, n_feat
    H_exclude, W_exclude = H % 5, W % 5
    x_in_2d = x_in.reshape(H, W, *x_in.shape[1:])
    if H_exclude > 0 and W_exclude > 0:
        x_slice = x_in_2d[:-H_exclude, :-W_exclude]
    elif H_exclude > 0: 
        x_slice = x_in_2d[:-H_exclude, :]
    elif W_exclude > 0:
        x_slice = x_in_2d[:, :-W_exclude]
    else:
        x_slice = x_in_2d
    h, w, *xxx = x_slice.shape
    assert h % block_length == 0 and w % block_length == 0, "h 和 w 必须能被 block_size 整除"
    # 重塑张量为 [h // block_size, block_size, w // block_size, block_size, xxx]
    x_slice = x_slice.view(h // block_length, block_length, w // block_length, block_length, *xxx)
    # 调整维度顺序为 [block_groups, block_size, block_size, xxx]
    x_slice = x_slice.permute(0, 2, 1, 3, *range(4, 4 + len(xxx)))
    # 合并前两维为 block_groups
    x = x_slice.reshape(-1, block_length*block_length, *xxx) # block_groups, block_size, n_samples, n_views, n_feat
    
    block_groups = x.shape[0]
    if out: x = x.squeeze(3)
    split_feats = [x[i] for i in range(block_groups)]
    return split_feats, h, w

def inverse_reshape(tensor, h, w, block_length=5):
    # 获取输入张量的形状
    block_groups, _, *xxx = tensor.shape
    # 恢复 block_size 的二维形状
    reshaped = tensor.view(block_groups, block_length, block_length, *xxx)
    # 将 block_groups 拆分为 [h // block_size, w // block_size]
    reshaped = reshaped.view(h // block_length, w // block_length, block_length, block_length, *xxx)
    # 调整维度顺序为 [h // block_size, w // block_size, block_size, block_size, xxx]
    permuted = reshaped.permute(0, 2, 1, 3, *range(4, 4 + len(xxx)))
    # 合并维度为 [h, w, xxx]
    result = permuted.reshape(h, w, *xxx)
    return result

class MOE(nn.Module):
    def __init__(self):
        super(MOE, self).__init__()
        self.mof_l2_loss = nn.MSELoss()

    def forward(self, H, W, rgb_feat, rgb_in, weights, out_org, mask_all, use_moe_block=False):
        if not self.training and not use_moe_block:
            return out_org, torch.tensor(0.0).to(out_org.device)
        block_size = 25
        split_rgb_feats, h, w = reshape(rgb_feat, H, W)
        split_weights, _, _ = reshape(weights, H, W)
        split_mask_all, _, _ = reshape(mask_all, H, W)
        split_rgb_ins, _, _ = reshape(rgb_in, H, W)
        split_outs, _, _ = reshape(out_org.clone().unsqueeze(2), H, W, out=True) # n_col_blocks, block_size*block_size, n_feat
        positions = [0, 2, 4, 10, 12, 14, 20, 22, 24]
        # positions = [1,3,5,6,7,8,9,11,13,15,16,17,18,19,21,23]
        # positions = [0,1,2,3,4,5,6,7,8,9,10,11,   13,14,15,16,17,18,19,20,21,22,23,24]
        for i in range(len(split_outs)):
            split_outs[i] = self.mof_func_distance(positions, split_rgb_feats[i], split_rgb_ins[i], split_weights[i], split_outs[i], split_mask_all[i], block_size)
        out_slice = torch.stack(split_outs, dim=0) # block_groups, block_size, n_samples, n_feat
        out_slice = inverse_reshape(out_slice, h, w)
        out_org_2d = out_org.reshape(H, W, *out_org.shape[1:])
        H_exclude, W_exclude = H % 5, W % 5
        if H_exclude > 0 and W_exclude > 0:
            out_org_2d[:H-H_exclude, :W-W_exclude] = out_slice
        elif H_exclude > 0: 
            out_org_2d[:H-H_exclude, :] = out_slice
        elif W_exclude > 0:
            out_org_2d[:, :W-W_exclude] = out_slice
        else:
            out_org_2d = out_slice
        out = out_org_2d.reshape(-1, *out_org.shape[1:])  # 恢复为原始形状
        mof_l2_loss = self.mof_l2_loss(out, out_org)
        return out, mof_l2_loss
    
    def mof_func_avg(self, positions, rgb_feat, rgb_in, weight, out, mask, block_size):
        n_rays, n_samples, n_views, n_feat = rgb_feat.shape
        known_len = len(positions)
        unknown_len = block_size - known_len
        all_positions = list(range(block_size))
        exclude_positions = [pos for pos in all_positions if pos not in positions]
        
        known_rgb_weight = weight[positions].unsqueeze(0).squeeze(-1)
        known_rgb = rgb_in[exclude_positions]
        known_mask = mask[exclude_positions]
        known_sigma = out[positions, :, 3:].unsqueeze(0)
        
        avg_rgb_weight_weight = torch.full((unknown_len, known_len, n_samples, n_views), 1.0/known_len).to(known_rgb_weight.device)
        # print(avg_rgb_weight_weight.shape, known_rgb_weight.shape, known_sigma.shape)
        avg_rgb_weight = avg_rgb_weight_weight * known_rgb_weight
        avg_rgb_weight = avg_rgb_weight.sum(dim=1).unsqueeze(-1)
        rgb_out = torch.sum(known_rgb*avg_rgb_weight, dim=2)

        avg_sigma_weight = torch.full((unknown_len, known_len, n_samples, 1), 1.0/known_len).to(known_sigma.device)
        avg_sigma = avg_sigma_weight * known_sigma
        avg_sigma = avg_sigma.sum(dim=1)
        # 对于每个样本，对张量 b 进行加权求和
        for i in range(len(exclude_positions)):
            out[exclude_positions[i],:,:3] = rgb_out[i]
            out[exclude_positions[i],:,3:] = avg_sigma[i]
        return out
    
    def mof_func_distance(self, positions, rgb_feat, rgb_in, weight, out, mask, block_size):
        n_rays, n_samples, n_views, n_feat = rgb_feat.shape
        known_len = len(positions)
        unknown_len = block_size - known_len
        all_positions = list(range(block_size))
        exclude_positions = [pos for pos in all_positions if pos not in positions]
        
        known_rgb_weight = weight[positions].unsqueeze(0).squeeze(-1)
        known_rgb = rgb_in[exclude_positions]
        known_mask = mask[exclude_positions]
        known_sigma = out[positions, :, 3:].unsqueeze(0)
        
        # 计算5x5网格中的坐标
        def get_2d_position(pos):
            return (pos // 5, pos % 5)
        
        # 计算基于距离的权重
        distance_weights = torch.zeros((unknown_len, known_len)).to(known_sigma.device)
        for i, exclude_pos in enumerate(exclude_positions):
            exclude_y, exclude_x = get_2d_position(exclude_pos)
            for j, known_pos in enumerate(positions):
                known_y, known_x = get_2d_position(known_pos)
                # 计算欧几里得距离
                distance = ((exclude_y - known_y) ** 2 + (exclude_x - known_x) ** 2) ** 0.5
                # 使用距离的倒数作为权重，加上小的epsilon避免除零
                distance_weights[i, j] = 1.0 / (distance + 1e-6)
        
        # 归一化权重，使每行的权重和为1
        distance_weights = distance_weights / distance_weights.sum(dim=1, keepdim=True)
        
        avg_rgb_weight_weight = torch.full((unknown_len, known_len, n_samples, n_views), 1.0/known_len).to(known_rgb_weight.device)
        # print(avg_rgb_weight_weight.shape, known_rgb_weight.shape, known_sigma.shape)
        avg_rgb_weight = avg_rgb_weight_weight * known_rgb_weight
        avg_rgb_weight = avg_rgb_weight.sum(dim=1).unsqueeze(-1)
        rgb_out = torch.sum(known_rgb*avg_rgb_weight, dim=2)

        # 使用基于距离的权重进行体密度加权平均
        avg_sigma_weight = distance_weights.unsqueeze(2).unsqueeze(3).expand(-1, -1, n_samples, 1)
        avg_sigma = avg_sigma_weight * known_sigma
        avg_sigma = avg_sigma.sum(dim=1)
        # 对于每个样本，对张量 b 进行加权求和
        for i in range(len(exclude_positions)):
            out[exclude_positions[i],:,:3] = rgb_out[i]
            out[exclude_positions[i],:,3:] = avg_sigma[i]
        return out
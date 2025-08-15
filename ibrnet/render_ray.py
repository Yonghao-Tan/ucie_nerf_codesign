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


import torch
from collections import OrderedDict
import numpy as np
from .pruning_utils import (
    apply_source_view_pruning, 
    apply_source_view_pruning_sparse_vectorized
)

########################################################################################################################
# helper functions for nerf ray rendering
########################################################################################################################


def sample_pdf(bins, weights, N_samples, det=False):
    '''
    :param bins: tensor of shape [N_rays, M+1], M is the number of bins
    :param weights: tensor of shape [N_rays, M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [N_rays, N_samples]
    '''

    M = weights.shape[1]
    weights += 1e-5
    # Get pdf
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)    # [N_rays, M]
    cdf = torch.cumsum(pdf, dim=-1)  # [N_rays, M]
    cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf], dim=-1) # [N_rays, M+1]

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., N_samples, device=bins.device)
        u = u.unsqueeze(0).repeat(bins.shape[0], 1)       # [N_rays, N_samples]
    else:
        u = torch.rand(bins.shape[0], N_samples, device=bins.device)

    # Invert CDF
    above_inds = torch.zeros_like(u, dtype=torch.long)       # [N_rays, N_samples]
    for i in range(M):
        above_inds += (u >= cdf[:, i:i+1]).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds-1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=2)     # [N_rays, N_samples, 2]

    cdf = cdf.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    bins = bins.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    # t = (u-cdf_g[:, :, 0]) / (cdf_g[:, :, 1] - cdf_g[:, :, 0] + TINY_NUMBER)  # [N_rays, N_samples]
    # fix numeric issue
    denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]      # [N_rays, N_samples]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[:, :, 0]) / denom

    samples = bins_g[:, :, 0] + t * (bins_g[:, :, 1]-bins_g[:, :, 0])

    return samples


def sample_along_camera_ray(ray_o, ray_d, depth_range,
                            N_samples,
                            inv_uniform=False,
                            det=False):
    '''
    :param ray_o: origin of the ray in scene coordinate system; tensor of shape [N_rays, 3]
    :param ray_d: homogeneous ray direction vectors in scene coordinate system; tensor of shape [N_rays, 3]
    :param depth_range: [near_depth, far_depth]
    :param inv_uniform: if True, uniformly sampling inverse depth
    :param det: if True, will perform deterministic sampling
    :return: tensor of shape [N_rays, N_samples, 3]
    '''
    # will sample inside [near_depth, far_depth]
    # assume the nearest possible depth is at least (min_ratio * depth)
    near_depth_value = depth_range[0, 0]
    far_depth_value = depth_range[0, 1]
    assert near_depth_value > 0 and far_depth_value > 0 and far_depth_value > near_depth_value

    near_depth = near_depth_value * torch.ones_like(ray_d[..., 0])

    far_depth = far_depth_value * torch.ones_like(ray_d[..., 0])
    if inv_uniform:
        start = 1. / near_depth     # [N_rays,]
        step = (1. / far_depth - start) / (N_samples-1)
        inv_z_vals = torch.stack([start+i*step for i in range(N_samples)], dim=1)  # [N_rays, N_samples]
        z_vals = 1. / inv_z_vals
    else:
        start = near_depth
        step = (far_depth - near_depth) / (N_samples-1)
        z_vals = torch.stack([start+i*step for i in range(N_samples)], dim=1)  # [N_rays, N_samples]
    
    if not det:
        # get intervals between samples
        mids = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, 0:1], mids], dim=-1)
        # uniform samples in those intervals
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand   # [N_rays, N_samples]
    
    ray_d = ray_d.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, 3]
    ray_o = ray_o.unsqueeze(1).repeat(1, N_samples, 1)
    pts = z_vals.unsqueeze(2) * ray_d + ray_o       # [N_rays, N_samples, 3]
    return pts, z_vals


########################################################################################################################
# ray rendering of nerf
########################################################################################################################

def raw2outputs(raw, z_vals, mask, white_bkgd=False):
    '''
    :param raw: raw network output; tensor of shape [N_rays, N_samples, 4]
    :param z_vals: depth of point samples along rays; tensor of shape [N_rays, N_samples]
    :param ray_d: [N_rays, 3]
    :return: {'rgb': [N_rays, 3], 'depth': [N_rays,], 'weights': [N_rays,], 'depth_std': [N_rays,]}
    '''
    rgb = raw[:, :, :3]     # [N_rays, N_samples, 3]
    sigma = raw[:, :, 3]    # [N_rays, N_samples]

    # note: we did not use the intervals here, because in practice different scenes from COLMAP can have
    # very different scales, and using interval can affect the model's generalization ability.
    # Therefore we don't use the intervals for both training and evaluation.
    sigma2alpha = lambda sigma, dists: 1. - torch.exp(-sigma)

    # point samples are ordered with increasing depth
    # interval between samples
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat((dists, dists[:, -1:]), dim=-1)  # [N_rays, N_samples]

    alpha = sigma2alpha(sigma, dists)  # [N_rays, N_samples]

    # Eq. (3): T
    T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[:, :-1]   # [N_rays, N_samples-1]
    T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)  # [N_rays, N_samples]

    # maths show weights, and summation of weights along a ray, are always inside [0, 1]
    weights = alpha * T     # [N_rays, N_samples]
    rgb_map = torch.sum(weights.unsqueeze(2) * rgb, dim=1)  # [N_rays, 3]

    if white_bkgd:
        rgb_map = rgb_map + (1. - torch.sum(weights, dim=-1, keepdim=True))

    mask = mask.float().sum(dim=1) > 8  # should at least have 8 valid observation on the ray, otherwise don't consider its loss
    depth_map = torch.sum(weights * z_vals, dim=-1)     # [N_rays,]

    ret = OrderedDict([('rgb', rgb_map),
                       ('depth', depth_map),
                       ('weights', weights),                # used for importance sampling of fine samples
                       ('mask', mask),
                       ('alpha', alpha),
                       ('z_vals', z_vals)
                       ])

    return ret

def share_center_to_block(tensor, window_size=5):
    """
    将每个5x5 block的中心位置[2, 2]的n_samples值分享给该block的其他24个位置。

    参数:
        tensor (torch.Tensor): 输入张量，形状为 [rays, n_samples]。
        h (int): 图像的高度。
        w (int): 图像的宽度。

    返回:
        torch.Tensor: 处理后的张量，形状仍为 [rays, n_samples]。
    """
    # 确保输入张量形状正确
    h, w = tensor.shape[:2]
    assert tensor.dim() == 3, "输入张量必须是三维的，形状为 [rays_h, rays_w, n_samples]"
    assert h * w == tensor.size(0) * tensor.size(1), "rays 必须等于 h * w"
    assert h % window_size == 0 and w % window_size == 0, "h 和 w 必须能被5整除"

    # 创建一个空的输出张量
    output = torch.zeros_like(tensor)
    # 遍历每个5x5的block
    for i in range(0, h, window_size):  # 按高度方向步进5
        for j in range(0, w, window_size):  # 按宽度方向步进5
            # 提取当前block的中心位置 [2, 2]
            center_value = tensor[i + 2, j + 2, :]  # 形状为 [n_samples]
            # 将中心值广播给当前block的所有位置
            output[i:i+window_size, j:j+window_size, :] = center_value.unsqueeze(0).unsqueeze(0)  # 广播到 [5, 5, n_samples]
    return output

def render_rays(ray_batch,
                model,
                featmaps,
                projector,
                N_samples,
                inv_uniform=False,
                N_importance=0,
                det=False,
                white_bkgd=False):
    '''
    :param ray_batch: {'ray_o': [N_rays, 3] , 'ray_d': [N_rays, 3], 'view_dir': [N_rays, 2]}
    :param model:  {'net_coarse':  , 'net_fine': }
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :param det: if True, will deterministicly sample depths
    :return: {'outputs_coarse': {}, 'outputs_fine': {}}
    '''

    ret = {'outputs_coarse': None,
           'outputs_fine': None}

    # pts: [N_rays, N_samples, 3]
    # z_vals: [N_rays, N_samples]
    pts, z_vals = sample_along_camera_ray(ray_o=ray_batch['ray_o'],
                                          ray_d=ray_batch['ray_d'],
                                          depth_range=ray_batch['depth_range'],
                                          N_samples=N_samples, inv_uniform=inv_uniform, det=det)
    N_rays, N_samples = pts.shape[:2]
    
    rgb_feat, ray_diff, mask = projector.compute(pts, ray_batch['camera'],
                                                 ray_batch['src_rgbs'],
                                                 ray_batch['src_cameras'],
                                                 featmaps=featmaps[0])  # [N_rays, N_samples, N_views, x]
    pixel_mask = mask[..., 0].sum(dim=2) > 1   # [N_rays, N_samples], should at least have 2 observations
    if mask.shape[2] != 8:
        print('Skip sv pruning!')
        model.sv_prune = False
    if model.sv_prune:
        blending_weights_valid, raw_coarse, mask = model.net_coarse(rgb_feat, ray_diff, mask, return_sv_prune=True)   # [N_rays, N_samples, 4]
        mask_coarse = mask.clone()
    else:
        raw_coarse = model.net_coarse(rgb_feat, ray_diff, mask)   # [N_rays, N_samples, 4]
    outputs_coarse = raw2outputs(raw_coarse, z_vals, pixel_mask,
                                 white_bkgd=white_bkgd)
    ret['outputs_coarse'] = outputs_coarse

    if N_importance > 0:
        assert model.net_fine is not None
        # detach since we would like to decouple the coarse and fine networks
        weights = outputs_coarse['weights'].clone().detach()            # [N_rays, N_samples]
        if inv_uniform:
            inv_z_vals = 1. / z_vals
            inv_z_vals_mid = .5 * (inv_z_vals[:, 1:] + inv_z_vals[:, :-1])   # [N_rays, N_samples-1]
            weights = weights[:, 1:-1]      # [N_rays, N_samples-2]
            inv_z_vals = sample_pdf(bins=torch.flip(inv_z_vals_mid, dims=[1]),
                                    weights=torch.flip(weights, dims=[1]),
                                    N_samples=N_importance, det=det)  # [N_rays, N_importance]
            z_samples = 1. / inv_z_vals
        else:
            raise NotImplementedError("Only inverse uniform sampling is implemented for now.")

        z_vals_coarse = z_vals.clone()
        # print(z_vals_coarse.shape, mask.shape)
        # visualize_depth_samples(z_vals_coarse[0], z_samples[0], save_path='test_dist.png')
        # exit()
        z_vals = torch.cat((z_vals, z_samples), dim=-1)  # [N_rays, N_samples + N_importance]
        z_vals, _ = torch.sort(z_vals, dim=-1)
        H, W = ray_batch['H'], ray_batch['W']
        assert z_vals.shape[0] >= W
        H = z_vals.shape[0] // W
        window_size = 5
        if model.sample_point_sparsity: 
            H_exclude, W_exclude = H % window_size, W % window_size
            z_vals_2d = z_vals.reshape(H, W, -1)
            if H_exclude > 0 and W_exclude > 0:
                z_vals_slice = z_vals_2d[:-H_exclude, :-W_exclude, :]
                # print('a', z_vals_slice.shape)
                z_vals_slice = share_center_to_block(z_vals_slice, window_size=window_size)
                z_vals_2d[:H-H_exclude, :W-W_exclude, :] = z_vals_slice
            elif H_exclude > 0:
                z_vals_slice = z_vals_2d[:-H_exclude, :, :]
                # print('b', z_vals_slice.shape)
                z_vals_slice = share_center_to_block(z_vals_slice, window_size=window_size)
                z_vals_2d[:H-H_exclude, :, :] = z_vals_slice
            elif W_exclude > 0:
                z_vals_slice = z_vals_2d[:, :-W_exclude, :]
                # print('c', z_vals_slice.shape)
                z_vals_slice = share_center_to_block(z_vals_slice, window_size=window_size)
                z_vals_2d[:, :W-W_exclude, :] = z_vals_slice
            else:
                # print('q', H, W, z_vals.shape[:1])
                z_vals_2d = share_center_to_block(z_vals_2d)
            z_vals = z_vals_2d.reshape(-1, z_vals_2d.shape[-1])  # [N_rays, N_samples + N_importance]

        small_variance_indices = None

        # 输出方差较小的20%索引
        N_total_samples = N_samples + N_importance

        viewdirs = ray_batch['ray_d'].unsqueeze(1).repeat(1, N_total_samples, 1)
        ray_o = ray_batch['ray_o'].unsqueeze(1).repeat(1, N_total_samples, 1)
        pts = z_vals.unsqueeze(2) * viewdirs + ray_o  # [N_rays, N_samples + N_importance, 3]

        rgb_feat_sampled, ray_diff, mask = projector.compute(pts, ray_batch['camera'],
                                                             ray_batch['src_rgbs'],
                                                             ray_batch['src_cameras'],
                                                             featmaps=featmaps[1])

        pixel_mask = mask[..., 0].sum(dim=2) > 1  # [N_rays, N_samples]. should at least have 2 observations
        
        # Apply source view pruning if we have coarse stage weights and mask
        if model.sv_prune and 'blending_weights_valid' in locals() and 'mask_coarse' in locals():
            # try:
                # Choose pruning method based on sample_point_sparsity setting
                if hasattr(model, 'sample_point_sparsity') and model.sample_point_sparsity:
                    # Use sparse pruning for sample_point_sparsity mode
                    # Follow the same edge case handling as z_vals processing
                    H_exclude, W_exclude = H % window_size, W % window_size
                    
                    # Reshape all data to 2D format like z_vals processing
                    z_vals_coarse_2d = z_vals_coarse.reshape(H, W, -1)
                    z_samples_2d = z_samples.reshape(H, W, -1)
                    
                    # Get shape info for proper reshaping
                    _, N_coarse_samples, N_views, _ = blending_weights_valid.shape
                    _, N_coarse_samples2, N_views2, _ = mask_coarse.shape  
                    _, N_total_samples, N_views3, _ = mask.shape
                    
                    blending_weights_2d = blending_weights_valid.reshape(H, W, N_coarse_samples, N_views, 1)
                    mask_coarse_2d = mask_coarse.reshape(H, W, N_coarse_samples2, N_views2, 1)
                    mask_2d = mask.reshape(H, W, N_total_samples, N_views3, 1)
                    
                    if H_exclude > 0 and W_exclude > 0:
                        # Case a: Both H and W have remainders
                        effective_H, effective_W = H - H_exclude, W - W_exclude
                        z_vals_coarse_slice = z_vals_coarse_2d[:-H_exclude, :-W_exclude, :].reshape(-1, z_vals_coarse_2d.shape[-1])
                        z_samples_slice = z_samples_2d[:-H_exclude, :-W_exclude, :].reshape(-1, z_samples_2d.shape[-1])
                        weights_slice = blending_weights_2d[:-H_exclude, :-W_exclude, :, :, :].reshape(-1, N_coarse_samples, N_views, 1)
                        mask_coarse_slice = mask_coarse_2d[:-H_exclude, :-W_exclude, :, :, :].reshape(-1, N_coarse_samples2, N_views2, 1)
                        mask_slice = mask_2d[:-H_exclude, :-W_exclude, :, :, :].reshape(-1, N_total_samples, N_views3, 1)
                        
                        mask_pruned = apply_source_view_pruning_sparse_vectorized(
                            z_vals_coarse_slice, z_samples_slice, weights_slice, mask_coarse_slice, mask_slice,
                            effective_H, effective_W, window_size=window_size, top_k=model.sv_top_k
                        )
                        mask_2d[:-H_exclude, :-W_exclude, :, :, :] = mask_pruned.reshape(effective_H, effective_W, N_total_samples, N_views3, 1)
                        print(f"Applied sparse pruning (case a): {effective_H}x{effective_W}")
                        
                    elif H_exclude > 0:
                        # Case b: Only H has remainder
                        effective_H = H - H_exclude
                        z_vals_coarse_slice = z_vals_coarse_2d[:-H_exclude, :, :].reshape(-1, z_vals_coarse_2d.shape[-1])
                        z_samples_slice = z_samples_2d[:-H_exclude, :, :].reshape(-1, z_samples_2d.shape[-1])
                        weights_slice = blending_weights_2d[:-H_exclude, :, :, :, :].reshape(-1, N_coarse_samples, N_views, 1)
                        mask_coarse_slice = mask_coarse_2d[:-H_exclude, :, :, :, :].reshape(-1, N_coarse_samples2, N_views2, 1)
                        mask_slice = mask_2d[:-H_exclude, :, :, :, :].reshape(-1, N_total_samples, N_views3, 1)
                        
                        mask_pruned = apply_source_view_pruning_sparse_vectorized(
                            z_vals_coarse_slice, z_samples_slice, weights_slice, mask_coarse_slice, mask_slice,
                            effective_H, W, window_size=window_size, top_k=model.sv_top_k
                        )
                        mask_2d[:-H_exclude, :, :, :, :] = mask_pruned.reshape(effective_H, W, N_total_samples, N_views3, 1)
                        print(f"Applied sparse pruning (case b): {effective_H}x{W}")
                        
                    elif W_exclude > 0:
                        # Case c: Only W has remainder
                        effective_W = W - W_exclude
                        z_vals_coarse_slice = z_vals_coarse_2d[:, :-W_exclude, :].reshape(-1, z_vals_coarse_2d.shape[-1])
                        z_samples_slice = z_samples_2d[:, :-W_exclude, :].reshape(-1, z_samples_2d.shape[-1])
                        weights_slice = blending_weights_2d[:, :-W_exclude, :, :, :].reshape(-1, N_coarse_samples, N_views, 1)
                        mask_coarse_slice = mask_coarse_2d[:, :-W_exclude, :, :, :].reshape(-1, N_coarse_samples2, N_views2, 1)
                        mask_slice = mask_2d[:, :-W_exclude, :, :, :].reshape(-1, N_total_samples, N_views3, 1)
                        
                        mask_pruned = apply_source_view_pruning_sparse_vectorized(
                            z_vals_coarse_slice, z_samples_slice, weights_slice, mask_coarse_slice, mask_slice,
                            H, effective_W, window_size=window_size, top_k=model.sv_top_k
                        )
                        mask_2d[:, :-W_exclude, :, :, :] = mask_pruned.reshape(H, effective_W, N_total_samples, N_views3, 1)
                        print(f"Applied sparse pruning (case c): {H}x{effective_W}")
                        
                    else:
                        # Case d: No remainders - full processing
                        mask = apply_source_view_pruning_sparse_vectorized(
                            z_vals_coarse, z_samples, blending_weights_valid, mask_coarse, mask, 
                            H, W, window_size=window_size, top_k=model.sv_top_k
                        )
                        print(f"Applied sparse pruning (case d): {H}x{W}")
                        
                    # Reshape back to 1D if needed
                    if H_exclude > 0 or W_exclude > 0:
                        mask = mask_2d.reshape(-1, N_total_samples, N_views3, 1)
                        
                else:
                    # Use standard pruning for non-sparse mode
                    mask = apply_source_view_pruning(
                        z_vals_coarse, z_samples, blending_weights_valid, mask_coarse, mask, top_k=model.sv_top_k
                    )
                    print(f"Applied standard source view pruning: mask shape {mask.shape}")
            # except Exception as e:
            #     print(f"Source view pruning failed: {e}, using original mask")
        
        if model.use_moe and H % 5 == 0:
            rgb_feat, rgb_in, blending_weights_valid, raw_fine_org, mask = model.net_fine(rgb_feat_sampled, ray_diff, mask, return_moe=True)
            raw_fine, mof_l2_loss = model.moe(
                H, W,
                rgb_feat,
                rgb_in,
                blending_weights_valid,
                raw_fine_org,
                mask,
                use_moe_block=True
            )
            ret['mof_l2_loss'] = mof_l2_loss
        else:
            raw_fine = model.net_fine(rgb_feat_sampled, ray_diff, mask, return_moe=False)
            
        outputs_fine = raw2outputs(raw_fine, z_vals, pixel_mask,
                                   white_bkgd=white_bkgd)
        ret['outputs_fine'] = outputs_fine
        if small_variance_indices is not None:
            ret['outputs_fine']['rgb'][small_variance_indices] = ret['outputs_coarse']['rgb'][small_variance_indices]

    return ret

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

import numpy as np
import torch
import torch.nn.functional as F


rng = np.random.RandomState(234)

########################################################################################################################
# ray batch sampling
########################################################################################################################


def parse_camera(params):
    H = params[:, 0]
    W = params[:, 1]
    intrinsics = params[:, 2:18].reshape((-1, 4, 4))
    c2w = params[:, 18:34].reshape((-1, 4, 4))
    return W, H, intrinsics, c2w


def dilate_img(img, kernel_size=20):
    import cv2
    assert img.dtype == np.uint8
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilation = cv2.dilate(img / 255, kernel, iterations=1) * 255
    return dilation


class RaySamplerSingleImage(object):
    def __init__(self, data, device, resize_factor=1, render_stride=1, sr=False, use_moe=False):
        super().__init__()
        self.render_stride = render_stride
        self.rgb = data['rgb'] if 'rgb' in data.keys() else None
        # self.rgb_lr = data['rgb_lr'] if 'rgb_lr' in data.keys() else None
        self.rgb_hr = data['rgb'] if 'rgb' in data.keys() else None # sr备用
        self.camera = data['camera']
        self.rgb_path = data['rgb_path']
        self.depth_range = data['depth_range']
        self.device = device
        W, H, self.intrinsics, self.c2w_mat = parse_camera(self.camera)
        self.batch_size = len(self.camera)

        self.H = int(H[0])
        self.W = int(W[0])
        self.sr = sr
        self.use_moe = use_moe
        
        # half-resolution output
        if resize_factor != 1:
            self.W = int(self.W * resize_factor)
            self.H = int(self.H * resize_factor)
            self.intrinsics[:, :2, :3] *= resize_factor
            if self.rgb is not None:
                self.rgb = F.interpolate(self.rgb.permute(0, 3, 1, 2), scale_factor=resize_factor).permute(0, 2, 3, 1) # TODO 这个不用bilinear/bicubic吗, 默认是nearest

        self.rays_o, self.rays_d = self.get_rays_single_image(self.H, self.W, self.intrinsics, self.c2w_mat)
        if self.rgb is not None:
            self.rgb = self.rgb.reshape(-1, 3)
        if self.rgb_hr is not None:
            self.rgb_hr = self.rgb_hr.reshape(-1, 3)

        if 'src_rgbs' in data.keys():
            self.src_rgbs = data['src_rgbs']
        else:
            self.src_rgbs = None
        if 'src_cameras' in data.keys():
            self.src_cameras = data['src_cameras']
        else:
            self.src_cameras = None

    def get_rays_single_image(self, H, W, intrinsics, c2w):
        '''
        :param H: image height
        :param W: image width
        :param intrinsics: 4 by 4 intrinsic matrix
        :param c2w: 4 by 4 camera to world extrinsic matrix
        :return:
        '''
        u, v = np.meshgrid(np.arange(W)[::self.render_stride], np.arange(H)[::self.render_stride])
        u = u.reshape(-1).astype(dtype=np.float32)  # + 0.5    # add half pixel
        v = v.reshape(-1).astype(dtype=np.float32)  # + 0.5
        pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)
        pixels = torch.from_numpy(pixels)
        batched_pixels = pixels.unsqueeze(0).repeat(self.batch_size, 1, 1)

        rays_d = (c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)).transpose(1, 2)
        rays_d = rays_d.reshape(-1, 3)
        rays_o = c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[0], 1).reshape(-1, 3)  # B x HW x 3
        return rays_o, rays_d

    def get_all(self):
        ret = {'ray_o': self.rays_o.cuda(),
               'ray_d': self.rays_d.cuda(),
               'depth_range': self.depth_range.cuda(),
               'camera': self.camera.cuda(),
               'rgb': self.rgb.cuda() if self.rgb is not None else None,
               'rgb_hr': self.rgb_hr.cuda() if self.rgb_hr is not None else None,
               'src_rgbs': self.src_rgbs.cuda() if self.src_rgbs is not None else None,
               'src_cameras': self.src_cameras.cuda() if self.src_cameras is not None else None,
        }
        return ret

    def sample_random_pixel(self, N_rand, sample_mode, center_ratio=0.8):
        if sample_mode == 'center':
            border_H = int(self.H * (1 - center_ratio) / 2.)
            border_W = int(self.W * (1 - center_ratio) / 2.)

            # pixel coordinates
            u, v = np.meshgrid(np.arange(border_H, self.H - border_H),
                               np.arange(border_W, self.W - border_W))
            u = u.reshape(-1)
            v = v.reshape(-1)

            select_inds = rng.choice(u.shape[0], size=(N_rand,), replace=False)
            select_inds = v[select_inds] + self.W * u[select_inds]

        elif sample_mode == 'uniform': # use
            # Random from one image
            select_inds = rng.choice(self.H*self.W, size=(N_rand,), replace=False)
        elif sample_mode == 'block_single':  # for super-resolution & moe
            if self.sr and self.use_moe:  # When sr (super-resolution) is True
                block_width = rng.randint(4, 8) * 5
                # Ensure block_width is even # TODO why?
                # if block_width % 2 != 0:
                #     block_width += 1
                select_inds_rgb = []  # Initialize a new list for rgb indices
                block_height = block_width
                rgb_block_width = block_width * 2  # For select_inds_rgb, use a block twice the size
                rgb_block_height = rgb_block_width
            elif self.sr:  # When sr (super-resolution) is True
                block_width = rng.randint(16, 33)  # Regular block width for non-sr mode
                # Ensure block_width is even
                if block_width % 2 != 0:
                    block_width += 1
                select_inds_rgb = []  # Initialize a new list for rgb indices
                block_height = block_width
                rgb_block_width = block_width * 2  # For select_inds_rgb, use a block twice the size
                rgb_block_height = rgb_block_width
            elif self.use_moe:
                # candidate_values = [20, 25, 30, 35, 40]
                # block_height = np.random.choice(candidate_values)
                block_height = 5
                block_width = rng.randint(10, 25) * 5
            else:
                block_height = rng.randint(16, 33)  # Regular block width for non-sr mode
                block_width = rng.randint(16, 33)  # Regular block width for non-sr mode
            self.H_real, self.W_real = block_height, block_width
            # Ensure the selected start positions leave space for the expanded block (for sr=True)
            if self.sr:
                max_row_start = self.H - rgb_block_height  # Ensure there's enough space for the larger block
                max_col_start = self.W - rgb_block_width  # Same for the width dimension
            else:
                max_row_start = self.H - block_height  # Normal size when sr=False
                max_col_start = self.W - block_width  # Same for the width dimension

            # Randomly select a start position, making sure it's within bounds for the original block
            select_row_start = rng.randint(block_height // 2 + 1, max_row_start)
            select_col_start = rng.randint(block_width // 2 + 1, max_col_start)

            # Generate select_inds for the original block (block_width x block_width)
            select_inds = []
            for i in range(block_height):
                for j in range(block_width):
                    select_inds.append((select_row_start + i) * self.W + (select_col_start + j))

            # If sr=True, generate select_inds_rgb for the larger block (rgb_block_width x rgb_block_width)
            if self.sr:
                select_inds_rgb = []
                # Use the original block's center (select_row_start + block_width//2, select_col_start + block_width//2)
                center_row = select_row_start * 2
                center_col = select_col_start * 2
                
                # Generate select_inds_rgb (expand the block to rgb_block_width)
                for i in range(rgb_block_height):
                    for j in range(rgb_block_width):
                        select_inds_rgb.append((center_row + i) * self.W * 2 + (center_col + j))
                return select_inds, select_inds_rgb
            # Now, select_inds contains the original block, and select_inds_rgb contains the expanded block
            return select_inds
        elif sample_mode == 'block_random':
            block_num = 20
            block_width = 5
            select_inds = []
            block_cnt = 0
            select_inds_start = -100
            while (block_cnt < block_num):
                while (select_inds_start in select_inds or select_inds_start <= 0 or select_inds_start % self.W >= self.W - block_width or select_inds_start // self.W >= self.H - block_width):
                    select_inds_start = rng.choice(self.H*self.W, size=(1,), replace=False)
                    select_inds_start = select_inds_start[0]
                for i in range(block_width):
                    for j in range(block_width):
                        select_inds.append(select_inds_start + i * self.W + j)
                block_cnt += 1
        else:
            raise Exception("unknown sample mode!")

        return select_inds

    def random_sample(self, N_rand, sample_mode, center_ratio=0.8):
        '''
        :param N_rand: number of rays to be casted
        :return:
        '''
        if not self.sr:
            select_inds = self.sample_random_pixel(N_rand, sample_mode, center_ratio)
        else:
            select_inds, select_inds_rgb = self.sample_random_pixel(N_rand, sample_mode, center_ratio)

        rays_o = self.rays_o[select_inds]
        rays_d = self.rays_d[select_inds]

        if self.rgb is not None:
            if not self.sr: 
                rgb = self.rgb[select_inds]
                rgb_hr = None
            else: 
                rgb = self.rgb[select_inds]
                rgb_hr = self.rgb_hr[select_inds_rgb]
                # print(select_inds)
                # print(select_inds_rgb)
                # exit()
        else:
            rgb = None

        ret = {'ray_o': rays_o.cuda(),
               'ray_d': rays_d.cuda(),
               'camera': self.camera.cuda(),
               'depth_range': self.depth_range.cuda(),
               'rgb': rgb.cuda() if rgb is not None else None,
               'rgb_hr': rgb_hr.cuda() if rgb_hr is not None else None,
               'src_rgbs': self.src_rgbs.cuda() if self.src_rgbs is not None else None,
               'src_cameras': self.src_cameras.cuda() if self.src_cameras is not None else None,
               'selected_inds': select_inds
        }
        return ret

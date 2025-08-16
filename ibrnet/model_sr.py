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
import os
from ibrnet.mlp_network import IBRNet
from ibrnet.feature_network import ResUNet
from ibrnet.Omni_SR.components.OmniSR import OmniSR
from ibrnet.moe_network import MOE

def de_parallel(model):
    return model.module if hasattr(model, 'module') else model

########################################################################################################################
# creation/saving/loading of nerf
########################################################################################################################


class IBRNetModel(object):
    def __init__(self, args, load_opt=True, load_scheduler=True, load_psnr=True):
        self.args = args
        self.sr = args.sr
        device = torch.device('cuda:{}'.format(args.local_rank))
        # create coarse IBRNet
        self.net_coarse = IBRNet(args,
                                 in_feat_ch=self.args.coarse_feat_dim,
                                 n_samples=self.args.N_samples).to(device)
        if args.coarse_only:
            self.net_fine = None
        else:
            # create coarse IBRNet
            self.net_fine = IBRNet(args,
                                   in_feat_ch=self.args.fine_feat_dim,
                                   n_samples=self.args.N_samples+self.args.N_importance,
                                   use_moe=args.use_moe).to(device)
        self.sample_point_sparsity = args.sample_point_sparsity
        self.use_moe = args.use_moe
        self.sv_prune = args.sv_prune
        self.sv_top_k = args.sv_top_k
        self.sample_point_group_size = args.sample_point_group_size
        if self.use_moe:
            self.moe = MOE().cuda()

        # create feature extraction network
        self.feature_net = ResUNet(coarse_out_ch=self.args.coarse_feat_dim,
                                   fine_out_ch=self.args.fine_feat_dim,
                                   coarse_only=self.args.coarse_only).to(device)
        
        # 3 3 64 5 2 True
        if self.sr:
            kwargs = {'upsampling': 2, 'kernel_size': 17, 'res_num': 5, 'block_num': 1, 'bias': True, 'block_script_name': 'OSA', 'block_class_name': 'OSA_Block', 'window_size': 8, 'pe': True, 'ffn_bias': True}
            self.sr_net = OmniSR(3, 3, 64, **kwargs).to(device)
            # total_params = sum(p.numel() for p in self.sr_net.parameters())
            # print(f"Total Parameters: {total_params/(1024*1024):.2f}M")
            # from thop import profile
            # input_tensor = torch.randn(1, 3, 16, 16).to(device)
            # # 计算 FLOPs 和参数量
            # flops, params = profile(self.sr_net, inputs=(input_tensor,))
            # print(f"FLOPs: {flops * 2 / 1e12}TFLOPs")
            # print(f"Parameters: {params / 1e6}M")
            # input_tensor = torch.randn(1, 3, 400, 400).to(device)
            # # 计算 FLOPs 和参数量
            # flops, params = profile(self.sr_net, inputs=(input_tensor,))
            # print(f"FLOPs: {flops * 2 / 1e12}TFLOPs")
            # print(f"Parameters: {params / 1e6}M")

        # optimizer and learning rate scheduler
        learnable_params = list(self.net_coarse.parameters())
        learnable_params += list(self.feature_net.parameters())
        if self.net_fine is not None:
            learnable_params += list(self.net_fine.parameters())
        if self.sr:
            learnable_params += list(self.sr_net.parameters())

        if self.net_fine is not None:
            if not self.sr:
                self.optimizer = torch.optim.Adam([
                    {'params': self.net_coarse.parameters()},
                    {'params': self.net_fine.parameters()},
                    {'params': self.feature_net.parameters(), 'lr': args.lrate_feature}],
                    lr=args.lrate_mlp)
            else:
                self.optimizer = torch.optim.Adam([
                    {'params': self.net_coarse.parameters()},
                    {'params': self.net_fine.parameters()},
                    {'params': self.feature_net.parameters(), 'lr': args.lrate_feature},
                    {'params': self.sr_net.parameters(), 'lr': args.lrate_sr}],
                    lr=args.lrate_mlp)
        else:
            self.optimizer = torch.optim.Adam([
                {'params': self.net_coarse.parameters()},
                {'params': self.feature_net.parameters(), 'lr': args.lrate_feature}],
                lr=args.lrate_mlp)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=args.lrate_decay_steps,
                                                         gamma=args.lrate_decay_factor)

        out_folder = os.path.join(args.rootdir, 'out', args.expname)
        self.start_step = self.load_from_ckpt(out_folder,
                                              load_opt=load_opt,
                                              load_scheduler=load_scheduler,
                                              load_psnr=load_psnr)

        if args.distributed:
            self.net_coarse = torch.nn.parallel.DistributedDataParallel(
                self.net_coarse,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )

            self.feature_net = torch.nn.parallel.DistributedDataParallel(
                self.feature_net,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )

            if self.net_fine is not None:
                self.net_fine = torch.nn.parallel.DistributedDataParallel(
                    self.net_fine,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank
                )
            
            if self.sr:
                self.sr_net = torch.nn.parallel.DistributedDataParallel(
                    self.sr_net,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank
                )

    def switch_to_eval(self):
        self.net_coarse.eval()
        self.feature_net.eval()
        if self.net_fine is not None:
            self.net_fine.eval()
        if self.sr:
            self.sr_net.eval()
        if self.use_moe: self.moe.eval()

    def switch_to_train(self):
        self.net_coarse.train()
        self.feature_net.train()
        if self.net_fine is not None:
            self.net_fine.train()
        if self.sr:
            self.sr_net.train()
        if self.use_moe: self.moe.train()

    def save_model(self, filename, psnr=None):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(),
                   'net_coarse': de_parallel(self.net_coarse).state_dict(),
                   'feature_net': de_parallel(self.feature_net).state_dict()
                   }

        if self.net_fine is not None:
            to_save['net_fine'] = de_parallel(self.net_fine).state_dict()
        
        if self.sr:
            to_save['sr_net'] = de_parallel(self.sr_net).state_dict()
        if psnr is not None:
            to_save['psnr'] = psnr

        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True, load_psnr=True):
        if self.args.distributed:
            to_load = torch.load(filename, map_location='cuda:{}'.format(self.args.local_rank))
        else:
            to_load = torch.load(filename)

        if load_opt:
            self.optimizer.load_state_dict(to_load['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load['scheduler'])

        self.net_coarse.load_state_dict(to_load['net_coarse'])
        self.feature_net.load_state_dict(to_load['feature_net'])

        if self.net_fine is not None and 'net_fine' in to_load.keys():
            self.net_fine.load_state_dict(to_load['net_fine'])

        if self.sr:
            self.sr_net.load_state_dict(to_load['sr_net'], strict=False)
        if load_psnr:
            self.psnr = to_load.get('psnr', None)  # 如果 'psnr' 不存在，返回 None
        else:
            self.psnr = None

    def load_from_ckpt(self, out_folder,
                       load_opt=True,
                       load_scheduler=True,
                       force_latest_ckpt=False,
                       load_psnr=True):
        '''
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        '''

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f)
                     for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler, load_psnr)
            # step = int(fpath[-10:-4])
            # print('Reloading from {}, starting at step={}'.format(fpath, step)) # TODO
            print('Reloading from {}'.format(fpath))
            step = 0
        else:
            print('No ckpts found, training from scratch...')
            step = 0

        return step



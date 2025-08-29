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
import imageio

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from ibrnet.data_loaders import dataset_dict
from ibrnet.quant_lsq import replace_linear_with_quantized
from ibrnet.render_ray import render_rays
from ibrnet.render_image import render_single_image
from ibrnet.model_sr import IBRNetModel
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.criterion import Criterion
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, cycle, img2psnr, save_current_code
import config
import torch.distributed as dist
from ibrnet.projection import Projector
from ibrnet.data_loaders.create_training_dataset import create_training_dataset


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def save_img(img_in, name, op=False):
    fine_pred_rgb = img_in
    patch_size = np.sqrt(fine_pred_rgb.shape[0]).astype(np.int)
    fine_pred_rgb = fine_pred_rgb.reshape(patch_size, patch_size, 3).detach().cpu()
    if op:
        fine_pred_rgb = torch.nn.functional.interpolate(fine_pred_rgb.permute(2, 0, 1).unsqueeze(0), scale_factor=0.5)[0].permute(1, 2, 0)
    coarse_pred_rgb = (255 * np.clip(fine_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
    imageio.imwrite(name, coarse_pred_rgb)
    
def train(args):

    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, 'pretrained', args.expname)
    # 在程序初始化阶段清空 eval.log 文件
    log_file_path = os.path.join(out_folder, 'eval.log')
    if args.local_rank == 0:  # 只在 Rank 0 上操作
        if os.path.exists(log_file_path):  # 检查文件是否存在
            with open(log_file_path, 'w') as log_file:
                log_file.write("")  # 清空文件内容
    print('outputs will be saved to {}'.format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    # save the args and config files
    f = os.path.join(out_folder, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    if args.config is not None:
        f = os.path.join(out_folder, 'config.txt')
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    # create training dataset
    train_dataset, train_sampler = create_training_dataset(args)
    # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
    # please use distributed parallel on multiple GPUs to train multiple target views per batch
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                               worker_init_fn=lambda _: np.random.seed(),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               shuffle=True if train_sampler is None else False)

    # create validation dataset
    val_dataset = dataset_dict[args.eval_dataset](args, 'validation',
                                                  scenes=args.eval_scenes)

    val_loader = DataLoader(val_dataset, batch_size=1)
    val_loader_iterator = iter(cycle(val_loader))

    # Create IBRNet model
    model = IBRNetModel(args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler, load_psnr=not args.no_load_psnr)
    # create projector
    projector = Projector(device=device)

    # Create criterion
    criterion = Criterion()
    tb_dir = os.path.join(args.rootdir, 'logs/', args.expname)
    if args.local_rank == 0:
        writer = SummaryWriter(tb_dir)
        print('saving tensorboard files to {}'.format(tb_dir))
    scalars_to_log = {}

    global_step = model.start_step + 1
    epoch = 0
    
    print("loaded psnr:", model.psnr)
    best_psnr = 0.0 if model.psnr is None else model.psnr
    while global_step < model.start_step + args.n_iters + 1:
        np.random.seed()
        for train_data in train_loader:
            time0 = time.time()
            if args.distributed:
                train_sampler.set_epoch(epoch)

            # Start of core optimization loop
            ray_sampler = RaySamplerSingleImage(train_data, args, device, resize_factor=args.resize_factor)
            N_rand = int(1.0 * args.N_rand * args.num_source_views / train_data['src_rgbs'][0].shape[0])
            ray_batch = ray_sampler.random_sample(N_rand,
                                                  sample_mode=args.sample_mode,
                                                  center_ratio=args.center_ratio
                                                  )
            H, W = ray_sampler.H, ray_sampler.W # TODO
            H, W = ray_sampler.H_real, ray_sampler.W_real # TODO
            ray_batch['H'] = H
            ray_batch['W'] = W

            featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
            
            ret = render_rays(ray_batch=ray_batch,
                              model=model,
                              projector=projector,
                              featmaps=featmaps,
                              N_samples=args.N_samples,
                              inv_uniform=args.inv_uniform,
                              N_importance=args.N_importance,
                              det=args.det,
                              white_bkgd=args.white_bkgd)

            # compute loss
            model.optimizer.zero_grad()
            loss, scalars_to_log = criterion(ret['outputs_coarse'], ray_batch, scalars_to_log)

            if ret['outputs_fine'] is not None: # 是coarse/fine/sr三个loss，还是说coarse/fine+sr两个loss
                fine_loss, scalars_to_log = criterion(ret['outputs_fine'], ray_batch, scalars_to_log) # TODO 这里不需要sr参数，因为这个fine经过了sr，他的分辨率就是跟rgb一样的
                loss += fine_loss
            if args.sr:
                sr_input = ret['outputs_fine']['rgb']
                patch_size = np.sqrt(sr_input.shape[0]).astype(np.int)
                sr_input = sr_input.reshape(patch_size, patch_size, 3).unsqueeze(0).permute(0, 3, 1, 2)
                sr_output = model.sr_net(sr_input)
                sr_output = sr_output.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
                ret['outputs_fine']['sr'] = sr_output
                fine_loss, scalars_to_log = criterion(sr_output, ray_batch, scalars_to_log, sr=True) # TODO 这里不需要sr参数，因为这个fine经过了sr，他的分辨率就是跟rgb一样的
                loss += fine_loss

            loss.backward()
            scalars_to_log['loss'] = loss.item()
            model.optimizer.step()
            model.scheduler.step()

            scalars_to_log['lr'] = model.scheduler.get_last_lr()[0]
            # end of core optimization loop
            dt = time.time() - time0

            
            if global_step % args.i_test == 0:
                print('Evaluating...')
                fine_psnr = eval(args, model, device)
                if args.local_rank == 0:  # 主进程执行评估

                    print('PSNR at {} with psnr {}, while best is {}'.format(global_step, fine_psnr, best_psnr))

                    # 记录日志
                    log_file_path = os.path.join(out_folder, 'eval.log')
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(f"Step: {global_step}, PSNR: {fine_psnr}, Best PSNR: {best_psnr}\n")

                    if fine_psnr > best_psnr:
                        best_psnr = fine_psnr
                        print('Saving best model at {} to {} with psnr {}'.format(global_step, out_folder, best_psnr))

                        # 更新日志
                        with open(log_file_path, 'a') as log_file:
                            log_file.write(f"New Best PSNR: {best_psnr} at Step: {global_step}\n")

                        # 保存模型
                        fpath = os.path.join(out_folder, 'model_best.pth')
                        model.save_model(fpath)
                        
            # Rest is logging
            if args.local_rank == 0:
                if global_step % args.i_print == 0 or global_step < 10:
                    # write mse and psnr stats
                    coarse_rgb_gt = ray_batch['rgb'] if args.sr else ray_batch['rgb'].squeeze(0)
                    mse_error = img2mse(ret['outputs_coarse']['rgb'], ray_batch['rgb']).item()
                    scalars_to_log['coarse-loss'] = mse_error
                    scalars_to_log['coarse-psnr'] = mse2psnr(mse_error)
                    if ret['outputs_fine'] is not None:
                        mse_error = img2mse(ret['outputs_fine']['rgb'], ray_batch['rgb']).item()
                        scalars_to_log['fine-loss'] = mse_error
                        scalars_to_log['fine-psnr'] = mse2psnr(mse_error)
                        
                    if args.sr:
                        save_img(ray_batch['rgb'], './test_image_gt.png')
                        mse_error = img2mse(ret['outputs_fine']['sr'], ray_batch['rgb_hr']).item()
                        scalars_to_log['sr-loss'] = mse_error
                        scalars_to_log['sr-psnr'] = mse2psnr(mse_error)
                        tmp_fine_rgb = ret['outputs_fine']['rgb']
                        patch_size = np.sqrt(tmp_fine_rgb.shape[0]).astype(np.int)
                        tmp_fine_rgb = tmp_fine_rgb.reshape(patch_size, patch_size, 3)
                        tmp_fine_rgb = torch.nn.functional.interpolate(tmp_fine_rgb.permute(2, 0, 1).unsqueeze(0), scale_factor=2.0)[0].permute(1, 2, 0).reshape(-1, 3)
                        mse_error = img2mse(tmp_fine_rgb, ray_batch['rgb_hr']).item()
                        # scalars_to_log['interpo-sr-loss'] = mse_error
                        scalars_to_log['interpo-sr-psnr'] = mse2psnr(mse_error)

                    logstr = '{} Epoch: {}  step: {} '.format(args.expname, epoch, global_step)
                    for k in scalars_to_log.keys():
                        logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                        writer.add_scalar(k, scalars_to_log[k], global_step)
                    print(logstr)
                    print('each iter time {:.05f} seconds'.format(dt))

                if global_step % args.i_weights == 0:
                    print('Saving checkpoints at {} to {}...'.format(global_step, out_folder))
                    fpath = os.path.join(out_folder, 'model_{:06d}.pth'.format(global_step))
                    model.save_model(fpath)

                if global_step % args.i_img == 0 and not args.sr:
                    print('Logging a random validation view...')
                    val_data = next(val_loader_iterator)
                    tmp_ray_sampler = RaySamplerSingleImage(val_data, args, device, render_stride=args.render_stride)
                    H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
                    gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
                    log_view_to_tb(writer, global_step, args, model, tmp_ray_sampler, projector,
                                   gt_img, render_stride=args.render_stride, prefix='val/')
                    torch.cuda.empty_cache()

                    print('Logging current training view...')
                    tmp_ray_train_sampler = RaySamplerSingleImage(train_data, args, device,
                                                                  render_stride=1)
                    H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
                    gt_img = tmp_ray_train_sampler.rgb.reshape(H, W, 3)
                    log_view_to_tb(writer, global_step, args, model, tmp_ray_train_sampler, projector,
                                   gt_img, render_stride=1, prefix='train/')
            global_step += 1
            if global_step > model.start_step + args.n_iters + 1:
                break
        epoch += 1


def log_view_to_tb(writer, global_step, args, model, ray_sampler, projector, gt_img,
                   render_stride=1, prefix=''):
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()
        if model.feature_net is not None:
            featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
        else:
            featmaps = [None, None]
        ret = render_single_image(ray_sampler=ray_sampler,
                                  ray_batch=ray_batch,
                                  model=model,
                                  projector=projector,
                                  chunk_size=args.chunk_size,
                                  N_samples=args.N_samples,
                                  inv_uniform=args.inv_uniform,
                                  det=True,
                                  N_importance=args.N_importance,
                                  white_bkgd=args.white_bkgd,
                                  render_stride=render_stride,
                                  featmaps=featmaps)

    average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))

    if args.render_stride != 1:
        gt_img = gt_img[::render_stride, ::render_stride]
        average_im = average_im[::render_stride, ::render_stride]

    rgb_gt = img_HWC2CHW(gt_img)
    average_im = img_HWC2CHW(average_im)

    rgb_pred = img_HWC2CHW(ret['outputs_coarse']['rgb'].detach().cpu())

    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
    rgb_im = torch.zeros(3, h_max, 3*w_max)
    rgb_im[:, :average_im.shape[-2], :average_im.shape[-1]] = average_im
    rgb_im[:, :rgb_gt.shape[-2], w_max:w_max+rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, :rgb_pred.shape[-2], 2*w_max:2*w_max+rgb_pred.shape[-1]] = rgb_pred

    depth_im = ret['outputs_coarse']['depth'].detach().cpu()
    acc_map = torch.sum(ret['outputs_coarse']['weights'], dim=-1).detach().cpu()

    if ret['outputs_fine'] is None:
        depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
        acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))
    else:
        rgb_fine = img_HWC2CHW(ret['outputs_fine']['rgb'].detach().cpu())
        rgb_fine_ = torch.zeros(3, h_max, w_max)
        rgb_fine_[:, :rgb_fine.shape[-2], :rgb_fine.shape[-1]] = rgb_fine
        rgb_im = torch.cat((rgb_im, rgb_fine_), dim=-1)
        depth_im = torch.cat((depth_im, ret['outputs_fine']['depth'].detach().cpu()), dim=-1)
        depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
        acc_map = torch.cat((acc_map, torch.sum(ret['outputs_fine']['weights'], dim=-1).detach().cpu()), dim=-1)
        acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))

    # write the pred/gt rgb images and depths
    writer.add_image(prefix + 'rgb_gt-coarse-fine', rgb_im, global_step)
    writer.add_image(prefix + 'depth_gt-coarse-fine', depth_im, global_step)
    writer.add_image(prefix + 'acc-coarse-fine', acc_map, global_step)

    # write scalar
    pred_rgb = ret['outputs_fine']['rgb'] if ret['outputs_fine'] is not None else ret['outputs_coarse']['rgb']
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    writer.add_scalar(prefix + 'psnr_image', psnr_curr_img, global_step)

    model.switch_to_train()

def eval(args, model, device):
    model.switch_to_eval()
    projector = Projector(device=device)
    scene_name = 'fern'
    test_dataset = dataset_dict['llff_test'](args, 'test', scenes=scene_name)
    test_loader = DataLoader(test_dataset, batch_size=1)

    total_fine_psnr = 0.
    cnt = 0
    for i, data in enumerate(test_loader):
        if i > 0: continue
        rgb_path = data['rgb_path'][0]
        file_id = os.path.basename(rgb_path).split('.')[0]
        # if file_id != 'image000': continue
        
        with torch.no_grad():
            ray_sampler = RaySamplerSingleImage(data, args, device, resize_factor=args.resize_factor)
            ray_batch = ray_sampler.get_all()
            featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
            
            H, W = ray_sampler.H, ray_sampler.W
            ray_batch['H'] = H
            ray_batch['W'] = W
            args.chunk_size = args.chunk_height * W

            ret = render_single_image(ray_sampler=ray_sampler,
                                      ray_batch=ray_batch,
                                      model=model,
                                      projector=projector,
                                      chunk_size=args.chunk_size,
                                      det=True,
                                      N_samples=args.N_samples,
                                      inv_uniform=args.inv_uniform,
                                      N_importance=args.N_importance,
                                      white_bkgd=args.white_bkgd,
                                      featmaps=featmaps)
            gt_rgb = data['rgb'][0]
            gt_rgb_np = gt_rgb.numpy()[None, ...]
            gt_rgb_hr = data['rgb'][0]
            gt_rgb_hr_np = gt_rgb_hr.numpy()[None, ...]

            if not args.sr:
                fine_pred_rgb = ret['outputs_fine']['rgb'].detach().cpu()
                fine_pred_rgb_np = np.clip(fine_pred_rgb.numpy()[None, ...], a_min=0., a_max=1.)
                mse = np.mean((gt_rgb_np - fine_pred_rgb_np) ** 2)
            else:
                sr_input = ret['outputs_fine']['rgb'].to(device) # H, W, C
                sr_input = sr_input.unsqueeze(0).permute(0, 3, 1, 2) # 1, C, H, W
                interpo_output = torch.nn.functional.interpolate(sr_input, scale_factor=2, mode='bicubic', align_corners=False).squeeze(0).permute(1, 2, 0)
                # sr_output = model.sr_net(sr_input)
                
                tile = args.window_size * 2
                tile_overlap = 0
                scale = 2
                b, c, h, w = sr_input.size()
                tile = min(tile, h, w)
                stride = tile - tile_overlap
                h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
                w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
                E = torch.zeros(b, c, h*scale, w*scale).type_as(sr_input)
                W = torch.zeros_like(E)

                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        in_patch = sr_input[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                        out_patch = model.sr_net(in_patch)
                        if isinstance(out_patch, list):
                            out_patch = out_patch[-1]
                        out_patch_mask = torch.ones_like(out_patch)

                        E[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch)
                        W[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch_mask)
                sr_output = E.div_(W)
                
                sr_output = sr_output.squeeze(0).permute(1, 2, 0)
                fine_pred_rgb = sr_output.detach().cpu()
                fine_pred_rgb_np = np.clip(fine_pred_rgb.numpy()[None, ...], a_min=0., a_max=1.)
                fine_interpo_rgb = interpo_output.detach().cpu()
                fine_interpo_rgb_np = np.clip(fine_interpo_rgb.numpy()[None, ...], a_min=0., a_max=1.)
                fine_pred_rgb = (255 * np.clip(fine_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                mse = np.mean((gt_rgb_hr_np - fine_pred_rgb_np) ** 2)
                
            if mse == 0:
                return float('inf')  # 如果两张图像完全一样，则 PSNR 是无限大
            max_pixel = 1.0
            fine_psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            # print(fine_psnr)
            total_fine_psnr += fine_psnr
            cnt += 1
    model.switch_to_train()
    return total_fine_psnr / cnt

if __name__ == '__main__':
    parser = config.config_parser()
    args = parser.parse_args()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    train(args)

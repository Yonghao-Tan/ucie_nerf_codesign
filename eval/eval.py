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


import sys
sys.path.append('../')
import imageio
from config import config_parser
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.render_image import render_single_image
from ibrnet.model_sr import IBRNetModel
from utils import *
from ibrnet.projection import Projector
from ibrnet.data_loaders import dataset_dict
import tensorflow as tf
from lpips_tensorflow import lpips_tf
from torch.utils.data import DataLoader

# os.environ["CUDA_VISIBLE_DEVICES"]="0"


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    args.distributed = False

    # Create IBRNet model
    model = IBRNetModel(args, load_scheduler=False, load_opt=False)
    eval_dataset_name = args.eval_dataset
    extra_out_dir = '{}/{}'.format(eval_dataset_name, args.expname)
    extra_out_dir = extra_out_dir + '_sr' if args.sr else extra_out_dir
    print("saving results to eval/{}...".format(extra_out_dir))
    os.makedirs(extra_out_dir, exist_ok=True)

    projector = Projector(device='cuda:0')

    assert len(args.eval_scenes) == 1, "only accept single scene"
    scene_name = args.eval_scenes[0]
    # out_scene_dir = os.path.join(extra_out_dir, '{}_{:06d}'.format(scene_name, model.start_step))
    out_scene_dir = os.path.join(extra_out_dir, f'{scene_name}')
    os.makedirs(out_scene_dir, exist_ok=True)

    test_dataset = dataset_dict[args.eval_dataset](args, 'test', scenes=args.eval_scenes, factor=4)
    # fern, f=1: 21.62, 21.68, 21.47
    # fern, f=4: 23.35, 23.25
    # fern, f=8: 24.40, 23.89
    
    # trex, f=1: 23.42, 23.47, 23.04
    save_prefix = scene_name
    test_loader = DataLoader(test_dataset, batch_size=1)
    total_num = len(test_loader)
    results_dict = {scene_name: {}}
    sum_coarse_psnr = 0
    sum_fine_psnr = 0
    sum_sr_psnr = 0
    sum_interpo_psnr = 0
    running_mean_coarse_psnr = 0
    running_mean_fine_psnr = 0
    running_mean_sr_psnr = 0
    running_mean_interpo_psnr = 0
    sum_coarse_lpips = 0
    sum_fine_lpips = 0
    running_mean_coarse_lpips = 0
    running_mean_fine_lpips = 0
    sum_coarse_ssim = 0
    sum_fine_ssim = 0
    running_mean_coarse_ssim = 0
    running_mean_fine_ssim = 0

    pred_ph = tf.placeholder(tf.float32)
    gt_ph = tf.placeholder(tf.float32)
    distance_t = lpips_tf.lpips(pred_ph, gt_ph, model='net-lin', net='vgg')
    ssim_tf = tf.image.ssim(pred_ph, gt_ph, max_val=1.)
    psnr_tf = tf.image.psnr(pred_ph, gt_ph, max_val=1.)

    for i, data in enumerate(test_loader):
        rgb_path = data['rgb_path'][0]
        # file_id = os.path.basename(rgb_path).split('.')[0] # TODO
        file_id = i
        src_rgbs = data['src_rgbs'][0].cpu().numpy()

        averaged_img = (np.mean(src_rgbs, axis=0) * 255.).astype(np.uint8)
        # imageio.imwrite(os.path.join(out_scene_dir, '{}_average.png'.format(file_id)), averaged_img)

        model.switch_to_eval()
        with torch.no_grad():
            ray_sampler = RaySamplerSingleImage(data, device='cuda:0', resize_factor=args.resize_factor, sr=args.sr)
            ray_batch = ray_sampler.get_all()
            featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
            H, W = ray_sampler.H, ray_sampler.W
            ray_batch['H'] = H
            ray_batch['W'] = W
            # TODO
            args.chunk_size = args.chunk_height * W
            assert args.chunk_height % args.window_size == 0
            # print(args.chunk_size, args.chunk_height, W)
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
            gt_rgb_hr = data['rgb'][0]
            gt_rgb_hr_np = gt_rgb_hr.numpy()[None, ...]
            gt_rgb = data['rgb'][0]
            if args.sr: gt_rgb = torch.nn.functional.interpolate(gt_rgb.unsqueeze(0).permute(0, 3, 1, 2), scale_factor=0.5).permute(0, 2, 3, 1)[0]
            coarse_pred_rgb = ret['outputs_coarse']['rgb'].detach().cpu()
            coarse_err_map = torch.sum((coarse_pred_rgb - gt_rgb) ** 2, dim=-1).numpy()
            coarse_err_map_colored = (colorize_np(coarse_err_map, range=(0., 1.)) * 255).astype(np.uint8)
            # imageio.imwrite(os.path.join(out_scene_dir, '{}_err_map_coarse.png'.format(file_id)), coarse_err_map_colored)
            coarse_pred_rgb_np = np.clip(coarse_pred_rgb.numpy()[None, ...], a_min=0., a_max=1.)
            gt_rgb_np = gt_rgb.numpy()[None, ...]

            # different implementation of the ssim and psnr metrics can be different.
            # we use the tf implementation for evaluating ssim and psnr to match the setup of NeRF paper.
            with tf.Session() as session:
                coarse_lpips = session.run(distance_t, feed_dict={pred_ph: coarse_pred_rgb_np, gt_ph: gt_rgb_np})[0]
                coarse_ssim = session.run(ssim_tf, feed_dict={pred_ph: coarse_pred_rgb_np, gt_ph: gt_rgb_np})[0]
                coarse_psnr = session.run(psnr_tf, feed_dict={pred_ph: coarse_pred_rgb_np, gt_ph: gt_rgb_np})[0]

            # saving outputs ...
            coarse_pred_rgb = (255 * np.clip(coarse_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_pred_coarse.png'.format(file_id)), coarse_pred_rgb)

            gt_rgb_np_uint8 = (255 * np.clip(gt_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_gt_rgb.png'.format(file_id)), gt_rgb_np_uint8)

            coarse_pred_depth = ret['outputs_coarse']['depth'].detach().cpu()
            # imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_coarse.png'.format(file_id)), (coarse_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
            coarse_pred_depth_colored = colorize_np(coarse_pred_depth,
                                                    range=tuple(data['depth_range'].squeeze().cpu().numpy()))
            # imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_vis_coarse.png'.format(file_id)), (255 * coarse_pred_depth_colored).astype(np.uint8))
            coarse_acc_map = torch.sum(ret['outputs_coarse']['weights'], dim=-1).detach().cpu()
            coarse_acc_map_colored = (colorize_np(coarse_acc_map, range=(0., 1.)) * 255).astype(np.uint8)
            # imageio.imwrite(os.path.join(out_scene_dir, '{}_acc_map_coarse.png'.format(file_id)), coarse_acc_map_colored)

            sum_coarse_psnr += coarse_psnr
            running_mean_coarse_psnr = sum_coarse_psnr / (i + 1)
            sum_coarse_lpips += coarse_lpips
            running_mean_coarse_lpips = sum_coarse_lpips / (i + 1)
            sum_coarse_ssim += coarse_ssim
            running_mean_coarse_ssim = sum_coarse_ssim / (i + 1)

            if ret['outputs_fine'] is not None:
                fine_pred_rgb = ret['outputs_fine']['rgb'].detach().cpu()
                fine_pred_rgb_np = np.clip(fine_pred_rgb.numpy()[None, ...], a_min=0., a_max=1.)

                with tf.Session() as session:
                    fine_lpips = session.run(distance_t, feed_dict={pred_ph: fine_pred_rgb_np, gt_ph: gt_rgb_np})[0]
                    fine_ssim = session.run(ssim_tf, feed_dict={pred_ph: fine_pred_rgb_np, gt_ph: gt_rgb_np})[0]
                    fine_psnr = session.run(psnr_tf, feed_dict={pred_ph: fine_pred_rgb_np, gt_ph: gt_rgb_np})[0]

                fine_err_map = torch.sum((fine_pred_rgb - gt_rgb) ** 2, dim=-1).numpy()
                fine_err_map_colored = (colorize_np(fine_err_map, range=(0., 1.)) * 255).astype(np.uint8)
                # imageio.imwrite(os.path.join(out_scene_dir, '{}_err_map_fine.png'.format(file_id)), fine_err_map_colored)

                fine_pred_rgb = (255 * np.clip(fine_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                imageio.imwrite(os.path.join(out_scene_dir, '{}_pred_fine.png'.format(file_id)), fine_pred_rgb)
                fine_pred_depth = ret['outputs_fine']['depth'].detach().cpu()
                # imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_fine.png'.format(file_id)), (fine_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
                fine_pred_depth_colored = colorize_np(fine_pred_depth,
                                                      range=tuple(data['depth_range'].squeeze().cpu().numpy()))
                # imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_vis_fine.png'.format(file_id)), (255 * fine_pred_depth_colored).astype(np.uint8))
                fine_acc_map = torch.sum(ret['outputs_fine']['weights'], dim=-1).detach().cpu()
                fine_acc_map_colored = (colorize_np(fine_acc_map, range=(0., 1.)) * 255).astype(np.uint8)
                # imageio.imwrite(os.path.join(out_scene_dir, '{}_acc_map_fine.png'.format(file_id)), fine_acc_map_colored)
            else:
                fine_ssim = fine_lpips = fine_psnr = 0.

            if args.sr:
                sr_input = ret['outputs_fine']['rgb'].to('cuda:0') # H, W, C
                sr_input = sr_input.unsqueeze(0).permute(0, 3, 1, 2) # 1, C, H, W
                interpo_output = torch.nn.functional.interpolate(sr_input, scale_factor=2, mode='bicubic', align_corners=False).squeeze(0).permute(1, 2, 0)
                # sr_output = model.sr_net(sr_input)
                
                # tile = 10
                tile = 2 * args.window_size # TODO important!
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
                imageio.imwrite(os.path.join(out_scene_dir, '{}_pred_sr.png'.format(file_id)), fine_pred_rgb)
                gt_rgb_hr_np_uint8 = (255 * np.clip(gt_rgb_hr.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                imageio.imwrite(os.path.join(out_scene_dir, '{}_gt_rgb_hr.png'.format(file_id)), gt_rgb_hr_np_uint8)
                with tf.Session() as session:
                    sr_psnr = session.run(psnr_tf, feed_dict={pred_ph: fine_pred_rgb_np, gt_ph: gt_rgb_hr_np})[0]
                    interpo_psnr = session.run(psnr_tf, feed_dict={pred_ph: fine_interpo_rgb_np, gt_ph: gt_rgb_hr_np})[0]
                sum_sr_psnr += sr_psnr
                running_mean_sr_psnr = sum_sr_psnr / (i + 1)
                sum_interpo_psnr += interpo_psnr
                running_mean_interpo_psnr = sum_interpo_psnr / (i + 1)
                
                
            sum_fine_psnr += fine_psnr
            running_mean_fine_psnr = sum_fine_psnr / (i + 1)
            sum_fine_lpips += fine_lpips
            running_mean_fine_lpips = sum_fine_lpips / (i + 1)
            sum_fine_ssim += fine_ssim
            running_mean_fine_ssim = sum_fine_ssim / (i + 1)

            if not args.sr:
                print("==================\n"
                    "{}, curr_id: {} \n"
                    "current coarse psnr: {:03f}, current fine psnr: {:03f} \n"
                    "running mean coarse psnr: {:03f}, running mean fine psnr: {:03f} \n"
                    "current coarse ssim: {:03f}, current fine ssim: {:03f} \n"
                    "running mean coarse ssim: {:03f}, running mean fine ssim: {:03f} \n" 
                    "current coarse lpips: {:03f}, current fine lpips: {:03f} \n"
                    "running mean coarse lpips: {:03f}, running mean fine lpips: {:03f} \n"
                    "===================\n"
                    .format(scene_name, file_id,
                            coarse_psnr, fine_psnr,
                            running_mean_coarse_psnr, running_mean_fine_psnr,
                            coarse_ssim, fine_ssim,
                            running_mean_coarse_ssim, running_mean_fine_ssim,
                            coarse_lpips, fine_lpips,
                            running_mean_coarse_lpips, running_mean_fine_lpips
                            ))
            else:
                print("==================\n"
                    "{}, curr_id: {} \n"
                    "current coarse psnr: {:03f}, current fine psnr: {:03f}, current sr psnr: {:03f}, current interpo psnr: {:03f} \n"
                    "running mean coarse psnr: {:03f}, running mean fine psnr: {:03f}, running mean sr psnr: {:03f}, running mean interpo psnr: {:03f} \n"
                    "current coarse ssim: {:03f}, current fine ssim: {:03f} \n"
                    "running mean coarse ssim: {:03f}, running mean fine ssim: {:03f} \n" 
                    "current coarse lpips: {:03f}, current fine lpips: {:03f} \n"
                    "running mean coarse lpips: {:03f}, running mean fine lpips: {:03f} \n"
                    "===================\n"
                    .format(scene_name, file_id,
                            coarse_psnr, fine_psnr, sr_psnr, interpo_psnr,
                            running_mean_coarse_psnr, running_mean_fine_psnr, running_mean_sr_psnr, running_mean_interpo_psnr,
                            coarse_ssim, fine_ssim,
                            running_mean_coarse_ssim, running_mean_fine_ssim,
                            coarse_lpips, fine_lpips,
                            running_mean_coarse_lpips, running_mean_fine_lpips
                            ))

            if not args.sr:
                results_dict[scene_name][file_id] = {'coarse_psnr': coarse_psnr,
                                                    'fine_psnr': fine_psnr,
                                                    'coarse_ssim': coarse_ssim,
                                                    'fine_ssim': fine_ssim,
                                                    'coarse_lpips': coarse_lpips,
                                                    'fine_lpips': fine_lpips,
                                                    }
            else:
                results_dict[scene_name][file_id] = {'coarse_psnr': coarse_psnr,
                                                    'fine_psnr': fine_psnr,
                                                    'sr_psnr': sr_psnr,
                                                    'interpo_psnr': interpo_psnr,
                                                    'coarse_ssim': coarse_ssim,
                                                    'fine_ssim': fine_ssim,
                                                    'coarse_lpips': coarse_lpips,
                                                    'fine_lpips': fine_lpips,
                                                    }

    mean_coarse_psnr = sum_coarse_psnr / total_num
    mean_fine_psnr = sum_fine_psnr / total_num
    mean_sr_psnr = sum_sr_psnr / total_num
    mean_interpo_psnr = sum_interpo_psnr / total_num
    mean_coarse_lpips = sum_coarse_lpips / total_num
    mean_fine_lpips = sum_fine_lpips / total_num
    mean_coarse_ssim = sum_coarse_ssim / total_num
    mean_fine_ssim = sum_fine_ssim / total_num

    if not args.sr:
        print('------{}-------\n'
            'final coarse psnr: {}, final fine psnr: {}\n'
            'fine coarse ssim: {}, final fine ssim: {} \n'
            'final coarse lpips: {}, fine fine lpips: {} \n'
            .format(scene_name, mean_coarse_psnr, mean_fine_psnr,
                    mean_coarse_ssim, mean_fine_ssim,
                    mean_coarse_lpips, mean_fine_lpips,
                    ))
    else:
        print('------{}-------\n'
            'final coarse psnr: {}, final fine psnr: {}, final sr psnr: {}, final interpo psnr: {}\n'
            'fine coarse ssim: {}, final fine ssim: {} \n'
            'final coarse lpips: {}, fine fine lpips: {} \n'
            .format(scene_name, mean_coarse_psnr, mean_fine_psnr, mean_sr_psnr, mean_interpo_psnr,
                    mean_coarse_ssim, mean_fine_ssim,
                    mean_coarse_lpips, mean_fine_lpips,
                    ))

    results_dict[scene_name]['coarse_mean_psnr'] = mean_coarse_psnr
    results_dict[scene_name]['fine_mean_psnr'] = mean_fine_psnr
    results_dict[scene_name]['coarse_mean_ssim'] = mean_coarse_ssim
    results_dict[scene_name]['fine_mean_ssim'] = mean_fine_ssim
    results_dict[scene_name]['coarse_mean_lpips'] = mean_coarse_lpips
    results_dict[scene_name]['fine_mean_lpips'] = mean_fine_lpips
    if args.sr:
        results_dict[scene_name]['sr_mean_psnr'] = mean_sr_psnr
        results_dict[scene_name]['interpo_mean_psnr'] = mean_interpo_psnr

    f = open("{}/psnr_{}_{}.txt".format(extra_out_dir, save_prefix, model.start_step), "w")
    f.write(str(results_dict))
    f.close()


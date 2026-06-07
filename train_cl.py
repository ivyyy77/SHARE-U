#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
# import sys
from scene.dataset_readers import fetchPly
from utils.graph_utils import *
from utils.point_utils import HumanSegmentationDataset
from model import libcore

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import torch
import seaborn as sns
import pandas as pd
from umap import UMAP
from random import randint
from sklearn.preprocessing import StandardScaler
from utils.mapper import types_mapping, correlation
from utils.loss_utils import *
from utils.tensor_utils import normalize_tensor, change_tensor
from gaussian_renderer._init_ori import render
# from gaussian_renderer import render
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
from scene import Scene, GaussianModel
from scene.gaussian_model import build_scaling_rotation
from utils.general_utils import safe_state, op_sigmoid
from utils.graphics_utils import world_to_camera, world_to_pixel
from utils.image_utils import *
# from
import uuid
import imageio
import torchvision
import numpy as np
import cv2
import pickle
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import clip
import matplotlib.colors as mcolors
from skopt import gp_minimize
from torch.fft import fft
from scipy.fft import fftshift
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity



try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import lpips

loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))
# loss_fn_vgg = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(torch.device('cuda', torch.cuda.current_device()))
import time
import torch.nn.functional as F


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             loss_type, elaboration_iterations):
    # base_colors = list(mcolors.TABLEAU_COLORS.values())  # 预定义颜色表
    fixed_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#f7b6d2", "#c7c7c7", "#ff9896", "#aec7e8", "#98df8a"
    ]
    # 归一化颜色
    colors = [mcolors.to_rgb(c) for c in fixed_colors]  # 归一化到 [0,1] 的 RGB 值
    first_iter = 0
    if dataset.actor_gender == 'neutral':
        opacity_thre = 0.005
        kl_thre = 0.4
        # kl_thre = 0.55    # ablation for GauHuman
        # elaboration_iterations = 800     # pcl
        # elaboration_iterations = 1200     # pcl
        # elaboration_iteration = 800     # fourier
        lambda_lpips = 0.08
    else:
        opacity_thre = 0.01
        opt.iterations = 3000
        opt.densify_until_iter = 1200 # 1500
        elaboration_iterations = 2000
        elaboration_iteration = 2000
        kl_thre = 0.1
        lambda_lpips = 0.05

    pipe.add = False
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender,
                              pipe.add, pipe.mcinfo)
    scene = Scene(dataset, gaussians)
    # cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    gaussians.training_setup(opt)
    if pipe.mcinfo:
        gaussians.mcinfo_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # pipe.contrastive = True

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    Ll1_loss_for_log = 0.0
    mask_loss_for_log = 0.0
    ssim_loss_for_log = 0.0
    lpips_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # lpips_test_lst = []
    pre_graph = None
    elapsed_time = 0
    pcd_path = './assets/smpl_semantic.ply'  # sample input scene
    # pcd_path = './assets/manohd_semantic.ply'
    file_list = [pcd_path]  # for now just the demo scene
    # pre_dataset = HumanSegmentationDataset(file_list=file_list)
    pre_dataset = HumanSegmentationDataset(file_list=file_list)
    _, _, labels = pre_dataset.load_pc(pcd_path)
    gaussians.frozen_labels = labels.cuda()
    clip_model, _ = clip.load("assets/ViT-B-16.pt", device='cuda', jit=False)

    gaussians._objects_dc = F.one_hot(gaussians.frozen_labels.to(torch.int64), num_classes=16).unsqueeze(1).to(
        torch.float32)

    # objects_before = gaussians._objects_dc
    # frozen_labels = torch.argmax(gaussians._objects_dc.squeeze(1), dim=1)

    viewpoint_stack = scene.getTrainCameras().copy()

    number = 0
    viewpoint_stack_list = {}
    for i in range(50):
        viewpoint_stack_list[i] = viewpoint_stack.copy()

    gaussians.prev_state = {}
    gaussians.prev_grads = {}


    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        segmentation_image = np.zeros((512, 512, 3))  # RGB 结果
        xyz_lr = gaussians.update_learning_rate(iteration)
        # background = torch.rand((3,), dtype=torch.float32, device='cuda')
        # gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            # gaussians.save_ply(scene.model_path)
        # Start timer
        start_time = time.time()

        # Pick a random Camera
        if len(viewpoint_stack) == 0:
            viewpoint_stack = viewpoint_stack_list[number]
            number = number + 1
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        # Semantic supervision loss
        frozen_labels = torch.argmax(gaussians._objects_dc.squeeze(1), dim=1)

        render_pkg = render(iteration, viewpoint_cam, gaussians, pipe, background)
        image, alpha, viewspace_point_tensor, visibility_filter, radii, objects = render_pkg["render"], \
                render_pkg["render_alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg[
                "radii"], render_pkg["render_object"]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        gt_image = viewpoint_cam.original_image.cuda()  # [3,512,512]

        bkgd_mask = viewpoint_cam.bkgd_mask.cuda()  # [1, 512, 512]
        bound_mask = viewpoint_cam.bound_mask.cuda()  # [1, 512, 512]
        
        Ll1 = l1_loss(image.permute(1, 2, 0)[bound_mask[0] == 1], gt_image.permute(1, 2, 0)[bound_mask[0] == 1])
        
        mask_loss = l2_loss(alpha[bound_mask == 1], bkgd_mask[bound_mask == 1])

        
        x, y, w, h = cv2.boundingRect(bound_mask[0].cpu().numpy().astype(np.uint8))
        img_pred = image[:, y:y + h, x:x + w].unsqueeze(0)
        img_gt = gt_image[:, y:y + h, x:x + w].unsqueeze(0)
        # ssim loss
        ssim_loss = ssim(img_pred, img_gt)
        # lipis loss
        lpips_loss = loss_fn_vgg(img_pred, img_gt).reshape(-1)
        # if iteration > elaboration_iterations or pipe.fourier:
        if (pipe.fourier and pipe.mcinfo) and iteration > elaboration_iterations:
            loss = Ll1 + 0.05 * mask_loss + 0.08 * (1.0 - ssim_loss) + lambda_lpips * lpips_loss 
        else:
            loss = Ll1 + 0.1 * mask_loss + 0.01 * (1.0 - ssim_loss) + 0.02 * lpips_loss


        if iteration == elaboration_iterations:
            gaussians.idx = gaussians.sample_gs(iteration)
        if iteration >= elaboration_iterations and iteration % 100 == 0:
            H_train = scene.uncertainty_est(iteration, viewpoint_stack_list[49], pipe, background, render)
            gaussians.H = H_train.unsqueeze(1)
        elif iteration > opt.densify_until_iter and iteration % 50 == 0:
            H_train = scene.uncertainty_est(iteration, viewpoint_stack_list[49], pipe, background, render)
            gaussians.H = H_train.unsqueeze(1)
        else:
            H_train = None


        if iteration > elaboration_iterations and pipe.mcinfo: 
            if (iteration-1) % 100 == 0 or iteration % 50 == 0:
                gaussians.idx = gaussians.sample_gs(iteration)
                gaussians.cal_ambiguity(K=gaussians.k)   
            features = torch.cat(
                (gaussians._xyz, gaussians._features_dc.squeeze(1), gaussians._opacity, gaussians._rotation, gaussians._scaling), dim=1)
            
            x_ref, x_pos, x_neg, H_ref, H_pos, H_neg, pos_ids, neg_ids = (gaussians.gen.sample
                            (gaussians, features, correlation, frozen_labels, n=1500, n_neg=16, oversampling_factor=10))
            
            mu_ref, kappa_ref, emb_ref = gaussians.enc(x_ref, H_ref)
            mu_pos, kappa_pos, emb_pos = gaussians.enc(x_pos, H_pos)
            mu_neg, kappa_neg, emb_neg = gaussians.enc(x_neg, H_neg)            

            nce_loss = gaussians.loss.InfoNCE_loss(emb_ref, emb_pos, emb_neg)
            loss += nce_loss
        

        if iteration > elaboration_iterations and pipe.fourier:
        
            object_type = torch.argmax(objects.permute(1, 2, 0), dim=2)
            fourier_loss = None
            for j in types_mapping.keys():
                if j == 0:
                    continue
                object_mask = (object_type == j).float().unsqueeze(0)
                img_part = image.permute(1, 2, 0)[object_mask[0] == 1]
                gt_part = gt_image.permute(1, 2, 0)[object_mask[0] == 1]
                if img_part.size(0) == 0:
                    continue
                
                img_part = torch.fft.fft2(img_part)
                gt_part = torch.fft.fft2(gt_part)

                fourier_loss_ = F.l1_loss(img_part, gt_part)

                fourier_loss = fourier_loss_ if fourier_loss is None else fourier_loss + fourier_loss_
            
            loss += 3e-4 * fourier_loss

        
        loss.backward(retain_graph=True)

        # end time
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time += (end_time - start_time)

        iter_end.record()

        try:
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        except RuntimeError as e:
            print(f"Error occurred: {e}")
            raise
        # ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        Ll1_loss_for_log = 0.4 * Ll1.item() + 0.6 * Ll1_loss_for_log
        mask_loss_for_log = 0.4 * mask_loss.item() + 0.6 * mask_loss_for_log
        ssim_loss_for_log = 0.4 * ssim_loss.item() + 0.6 * ssim_loss_for_log
        lpips_loss_for_log = 0.4 * lpips_loss.item() + 0.6 * lpips_loss_for_log

        with torch.no_grad():
            if iteration % 10 == 0:
                progress_bar.set_postfix({"#pts": gaussians._xyz.shape[0], "Ll1 Loss": f"{Ll1_loss_for_log:.{3}f}",
                                          "mask Loss": f"{mask_loss_for_log:.{2}f}",
                                          "ssim": f"{ssim_loss_for_log:.{2}f}", "lpips": f"{lpips_loss_for_log:.{2}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Start timer
            start_time = time.time()
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration >= opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    prev_iter = iteration 
                    gaussians.grads_pre = gaussians.xyz_gradient_accum / gaussians.denom
                    gaussians.prev_state.update({f'{prev_iter}': gaussians._objects_dc})
                    gaussians.prev_grads.update({f'{prev_iter}': gaussians.grads_pre})

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opacity_thre, scene.cameras_extent, size_threshold,
                            kl_threshold=kl_thre, t_vertices=viewpoint_cam.big_pose_world_vertex, iter=iteration, H=H_train)
                            
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                if iteration > elaboration_iterations and pipe.mcinfo:
                    gaussians.mcinfo_optimizer.step()
                    gaussians.mc_scheduler.step()
                    gaussians.mcinfo_optimizer.zero_grad(set_to_none=True)


            # end time
            end_time = time.time()
            # Calculate elapsed time
            elapsed_time += (end_time - start_time)

            # if (iteration in checkpoint_iterations):
            if (iteration in testing_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")



def prepare_output_and_logger(args):
    if not args.model_path:
        
        args.model_path = os.path.join("./output/", args.exp_name)

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': scene.getTrainCameras()})

        smpl_rot = {}
        smpl_rot['train'], smpl_rot['test'] = {}, {}
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    smpl_rot[config['name']][viewpoint.pose_id] = {}
                    render_output = renderFunc(iteration, viewpoint, scene.gaussians, *renderArgs, return_smpl_rot=True)
                    image = torch.clamp(render_output["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    bound_mask = viewpoint.bound_mask
                    image.permute(1, 2, 0)[bound_mask[0] == 0] = 0 if renderArgs[1].sum().item() == 0 else 1
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += loss_fn_vgg(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()

                    smpl_rot[config['name']][viewpoint.pose_id]['transforms'] = render_output['transforms']
                    smpl_rot[config['name']][viewpoint.pose_id]['translation'] = render_output['translation']

                l1_test /= len(config['cameras'])
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {} #{}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'],
                                                                                             len(config['cameras']),
                                                                                             l1_test, psnr_test,
                                                                                             ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        # Store data (serialize)
        save_path = os.path.join(scene.model_path, 'smpl_rot', f'iteration_{iteration}')
        os.makedirs(save_path, exist_ok=True)
        with open(save_path + "/smpl_rot.pickle", 'wb') as handle:
            pickle.dump(smpl_rot, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        # torch.cuda.empty_cache()   # original




if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=7004)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[1200， 2000])
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[1200， 2000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--elaboration_iterations", type=int, default=700)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.exp_name, 
             args.elaboration_iterations)
    # All done
    print("\nTraining complete.")

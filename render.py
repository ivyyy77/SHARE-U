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

import torch
import numpy as np
from scene import Scene
import os
import time
import pickle
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
# from gaussian_renderer._init_ori import render
import torchvision
import torchvision.utils
# from torchvision import utils
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import matplotlib.colors as mcolors
from utils.mapper import types_mapping, correlation

from utils.image_utils import psnr
from utils.loss_utils import ssim
import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    fixed_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#f7b6d2", "#c7c7c7", "#ff9896", "#aec7e8", "#98df8a"
    ]
    # 归一化颜色
    colors = [mcolors.to_rgb(c) for c in fixed_colors]
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    heatmap_path = os.path.join(model_path, name, "ours_{}".format(iteration), "seg")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(heatmap_path, exist_ok=True)

    # Load data (deserialize)
    with open(model_path + '/smpl_rot/' + f'iteration_{iteration}/' + 'smpl_rot.pickle', 'rb') as handle:
        smpl_rot = pickle.load(handle)

    rgbs = []
    rgbs_gt = []
    heatmap_list = []
    elapsed_time = 0

    # background = torch.rand((3,), dtype=torch.float32, device='cuda')
    for _, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :].cuda()
        bound_mask = view.bound_mask
        # bkgd_mask = view.bkgd_mask.to('cuda')
        # # gt = gt.permute(1, 2, 0)[bkgd_mask[0] == 1]
        # bkgd_mask = bkgd_mask.squeeze(0)  # (540, 540)
        # bkgd_mask = bkgd_mask.unsqueeze(-1).repeat(1, 1, 3)  # (540, 540, 3)
        #
        # new_bg = background.unsqueeze(0).unsqueeze(0).repeat(540, 540, 1)  # (540, 540, 3)
        # # 合并前景和背景
        # gt = gt.permute(1,2,0) * bkgd_mask + new_bg * (1 - bkgd_mask)

        transforms, translation = smpl_rot[name][view.pose_id]['transforms'], smpl_rot[name][view.pose_id]['translation']

        # Start timer
        start_time = time.time() 
        render_output = render(iteration, view, gaussians, pipeline, background, transforms=transforms, translation=translation)
        
        # end time
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time += end_time - start_time

        rendering = render_output["render"]
        # objects = render_output["render_object"]
        # segmentation_image = np.zeros((512, 512, 3))
        # object_type = torch.argmax(objects.permute(1, 2, 0), dim=2)
        # for j in types_mapping.keys():
        #     if j == 0:
        #         continue
            # mask = (object_type == j).cpu().numpy()  # 获取当前类别的 mask
            # segmentation_image[mask] = colors[j - 1][:3]
        # segmentation_image = torch.tensor(segmentation_image).permute(2, 0, 1)

        rendering.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1
        # rendering.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1

        heatmap = render_output['heatmap']
        rgbs.append(rendering)
        rgbs_gt.append(gt)
        heatmap_list.append(heatmap)

    # Calculate elapsed time
    print("Elapsed time: ", elapsed_time, " FPS: ", len(views)/elapsed_time) 

    psnrs = 0.0
    ssims = 0.0
    lpipss = 0.0

    # from utils.ffmpeg import images_to_video
    # images_to_video(
    #     torch.stack(rgbs, axis=0).permute(0,2,3,1).cpu().numpy(),
    #     output_path='./debug_video.mp4',
    #     fps=24,
    #     gradio_codec=False,
    #     verbose=True,
    # )

    for id in range(len(views)):
        rendering = rgbs[id]
        gt = rgbs_gt[id]
        heatmap_id = heatmap_list[id]
        rendering = torch.clamp(rendering, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)
        # heatmap_list_id = torch.clamp(heatmap_id, 0.0, 1.0)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(id) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(id) + ".png"))
        torchvision.utils.save_image(heatmap_id, os.path.join(heatmap_path, '{0:05d}'.format(id) + ".png"))
        #
        # metrics
        psnrs += psnr(rendering, gt).mean().double()
        ssims += ssim(rendering, gt).mean().double()
        lpipss += loss_fn_vgg(rendering, gt).mean().double()

    psnrs /= len(views)   
    ssims /= len(views)
    lpipss /= len(views)  

    # evalution metrics
    print("\n[ITER {}] Evaluating {} #{}: PSNR {} SSIM {} LPIPS {}".format(iteration, name, len(views), psnrs, ssims, lpipss))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender,
                                  pipeline.add, pipeline.mcinfo)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # print('------------------------------------------', dataset.source_path)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # background = torch.rand((3,), dtype=torch.float32, device='cuda')

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
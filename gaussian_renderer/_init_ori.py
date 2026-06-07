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
import os.path

import torch
import math
import diff_gaussian_rasterization as dgr
import diff_gaussian_rasterization_obj as dgro
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from pytorch3d.ops import knn_points
import cv2
import numpy as np
import matplotlib.pyplot as plt
# from gaussian_renderer.normal_render import Renderer
from process_smpl import *
import matplotlib.colors as mcolors
from utils.mapper import types_mapping
from model import libcore



def render(iteration, viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None, return_smpl_rot=False, transforms=None, translation=None, query_pose_cam=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass


    fixed_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#f7b6d2", "#c7c7c7", "#ff9896", "#aec7e8", "#98df8a"
    ]
    # 归一化颜色
    colors = [mcolors.to_rgb(c) for c in fixed_colors]

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = dgr.GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    raster_settings_obj = dgro.GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    # rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rasterizer = dgr.GaussianRasterizer(raster_settings=raster_settings)
    rasterizer_obj = dgro.GaussianRasterizer(raster_settings=raster_settings_obj)

    if pipe.add and iteration > 800:
        # pc, embedding = pc.non_rigid(pc, iteration, viewpoint_camera)
        embedding = pc.non_rigid(pc, iteration, viewpoint_camera)
        means3D = pc.get_xyz
    else:
        means3D = pc.get_xyz


    if not pc.motion_offset_flag:
        _, means3D, _, transforms, _ = pc.coarse_deform_c2source(means3D[None], viewpoint_camera.smpl_param,
                                                                 viewpoint_camera.big_pose_smpl_param,
                                                                 viewpoint_camera.big_pose_world_vertex[None])
    else:
        if transforms is None:
            # pose offset
            # dst_posevec = viewpoint_camera.smpl_param['poses'][:, 3:]
            dst_posevec = viewpoint_camera.smpl_param['poses'][:, 3:]
            pose_out = pc.pose_decoder(dst_posevec)
            correct_Rs = pose_out['Rs']

            # SMPL lbs weights
            lbs_weights = pc.lweight_offset_decoder(means3D[None].detach())
            lbs_weights = lbs_weights.permute(0, 2, 1)

            # transform points

            _, means3D, _, transforms, translation, global_transformation\
                = pc.coarse_deform_c2source(
                means3D[None], viewpoint_camera.smpl_param, viewpoint_camera.big_pose_smpl_param,
                viewpoint_camera.big_pose_world_vertex[None], lbs_weights=lbs_weights, correct_Rs=correct_Rs,
                return_transl=return_smpl_rot)
        else:
            correct_Rs = None
            means3D = torch.matmul(transforms, means3D[..., None]).squeeze(-1) + translation


    means3D = means3D.squeeze()
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp_ = pc.get_covariance(scaling_modifier, transforms.squeeze())  #
        cov3D_precomp = cov3D_precomp_[0]
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation


    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    sh_objs = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
            sh_objs = pc.get_objects
    else:
        colors_precomp = override_color


    # # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth, alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    rendered_image_obj, radii_obj, rendered_objects = rasterizer_obj(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        sh_objs=sh_objs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        # scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    if pipe.contrastive and query_pose_cam is not None:
        dst_posevec_query = query_pose_cam.smpl_param['poses'][:, 3:]
        pose_out_query = pc.pose_decoder(dst_posevec_query)
        correct_Rs_query = pose_out_query['Rs']
        # _, means3D, _, transforms, translation
        _, means3D_query, _, transform_query, _, global_transformation_query, query_A, query_shape_off, query_pose_off\
            = pc.coarse_deform_c2source(
            pc.get_xyz[None], query_pose_cam.smpl_param, query_pose_cam.big_pose_smpl_param,
            query_pose_cam.big_pose_world_vertex[None], lbs_weights=lbs_weights, correct_Rs=correct_Rs_query,
            return_transl=return_smpl_rot)
        contrastive_gau, cano_gaussian_contrastive = pc.contrastive_deform(
            means3D_query.squeeze(), query_A.squeeze(), current_A.squeeze(),
            global_transformation_query, global_transformation,
            query_shape_off, current_shape_off, query_pose_off, current_pose_off)  # tensor  1704, 3

        cov3D_precomp_query = pc.get_cov_contrastive(cov3D_precomp_[1], transform_query.squeeze(), transforms.squeeze())

        rendered_image_query, _, _, alpha_contrastive = rasterizer(
            means3D=contrastive_gau.squeeze(),
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp_query)


    if query_pose_cam is not None:
        return {
            "render_query": rendered_image_query,  # contrastive
            "render": rendered_image,
            "render_depth": depth,
            "render_alpha": alpha,
            "render_alpha_contrastive": alpha_contrastive,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "transforms": transforms,
            "translation": translation,
            "deform_matrix": current_A,
            "correct_Rs": correct_Rs,
            "posed_Gaussian_query": contrastive_gau.squeeze(), 
            "cano_gau_contrastive": cano_gaussian_contrastive.squeeze(),  
            "render_object": rendered_objects,
            'posed_gau': means3D,

    else:

        # print("rendered_image={}".format(rendered_image))
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
        # return {"render":heatmap,
                "render_depth": depth,
                "render_alpha": alpha,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                'posed_gau': means3D,
                "transforms": transforms,
                "translation": translation,
                "correct_Rs": correct_Rs,
                "render_object": rendered_objects,
                }

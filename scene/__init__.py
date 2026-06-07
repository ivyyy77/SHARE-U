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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel

from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
from utils.system_utils import mkdir_p
from tqdm import tqdm
from einops import reduce
from utils.loss_utils import *

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        # self.non_rigid = gaussians.non_rigid.cuda()

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        # print(args.source_path)
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif 'zju_mocap_refine' in args.source_path: #os.path.exists(os.path.join(args.source_path, "annots.npy")):
            print("Found annots.json file, assuming ZJU_MoCap_refine data set!")
            scene_info = sceneLoadTypeCallbacks["ZJU_MoCap_refine"](args.source_path, args.white_background, args.exp_name, args.eval)
        elif 'monocap' in args.source_path:
            print("assuming MonoCap data set!")
            scene_info = sceneLoadTypeCallbacks["MonoCap"](args.source_path, args.white_background, args.exp_name, args.eval)
        elif 'dna_rendering' in args.source_path:
            print("assuming dna_rendering data set!")
            scene_info = sceneLoadTypeCallbacks["dna_rendering"](args.source_path, args.white_background, args.exp_name, args.eval)
        elif 'people_snapshot' in args.source_path:
            print("assuming PeopleSnapShot data set!")
            scene_info = sceneLoadTypeCallbacks["peoplesnap"](args.source_path, args.white_background, args.exp_name, args.eval)
        elif 'xhuman' in args.source_path:
            print("assuming XHuman data set!")
            scene_info = sceneLoadTypeCallbacks["xhuman"](args.source_path, args.white_background, args.exp_name, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        # if 'people_snapshot' not in args.source_path:
        #     self.cameras_extent = scene_info.nerf_normalization["radius"]
        # else:
        #     self.cameras_extent = 1.0

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        if self.gaussians.motion_offset_flag:
            model_path = os.path.join(self.model_path, "mlp_ckpt", "iteration_" + str(self.loaded_iter), "ckpt.pth")
            if os.path.exists(model_path):
                ckpt = torch.load(model_path, map_location='cuda:0')
                self.gaussians.pose_decoder.load_state_dict(ckpt['pose_decoder'])
                self.gaussians.lweight_offset_decoder.load_state_dict(ckpt['lweight_offset_decoder'])

    def save_posed_gaussians_contrastive(self, iteration, posed_gaussian):
        point_cloud_path = os.path.join(self.model_path, "posed_gau".format(iteration))
        self.gaussians.save_posed_gaussian_ply(os.path.join(point_cloud_path, f"1-xhuman-{iteration}.ply"),
                                               posed_gaussian)

    def uncertainty_est(self, iteration, viewpoint_cams, pipe, background, render):
        # uncertainty
        # filter_out_grad = ["rotation", "opacity", "scale"]
        # filter_out_grad = ["xyz", "scale"]
        filter_out_grad = ["rotation", "opacity", "scale", "xyz", "feature_dc"]
        name2idx = {"xyz": 0, "rgb": 1, "feature_dc": 2, "scale": 3, "rotation": 4, "opacity": 5}
        filter_out_idx = [name2idx[k] for k in filter_out_grad]

        gaussians_params = self.gaussians.capture()[1:7]
        gaussians_params = [p for i, p in enumerate(gaussians_params) if i not in filter_out_idx]

        # H_train = torch.zeros(sum(p.numel() for p in gaussians_params), device=gaussians_params[0].device, dtype=gaussians_params[0].dtype)
        H_train = torch.zeros(gaussians_params[0].shape[0], device=gaussians_params[0].device,
                              dtype=gaussians_params[0].dtype)

        # Run hessian on training set
        # for i, cam in enumerate(tqdm(viewpoint_cams, desc="Calculating diagonal Hessian on training views")):
        for i, cam in enumerate(viewpoint_cams):

            render_pkg = render(iteration, cam, self.gaussians, pipe, background)
            pred_img = render_pkg["render"]
            # gt_image = cam.original_image.cuda()
            # bound_mask = cam.bound_mask.cuda()
            # loss = l1_loss(pred_img.permute(1, 2, 0)[bound_mask[0] == 1], gt_image.permute(1, 2, 0)[bound_mask[0] == 1])
            # loss.backward(gradient=torch.ones_like(pred_img))
            pred_img.backward(gradient=torch.ones_like(pred_img))
            H_train += sum([reduce(torch.square(p.grad.detach()), "n ... -> n", "sum") for p in gaussians_params])

            self.gaussians.optimizer.zero_grad(set_to_none=True)

        return H_train


    # def uncertainty_est(self, iteration, viewpoint_cams, pipe, background, render):
    #     filter_out_grad = ["rotation", "opacity", "scale", "xyz", "feature_dc"]
    #     name2idx = {"xyz": 0, "rgb": 1, "feature_dc": 2, "scale": 3, "rotation": 4, "opacity": 5}
    #     filter_out_idx = [name2idx[k] for k in filter_out_grad]
    #
    #     gaussians_params = self.gaussians.capture()[1:7]
    #     gaussians_params = [p for i, p in enumerate(gaussians_params) if i not in filter_out_idx]
    #
    #     H_train = torch.zeros(gaussians_params[0].shape[0],
    #                           device=gaussians_params[0].device,
    #                           dtype=gaussians_params[0].dtype)
    #
    #     # 避免重复计算梯度
    #     gradients = []
    #     for cam in viewpoint_cams:
    #         with torch.no_grad():  # 关闭梯度计算，避免计算图构建
    #             render_pkg = render(iteration, cam, self.gaussians, pipe, background)
    #             pred_img = render_pkg["render"]
    #
    #         # 计算梯度，不存储计算图，减少显存占用
    #         grads = torch.autograd.grad(pred_img, gaussians_params,
    #                                     grad_outputs=torch.ones_like(pred_img),
    #                                     retain_graph=False, create_graph=False)
    #
    #         gradients.append([torch.square(g.detach()) for g in grads])
    #
    #     # 直接在 GPU 上累加
    #     for grads in gradients:
    #         H_train += sum(grads)
    #
    #     return H_train


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

        if self.gaussians.motion_offset_flag:
            model_path = os.path.join(self.model_path, "mlp_ckpt", "iteration_" + str(iteration), "ckpt.pth")
            mkdir_p(os.path.dirname(model_path))
            torch.save({
                'iter': iteration,
                'pose_decoder': self.gaussians.pose_decoder.state_dict(),
                'lweight_offset_decoder': self.gaussians.lweight_offset_decoder.state_dict(),
            }, model_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
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
import pickle

import torch
import torch.nn.functional as F
from knn_cuda import KNN
from networkx.classes import selfloop_edges
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from nets.mlp_delta_body_pose import BodyPoseRefiner
from nets.mlp_delta_weight_lbs import LBSOffsetDecoder
from nets.non_rigid import HashGridwithMLP
from process_smpl import *
from utils.general_utils import WarmupCosineLR
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_scaling
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.graph_utils import GAT
from utils.graphics_utils import BasicPointCloud
from utils.loss_cl import Encoder, MCInfoNCE, Generator
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p
from pytorch3d.transforms import axis_angle_to_matrix
from utils.smplx.smplx import SMPLLayer
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, transform):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance_ = L @ L.transpose(1, 2)
            if transform is not None:
                actual_covariance = transform @ actual_covariance_
                actual_covariance = actual_covariance @ transform.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm, actual_covariance_

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, smpl_type: str, motion_offset_flag: bool, actor_gender: str, add: bool, mcinfo: bool):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._objects_dc = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.num_objects = 16
        self.frozen_labels = torch.empty(0)

        self.device = torch.device('cuda', torch.cuda.current_device())
        if smpl_type == 'smpl':
            neutral_smpl_path = os.path.join('assets', f'SMPL_{actor_gender.upper()}.pkl')
            self.SMPL_NEUTRAL = SMPL_to_tensor(read_pickle(neutral_smpl_path), device=self.device)
        elif smpl_type == 'smplx':
            neutral_smpl_path = os.path.join('assets/models/smplx', f'SMPLX_{actor_gender.upper()}.npz')
            params_init = dict(np.load(neutral_smpl_path, allow_pickle=True))
            self.SMPL_NEUTRAL = SMPL_to_tensor(params_init, device=self.device)

        self.gender = actor_gender
        self.smpl_model_big = SMPLLayer(model_path=f'assets/SMPL_{actor_gender.upper()}.pkl')

        self.lambda_legs = -5
        self.lambda_body = -5
        self.lambda_face = -5
        self.lambda_hand = -5

        self.e = 4e-4
        self.e1 = -2e-3
        self.e_legs = 3.5e-4
        self.e_body = 3.5e-4
        self.e_face = 3.4e-4
        self.e_hand = 3.5e-4


        self.hyperparameter = [self.lambda_hand, self.lambda_face, self.lambda_legs, self.lambda_body,
                               self.e_hand, self.e_face, self.e_legs, self.e_body]

        self.lambda_range = [(-3, -7), (-3, -7), (-3, -7), (-3, -7)]
        self.e_range = [(2e-4, 4e-4), (2e-4, 4e-4), (2e-4, 4e-4), (2e-4, 4e-4)]

        self.mcinfo = mcinfo
        if self.mcinfo:
            self.gen = Generator(dim_x=14, dim_hidden=10, dim_z=10, n_hidden=1, pos_kappa=20, post_kappa_min=18, post_kappa_max=32,
                                 family="vmf", has_joint_backbone=False)
            self.enc = Encoder(dim_x=14, dim_z=32, post_kappa_min=16, post_kappa_max=32, dim_hidden=64,
                               x_samples=self.gen._sample_x(1000), has_joint_backbone=False)

            self.enc.train()
            self.loss = MCInfoNCE(kappa_init=32, n_samples=512)
            self.loss.train()


        self.knn = KNN(k=1, transpose_mode=True)
        self.knn_near_2 = KNN(k=2, transpose_mode=True)
        self.knn_near_obj = KNN(k=10, transpose_mode=True)
        self.k = 10
        self.knn_near_semantic = KNN(k=self.k+1, transpose_mode=True)

        self.GAT = GAT(input_dim=14, hidden_dim=64, output_dim=64).to('cuda').train()
        self.optimizer_GAT = torch.optim.Adam(self.GAT.parameters(), lr=0.001)

        self.motion_offset_flag = motion_offset_flag
        if self.motion_offset_flag:
            total_bones = self.SMPL_NEUTRAL['weights'].shape[-1]
            self.pose_decoder = BodyPoseRefiner(total_bones=total_bones, embedding_size=3 * (total_bones - 1),
                                                mlp_width=128, mlp_depth=2)
            self.pose_decoder.to(self.device)

            self.lweight_offset_decoder = LBSOffsetDecoder(total_bones=total_bones)
            self.lweight_offset_decoder.to(self.device)

        self.add = add
        if self.add:
            self.non_rigid = HashGridwithMLP().to(self.device)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.pose_decoder,
            self.lweight_offset_decoder,
            self._objects_dc
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self._objects_dc,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale,
         self.pose_decoder,
         self.lweight_offset_decoder) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_objects(self):
        return self._objects_dc

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1, transform=None):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation, transform)

    def get_cov_contrastive_ori(self, query_cov, transform_query, transform_current):
        cov_recover = torch.inverse(transform_query) @ query_cov
        transform_transpose = transform_query.transpose(1,2)
        cov_recover = cov_recover @ torch.inverse(transform_transpose[...,:3,:3])

        cov_contrastive = transform_current @ cov_recover
        cov_contrastive = cov_contrastive @ transform_current.transpose(1,2)
        symm = strip_symmetric(cov_contrastive)
        return symm

    def get_cov_contrastive(self, cano_cov, transform_query, transform_current):

        cov_contrastive = transform_current @ cano_cov
        cov_contrastive = cov_contrastive @ transform_current.transpose(1,2)
        symm = strip_symmetric(cov_contrastive)
        return symm


    def clone(self):
        cloned = GaussianModel(self.max_sh_degree, 'smpl', self.motion_offset_flag, 'neutral',
                               self.add, self.mcinfo)

        properties = ["active_sh_degree",
                      "non_rigid_feature"]
        for property in properties:
            if hasattr(self, property):
                setattr(cloned, property, getattr(self, property))

        parameters = ["_xyz",
                      "_features_dc",
                      "_features_rest",
                      "_scaling",
                      "_rotation",
                      "_opacity",
                      '_objects_dc']
        for parameter in parameters:
            setattr(cloned, parameter, getattr(self, parameter) + 0.)

        return cloned

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        fused_objects = RGB2SH(torch.rand((fused_point_cloud.shape[0], self.num_objects), device="cuda"))
        fused_objects = fused_objects[:, :, None]

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._objects_dc = torch.zeros([self.get_xyz.shape[0],1,16], device="cuda")
        self.frozen_labels = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def save_posed_gaussian_ply(self, path, posed_gaussian):
        mkdir_p(os.path.dirname(path))

        xyz = posed_gaussian.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()

        scale = self._scaling.detach().cpu().numpy()

        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_wo_obj()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        if not self.motion_offset_flag:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': [self._objects_dc], 'lr': training_args.feature_lr, "name": "obj_dc"},
            ]
        else:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': self.pose_decoder.parameters(), 'lr': training_args.pose_refine_lr, "name": "pose_decoder"},
                {'params': self.lweight_offset_decoder.parameters(), 'lr': training_args.lbs_offset_lr,
                 "name": "lweight_offset_decoder"},

            ]
            if self.add:
                l.append({'params': [p for n, p in self.non_rigid.named_parameters() if 'latent' not in n],
                 'lr': training_args.non_rigid_lr, "name": "non_rigid_field"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def mcinfo_setup(self, training_args):
        if self.mcinfo:
            l = [{'params': list(self.enc.mu_net.parameters()),
                  'lr': training_args.mcinfo_lr, "name": "mcinfo"}]

        self.mcinfo_optimizer = torch.optim.AdamW(l, lr=0.0, eps=1e-15)
        self.mc_scheduler = WarmupCosineLR(self.mcinfo_optimizer, warmup_epochs=0, max_epochs=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._objects_dc.shape[1] * self._objects_dc.shape[2]):
            l.append('obj_dc_{}'.format(i))
        return l

    def construct_list_of_attributes_wo_obj(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l


    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_wo_obj()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])


        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.frozen_labels = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation', "obj_dc"]:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._objects_dc = self._objects_dc[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def prune_points_(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._objects_dc = self._objects_dc[valid_points_mask]
        self.H = self.H[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]


    def cat_tensors_to_optimizer_ori(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation', "obj_dc"]:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                        dim=0)
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation']:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                        dim=0)
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(False))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_objects_dc):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        ori_obj_dc = self._objects_dc
        self._objects_dc = torch.cat((ori_obj_dc, new_objects_dc), dim=0)


        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def densification_postfix_(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_objects_dc, new_H):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        ori_obj_dc = self._objects_dc
        self._objects_dc = torch.cat((ori_obj_dc, new_objects_dc), dim=0)

        self.H = torch.cat((self.H.squeeze(), new_H), dim=0)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_objects_dc = self._objects_dc[selected_pts_mask].repeat(N, 1, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_objects_dc)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, objects_before, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_objects_dc = objects_before[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_objects_dc)

    def kl_densify_and_clone(self, grads, grad_threshold, scene_extent, kl_threshold=0.4, H=None):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        _, point_ids = self.knn_near_2(self._xyz[None].detach(), self._xyz[None].detach())
        xyz = self._xyz[point_ids[0]].detach()
        rotation_q = self._rotation[point_ids[0]].detach()
        scaling_diag = self.get_scaling[point_ids[0]].detach()

        xyz_0 = xyz[:, 0].reshape(-1, 3)
        rotation_0_q = rotation_q[:, 0].reshape(-1, 4)
        scaling_diag_0 = scaling_diag[:, 0].reshape(-1, 3)

        xyz_1 = xyz[:, 1:].reshape(-1, 3)
        rotation_1_q = rotation_q[:, 1:].reshape(-1, 4)
        scaling_diag_1 = scaling_diag[:, 1:].reshape(-1, 3)

        kl_div = self.kl_div(xyz_0, rotation_0_q, scaling_diag_0, xyz_1, rotation_1_q, scaling_diag_1)
        self.kl_selected_pts_mask = kl_div > kl_threshold

        selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask

        print("[kl clone]: ", (selected_pts_mask & self.kl_selected_pts_mask).sum().item())
        stds = self.get_scaling[selected_pts_mask]
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask])
        new_rotation = self._rotation[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_objects_dc = self._objects_dc[selected_pts_mask]
        if H is not None:
            self.H = torch.cat((H, H[selected_pts_mask]), dim=0).unsqueeze(1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_objects_dc)

    def kl_densify_and_split(self, grads, grad_threshold, scene_extent, kl_threshold=0.4, N=2, H=None):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        _, point_ids = self.knn_near_2(self._xyz[None].detach(), self._xyz[None].detach())
        xyz = self._xyz[point_ids[0]].detach()
        rotation_q = self._rotation[point_ids[0]].detach()
        scaling_diag = self.get_scaling[point_ids[0]].detach()

        xyz_0 = xyz[:, 0].reshape(-1, 3)
        rotation_0_q = rotation_q[:, 0].reshape(-1, 4)
        scaling_diag_0 = scaling_diag[:, 0].reshape(-1, 3)

        xyz_1 = xyz[:, 1:].reshape(-1, 3)
        rotation_1_q = rotation_q[:, 1:].reshape(-1, 4)
        scaling_diag_1 = scaling_diag[:, 1:].reshape(-1, 3)

        kl_div = self.kl_div(xyz_0, rotation_0_q, scaling_diag_0, xyz_1, rotation_1_q, scaling_diag_1)
        self.kl_selected_pts_mask = kl_div > kl_threshold

        selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask

        print("[kl split]: ", (selected_pts_mask & self.kl_selected_pts_mask).sum().item())

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_objects_dc = self._objects_dc[selected_pts_mask].repeat(N, 1, 1)
        if H is not None:
            self.H = torch.cat((self.H, self.H[selected_pts_mask].repeat(N,1)), dim=0)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_objects_dc)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def kl_merge(self, grads, grad_threshold, scene_extent, kl_threshold=0.1, H=None):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        _, point_ids = self.knn_near_2(self._xyz[None].detach(), self._xyz[None].detach())
        xyz = self._xyz[point_ids[0]].detach()
        rotation_q = self._rotation[point_ids[0]].detach()
        scaling_diag = self.get_scaling[point_ids[0]].detach()

        xyz_0 = xyz[:, 0].reshape(-1, 3)
        rotation_0_q = rotation_q[:, 0].reshape(-1, 4)
        scaling_diag_0 = scaling_diag[:, 0].reshape(-1, 3)

        xyz_1 = xyz[:, 1:].reshape(-1, 3)
        rotation_1_q = rotation_q[:, 1:].reshape(-1, 4)
        scaling_diag_1 = scaling_diag[:, 1:].reshape(-1, 3)

        kl_div = self.kl_div(xyz_0, rotation_0_q, scaling_diag_0, xyz_1, rotation_1_q, scaling_diag_1)
        self.kl_selected_pts_mask = kl_div < kl_threshold

        selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask

        print("[kl merge]: ", (selected_pts_mask & self.kl_selected_pts_mask).sum().item())

        if selected_pts_mask.sum() >= 1:
            selected_point_ids = point_ids[0][selected_pts_mask]
            new_xyz = self.get_xyz[selected_point_ids].mean(1)
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_point_ids][:, 0] / 0.8)
            new_rotation = self._rotation[selected_point_ids][:, 0]
            new_features_dc = self._features_dc[selected_point_ids].mean(1)
            new_features_rest = self._features_rest[selected_point_ids].mean(1)
            new_opacity = self._opacity[selected_point_ids].mean(1)
            new_objects_dc = self._objects_dc[selected_pts_mask].mean(1)
            new_objects_dc = new_objects_dc.unsqueeze(1)
            if H is not None:
                self.H = torch.cat((self.H, self.H[selected_point_ids].mean(1)), dim=0)
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling,
                                       new_rotation, new_objects_dc)


            selected_pts_mask[selected_point_ids[:, 1]] = True
            prune_filter = torch.cat((selected_pts_mask, torch.zeros(new_xyz.shape[0], device="cuda", dtype=bool)))
            self.prune_points(prune_filter)



    def kl_densify_and_clone_semantic(self, grads, grad_threshold, scene_extent, iter, kl_threshold=0.4, H=None):

        if iter == 500:
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling,
                                                            dim=1).values <= self.percent_dense * scene_extent)
        else:
            num_classes = self._objects_dc.squeeze().size(1)
            selected_pts_mask = torch.zeros(grads.size(0), dtype=torch.bool, device=grads.device)
            frozen_labels = torch.argmax(self._objects_dc.squeeze(1), dim=1)

            prev_iter = iter - 100
            prev_state = self.prev_state[str(prev_iter)]
            prev_grads = self.prev_grads[str(prev_iter)]
            frozen_labels_prev = torch.argmax(prev_state.squeeze(1), dim=1)

            for class_id in range(num_classes):
                if class_id == 0:
                    continue
                class_mask = (frozen_labels == class_id)
                class_mask_prev = (frozen_labels_prev == class_id)
                class_grads = torch.norm(grads, dim=-1)[class_mask]
                grads_pre = prev_grads[class_mask_prev]


                lambdas_ori = class_mask.sum().item() * (-2e-2)
                lambdas = class_mask.sum().item() * (-2e-3)

                delta_mean = class_grads.mean() - grads_pre.squeeze().mean()
                threshold_ori = self.e - lambdas_ori * torch.abs(delta_mean)
                threshold = self.e - lambdas * class_grads.mean()



                grad_history = torch.norm(class_grads.unsqueeze(1), dim=-1)
                class_selected_mask = torch.where(grad_history >= threshold, True, False)
                class_selected_mask = class_selected_mask & (
                            torch.max(self.get_scaling[class_mask], dim=1).values <= self.percent_dense * scene_extent)

                selected_pts_mask[class_mask] = class_selected_mask



        _, point_ids = self.knn_near_2(self._xyz[None].detach(), self._xyz[None].detach())
        xyz = self._xyz[point_ids[0]].detach()
        rotation_q = self._rotation[point_ids[0]].detach()
        scaling_diag = self.get_scaling[point_ids[0]].detach()

        xyz_0 = xyz[:, 0].reshape(-1, 3)
        rotation_0_q = rotation_q[:, 0].reshape(-1, 4)
        scaling_diag_0 = scaling_diag[:, 0].reshape(-1, 3)

        xyz_1 = xyz[:, 1:].reshape(-1, 3)
        rotation_1_q = rotation_q[:, 1:].reshape(-1, 4)
        scaling_diag_1 = scaling_diag[:, 1:].reshape(-1, 3)

        kl_div = self.kl_div(xyz_0, rotation_0_q, scaling_diag_0, xyz_1, rotation_1_q, scaling_diag_1)
        self.kl_selected_pts_mask = kl_div > kl_threshold

        selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask

        print("[kl clone]: ", (selected_pts_mask).sum().item())
        stds = self.get_scaling[selected_pts_mask]
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask])
        new_rotation = self._rotation[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_objects_dc = self._objects_dc[selected_pts_mask]

        if H is not None:
            self.H = torch.cat((self.H, self.H[selected_pts_mask]), dim=0)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_objects_dc)






    def kl_densify_and_split_semantic(self, grads, grad_threshold, scene_extent, iter, kl_threshold=0.4, N=2, H=None):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()

        if iter == 500:
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling,
                                                            dim=1).values > self.percent_dense * scene_extent)

        else:
            num_classes = self._objects_dc.squeeze().size(1)
            selected_pts_mask = torch.zeros(padded_grad.size(0), dtype=torch.bool, device=grads.device)
            frozen_labels = torch.argmax(self._objects_dc.squeeze(1), dim=1)

            prev_iter = iter - 100
            prev_state = self.prev_state[str(prev_iter)]
            prev_grads = self.prev_grads[str(prev_iter)]
            frozen_labels_prev = torch.argmax(prev_state.squeeze(1), dim=1)

            for class_id in range(num_classes):
                if class_id == 0:
                    continue
                class_mask = (frozen_labels == class_id)
                class_mask_prev = (frozen_labels_prev == class_id)
                class_grads = torch.norm(padded_grad.unsqueeze(1), dim=-1)[class_mask]
                grads_pre = prev_grads[class_mask_prev]


                lambdas_ori = class_mask.sum().item() * (-2e-2)
                lambdas = class_mask.sum().item() * (-2e-3)

                delta_mean = class_grads.mean() - grads_pre.squeeze().mean()
                threshold_ori = self.e - lambdas_ori * torch.abs(delta_mean)
                threshold = self.e - lambdas * class_grads.mean()

                class_selected_mask = torch.where(class_grads >= threshold, True, False)
                class_selected_mask = class_selected_mask & (
                            torch.max(self.get_scaling[class_mask], dim=1).values > self.percent_dense * scene_extent)

                selected_pts_mask[class_mask] = class_selected_mask


        _, point_ids = self.knn_near_2(self._xyz[None].detach(), self._xyz[None].detach())
        xyz = self._xyz[point_ids[0]].detach()
        rotation_q = self._rotation[point_ids[0]].detach()
        scaling_diag = self.get_scaling[point_ids[0]].detach()

        xyz_0 = xyz[:, 0].reshape(-1, 3)
        rotation_0_q = rotation_q[:, 0].reshape(-1, 4)
        scaling_diag_0 = scaling_diag[:, 0].reshape(-1, 3)

        xyz_1 = xyz[:, 1:].reshape(-1, 3)
        rotation_1_q = rotation_q[:, 1:].reshape(-1, 4)
        scaling_diag_1 = scaling_diag[:, 1:].reshape(-1, 3)

        kl_div = self.kl_div(xyz_0, rotation_0_q, scaling_diag_0, xyz_1, rotation_1_q, scaling_diag_1)
        self.kl_selected_pts_mask = kl_div > kl_threshold

        selected_pts_mask = selected_pts_mask & self.kl_selected_pts_mask

        print("[kl split]: ", (selected_pts_mask).sum().item())

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_objects_dc = self._objects_dc[selected_pts_mask].repeat(N, 1, 1)
        if H is not None:
            self.H = torch.cat((self.H, self.H[selected_pts_mask].repeat(N,1)), dim=0)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_objects_dc)


        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)



    def normal_cdf(self, xyz, mean, std_dev):
        standardized_x = (xyz - mean) / (std_dev * torch.sqrt(torch.tensor(2.0, device="cuda")))
        erf_val = torch.erf(standardized_x)
        cdf = 0.5 * ( 1 + erf_val )
        return cdf


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, kl_threshold=0.4, t_vertices=None, iter=None, H=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0



        if H is None:
            self.kl_densify_and_clone_semantic(grads, max_grad, extent, iter)
            self.kl_densify_and_split_semantic(grads, max_grad, extent, iter)
            self.kl_merge(grads, max_grad, extent, 0.1)
        else:
            self.kl_densify_and_clone_semantic(grads, max_grad, extent, iter, H=H)
            self.kl_densify_and_split_semantic(grads, max_grad, extent, iter, H=H)
            self.kl_merge(grads, max_grad, extent, 0.1, H=H)

        self.obj_finetune(iter)


        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        distance, _ = self.knn(t_vertices[None], self._xyz[None].detach())
        distance = distance.view(distance.shape[0], -1)
        threshold = 0.05
        pts_mask = (distance > threshold).squeeze()

        prune_mask = prune_mask | pts_mask


        print('total points num: ', self._xyz.shape[0], 'prune num: ', prune_mask.sum().item())

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


    def kl_div(self, mu_0, rotation_0_q, scaling_0_diag, mu_1, rotation_1_q, scaling_1_diag):

        rotation_0 = build_rotation(rotation_0_q)
        scaling_0 = build_scaling(scaling_0_diag)
        L_0 = rotation_0 @ scaling_0
        cov_0 = L_0 @ L_0.transpose(1, 2)

        rotation_1 = build_rotation(rotation_1_q)
        scaling_1_inv = build_scaling(1 / scaling_1_diag)
        L_1_inv = rotation_1 @ scaling_1_inv
        cov_1_inv = L_1_inv @ L_1_inv.transpose(1, 2)

        mu_diff = mu_1 - mu_0

        kl_div_0 = torch.vmap(torch.trace)(cov_1_inv @ cov_0)
        kl_div_1 = mu_diff[:, None].matmul(cov_1_inv).matmul(mu_diff[..., None]).squeeze()
        kl_div_2 = torch.log(torch.prod((scaling_1_diag / scaling_0_diag) ** 2, dim=1))
        kl_div = 0.5 * (kl_div_0 + kl_div_1 + kl_div_2 - 3)
        return kl_div


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1


    def coarse_deform_c2source(self, query_pts, params, t_params, t_vertices, lbs_weights=None, correct_Rs=None,
                               return_transl=False):
        bs = query_pts.shape[0]
        joints_num = self.SMPL_NEUTRAL['weights'].shape[-1]
        vertices_num = t_vertices.shape[1]
        smpl_pts = t_vertices

        _, vert_ids = self.knn(smpl_pts.float(), query_pts.float())
        if lbs_weights is None:
            bweights = self.SMPL_NEUTRAL['weights'][vert_ids].view(*vert_ids.shape[:2],
                                                                   joints_num)
        else:
            bweights = self.SMPL_NEUTRAL['weights'][vert_ids].view(*vert_ids.shape[:2], joints_num)
            bweights = torch.log(bweights + 1e-9) + lbs_weights
            bweights = F.softmax(bweights, dim=-1)

        big_pose_params = t_params
        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, big_pose_params)
        A = torch.matmul(bweights, A.reshape(bs, joints_num, -1))
        A = torch.reshape(A, (bs, -1, 4, 4))
        query_pts = query_pts - A[..., :3, 3]
        R_inv = torch.inverse(A[..., :3, :3].float())
        query_pts = torch.matmul(R_inv, query_pts[..., None]).squeeze(-1)

        transforms = R_inv

        translation = None
        if return_transl:
            translation = -A[..., :3, 3]
            translation = torch.matmul(R_inv, translation[..., None]).squeeze(-1)

        self.mean_shape = True
        if self.mean_shape:

            posedirs = self.SMPL_NEUTRAL['posedirs'].cuda().float()
            pose_ = big_pose_params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = pose_.shape[0]
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
            pose_offsets = torch.matmul(pose_feature.unsqueeze(1),
                                        posedirs.view(vertices_num * 3, -1).transpose(1, 0).unsqueeze(0)).view(
                batch_size, -1, 3)
            pose_offsets = torch.gather(pose_offsets, 1, vert_ids.expand(-1, -1, 3))
            query_pts = query_pts - pose_offsets

            if return_transl:
                translation -= pose_offsets

            shapedirs = self.SMPL_NEUTRAL['shapedirs'][..., :params['shapes'].shape[-1]]
            shape_offset = torch.matmul(shapedirs.unsqueeze(0),
                                        torch.reshape(params['shapes'].cuda(), (batch_size, 1, -1, 1))).squeeze(-1)
            shape_offset = torch.gather(shape_offset, 1, vert_ids.expand(-1, -1, 3))
            query_pts = query_pts + shape_offset

            if return_transl:
                translation += shape_offset

            posedirs = self.SMPL_NEUTRAL['posedirs']
            pose_ = params['poses']
            ident = torch.eye(3).cuda().float()
            batch_size = pose_.shape[0]
            rot_mats = batch_rodrigues(pose_.view(-1, 3)).view([batch_size, -1, 3, 3])

            if correct_Rs is not None:
                rot_mats_no_root = rot_mats[:, 1:]
                rot_mats_no_root = torch.matmul(rot_mats_no_root.reshape(-1, 3, 3),
                                                correct_Rs.reshape(-1, 3, 3)).reshape(-1, joints_num - 1, 3, 3)
                rot_mats = torch.cat([rot_mats[:, 0:1], rot_mats_no_root], dim=1)

            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
            pose_offsets = torch.matmul(pose_feature.unsqueeze(1),
                                        posedirs.view(vertices_num * 3, -1).transpose(1, 0).unsqueeze(0)).view(
                batch_size, -1, 3)
            pose_offsets = torch.gather(pose_offsets, 1, vert_ids.expand(-1, -1, 3))
            query_pts = query_pts + pose_offsets

            if return_transl:
                translation += pose_offsets

        A, R, Th, joints = get_transform_params_torch(self.SMPL_NEUTRAL, params, rot_mats=rot_mats)

        self.s_A = A
        A = torch.matmul(bweights, self.s_A.reshape(bs, joints_num, -1))
        A = torch.reshape(A, (bs, -1, 4, 4))
        can_pts = torch.matmul(A[..., :3, :3], query_pts[..., None]).squeeze(-1)
        smpl_src_pts = can_pts + A[..., :3, 3]

        transforms = torch.matmul(A[..., :3, :3], transforms)

        if return_transl:
            translation = torch.matmul(A[..., :3, :3], translation[..., None]).squeeze(-1) + A[..., :3, 3]

        R_inv = torch.inverse(R)
        world_src_pts = torch.matmul(smpl_src_pts, R_inv) + Th

        transforms = torch.matmul(R, transforms)

        if return_transl:
            translation = torch.matmul(translation, R_inv).squeeze(-1) + Th



        return smpl_src_pts, world_src_pts, bweights, transforms, translation, smpl_src_pts


    def save_ply_ablation(self, path, idx):
        mkdir_p(os.path.dirname(path))

        idx = np.array(idx.cpu())

        xyz = self._xyz.detach().cpu().numpy()
        total_points = xyz.shape[0]

        yellow_red_cmap = LinearSegmentedColormap.from_list("yellow_red", ["yellow", "red"])

        dist_sq = self.get_opacity.unsqueeze(0).squeeze(-1)
        min_val = 0.0000
        max_val = 1.0000
        dist_sq = torch.clip(dist_sq, min_val, max_val)
        dist_sq = (dist_sq - min_val) / (max_val - min_val)
        dist_sq = dist_sq.unsqueeze(-1)
        dist_sq = dist_sq.repeat(1, 1, 3)
        dist_sq = dist_sq.detach().cpu().numpy()


        dist_sq_normalized = (dist_sq[:, :, 0] * 255).astype(np.uint8)
        dist_sq = yellow_red_cmap(dist_sq_normalized / 255.0)[:, :, :3]


        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        opacities = np.zeros_like(opacities)
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        red = torch.tensor(dist_sq.squeeze())
        f_dc_red = RGB2SH(red.float().cuda()).to('cpu')

        default_normals = np.zeros_like(normals[0])
        default_f_dc = np.zeros_like(f_dc[0])
        default_f_rest = np.zeros_like(f_rest[0])
        default_opacity = np.array([0.0])
        default_scale = np.full_like(scale[0], -10)
        default_rotation = np.zeros_like(rotation[0])
        default_rotation[0] = 1

        attributes = np.empty((total_points, xyz.shape[1] + normals.shape[1] +
                               f_dc.shape[1] + f_rest.shape[1] + opacities.shape[1] +
                               scale.shape[1] + rotation.shape[1]))

        for i in range(total_points):
            if i in idx:
                attributes[i] = np.concatenate((xyz[i], normals[i], f_dc_red[i], f_rest[i],
                                                opacities[i], scale[i], rotation[i]))
            else:
                attributes[i] = np.concatenate((xyz[i].squeeze(), default_normals, default_f_dc, default_f_rest,
                                                default_opacity, default_scale, default_rotation))

        dtype_full = [(attributes, 'f4') for attributes in self.construct_list_of_attributes_wo_obj()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)



    def obj_finetune(self, iteration):
        _, knn_indices = self.knn_near_obj(self._xyz[None].detach(), self._xyz[None].detach())
        obj_dc_neighbors = self._objects_dc[knn_indices[0]].detach()
        neighbor_class_counts = obj_dc_neighbors.sum(dim=1)

        most_common_classes = neighbor_class_counts.argmax(dim=-1)

        updated_obj_dc = F.one_hot(most_common_classes, num_classes=16).float()
        original_obj_dc = self._objects_dc

        num_changed_points = (original_obj_dc.argmax(dim=-1) != updated_obj_dc.argmax(dim=-1)).sum().item()
        print(f'{num_changed_points} object changed')
        self._objects_dc = updated_obj_dc

    def sample_gs(self, iter):
        probs = self.get_opacity.squeeze(-1)
        sampled_idxs = torch.multinomial(probs, 1000, replacement=False)


        if iter == 1000 or iter == 1100 or iter ==1200:
            random_idxs = torch.randint(0, self._xyz.shape[0], (1000,), device=self._xyz.device)
            self.plot_probs_distribution_multi(probs, sampled_idxs, random_idxs)


        return sampled_idxs



    def plot_probs_distribution_multi(self, probs, sampled_idxs1, sampled_idxs2, num_bins=5):
        """
        Plot opacity histograms for all Gaussians and two sampled subsets.
        """
        min_val, max_val = probs.min().item(), probs.max().item()
        bin_edges = torch.linspace(min_val, max_val, steps=num_bins + 1)

        original_hist = torch.histc(probs, bins=num_bins, min=min_val, max=max_val)
        total_original = original_hist.sum().item()

        sampled_probs1 = probs[sampled_idxs1]
        sampled_probs2 = probs[sampled_idxs2]
        sampled_hist1 = torch.histc(sampled_probs1, bins=num_bins, min=min_val, max=max_val)
        sampled_hist2 = torch.histc(sampled_probs2, bins=num_bins, min=min_val, max=max_val)
        total_sampled1 = sampled_hist1.sum().item()
        total_sampled2 = sampled_hist2.sum().item()

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = (bin_edges[1] - bin_edges[0]).item() * 0.25

        plt.figure(figsize=(8, 6))

        plt.bar(bin_centers - width, original_hist.cpu().numpy(), width=width, label='All Gaussians', alpha=1,
                color='blue')
        plt.bar(bin_centers, sampled_hist1.cpu().numpy(), width=width, label='GOS', alpha=1,
                color='green')
        plt.bar(bin_centers + width, sampled_hist2.cpu().numpy(), width=width, label='GRS', alpha=1,
                color='orange')

        for center, count in zip(bin_centers - width, original_hist.cpu().numpy()):
            percentage = count / total_original * 100 if total_original > 0 else 0
            plt.text(center, count + 0.5, f"{percentage:.0f}%", ha='center', va='bottom', fontsize=12, color='blue')

        for center, count in zip(bin_centers, sampled_hist1.cpu().numpy()):
            percentage = count / total_sampled1 * 100 if total_sampled1 > 0 else 0
            plt.text(center, count + 0.5, f"{percentage:.0f}%", ha='center', va='bottom', fontsize=12, color='green')

        for center, count in zip(bin_centers + width, sampled_hist2.cpu().numpy()):
            percentage = count / total_sampled2 * 100 if total_sampled2 > 0 else 0
            plt.text(center, count + 0.5, f"{percentage:.0f}%", ha='center', va='bottom', fontsize=12, color='orange')

        plt.xlabel('Opacity Value', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.title('Gaussian Opacity Histogram', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()
        plt.savefig("opacity_histogram.png", dpi=300, bbox_inches='tight')
        plt.close()


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()

def SMPL_to_tensor(params, device):
    key_ = ['v_template', 'shapedirs', 'J_regressor', 'kintree_table', 'f', 'weights', "posedirs"]
    for key1 in key_:
        if key1 == 'J_regressor':
            if isinstance(params[key1], np.ndarray):
                params[key1] = torch.tensor(params[key1].astype(float), dtype=torch.float32, device=device)
            else:
                params[key1] = torch.tensor(params[key1].toarray().astype(float), dtype=torch.float32, device=device)
        elif key1 == 'kintree_table' or key1 == 'f':
            params[key1] = torch.tensor(np.array(params[key1]).astype(float), dtype=torch.long, device=device)
        else:
            params[key1] = torch.tensor(np.array(params[key1]).astype(float), dtype=torch.float32, device=device)
    return params

def batch_rodrigues_torch(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = torch.norm(poses + 1e-8, p=2, dim=1, keepdim=True)
    rot_dir = poses / angle

    cos = torch.cos(angle)[:, None]
    sin = torch.sin(angle)[:, None]

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), device=poses.device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1)
    K = K.reshape([batch_size, 3, 3])

    ident = torch.eye(3)[None].to(poses.device)
    rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)

    return rot_mat

def get_rigid_transformation_torch(rot_mats, joints, parents):
    """
    rot_mats: bs x 24 x 3 x 3
    joints: bs x 24 x 3
    parents: 24
    """
    bs, joints_num = joints.shape[0:2]
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = torch.cat([rot_mats, rel_joints[..., None]], dim=-1)
    padding = torch.zeros([bs, joints_num, 1, 4], device=rot_mats.device)
    padding[..., 3] = 1
    transforms_mat = torch.cat([transforms_mat, padding], dim=-2)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=1)

    padding = torch.zeros([bs, joints_num, 1], device=rot_mats.device)
    joints_homogen = torch.cat([joints, padding], dim=-1)
    rel_joints = torch.sum(transforms * joints_homogen[:, :, None], dim=3)
    transforms[..., 3] = transforms[..., 3] - rel_joints

    return transforms

def get_transform_params_torch(smpl, params, rot_mats=None, correct_Rs=None):
    """ obtain the transformation parameters for linear blend skinning
    """
    v_template = smpl['v_template']

    shapedirs = smpl['shapedirs']
    betas = params['shapes']
    v_shaped = v_template[None] + torch.sum(shapedirs[None][..., :betas.shape[-1]] * betas[:, None], axis=-1).float()

    if rot_mats is None:
        poses = params['poses'].unsqueeze(0).reshape(-1, 3)
        rot_mats = axis_angle_to_matrix(poses).view(params['poses'].unsqueeze(0).shape[0], -1, 3, 3)

        if correct_Rs is not None:
            rot_mats_no_root = rot_mats[:, 1:]
            rot_mats_no_root = torch.matmul(rot_mats_no_root.reshape(-1, 3, 3), correct_Rs.reshape(-1, 3, 3)).reshape(
                -1, rot_mats.shape[1] - 1, 3, 3)
            rot_mats = torch.cat([rot_mats[:, 0:1], rot_mats_no_root], dim=1)

    joints = torch.matmul(smpl['J_regressor'][None], v_shaped)

    parents = smpl['kintree_table'][0]
    A = get_rigid_transformation_torch(rot_mats, joints, parents)

    R = params['R']
    Th = params['Th']

    return A, R, Th, joints

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """


    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints.unsqueeze(-1), [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return rel_transforms

def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)

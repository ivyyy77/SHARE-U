import torch
import torch.nn as nn
import pytorch3d.transforms as tf

from nets.network_utils import (HierarchicalPoseEncoder,
                                  VanillaCondMLP,
                                  HannwCondMLP,
                                  HashGrid)
from utils.general_utils import quaternion_multiply

class NonRigidDeform(nn.Module):
    def __init__(self):
        super().__init__()
        # self.cfg = cfg

    def forward(self, gaussians, iteration, camera, compute_loss=True):
        raise NotImplementedError

class Identity(NonRigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)

    def forward(self, gaussians, iteration, camera, compute_loss=True):
        return gaussians, {}

class MLP(NonRigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.pose_encoder = HierarchicalPoseEncoder(**cfg.pose_encoder)
        d_cond = self.pose_encoder.n_output_dims

        # add latent code
        self.latent_dim = cfg.get('latent_dim', 0)
        if self.latent_dim > 0:
            d_cond += self.latent_dim
            self.frame_dict = metadata['frame_dict']
            self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)

        d_in = 3
        d_out = 3 + 3 + 4
        self.feature_dim = cfg.get('feature_dim', 0)
        d_out += self.feature_dim

        # output dimension: position + scale + rotation
        self.mlp = VanillaCondMLP(d_in, d_cond, d_out, cfg.mlp)
        self.aabb = metadata['aabb']

        self.delay = cfg.get('delay', 0)


    def forward(self, gaussians, iteration, camera, compute_loss=True):
        if iteration < self.delay:
            deformed_gaussians = gaussians.clone()
            if self.feature_dim > 0:
                setattr(deformed_gaussians, "non_rigid_feature", torch.zeros(gaussians.get_xyz.shape[0], self.feature_dim).cuda())
            return deformed_gaussians, {}

        rots = camera.rots
        Jtrs = camera.Jtrs
        pose_feat = self.pose_encoder(rots, Jtrs)

        if self.latent_dim > 0:
            frame_idx = camera.frame_id
            if frame_idx not in self.frame_dict:
                latent_idx = len(self.frame_dict) - 1
            else:
                latent_idx = self.frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx]).long().to(pose_feat.device)
            latent_code = self.latent(latent_idx)
            latent_code = latent_code.expand(pose_feat.shape[0], -1)
            pose_feat = torch.cat([pose_feat, latent_code], dim=1)

        xyz = gaussians.get_xyz
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        deformed_gaussians = gaussians.clone()
        deltas = self.mlp(xyz_norm, cond=pose_feat)

        delta_xyz = deltas[:, :3]
        delta_scale = deltas[:, 3:6]
        delta_rot = deltas[:, 6:10]

        deformed_gaussians._xyz = gaussians._xyz + delta_xyz

        scale_offset = self.cfg.get('scale_offset', 'logit')
        if scale_offset == 'logit':
            deformed_gaussians._scaling = gaussians._scaling + delta_scale
        elif scale_offset == 'exp':
            deformed_gaussians._scaling = torch.log(torch.clamp_min(gaussians.get_scaling + delta_scale, 1e-6))
        elif scale_offset == 'zero':
            delta_scale = torch.zeros_like(delta_scale)
            deformed_gaussians._scaling = gaussians._scaling
        else:
            raise ValueError

        rot_offset = self.cfg.get('rot_offset', 'add')
        if rot_offset == 'add':
            deformed_gaussians._rotation = gaussians._rotation + delta_rot
        elif rot_offset == 'mult':
            q1 = delta_rot
            q1[:, 0] = 1. # [1,0,0,0] represents identity rotation
            delta_rot = delta_rot[:, 1:]
            q2 = gaussians._rotation
            # deformed_gaussians._rotation = quaternion_multiply(q1, q2)
            deformed_gaussians._rotation = tf.quaternion_multiply(q1, q2)
        else:
            raise ValueError

        if self.feature_dim > 0:
            setattr(deformed_gaussians, "non_rigid_feature", deltas[:, 10:])

        if compute_loss:
            # regularization
            loss_xyz = torch.norm(delta_xyz, p=2, dim=1).mean()
            loss_scale = torch.norm(delta_scale, p=1, dim=1).mean()
            loss_rot = torch.norm(delta_rot, p=1, dim=1).mean()
            loss_reg = {
                'nr_xyz': loss_xyz,
                'nr_scale': loss_scale,
                'nr_rot': loss_rot
            }
        else:
            loss_reg = {}
        return deformed_gaussians, loss_reg


class HannwMLP(NonRigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.pose_encoder = HierarchicalPoseEncoder(**cfg.pose_encoder)
        # output dimension: position + scale + rotation
        self.mlp = HannwCondMLP(3, self.pose_encoder.n_output_dims, 3 + 3 + 4, cfg.mlp, dim_coord=3)
        self.aabb = metadata['aabb']


    def forward(self, gaussians, iteration, camera, compute_loss=True):
        rots = camera.rots
        Jtrs = camera.Jtrs
        pose_feat = self.pose_encoder(rots, Jtrs)

        xyz = gaussians.get_xyz
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        deformed_gaussians = gaussians.clone()
        deltas = self.mlp(xyz_norm, iteration, cond=pose_feat)

        if iteration < self.cfg.mlp.embedder.kick_in_iter:
            deltas = deltas * torch.zeros_like(deltas)

        delta_xyz = deltas[:, :3]
        delta_scale = deltas[:, 3:6]
        delta_rot = deltas[:, -4:]

        deformed_gaussians._xyz = gaussians._xyz + delta_xyz

        scale_offset = self.cfg.get('scale_offset', 'logit')
        if scale_offset == 'logit':
            deformed_gaussians._scaling = gaussians._scaling + delta_scale
        elif scale_offset == 'exp':
            deformed_gaussians._scaling = torch.log(torch.clamp_min(gaussians.get_scaling + delta_scale, 1e-6))
        elif scale_offset == 'zero':
            delta_scale = torch.zeros_like(delta_scale)
            deformed_gaussians._scaling = gaussians._scaling
        else:
            raise ValueError

        rot_offset = self.cfg.get('rot_offset', 'add')
        if rot_offset == 'add':
            deformed_gaussians._rotation = gaussians._rotation + delta_rot
        elif rot_offset == 'mult':
            q1 = delta_rot
            q1[:, 0] = 1.  # [1,0,0,0] represents identity rotation
            delta_rot = delta_rot[:, 1:]
            q2 = gaussians._rotation
            deformed_gaussians._rotation = quaternion_multiply(q1, q2)
        else:
            raise ValueError

        if compute_loss:
            # regularization
            loss_xyz = torch.norm(delta_xyz, p=2, dim=1).mean()
            loss_scale = torch.norm(delta_scale, p=1, dim=1).mean()
            loss_rot = torch.norm(delta_rot, p=1, dim=1).mean()
            loss_reg = {
                'nr_xyz': loss_xyz,
                'nr_scale': loss_scale,
                'nr_rot': loss_rot
            }
        else:
            loss_reg = {}
        return deformed_gaussians, loss_reg

class HashGridwithMLP(NonRigidDeform):
    def __init__(self):
        super().__init__()
        # self.pose_encoder = HierarchicalPoseEncoder()
        # d_cond = self.pose_encoder.n_output_dims
        d_cond = 4 + 45

        # add latent code
        self.latent_dim = 0
        if self.latent_dim > 0:
            d_cond += self.latent_dim
            self.frame_dict = metadata['frame_dict']
            self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)

        # d_out = 3 + 3 + 4
        d_out = 3
        self.d_out = d_out
        self.feature_dim = 0  # 16
        d_out += self.feature_dim

        # self.aabb = metadata['aabb']
        self.hashgrid = HashGrid()

        mlp = {}
        mlp.update({'n_neurons': 64,
        'n_hidden_layers': 3,
        'skip_in': [],
        'cond_in': [],
        # 'cond_in': [ 0 ],
        'multires': 6,  # 0,
        'last_layer_init': False})

        # mlp['n_neurons': 128,
        # 'n_hidden_layers': 3,
        # 'skip_in': [],
        # 'cond_in': [ 0 ],
        # 'multires': 0,
        # 'last_layer_init': False ]

        self.mlp = VanillaCondMLP(self.hashgrid.n_output_dims, d_cond, d_out, mlp)

        self.delay = 100 # 500

    def forward(self, gaussians, iteration, camera, compute_loss=True):
        if iteration < self.delay:
            deformed_gaussians = gaussians.clone()
            if self.feature_dim > 0:
                setattr(deformed_gaussians, "non_rigid_feature",
                        torch.zeros(gaussians.get_xyz.shape[0], self.feature_dim).cuda())
            return deformed_gaussians, {}

        # rots = camera.rots    # 1,24,9
        # Jtrs = camera.Jtrs    # 1,24,3
        # pose_feat = self.pose_encoder(rots, Jtrs)    # 1,144
        n = gaussians.get_xyz.size(0)

        if self.latent_dim > 0:
            frame_idx = camera.frame_id
            if frame_idx not in self.frame_dict:
                latent_idx = len(self.frame_dict) - 1
            else:
                latent_idx = self.frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx]).long().to(pose_feat.device)
            latent_code = self.latent(latent_idx)
            latent_code = latent_code.expand(pose_feat.shape[0], -1)
            pose_feat = torch.cat([pose_feat, latent_code], dim=1)

        xyz = gaussians.get_xyz
        xyz_norm = camera.aabb.normalize(xyz, sym=True)
        deformed_gaussians = gaussians.clone()
        feature_xyz = self.hashgrid(xyz_norm)

        fourier = enhance_high_frequency(feature_xyz)


        feature_gau = torch.cat((gaussians._features_dc.squeeze(1),
                                 gaussians._opacity, gaussians._features_rest.reshape(n, -1)), dim=1)   # N, 49

        deltas, embedding = self.mlp(feature_xyz, xyz, cond=feature_gau)    # N,26
        # deltas = self.mlp(feature_xyz, cond=pose_feat)    # N,26

        delta_xyz = deltas[:, :3]
        deformed_gaussians._xyz = gaussians._xyz + delta_xyz

        if self.d_out > 3:
            delta_scale = deltas[:, 3:6]
            delta_rot = deltas[:, 6:10]

            scale_offset = 'logit'
            if scale_offset == 'logit':
                deformed_gaussians._scaling = gaussians._scaling + delta_scale
            elif scale_offset == 'exp':
                deformed_gaussians._scaling = torch.log(torch.clamp_min(gaussians.get_scaling + delta_scale, 1e-6))
            elif scale_offset == 'zero':
                delta_scale = torch.zeros_like(delta_scale)
                deformed_gaussians._scaling = gaussians._scaling
            else:
                raise ValueError

            rot_offset = 'mult'
            if rot_offset == 'add':
                deformed_gaussians._rotation = gaussians._rotation + delta_rot
            elif rot_offset == 'mult':
                q1 = delta_rot
                q1[:, 0] = 1.  # [1,0,0,0] represents identity rotation
                delta_rot = delta_rot[:, 1:]
                q2 = gaussians._rotation
                # deformed_gaussians._rotation = quaternion_multiply(q1, q2)
                deformed_gaussians._rotation = tf.quaternion_multiply(q1, q2)
            else:
                raise ValueError

        if self.feature_dim > 0:
            setattr(deformed_gaussians, "non_rigid_feature", deltas[:, 10:])

        return deformed_gaussians, embedding

        # if compute_loss:
        #     # regularization
        #     loss_xyz = torch.norm(delta_xyz, p=2, dim=1).mean()
        #     loss_scale = torch.norm(delta_scale, p=1, dim=1).mean()
        #     loss_rot = torch.norm(delta_rot, p=1, dim=1).mean()
        #     loss_reg = {
        #         'nr_xyz': loss_xyz,
        #         'nr_scale': loss_scale,
        #         'nr_rot': loss_rot
        #     }
        # else:
        #     loss_reg = {}
        # return deformed_gaussians, loss_reg

def enhance_high_frequency(voxel_grid, cutoff_ratio=0.2):
    """
    使用3D傅里叶变换提取并增强高频部分。
    """
    # 3D傅里叶变换
    freq_grid = torch.fft.fftn(voxel_grid)
    freq_magnitude = torch.abs(freq_grid)

    # 获取频谱的大小
    center = torch.tensor(freq_grid.shape) // 2
    max_distance = torch.sqrt((center ** 2).sum())

    # 生成频率过滤器（高频增强）
    x, y, z = torch.meshgrid(
        torch.arange(-center[0], center[0]),
        torch.arange(-center[1], center[1]),
        torch.arange(-center[2], center[2])
    )
    distance = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    high_freq_filter = torch.where(distance > cutoff_ratio * max_distance, 1.0, 0.0)

    # 应用过滤器
    enhanced_freq_grid = freq_grid * high_freq_filter

    # 反傅里叶变换回到空间域
    enhanced_voxel_grid = torch.fft.ifftn(enhanced_freq_grid).real
    return enhanced_voxel_grid
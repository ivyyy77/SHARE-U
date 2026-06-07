import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn
from omegaconf import OmegaConf
import math

import torch.fft as fft


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


# def get_embedder(multires, input_dims=3):
def get_embedder(multires, input_dims=1):
    if multires == 0:
        return lambda x: x, input_dims
    assert multires > 0

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


class HannwEmbedder:
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)

        # get hann window weights
        if self.cfg.full_band_iter <= 0 or self.cfg.kick_in_iter >= self.cfg.full_band_iter:
            alpha = torch.tensor(N_freqs, dtype=torch.float32)
        else:
            kick_in_iter = torch.tensor(self.cfg.kick_in_iter,
                                        dtype=torch.float32)
            t = torch.clamp(self.kwargs['iter_val'] - kick_in_iter, min=0.)
            N = self.cfg.full_band_iter - kick_in_iter
            m = N_freqs
            alpha = m * t / N

        for freq_idx, freq in enumerate(freq_bands):
            w = (1. - torch.cos(np.pi * torch.clamp(alpha - freq_idx,
                                                    min=0., max=1.))) / 2.
            # print("freq_idx: ", freq_idx, "weight: ", w, "iteration: ", self.kwargs['iter_val'])
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq, w=w: w * p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_hannw_embedder(cfg, multires, iter_val,):
    embed_kwargs = {
        'include_input': False,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'periodic_fns': [torch.sin, torch.cos],
        'iter_val': iter_val
    }

    embedder_obj = HannwEmbedder(cfg, **embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

class HierarchicalPoseEncoder(nn.Module):
    '''Hierarchical encoder from LEAP.'''

    def __init__(self, num_joints=24, rel_joints=False, dim_per_joint=6, out_dim=-1):
        super().__init__()

        self.num_joints = num_joints
        self.rel_joints = rel_joints
        self.ktree_parents = np.array([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,
            9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21], dtype=np.int32)

        self.layer_0 = nn.Linear(9*num_joints + 3*num_joints, dim_per_joint)
        # self.layer_0 = nn.Linear(3*num_joints, dim_per_joint)
        dim_feat = 13 + dim_per_joint

        layers = []
        for idx in range(num_joints):
            layer = nn.Sequential(nn.Linear(dim_feat, dim_feat), nn.ReLU(), nn.Linear(dim_feat, dim_per_joint))

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

        if out_dim <= 0:
            self.out_layer = nn.Identity()
            self.n_output_dims = num_joints * dim_per_joint
        else:
            self.out_layer = nn.Linear(num_joints * dim_per_joint, out_dim)
            self.n_output_dims = out_dim

    def forward(self, rots, Jtrs, skinning_weight=None):
        batch_size = rots.size(0)

        if self.rel_joints:
            with torch.no_grad():
                Jtrs_rel = Jtrs.clone()
                Jtrs_rel[:, 1:, :] = Jtrs_rel[:, 1:, :] - Jtrs_rel[:, self.ktree_parents[1:], :]
                Jtrs = Jtrs_rel.clone()

        global_feat = torch.cat([rots.view(batch_size, -1), Jtrs.view(batch_size, -1)], dim=-1)
        global_feat = self.layer_0(global_feat)
        # global_feat = (self.layer_0.weight@global_feat[0]+self.layer_0.bias)[None]
        out = [None] * self.num_joints
        for j_idx in range(self.num_joints):
            rot = rots[:, j_idx, :]
            Jtr = Jtrs[:, j_idx, :]
            parent = self.ktree_parents[j_idx]
            if parent == -1:
                bone_l = torch.norm(Jtr, dim=-1, keepdim=True)
                in_feat = torch.cat([rot, Jtr, bone_l, global_feat], dim=-1)
                out[j_idx] = self.layers[j_idx](in_feat)
            else:
                parent_feat = out[parent]
                bone_l = torch.norm(Jtr if self.rel_joints else Jtr - Jtrs[:, parent, :], dim=-1, keepdim=True)
                in_feat = torch.cat([rot, Jtr, bone_l, parent_feat], dim=-1)
                out[j_idx] = self.layers[j_idx](in_feat)

        out = torch.cat(out, dim=-1)
        out = self.out_layer(out)
        return out

class VanillaCondMLP(nn.Module):
    def __init__(self, dim_in, dim_cond, dim_out, config, dim_coord=3):
        super(VanillaCondMLP, self).__init__()

        self.n_input_dims = dim_in
        self.n_output_dims = dim_out

        self.n_neurons, self.n_hidden_layers = config.get('n_neurons'), config.get('n_hidden_layers')

        self.config = config
        dims = [dim_in] + [self.n_neurons for _ in range(self.n_hidden_layers)] + [dim_out]

        self.embed_fn = None
        if config.get('multires') > 0:
            embed_fn, input_ch = get_embedder(config.get('multires'), input_dims=dim_in)
            self.embed_fn = embed_fn
            dims[0] = 64  # input_ch+52   # 676
            # self.embed_linear = nn.Linear(input_ch+52, 64)
            self.embed_linear = nn.Linear(input_ch+52, 64)

        self.last_layer_init = config.get('last_layer_init', False)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            if l + 1 in config.get('skip_in'):
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if l in config.get('cond_in'):
                lin = nn.Linear(dims[l] + dim_cond, out_dim)
            else:
                lin = nn.Linear(dims[l], out_dim)

            if self.last_layer_init and l == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=0., std=1e-5)
                torch.nn.init.constant_(lin.bias, val=0.)


            setattr(self, "lin" + str(l), lin)

        self.activation = nn.LeakyReLU()

    def forward(self, coords, xyz, cond=None):
        if cond is not None:
            cond = cond.expand(coords.shape[0], -1)
            # cond_gau = torch.cat((coords, cond), dim=1)
            cond_gau = torch.cat((xyz, cond), dim=1)

        if self.embed_fn is not None:
            features_embedded = self.embed_fn(cond_gau)      # (input_dim)-> 52 × ( 2 × multires + 1 )
            features_embedded = self.embed_linear(features_embedded)  # 降维
        else:
            features_embedded = coords
        x = features_embedded

        # Store embedded coordinates for output
        self.features_embedded = features_embedded  # 保存嵌入结果

        # x = coords_embedded
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.config.get('cond_in'):
                x = torch.cat([x, cond], 1)

            if l in self.config.get('skip_in'):
                x = torch.cat([x, coords_embedded], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x, self.features_embedded

def get_skinning_mlp(n_input_dims, n_output_dims, config):
    if config.otype == 'VanillaMLP':
        network = VanillaCondMLP(n_input_dims, 0, n_output_dims, config)
    else:
        raise ValueError

    return network


class HannwCondMLP(nn.Module):
    def __init__(self, dim_in, dim_cond, dim_out, config, dim_coord=3):
        super(HannwCondMLP, self).__init__()

        self.n_input_dims = dim_in
        self.n_output_dims = dim_out

        self.n_neurons, self.n_hidden_layers = config.n_neurons, config.n_hidden_layers

        self.config = config
        dims = [dim_in] + [self.n_neurons for _ in range(self.n_hidden_layers)] + [dim_out]

        self.embed_fn = None
        if config.multires > 0:
            _, input_ch = get_hannw_embedder(config.embedder, config.multires, 0)
            dims[0] = input_ch

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            if l + 1 in config.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if l in config.cond_in:
                lin = nn.Linear(dims[l] + dim_cond, out_dim)
            else:
                lin = nn.Linear(dims[l], out_dim)

            if l in config.cond_in:
                # Conditional input layer initialization
                torch.nn.init.constant_(lin.weight[:, -dim_cond:], 0.0)
            torch.nn.init.constant_(lin.bias, 0.0)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.ReLU()

    def forward(self, coords, iteration, cond=None):
        if cond is not None:
            cond = cond.expand(coords.shape[0], -1)

        if self.config.multires > 0:
            embed_fn, _ = get_hannw_embedder(self.config.embedder, self.config.multires, iteration)
            coords_embedded = embed_fn(coords)
        else:
            coords_embedded = coords

        x = coords_embedded
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.config.cond_in:
                x = torch.cat([x, cond], 1)

            if l in self.config.skip_in:
                x = torch.cat([x, coords_embedded], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x

def config_to_primitive(config, resolve=True):
    return OmegaConf.to_container(config, resolve=resolve)

class HashGrid(nn.Module):
    def __init__(self):
        super().__init__()
        config = {}

        config.update({'n_levels': 16,
            'n_features_per_level': 2,
            'n_features_per_level': 2,
            'log2_hashmap_size': 16,
            'base_resolution': 16,
            'per_level_scale': 1.447269237440378, # max reso 4096
            'max_resolution': 2048})

        # config['n_levels':16,
        #     'n_features_per_level': 2,
        #     'n_features_per_level': 2,
        #     'log2_hashmap_size': 16,
        #     'base_resolution': 16,
        #     'per_level_scale': 1.447269237440378, # max reso 4096
        #     'max_resolution': 2048 ]

        xL = config.get('max_resolution', -1)
        if xL > 0:
            # L = config.n_levels
            L = config.get('n_levels', -1)
            x0 = config.get('base_resolution', -1)
            # x0 = config.base_resolution
            # a = config.get('per_level_scale')
            a = float(np.exp(np.log(xL / x0) / (L - 1)))
            config.update({'per_level_scale': a })
        if not isinstance(config, OmegaConf):
            config = OmegaConf.create(config)
        self.encoding = tcnn.Encoding(3, config_to_primitive(config))
        # self.dencoding = tcnn.Encoding(3, config_to_primitive(config))
        self.n_output_dims = self.encoding.n_output_dims
        self.n_input_dims = self.encoding.n_input_dims
        self.grid_size = 64

    def forward(self, x):
        x = (x + 1.) * 0.5 # [-1, 1] => [0, 1]
        hash_features = self.encoding(x)

        # 2. 构造稠密的体素网格（假设输入为稀疏点）
        voxel_grid = self.construct_voxel_grid(hash_features, x)    # 64, 64, 64

        # 3. 应用 3D 傅里叶变换
        frequency_domain = fft.fftn(voxel_grid, dim=(0, 1, 2))  # 3D FFT
        high_freq = self.extract_high_frequency(frequency_domain)

        # 4. 返回增强后的高频特征
        return hash_features + high_freq


    # def forward(self, x):
    #     x = (x + 1.) * 0.5 # [-1, 1] => [0, 1]
    #
    #     return self.encoding(x)

    def construct_voxel_grid(self, hash_features, points):
        # 假设输入为稀疏体素，将其映射到稠密网格
          # 64^3 体素网格
        # voxel_grid = torch.zeros((grid_size, grid_size, grid_size), device=hash_features.device)
        # 创建与 hash_features 相同类型的体素网格
        voxel_grid = torch.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=hash_features.dtype, device=hash_features.device)

        # 填充稠密体素网格（仅举例，需根据实际点的位置填充）
        # 假设 hash_features 包含 (N, C) 特征，其中 N 是点的数量
        indices = (points * (self.grid_size - 1)).long()  # 将归一化的坐标转换为体素网格的索引

        # 避免超出体素网格范围
        indices = torch.clamp(indices, 0, self.grid_size - 1)

        # 遍历每个点并将其特征值赋值到体素网格中
        for i in range(points.shape[0]):
            # 获取当前点的体素网格位置
            x, y, z = indices[i]
            # 将该位置的特征值累加到网格中
            voxel_grid[x, y, z] += hash_features[i].sum()
        voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = hash_features.sum(dim=1)
        return voxel_grid

    def extract_high_frequency(self, frequency_domain):
        # 高频提取：假设低频部分在中心
        threshold = 0.5  # 仅保留高于一定频率的部分
        high_freq = frequency_domain.abs() * (frequency_domain.abs() > threshold)
        results = torch.real(fft.ifftn(high_freq, dim=(0, 1, 2)))  # 逆傅里叶变换
        return results



class AABB(torch.nn.Module):
    def __init__(self, coord_max, coord_min):
        super().__init__()
        self.register_buffer("coord_max", torch.from_numpy(coord_max).float())
        # self.register_buffer("coord_max", torch.from_numpy(coord_max).float())
        # self.register_buffer("coord_min", torch.from_numpy(coord_min).float())
        self.register_buffer("coord_min", torch.from_numpy(coord_min).float())

    def normalize(self, x, sym=False):
        # x = torch.tensor(x).to('cuda')
        x = x.clone().detach().requires_grad_(True)
        x = (x - self.coord_min) / (self.coord_max - self.coord_min)
        if sym:
            x = 2 * x - 1.
        return x

    def unnormalize(self, x, sym=False):
        if sym:
            x = 0.5 * (x + 1)
        x = x * (self.coord_max - self.coord_min) + self.coord_min
        return x

    def clip(self, x):
        return x.clip(min=self.coord_min, max=self.coord_max)

    def volume_scale(self):
        return self.coord_max - self.coord_min

    def scale(self):
        return math.sqrt((self.volume_scale() ** 2).sum() / 3.)



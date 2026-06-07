from torch import nn
import torch
import random
from random import randint
from utils.vmf_sampler import VonMisesFisher
from utils.utils import pairwise_cos_sims, pairwise_l2_dists, log_vmf_norm_const, construct_mlp, vmf_norm_ratio

from utils.vmf_sampler import VonMisesFisher
from torch.distributions.normal import Normal
from torch.distributions.laplace import Laplace
from nets.network_utils import get_embedder


class MCInfoNCE(nn.Module):
    def __init__(self, kappa_init=20, n_samples=16, device=torch.device('cuda:0')):
        super().__init__()

        self.n_samples = n_samples
        self.temperature = 0.2

    def forward(self, mu_ref, kappa_ref, mu_pos, kappa_pos, mu_neg, kappa_neg):
        mu_ref, mu_pos = mu_ref.unsqueeze(1), mu_pos.unsqueeze(1)
        kappa_ref, kappa_pos = kappa_ref.unsqueeze(1), kappa_pos.unsqueeze(1)

        if mu_neg.shape == 2:
            mu_neg, kappa_neg = mu_neg.unqsueeze(1), kappa_neg.unsqueeze(1)

        # Draw samples (new dimension 0 contains the samples)
        samples_ref = VonMisesFisher(mu_ref, kappa_ref).rsample(self.n_samples) # [n_MC, batch, n_pos, dim]
        samples_pos = VonMisesFisher(mu_pos, kappa_pos).rsample(self.n_samples)
        if mu_neg is not None:
            samples_neg = VonMisesFisher(mu_neg, kappa_neg).rsample(self.n_samples)
        else:
            # If we don't get negative samples, treat the next batch sample as negative sample
            samples_neg = torch.roll(samples_pos, 1, 1)

        # calculate the standard log contrastive loss for each vmf sample
        negs = torch.logsumexp(torch.sum(samples_ref * samples_neg, dim=3) * self.kappa - torch.log(torch.ones(1).cuda() * samples_neg.shape[2]), dim=2)
        log_denominator_pos = torch.logsumexp(torch.stack((torch.sum(samples_ref * samples_pos, dim=3).squeeze(2) * self.kappa, negs), dim=0), dim=0)     # 分母
        log_numerator_pos = torch.sum(samples_ref * samples_pos, dim=3) * self.kappa     # 分子
        log_py1_pos = log_numerator_pos - log_denominator_pos.unsqueeze(2)

        # Average over the samples (we actually want a logmeanexp, that's why we substract log(n_samples))
        log_py1_pos = torch.logsumexp(log_py1_pos, dim=0) - torch.log(torch.ones(1, device=self.kappa.device) * self.n_samples)

        # Calculate loss
        loss = torch.mean(log_py1_pos)
        return -loss

    def InfoNCE_loss(self, emb_ref, emb_pos, emb_neg):
        pos_similarity = torch.sum(emb_ref.unsqueeze(1) * emb_pos, dim=-1) / self.temperature  

        emb_neg = torch.roll(emb_neg, shifts=1, dims=0)  # (batch_size, n_neg, dim)

        neg_similarity = torch.sum(emb_ref.unsqueeze(1) * emb_neg, dim=-1) / self.temperature

        log_denominator = torch.logsumexp(torch.cat([pos_similarity, neg_similarity], dim=1), dim=1) 

        log_numerator = pos_similarity

        log_py1 = log_numerator - log_denominator.unsqueeze(1) 

        loss = -log_py1.mean()
        return loss


class ELK(nn.Module):
    def __init__(self, kappa_init=20, device=torch.device('cuda:0')):
        super().__init__()

        self.kappa = torch.nn.Parameter(torch.ones(1, device=device) * kappa_init, requires_grad=True)

    def log_ppk_vmf_vec(self, mu1, kappa1, mu2, kappa2):
        p = mu1.shape[-1]

        kappa3 = torch.linalg.norm(kappa1 * mu1 + kappa2 * mu2, dim=-1).unsqueeze(-1)
        ppk = log_vmf_norm_const(kappa1, p) + log_vmf_norm_const(kappa2, p) - log_vmf_norm_const(kappa3, p)
        ppk = ppk * self.kappa

        return ppk.squeeze(-1)

    def forward(self, mu_ref, kappa_ref, mu_pos, kappa_pos, mu_neg, kappa_neg):
        # mu_neg and mu_pos is of dimension [batch, n_neg, dim]
        # mu_ref is dimension [batch, dim]
        mu_ref = mu_ref.unsqueeze(1)
        kappa_ref = kappa_ref.unsqueeze(1)

        # Calculate similarities
        sim_pos = self.log_ppk_vmf_vec(mu_ref, kappa_ref, mu_pos, kappa_pos)
        if mu_neg is not None:
            sim_neg = self.log_ppk_vmf_vec(mu_ref, kappa_ref, mu_neg, kappa_neg)
        else:
            # If we don't get negative samples, treat the next batch sample as negative sample
            sim_neg = torch.roll(sim_pos, 1, 0)

        # Calculate loss
        loss = torch.mean(sim_pos, dim=1) - torch.logsumexp(torch.cat((sim_pos, sim_neg), dim=1), dim=1)
        loss = -torch.mean(loss)
        return loss


class HedgedInstance(nn.Module):
    def __init__(self, kappa_init=1, b_init=0, n_samples=16, device=torch.device('cuda:0')):
        super().__init__()

        self.n_samples = n_samples
        self.kappa = torch.nn.Parameter(torch.ones(1, device=device) * kappa_init, requires_grad=True) # kappa is "a" in the notation of their paper
        self.b = torch.nn.Parameter(torch.ones(1, device=device) * b_init, requires_grad=True)

    def forward(self, mu_ref, kappa_ref, mu_pos, kappa_pos, mu_neg, kappa_neg):
        mu_ref = mu_ref.unsqueeze(1)
        kappa_ref = kappa_ref.unsqueeze(1)

        # Draw samples (new dimension 0 contains the samples)
        samples_ref = VonMisesFisher(mu_ref, kappa_ref).rsample(self.n_samples) # [n_MC, batch, n_pos, dim]
        samples_pos = VonMisesFisher(mu_pos, kappa_pos).rsample(self.n_samples)
        if mu_neg is not None:
            samples_neg = VonMisesFisher(mu_neg, kappa_neg).rsample(self.n_samples)
        else:
            # If we don't get negative samples, treat the next batch sample as negative sample
            samples_neg = torch.roll(samples_pos, 1, 1)

        # calculate the standard log contrastive loss for each vmf sample
        py1_pos = torch.sigmoid(self.kappa * torch.sum(samples_ref * samples_pos, dim=-1) + self.b)
        py1_neg = torch.sigmoid(self.kappa * torch.sum(samples_ref * samples_neg, dim=-1) + self.b)

        # Average over the samples
        log_py1_pos = torch.mean(torch.log(py1_pos), dim=0)
        log_py0_neg = torch.mean(torch.log(1 - py1_neg), dim=0)

        # Calculate loss
        loss = torch.mean(log_py1_pos) + torch.mean(log_py0_neg) / log_py0_neg.shape[-1]
        return -loss


def smoothness_loss(x, z):
    x_dist = pairwise_l2_dists(x)
    z_dist = 1 - pairwise_cos_sims(z)/ 2

    loss = torch.mean((x_dist - z_dist)**2 * (torch.sqrt(torch.ones(1, device=x.device) * 2) - z_dist.detach())**4)

    return loss


class Encoder(nn.Module):
    def __init__(self, n_hidden=2, dim_x=10, dim_z=2, dim_hidden=64,
                 post_kappa_min=20, post_kappa_max=80, x_samples=None,
                 device=torch.device('cuda:0'), has_joint_backbone=False):
        super().__init__()

        # Save parameters
        self.device = device
        self.post_kappa_min = torch.tensor(post_kappa_min, device=device)
        self.post_kappa_max = torch.tensor(post_kappa_max, device=device)
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.embed_view, self.embed_view_cnl = get_embedder(multires=2)
        # self.embed_view, self.embed_view_cnl = get_embedder(multires=1)
        self.embed_view_cnl=None

        # Create networks
        self.has_joint_backbone = has_joint_backbone
        self.mu_net = construct_mlp(n_hidden=n_hidden, dim_x=dim_x, dim_z=dim_z, dim_hidden=dim_hidden, fourier=self.embed_view_cnl)
        self.mu_net = self.mu_net.to(device)

        # Turn on gradients
        for p in self.mu_net.parameters():
            p.requires_grad = True
        
    def forward(self, x, view=None):
        view_ = view / torch.norm(view, p=2)
        mu_ = self.mu_net(x)
        mu_ = mu_ * view_
        mu = mu_ / torch.norm(mu_, dim=-1).unsqueeze(-1)
        kappa = torch.norm(mu_, dim=-1).unsqueeze(-1)
        
        return mu, kappa, mu_

    def _rescale_kappa(self, x_samples=None, fourier=None):
        # Goal: Find scale and shift parameters to bring the kappas to the desired range
        # indicated by self.post_kappa_min and self.post_kappa_max
        if torch.isinf(self.post_kappa_min) or torch.isinf(self.post_kappa_max):
            self.kappa_upscale = torch.ones(1, device=self.device) * float("inf")
            self.kappa_add = torch.ones(1, device=self.device) * float("inf")
        else:
            if self.post_kappa_max <= self.post_kappa_min:
                raise("post_kappa_max has to be > post_kappa_min.")
            if x_samples is None:
                raise("Please provide x_samples to the encoder to know which region of x we're dealing with.")
            if fourier is not None:
                fourier = torch.rand((x_samples.size(0), self.embed_view_cnl), device=self.device)
                x_samples = torch.cat([x_samples, fourier], dim=-1)
            kappa_samples = torch.log(1 + torch.exp(self.kappa_net(x_samples)))
            sample_min = torch.min(kappa_samples)
            sample_max = torch.max(kappa_samples)

            self.kappa_upscale = (torch.log(self.post_kappa_max) - torch.log(self.post_kappa_min)) / (
                         sample_max - sample_min)
            self.kappa_add = torch.log(self.post_kappa_max) - self.kappa_upscale * sample_max


def smoothen_via_training(gen, print_progress=False):
    # Turn on gradients
    for p in gen.mu_net.parameters():
        p.requires_grad = True
    gen.train()

    # train
    optim = torch.optim.Adam(gen.parameters(), lr=0.01)
    running_loss = 0
    for b in range(5000):
        optim.zero_grad()
        with torch.no_grad():
            x = gen._sample_x(64)
        x.requires_grad = True
        mu, _ = gen(x)
        loss = 0
        loss += smoothness_loss(x, mu)
        running_loss += loss.detach()
        loss.backward()
        optim.step()

        if print_progress and b % 500 == 0:
            if b == 0:
                avg_loss = running_loss
            else:
                avg_loss = running_loss / 500
            print(f'Loss: {avg_loss}')
            running_loss = 0

    # Turn off gradients
    gen.eval()
    for p in gen.mu_net.parameters():
        p.requires_grad = False

    return gen


class Generator(nn.Module):
    def __init__(self, n_hidden=2, dim_x=10, dim_z=2, dim_hidden=32, pos_kappa=10, n_samples=10,
                 post_kappa_min=20, post_kappa_max=80, family="vmf", device=torch.device('cuda:0'), has_joint_backbone=False):
        super().__init__()

        # Save parameters
        self.device = device
        self.post_kappa_min = torch.tensor(post_kappa_min, device=device)
        self.post_kappa_max = torch.tensor(post_kappa_max, device=device)
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.pos_kappa = pos_kappa
        self.family = family

        # For sampling
        self.n_samples = n_samples
        self.denom_const = None  # will be calculated on demand below

        # Prepare negative sampling
        self.p_different_class = None

        # Create networks
        self.has_joint_backbone = has_joint_backbone
        self.mu_net = construct_mlp(n_hidden=n_hidden, dim_x=dim_x, dim_z=dim_z, dim_hidden=dim_hidden)
        self.kappa_net = construct_mlp(n_hidden=n_hidden - 1, dim_x=dim_x if not has_joint_backbone else dim_z, dim_z=1, dim_hidden=dim_hidden)
        self.mu_net = self.mu_net.to(device)
        self.kappa_net = self.kappa_net.to(device)

        # Turn off gradients
        for p in self.mu_net.parameters():
            p.requires_grad = False
        for p in self.kappa_net.parameters():
            p.requires_grad = False

        # Bring the kappa network to the correct range
        self.kappa_upscale = 1.
        self.kappa_add = 0.
        self._rescale_kappa()

        # smoothen_via_training(self)

    def forward(self, x):
        # Return posterior (z-space) means and kappas for a batch of x
        mu = self.mu_net(x)
        mu = mu / torch.norm(mu, dim=-1).unsqueeze(-1)
        kappa = torch.exp(self.kappa_upscale * torch.log(1 + torch.exp(self.kappa_net(x if not self.has_joint_backbone else mu))) + self.kappa_add)
        return mu, kappa

    def _rescale_kappa(self):
        # Goal: Find scale and shift parameters to bring the kappas to the desired range
        # indicated by self.post_kappa_min and self.post_kappa_max
        if torch.isinf(self.post_kappa_min) or torch.isinf(self.post_kappa_max):
            self.kappa_upscale = torch.ones(1, device=self.device) * float("inf")
            self.kappa_add = torch.ones(1, device=self.device) * float("inf")
        else:
            if self.post_kappa_max <= self.post_kappa_min:
                raise("post_kappa_max has to be > post_kappa_min.")
            x_samples = self._sample_x(1000)
            kappa_samples = torch.log(1 + torch.exp(self.kappa_net(x_samples)))
            sample_min = torch.min(kappa_samples)
            sample_max = torch.max(kappa_samples)

            self.kappa_upscale = (torch.log(self.post_kappa_max) - torch.log(self.post_kappa_min)) / (
                         sample_max - sample_min)
            self.kappa_add = torch.log(self.post_kappa_max) - self.kappa_upscale * sample_max

    def sample(self, gau, feature, correlation, frozen_label, n=64, n_neg=16, oversampling_factor=1, same_ref=False):
        pos_ids, neg_ids = self.sample_pos_neg_by_candidates_ambiguity(gau, correlation, n_neg)

        x_ref = torch.stack([feature[i] for i in gau.idx], dim=0).to('cuda')
        H_ref = torch.stack([gau.H[i] for i in gau.idx], dim=0).to('cuda')
        x_pos = torch.stack([feature[i] for i in pos_ids], dim=0).to('cuda')
        H_pos = torch.stack([gau.H[i] for i in pos_ids], dim=0).to('cuda')

        x_neg = torch.stack([feature[i] for i in neg_ids], dim=0).to('cuda')
        H_neg = torch.stack([gau.H[i] for i in neg_ids], dim=0).to('cuda')

        return x_ref, x_pos, x_neg, H_ref, H_pos, H_neg, pos_ids, neg_ids

    def sample_pos_neg_by_candidates(self, gau, frozen_label, correlation, n_neg):
        ref_ids = gau.idx
        batchsize = ref_ids.shape[0]

        ref_class = torch.argmax(torch.stack([gau._objects_dc[i] for i in ref_ids], dim=0).squeeze(1), dim=1).to('cuda')

        pos_class = torch.tensor([correlation[c] for c in ref_class], device='cuda')

        probs = gau.get_opacity.squeeze(-1)  

        mask = (frozen_label.unsqueeze(0) == pos_class.unsqueeze(1)).any(dim=2)

        pos_probs = torch.where(mask, probs, torch.zeros_like(probs))  
        sampled_idx_pos = torch.multinomial(pos_probs, num_samples=1).squeeze(1)  

        neg_probs = torch.where(~mask, probs, torch.zeros_like(probs))  # (batchsize, N)
        sampled_idx_neg = torch.multinomial(neg_probs, num_samples=n_neg, replacement=False) 

        return sampled_idx_pos, sampled_idx_neg

    def sample_pos_neg_by_candidates_ambiguity(self, gau, correlation, n_neg):
        ref_ids = gau.idx
        
        ref_class = torch.stack([gau.semantic_prob[i] for i in ref_ids], dim=0).to('cuda').squeeze(1).tolist()
        pos_class = torch.tensor([[correlation[cls] for cls in ref_class]], device="cuda").squeeze()

        probs = gau.get_opacity.squeeze(-1)
        semantic_prob = gau.semantic_prob  

        masks = (semantic_prob.unsqueeze(0) == pos_class.unsqueeze(1)).any(dim=2)

        pos_probs = probs.unsqueeze(0) * masks 
        sampled_global_idx_pos = torch.multinomial(pos_probs, num_samples=1).squeeze(1) 

        neg_probs = probs.unsqueeze(0) * (~masks) 
        sampled_global_idx_neg = torch.multinomial(neg_probs, num_samples=n_neg, replacement=False) 

        return sampled_global_idx_pos, sampled_global_idx_neg

    def _sample_pos_by_candidates(self, gau, correlation):

        ref_ids = gau.idx
        batchsize = ref_ids.shape[0]
        id_partner = torch.zeros(batchsize, device=self.device).long()
        needs_partner = torch.ones(batchsize, dtype=torch.uint8, device=self.device)
        every_class = torch.multinomial(gau.semantic_prob, num_samples=1)
        while torch.any(needs_partner):
            # Draw a class that we assume the reference belongs to
            ref_class = torch.multinomial(gau.semantic_prob[ref_ids], num_samples=1).squeeze(1)

            pos_class.append([correlation[ref_class[i].item()] for i in range(len(ref_class))])
            pos_class = torch.tensor(pos_class).squeeze()

            # See if we can find positive matches in that class
            is_ref_and_cand_pos = torch.bernoulli(gau.semantic_prob[:, pos_class].t())
            p_select_bigger0 = is_ref_and_cand_pos + (torch.sum(is_ref_and_cand_pos, dim=1) == 0).unsqueeze(1)
            chosen_idxes = torch.multinomial(p_select_bigger0, num_samples=1, replacement=False)

            n_matches = torch.sum(is_ref_and_cand_pos, dim=1)
            id_partner[torch.logical_and(needs_partner, n_matches > 0)] = chosen_idxes[torch.logical_and(needs_partner, n_matches > 0), 0]
            needs_partner[torch.logical_and(needs_partner, n_matches > 0)] = False

        return id_partner


    def _sample_neg(self, gau, n_neg):
        if self.p_different_class is None:
            # First time, calculate it:
            self.p_different_class = 1 - torch.matmul(self.plabels, self.plabels.t())

        ref_ids = gau.idx
        batchsize = ref_ids.shape[0]

        # Generate candidates until each z_ref has a sample
        partner_ids = torch.zeros((batchsize, n_neg), device=self.device)
        needs_partner = torch.ones((batchsize, n_neg), dtype=torch.uint8, device=self.device)
        while torch.any(needs_partner):
            # Limit ourselves to those samples that need partners (for efficiency)
            requires_partner = torch.any(needs_partner, dim=1)

            # Sample whether other samples are neg to the ref
            is_ref_and_cand_wanted = torch.bernoulli(self.p_different_class[ref_ids[requires_partner]])
            is_ref_and_cand_wanted = is_ref_and_cand_wanted.type(torch.uint8)
            # Choose samples
            # in is_ref_and_cand_wanted we might have rows with full 0. This crashes torch.multinomial.
            # In case we have no 1, give everything a one and then filter out everything again afterwards
            p_select_bigger0 = is_ref_and_cand_wanted.float() + (torch.sum(is_ref_and_cand_wanted, dim=1) == 0).unsqueeze(1)
            chosen_idxes = torch.multinomial(p_select_bigger0, n_neg, replacement=False)

            # Choose the actual matches for each ref sample:
            for sub_idx, overall_idx in enumerate(requires_partner.nonzero()[:, 0]):
                # sub_idx is the index with respect to those that require a partner (the first that requires a partner, the second, ...)
                # overall_idx is the general idx of those samples (e.g., 8, 17, 52, ...)
                # The chosen_idx will probably contain samples with probability 0, because we forced it to sample n things,
                # even if there were less than n possible 1s in the array.
                n_matches = torch.sum(is_ref_and_cand_wanted[sub_idx])
                n_needed = torch.sum(needs_partner[overall_idx, :])
                n_new_samples = torch.min(n_matches, n_needed).type(torch.int)
                if n_new_samples > 0:
                    # One trick we can use is that the prob-0 choices are always at the end
                    chosen_idx = chosen_idxes[sub_idx, :n_new_samples]
                    partner_ids[overall_idx, n_neg - n_needed:(n_neg - n_needed + n_new_samples)] = chosen_idx
                    needs_partner[overall_idx, n_neg - n_needed:(n_neg - n_needed + n_new_samples)] = False

        # The dataloader expects int on cpu
        partner_ids = partner_ids.cpu().type(torch.uint8)
        return partner_ids


    def _sample_pos_by_candidates_(self, gau, frozen_label, correlation):
        ref_ids = gau.idx
        batchsize = ref_ids.shape[0]
        id_partner = torch.zeros(batchsize, device=self.device).long()
        needs_partner = torch.ones(batchsize, dtype=torch.uint8, device=self.device)
        pos_class = []
        # while torch.any(needs_partner):
        # Draw a class that we assume the reference belongs to
        # ref_class = torch.multinomial(self.plabels[ref_ids], num_samples=1).squeeze(1)
        ref_class = torch.stack([gau._objects_dc[i] for i in ref_ids], dim=0).to('cuda')
        ref_class = torch.argmax(ref_class.squeeze(1), dim=1).tolist()
        pos_class.append([correlation[ref_class[i]] for i in range(len(ref_class))])
        pos_class = torch.tensor(pos_class).squeeze()

        probs = gau.get_opacity.squeeze(-1)

        mask = torch.zeros_like(probs, dtype=torch.bool, device='cuda')

        # Generate a probability mask for sampling
        for i in range(batchsize):

            pos_indices = pos_class[i]
            mask = (frozen_label.unsqueeze(1) == (pos_indices.to('cuda')).clone().detach()).any(dim=1)
            pos_probs = probs[mask]
            sampled_idx = torch.multinomial(pos_probs, num_samples=1)

            global_indices = torch.nonzero(mask).squeeze(1)
            sampled_global_idx = global_indices[sampled_idx]
            id_partner[i] = sampled_global_idx

        return id_partner, ref_class

    def _sample_x(self, n):
        # idx = torch.multinomial()
        return torch.rand((n, self.dim_x), device=self.device)

    def _sample_z_from_x(self, x):
        # Takes a batch of x, encodes their posteriors and draws from them
        mu, kappa = self.forward(x)
        if self.family == "vmf":
            z_distrs = VonMisesFisher(mu, kappa)
        elif self.family == "Gaussian":
            z_distrs = Normal(mu, 1/torch.sqrt(kappa))
        elif self.family == "Laplace":
            z_distrs = Laplace(mu, 1/kappa)
        z_samples = z_distrs.sample()
        z_samples = torch.nn.functional.normalize(z_samples, dim=-1)
        return z_samples

    def _sample_pos_neg_by_candidates_(self, z_ref, n_neg=1, oversampling_factor=1):
        # Sample x-candidates, encode them into z and try to find pos/neg matches to the reference points
        # Works if the area that z_pos covers inside the whole z space is relatively high.
        # z_ref - [batchsize, x_dim] batch of reference points
        # oversampling_factor - integer, how many candidates to generate to select x_pos and x_neg from.
        #                       Use a value as high as possible, otherwise need to resample

        x_pos, z_pos = self._sample_candidates(z_ref, n=1, want_pos=True, oversampling_factor=oversampling_factor)
        if n_neg > 0:
            x_neg, z_neg = self._sample_candidates(z_ref, n=n_neg, want_pos=False, oversampling_factor=oversampling_factor)
        else:
            x_neg = None
            z_neg = None

        return x_pos, x_neg, z_pos, z_neg

    def _sample_candidates(self, z_ref, n=1, want_pos=True, oversampling_factor=1):
        batchsize = z_ref.shape[0]

        # Generate candidates until each z_ref has a sample
        x_partner = torch.zeros((batchsize, n, self.dim_x), device=self.device)
        z_partner = torch.zeros((batchsize, n, self.dim_z), device=self.device)
        needs_partner = torch.ones((batchsize, n), dtype=torch.uint8, device=self.device)
        while torch.any(needs_partner):
            requires_partner = torch.any(needs_partner, dim=1)
            n_require_partner = torch.sum(requires_partner)
            x_cand = self._sample_x(n_require_partner * n * oversampling_factor)
            z_cand = self._sample_z_from_x(x_cand)

            # sample whether the candidates are pos/neg to the ref
            # Each x_ref has its own candidates
            cand_per_ref = z_cand.reshape(n_require_partner, n*oversampling_factor, z_cand.shape[-1])
            prob_ref_and_cand_pos = self._pos_prob(z_ref[requires_partner].unsqueeze(1), cand_per_ref)
            is_ref_and_cand_pos = torch.bernoulli(prob_ref_and_cand_pos)
            is_ref_and_cand_pos = is_ref_and_cand_pos.type(torch.uint8)
            is_ref_and_cand_wanted = is_ref_and_cand_pos == want_pos

            # Choose samples
            # in is_ref_and_cand_wanted we might have rows with full 0. This crashes torch.multinomial.
            # In case we have no 1, give everything a one and then filter out everything again afterwards
            p_select_bigger0 = is_ref_and_cand_wanted.float() + (torch.sum(is_ref_and_cand_wanted, dim=1) == 0).unsqueeze(1)
            chosen_idxes = torch.multinomial(p_select_bigger0, n, replacement=False)
            # Currently, chosen_idxes indices the columns per row.
            # We want to get back to the original indexing of the flattened x_cand and z_cand tensors:
            chosen_idxes = chosen_idxes + torch.arange(n_require_partner, device=chosen_idxes.device).unsqueeze(1) * n * oversampling_factor

            if n > 1:
                # If we need several samples, we need to fill in the tensor sample by sample, because we might have
                # a different amount of valid candidates per sample and cannot tensorize this indexing
                for sub_idx, overall_idx in enumerate(requires_partner.nonzero()[:,0]):
                    # sub_idx is the index with respect to those that require a partner (the first that requires a partner, the second, ...)
                    # overall_idx is the general idx of those samples (e.g., 8, 17, 52, ...)
                    # The chosen_idx will probably contain samples with probability 0, because we forced it to sample n things,
                    # even if there were less than n possible 1s in the array.
                    n_matches = torch.sum(is_ref_and_cand_wanted[sub_idx])
                    n_needed = torch.sum(needs_partner[overall_idx, :])
                    n_new_samples = torch.min(n_matches, n_needed).type(torch.int)
                    if n_new_samples > 0:
                        # One trick we can use is that the prob-0 choices are always at the end
                        chosen_idx = chosen_idxes[sub_idx,:n_new_samples]
                        x_partner[overall_idx, n - n_needed:(n - n_needed + n_new_samples)] = x_cand[chosen_idx, :]
                        z_partner[overall_idx, n - n_needed:(n - n_needed + n_new_samples)] = z_cand[chosen_idx, :]
                        needs_partner[overall_idx, n - n_needed:(n - n_needed + n_new_samples)] = False
            elif n == 1:
                # We can speed up the indexing by tensorizing it
                n_matches = torch.sum(is_ref_and_cand_wanted, dim=1)
                x_partner[requires_partner.nonzero()[n_matches > 0, 0], 0] = x_cand[chosen_idxes[n_matches > 0, 0], :]
                z_partner[requires_partner.nonzero()[n_matches > 0, 0], 0] = z_cand[chosen_idxes[n_matches > 0, 0], :]
                needs_partner[requires_partner.nonzero()[n_matches > 0, 0], 0] = False

        return x_partner, z_partner

    def _pos_prob(self, z1, z2):
        # Returns P(Y = 1|z_1, z_2) based on the P(z_2|Y=1, z_1) pos-vMF distribution
        # and the uniform distribution for negative samples
        # Input:
        #  z_1 - [batchsize_1, z_dim] tensor containing rowwise normalized zs
        #  z_2 - [batchsize_2, z_dim] tensor containing rowwise normalized zs
        # Output:
        #  [batchsize_1, batchsize_2] tensor containing probabilities P(Y=1) in [0, 1]

        # Calculate these constants here and not in the class init, because not all strategies need them
        if self.denom_const is None:
            self.denom_const = torch.tensor(vmf_norm_ratio(self.pos_kappa, self.dim_z), device=self.device)

        cos = torch.sum(z1 * z2, dim=-1)
        log_pos_dens = self.pos_kappa * cos
        log_neg_dens = self.denom_const

        return torch.exp(log_pos_dens - torch.logsumexp(
            torch.stack((log_pos_dens, log_neg_dens * torch.ones(log_pos_dens.shape, device=self.device)), dim=0), dim=0))


def smoothness_loss(x, z):
    x_dist = pairwise_l2_dists(x)
    z_dist = 1 - pairwise_cos_sims(z)/ 2

    loss = torch.mean((x_dist - z_dist)**2 * (torch.sqrt(torch.ones(1, device=x.device) * 2) - z_dist.detach())**4)

    return loss


class GTnet(nn.Module):
    def __init__(self, res_pos=3, res_view=10, num_hidden=3, width=64, pos_delta=False, num_moments=4):
        super().__init__()
        self.pos_delta = pos_delta
        self.num_moments = num_moments

        self.embed_pos, self.embed_pos_cnl = get_embedder(res_pos, 3)
        self.embed_view, self.embed_view_cnl = get_embedder(res_view, 3)
        in_cnl = self.embed_pos_cnl + self.embed_view_cnl + 7  # 7 for scales and rotations

        hiddens = [nn.Linear(width, width) if i % 2 == 0 else nn.ReLU()
                   for i in range((num_hidden - 1) * 2)]

        self.linears = nn.Sequential(
            nn.Linear(in_cnl, width),
            nn.ReLU(),
            *hiddens,
        ).to("cuda")
        if not pos_delta:  # Defocus
            self.s = nn.Linear(width, 3).to("cuda")
            self.r = nn.Linear(width, 4).to("cuda")
        else:  # Motion
            self.s = nn.Linear(width, 3 * (num_moments + 1)).to("cuda")
            self.r = nn.Linear(width, 4 * (num_moments + 1)).to("cuda")
            self.p = nn.Linear(width, 3 * num_moments).to("cuda")

        self.linears.apply(init_linear_weights)
        self.s.apply(init_linear_weights)
        self.r.apply(init_linear_weights)
        if pos_delta:
            self.p.apply(init_linear_weights)

    def forward(self, pos, scales, rotations, viewdirs):
        pos_delta = None
        pos = self.embed_pos(pos)
        viewdirs = self.embed_view(viewdirs)

        x = torch.cat([pos, viewdirs, scales, rotations], dim=-1)
        x1 = self.linears(x)

        scales_delta = self.s(x1)
        rotations_delta = self.r(x1)

        if self.pos_delta:
            pos_delta = self.p(x1)

        return scales_delta, rotations_delta, pos_delta



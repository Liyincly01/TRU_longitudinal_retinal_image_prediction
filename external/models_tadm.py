# SPDX-License-Identifier: GPL-3.0-only
"""2D FAF adaptation of TADM (Litrico et al., MICCAI 2024); see LICENSE-TADM."""

import math
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

class Mish(nn.Module):

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Block(nn.Module):

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        if groups == 0:
            self.block = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim_out, 3), Mish())
        else:
            self.block = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim_out, 3), nn.GroupNorm(groups, dim_out), Mish())

    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):

    def __init__(self, dim, dim_out, *, time_emb_dim=0, ages_emb_dim=64, groups=8):
        super().__init__()
        if time_emb_dim > 0:
            self.mlp = nn.Sequential(Mish(), nn.Linear(time_emb_dim, dim_out))
        else:
            self.mlp = None
        if ages_emb_dim > 0:
            self.mlp_diff_ages = nn.Sequential(Mish(), nn.Linear(ages_emb_dim, dim_out))
        else:
            self.mlp_diff_ages = None
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, diff_ages_embed=None):
        h = self.block1(x)
        if time_emb is not None and self.mlp is not None:
            h = h + self.mlp(time_emb)[:, :, None, None]
        if diff_ages_embed is not None and self.mlp_diff_ages is not None:
            h = h + self.mlp_diff_ages(diff_ages_embed.float())[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class Rezero(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g

class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        (b, c, h, w) = x.shape
        qkv = self.to_qkv(x)
        qkv = qkv.view(b, 3, self.heads, -1, h * w)
        (q, k, v) = (qkv[:, 0], qkv[:, 1], qkv[:, 2])
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = out.reshape(b, -1, h, w)
        return self.to_out(out)

class Downsample(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, 2))

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class ConditionEncoder(nn.Module):

    def __init__(self, in_channels=1, base_dim=64, num_blocks=4):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, base_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2)
        blocks = []
        for i in range(num_blocks):
            blocks.append(nn.Sequential(nn.Conv2d(base_dim, base_dim, 3, 1, 1), nn.LeakyReLU(0.2), nn.Conv2d(base_dim, base_dim, 3, 1, 1)))
        self.blocks = nn.ModuleList(blocks)
        self.trunk_conv = nn.Conv2d(base_dim, base_dim, 3, 1, 1)

    def forward(self, x, get_fea=True):
        feas = []
        fea = self.conv_first(x)
        fea_first = fea
        for block in self.blocks:
            fea = block(fea) * 0.2 + fea
            feas.append(fea)
        trunk = self.trunk_conv(fea)
        fea = fea_first + trunk
        feas.append(fea)
        if get_fea:
            return (x, feas)
        return x

class TADMUnet(nn.Module):

    def __init__(self, dim=64, out_dim=1, dim_mults=(1, 2, 4, 8), cond_dim=64):
        super().__init__()
        dims = [2, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        groups = 0
        n_cond_fea = 3
        self.cond_proj = nn.Conv2d(cond_dim * n_cond_fea, dim, 1)
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), Mish(), nn.Linear(dim * 4, dim))
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for (ind, (dim_in, dim_out)) in enumerate(in_out):
            is_last = ind >= num_resolutions - 1
            self.downs.append(nn.ModuleList([ResnetBlock(dim_in, dim_out, time_emb_dim=dim, ages_emb_dim=dim, groups=groups), ResnetBlock(dim_out, dim_out, time_emb_dim=dim, ages_emb_dim=dim, groups=groups), Downsample(dim_out) if not is_last else nn.Identity()]))
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, ages_emb_dim=dim, groups=groups)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, ages_emb_dim=dim, groups=groups)
        for (ind, (dim_in, dim_out)) in enumerate(reversed(in_out[1:])):
            is_last = ind >= num_resolutions - 1
            self.ups.append(nn.ModuleList([ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim, ages_emb_dim=dim, groups=groups), ResnetBlock(dim_in, dim_in, time_emb_dim=dim, ages_emb_dim=dim, groups=groups), Upsample(dim_in) if not is_last else nn.Identity()]))
        self.final_conv = nn.Sequential(Block(dim, dim, groups=groups), nn.Conv2d(dim, out_dim, 1))
        self.up_proj = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(1, dim, 3))

    def forward(self, x, time, cond_feas, img_baseline, diff_ages):
        t = self.time_pos_emb(time)
        t = self.mlp(t)
        diff_ages_embed = self.time_pos_emb(diff_ages)
        selected_feas = [cond_feas[1], cond_feas[3], cond_feas[4]]
        cond = self.cond_proj(torch.cat(selected_feas, dim=1))
        x = torch.cat([x, img_baseline], dim=1)
        h = []
        for (i, (resnet, resnet2, downsample)) in enumerate(self.downs):
            x = resnet(x, t, diff_ages_embed=diff_ages_embed)
            x = resnet2(x, t, diff_ages_embed=diff_ages_embed)
            if i == 0:
                x = x + cond
                x = x + self.up_proj(img_baseline)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t, diff_ages_embed=diff_ages_embed)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, diff_ages_embed=diff_ages_embed)
        for (resnet, resnet2, upsample) in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t, diff_ages_embed=diff_ages_embed)
            x = resnet2(x, t, diff_ages_embed=diff_ages_embed)
            x = upsample(x)
        return self.final_conv(x)

def extract(a, t, x_shape):
    (b, *_) = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *(1,) * (len(x_shape) - 1))

class TADMDiffusion(nn.Module):

    def __init__(self, denoise_fn, cond_net, timesteps=1000, loss_type='l1', beta_schedule='linear', beta_end=0.02, res_rescale=2.0):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.cond_net = cond_net
        self.loss_type = loss_type
        self.res_rescale = res_rescale
        if beta_schedule == 'linear':
            betas = np.linspace(0.0001, beta_end, timesteps, dtype=np.float64)
            betas[-1] = 0.999
        elif beta_schedule == 'cosine':
            steps = timesteps + 1
            x = np.linspace(0, steps, steps)
            alphas_cumprod = np.cos((x / steps + 0.008) / 1.008 * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = np.clip(betas, a_min=0, a_max=0.999)
        else:
            raise ValueError(f'Unknown beta_schedule: {beta_schedule}')
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.num_timesteps = int(timesteps)
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)))

    def img2res(self, x, baseline):
        return (x - baseline) * self.res_rescale

    def res2img(self, res, baseline):
        return res / self.res_rescale + baseline

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        t_cond = (t[:, None, None, None] >= 0).float()
        t_clamp = t.clamp_min(0)
        return (extract(self.sqrt_alphas_cumprod, t_clamp, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t_clamp, x_start.shape) * noise) * t_cond + x_start * (1 - t_cond)

    def predict_start_from_noise(self, x_t, t, noise):
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return (posterior_mean, posterior_variance, posterior_log_variance_clipped)

    def p_mean_variance(self, x, t, noise_pred, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)
        if clip_denoised:
            x_recon.clamp_(-2, 2)
        (model_mean, posterior_variance, posterior_log_variance) = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return (model_mean, posterior_variance, posterior_log_variance, x_recon)

    def forward(self, img_target, img_baseline, pixel_mask, diff_ages, t=None):
        x = img_target
        (b, *_, device) = (*x.shape, x.device)
        if t is None:
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        else:
            t = torch.LongTensor([t]).repeat(b).to(device)
        self.cond_net.eval()
        with torch.no_grad():
            (_, cond_feas) = self.cond_net(img_baseline, get_fea=True)
        residual = self.img2res(x, img_baseline)
        noise = torch.randn_like(residual)
        x_t = self.q_sample(x_start=residual, t=t, noise=noise)
        noise_pred = self.denoise_fn(x_t, t, cond_feas, img_baseline, diff_ages)
        if self.loss_type == 'l1':
            loss = (noise - noise_pred).abs()
        elif self.loss_type == 'l2':
            loss = (noise - noise_pred) ** 2
        else:
            raise NotImplementedError(self.loss_type)
        if pixel_mask is not None:
            loss = loss * pixel_mask
            num_valid = pixel_mask.sum().clamp(min=1)
            loss = loss.sum() / num_valid
        else:
            loss = loss.mean()
        return {'total': loss, 'noise_mse': loss.detach()}

    @torch.no_grad()
    def p_sample(self, x, t, cond_feas, img_baseline, diff_ages, clip_denoised=True):
        (b, *_, device) = (*x.shape, x.device)
        noise_pred = self.denoise_fn(x, t, cond_feas, img_baseline, diff_ages)
        (model_mean, _, model_log_variance, x0_pred) = self.p_mean_variance(x=x, t=t, noise_pred=noise_pred, clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *(1,) * (len(x.shape) - 1))
        return (model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0_pred)

    @torch.no_grad()
    def sample(self, img_baseline, diff_ages, num_steps=None):
        device = self.betas.device
        b = img_baseline.shape[0]
        shape = img_baseline.shape
        (_, cond_feas) = self.cond_net(img_baseline, get_fea=True)
        img = torch.randn(shape, device=device)
        steps = num_steps or self.num_timesteps
        if steps < self.num_timesteps:
            step_indices = np.linspace(0, self.num_timesteps - 1, steps, dtype=int)
        else:
            step_indices = list(range(self.num_timesteps))
        for i in tqdm(reversed(step_indices), desc='DDPM sampling', total=len(step_indices)):
            t_batch = torch.full((b,), i, device=device, dtype=torch.long)
            (img, x_recon) = self.p_sample(img, t_batch, cond_feas, img_baseline, diff_ages)
        img = self.res2img(img, img_baseline)
        return img.clamp(-1, 1)

    @torch.no_grad()
    def sample_ddim(self, img_baseline, diff_ages, num_steps=50, eta=0.0):
        device = self.betas.device
        b = img_baseline.shape[0]
        shape = img_baseline.shape
        (_, cond_feas) = self.cond_net(img_baseline, get_fea=True)
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(len(timesteps))), desc='DDIM sampling', total=len(timesteps)):
            t = timesteps[i]
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            noise_pred = self.denoise_fn(img, t_batch, cond_feas, img_baseline, diff_ages)
            alpha_bar_t = self.alphas_cumprod[t]
            alpha_bar_prev = self.alphas_cumprod[timesteps[i - 1]] if i > 0 else torch.tensor(1.0)
            x0_pred = (img - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            x0_pred = x0_pred.clamp(-2, 2)
            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * noise_pred
            noise = torch.randn_like(img) if i > 0 else 0
            img = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise
        img = self.res2img(img, img_baseline)
        return img.clamp(-1, 1)

    @torch.no_grad()
    def x0hat_predict(self, img_baseline, diff_ages, t_infer=200):
        device = self.betas.device
        b = img_baseline.shape[0]
        (_, cond_feas) = self.cond_net(img_baseline, get_fea=True)
        noise = torch.randn_like(img_baseline)
        t_batch = torch.full((b,), t_infer, device=device, dtype=torch.long)
        residual_start = torch.zeros_like(img_baseline)
        x_t = self.q_sample(x_start=residual_start, t=t_batch, noise=noise)
        noise_pred = self.denoise_fn(x_t, t_batch, cond_feas, img_baseline, diff_ages)
        x0_pred = self.predict_start_from_noise(x_t, t_batch, noise_pred)
        x0_pred = x0_pred.clamp(-2, 2)
        img = self.res2img(x0_pred, img_baseline)
        return img.clamp(-1, 1)

"""
2D-BrLP model components.

Adapted from BrLP (Puglisi et al., MIA 2025) for 2D retinal FAF prediction.
Components:
  1. VAE2D: KL-regularized 2D autoencoder (256×256 → 32×32×4 latent)
  2. DiffusionUNet2D: U-Net for latent diffusion with cross-attention Δt conditioning
  3. ControlNet2D: Subject-specific conditioning from baseline latent + Δt spatial map
  4. DDIMScheduler: Deterministic reverse-process sampler
  5. latent_average_stabilization: Average m independent denoised latents
  6. PatchDiscriminator: For VAE adversarial training
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class GroupNormSiLU(nn.Module):
    def __init__(self, channels, groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(min(groups, channels), channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(x))


class ResBlock2D(nn.Module):
    """ResNet block with optional time-embedding conditioning."""
    def __init__(self, in_ch, out_ch, emb_dim=None, groups=32):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(min(groups, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.emb_proj = (nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch * 2))
                         if emb_dim else None)

    def forward(self, x, emb=None):
        h = self.block1(x)
        if self.emb_proj is not None and emb is not None:
            scale, shift = self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
            h = h * (1 + scale) + shift
        h = self.block2(h)
        return h + self.res_conv(x)


class Downsample2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class SelfAttention2D(nn.Module):
    """Multi-head self-attention for spatial feature maps."""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.einsum("bhdn,bhem->bhnm", q, k) * (C // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class CrossAttention2D(nn.Module):
    """Cross-attention: spatial features attend to a context vector."""
    def __init__(self, channels, context_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.kv = nn.Linear(context_dim, channels * 2)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x, context):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(B, self.num_heads, self.head_dim, H * W)

        kv = self.kv(context)
        kv = kv.reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 4, 1)
        k, v = kv[0], kv[1]

        attn = torch.einsum("bhdn,bhds->bhns", q, k) * self.head_dim ** -0.5
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhns,bhds->bhdn", attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ContinuousTimeEmbedding(nn.Module):
    """Sinusoidal log(1+Δt) encoding — matches TRU pipeline."""
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, delta):
        delta_log = torch.log1p(delta)
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0, -math.log(100), half, device=delta.device))
        args = delta_log.unsqueeze(-1) * freqs
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


# =============================================================================
# 2D VAE (KL-AUTOENCODER)
# =============================================================================

class VAE2D(nn.Module):
    """2D KL-Autoencoder: 256×256×1 → 32×32×4 latent (8× spatial reduction)."""

    def __init__(self, in_channels=1, latent_channels=4,
                 num_channels=(64, 128, 256, 256), num_res_blocks=2):
        super().__init__()
        self.latent_channels = latent_channels

        enc_layers = [nn.Conv2d(in_channels, num_channels[0], 3, padding=1)]
        ch = num_channels[0]
        for i, out_ch in enumerate(num_channels):
            for _ in range(num_res_blocks):
                enc_layers.append(ResBlock2D(ch, out_ch))
                ch = out_ch
            if i < len(num_channels) - 1:
                enc_layers.append(Downsample2D(ch))
        enc_layers.extend([
            nn.GroupNorm(32, ch), nn.SiLU(),
            nn.Conv2d(ch, latent_channels * 2, 3, padding=1),
        ])
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = [nn.Conv2d(latent_channels, num_channels[-1], 3, padding=1)]
        ch = num_channels[-1]
        for i, out_ch in enumerate(reversed(num_channels)):
            for _ in range(num_res_blocks):
                dec_layers.append(ResBlock2D(ch, out_ch))
                ch = out_ch
            if i < len(num_channels) - 1:
                dec_layers.append(Upsample2D(ch))
        dec_layers.extend([
            nn.GroupNorm(32, ch), nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, padding=1),
        ])
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = h.chunk(2, dim=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar

    @torch.no_grad()
    def encode_to_latent(self, x):
        mean, _ = self.encode(x)
        return mean


# =============================================================================
# 2D DIFFUSION U-NET
# =============================================================================

class UNetLevel(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, num_res_blocks=2,
                 use_attention=False, use_cross_attn=False, context_dim=None):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.cross_attns = nn.ModuleList()
        ch = in_ch
        for _ in range(num_res_blocks):
            self.blocks.append(ResBlock2D(ch, out_ch, emb_dim))
            self.attns.append(SelfAttention2D(out_ch) if use_attention else nn.Identity())
            self.cross_attns.append(
                CrossAttention2D(out_ch, context_dim) if use_cross_attn else None
            )
            ch = out_ch

    def forward(self, x, emb, context=None):
        for block, attn, cross_attn in zip(self.blocks, self.attns, self.cross_attns):
            x = block(x, emb)
            x = attn(x)
            if cross_attn is not None and context is not None:
                x = cross_attn(x, context)
        return x


class DiffusionUNet2D(nn.Module):
    """2D Diffusion U-Net operating on (32×32×4) latents.
    3 resolution levels: 32→16→8. Channels (256, 512, 768). Attention at levels 2-3.
    """

    def __init__(self, in_channels=4, out_channels=4,
                 num_channels=(256, 512, 768), num_res_blocks=2,
                 attention_levels=(False, True, True),
                 context_dim=256):
        super().__init__()
        self.in_channels = in_channels
        time_dim = num_channels[0] * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(num_channels[0]),
            nn.Linear(num_channels[0], time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, num_channels[0], 3, padding=1)

        self.down_levels = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        ch = num_channels[0]
        for i, (out_ch, use_attn) in enumerate(zip(num_channels, attention_levels)):
            self.down_levels.append(
                UNetLevel(ch, out_ch, time_dim, num_res_blocks, use_attn,
                          use_cross_attn=use_attn, context_dim=context_dim)
            )
            ch = out_ch
            if i < len(num_channels) - 1:
                self.downsamplers.append(Downsample2D(ch))
            else:
                self.downsamplers.append(nn.Identity())

        self.mid = nn.ModuleList([
            ResBlock2D(ch, ch, time_dim),
            SelfAttention2D(ch),
            CrossAttention2D(ch, context_dim),
            ResBlock2D(ch, ch, time_dim),
        ])

        self.up_levels = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for i, (out_ch, use_attn) in enumerate(
            zip(reversed(num_channels), reversed(attention_levels))
        ):
            self.up_levels.append(
                UNetLevel(ch + out_ch, out_ch, time_dim, num_res_blocks, use_attn,
                          use_cross_attn=use_attn, context_dim=context_dim)
            )
            ch = out_ch
            if i < len(num_channels) - 1:
                self.upsamplers.append(Upsample2D(ch))
            else:
                self.upsamplers.append(nn.Identity())

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch), nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x, t, context, controlnet_residuals=None):
        emb = self.time_mlp(t)
        h = self.input_conv(x)

        skips = []
        for i, (level, down) in enumerate(zip(self.down_levels, self.downsamplers)):
            h = level(h, emb, context)
            if controlnet_residuals is not None and i < len(controlnet_residuals.get("down", [])):
                h = h + controlnet_residuals["down"][i]
            skips.append(h)
            if i < len(self.down_levels) - 1:
                h = down(h)

        h = self.mid[0](h, emb)
        h = self.mid[1](h)
        h = self.mid[2](h, context)
        h = self.mid[3](h, emb)
        if controlnet_residuals is not None and "mid" in controlnet_residuals:
            h = h + controlnet_residuals["mid"]

        for i, (level, up) in enumerate(zip(self.up_levels, self.upsamplers)):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = level(h, emb, context)
            if i < len(self.up_levels) - 1:
                h = up(h)

        return self.out(h)


# =============================================================================
# CONTROLNET 2D
# =============================================================================

class ControlNet2D(nn.Module):
    """ControlNet mirror of the U-Net encoder. Spatial condition is
    concat(baseline_latent, Δt_spatial_map); residuals are added into the
    U-Net encoder via zero-initialized projections.
    """

    def __init__(self, in_channels=4, cond_channels=5,
                 num_channels=(256, 512, 768), num_res_blocks=2,
                 attention_levels=(False, True, True),
                 context_dim=256):
        super().__init__()
        time_dim = num_channels[0] * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(num_channels[0]),
            nn.Linear(num_channels[0], time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, num_channels[0], 3, padding=1)
        self.cond_embed = nn.Sequential(
            nn.Conv2d(cond_channels, num_channels[0], 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(num_channels[0], num_channels[0], 3, padding=1),
        )

        self.down_levels = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.zero_convs = nn.ModuleList()
        ch = num_channels[0]
        for i, (out_ch, use_attn) in enumerate(zip(num_channels, attention_levels)):
            self.down_levels.append(
                UNetLevel(ch, out_ch, time_dim, num_res_blocks, use_attn,
                          use_cross_attn=use_attn, context_dim=context_dim)
            )
            ch = out_ch
            self.zero_convs.append(nn.Conv2d(ch, ch, 1))
            nn.init.zeros_(self.zero_convs[-1].weight)
            nn.init.zeros_(self.zero_convs[-1].bias)
            if i < len(num_channels) - 1:
                self.downsamplers.append(Downsample2D(ch))
            else:
                self.downsamplers.append(nn.Identity())

        self.mid = nn.ModuleList([
            ResBlock2D(ch, ch, time_dim),
            SelfAttention2D(ch),
            CrossAttention2D(ch, context_dim),
            ResBlock2D(ch, ch, time_dim),
        ])
        self.mid_zero_conv = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.mid_zero_conv.weight)
        nn.init.zeros_(self.mid_zero_conv.bias)

    def forward(self, x, t, context, controlnet_cond):
        emb = self.time_mlp(t)
        h = self.input_conv(x) + self.cond_embed(controlnet_cond)

        down_residuals = []
        for i, (level, down, zero_conv) in enumerate(
            zip(self.down_levels, self.downsamplers, self.zero_convs)
        ):
            h = level(h, emb, context)
            down_residuals.append(zero_conv(h))
            if i < len(self.down_levels) - 1:
                h = down(h)

        h = self.mid[0](h, emb)
        h = self.mid[1](h)
        h = self.mid[2](h, context)
        h = self.mid[3](h, emb)
        mid_residual = self.mid_zero_conv(h)

        return {"down": down_residuals, "mid": mid_residual}


# =============================================================================
# DDIM SCHEDULER
# =============================================================================

class DDIMScheduler:
    """DDIM scheduler with scaled-linear or linear beta schedule."""

    def __init__(self, num_train_timesteps=1000,
                 beta_start=0.0015, beta_end=0.0205,
                 schedule="scaled_linear"):
        self.num_train_timesteps = num_train_timesteps

        if schedule == "scaled_linear":
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5,
                                   num_train_timesteps) ** 2
        else:
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)

        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def add_noise(self, x0, noise, timesteps):
        acp = self.alphas_cumprod.to(x0.device)
        sqrt_acp = acp[timesteps].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_acp = (1 - acp[timesteps]).sqrt().view(-1, 1, 1, 1)
        return sqrt_acp * x0 + sqrt_one_minus_acp * noise

    def get_timesteps(self, num_inference_steps):
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = torch.arange(0, num_inference_steps) * step_ratio
        return timesteps.flip(0)

    def step(self, noise_pred, t, x_t, t_prev=None):
        acp = self.alphas_cumprod.to(x_t.device)
        alpha_t = acp[t]
        alpha_prev = (acp[t_prev] if t_prev is not None and t_prev >= 0
                      else torch.tensor(1.0, device=x_t.device))

        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()

        x0_pred = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        dir_xt = (1 - alpha_prev).sqrt() * noise_pred
        x_prev = alpha_prev.sqrt() * x0_pred + dir_xt
        return x_prev, x0_pred


@torch.no_grad()
def latent_average_stabilization(
    unet, controlnet, scheduler, controlnet_cond, context,
    latent_shape, device, num_inference_steps=50, m=5,
):
    """Run m independent DDIM reverse processes in parallel, average the latents."""
    cond_rep = controlnet_cond.repeat(m, 1, 1, 1)
    ctx_rep = context.repeat(m, 1, 1)

    z = torch.randn(m, *latent_shape, device=device)
    timesteps = scheduler.get_timesteps(num_inference_steps).to(device)

    for i, t in enumerate(timesteps):
        t_batch = torch.full((m,), t.item(), device=device, dtype=torch.long)
        t_prev = timesteps[i + 1].item() if i + 1 < len(timesteps) else 0
        cn_residuals = controlnet(z, t_batch, ctx_rep, cond_rep)
        noise_pred = unet(z, t_batch, ctx_rep, cn_residuals)
        z, _ = scheduler.step(noise_pred, t.item(), z, t_prev)

    return z.mean(dim=0, keepdim=True)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for VAE adversarial training."""
    def __init__(self, in_channels=1, num_channels=32, num_layers=3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, num_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        ]
        ch = num_channels
        for i in range(1, num_layers):
            out_ch = min(ch * 2, 256)
            layers.extend([
                nn.Conv2d(ch, out_ch, 4, stride=2, padding=1),
                nn.GroupNorm(min(32, out_ch), out_ch),
                nn.LeakyReLU(0.2),
            ])
            ch = out_ch
        layers.append(nn.Conv2d(ch, 1, 4, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

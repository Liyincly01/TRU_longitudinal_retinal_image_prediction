"""
Five-configuration training/inference wrappers (paper Table 3).

All four classes wrap the shared `MultiScaleTemporalUNet_v13` backbone so
that parameter count and compute are identical. The configurations differ
only in:

  • which image gets noised during training (I* vs I_N),
  • the noise schedule (nonlinear DDPM αᵗ vs linear αᵗ = 1 − t),
  • the training-target parameterization (v-prediction vs x̂₀-prediction),
  • the inference initialization.

Paper-native ↔ class name mapping:
  - Std-DDIM (50-step, 1-step)   →  StdVPredDiffusion
  - IA-Nonlinear                 →  IANonlinearDiffusion
  - IA-Linear                    →  IALinearDiffusion
  - TRU                          →  TRUWrapper
"""

import torch
import torch.nn as nn

from .backbone import extract_last_frame


# =============================================================================
# TRU — no noise, direct regression
# =============================================================================

class TRUWrapper(nn.Module):
    """Temporal Residual U-Net.

    Training: clean I_N → model → x̂₀; loss = masked MSE(x̂₀, I*).
    Inference: same single forward pass.

    The `t_fixed` argument (default 200) is passed through the shared
    backbone's SinusoidalPosEmb unchanged on every call. Because t is
    constant, the t-branch of each ResBlock FiLM collapses to a learned
    constant bias — functionally equivalent to a single-pair FiLM on
    delta. The argument exists only so that the backbone signature can
    be shared with the diffusion variants without architectural forks.
    """

    def __init__(self, model, t_fixed=200, device="cuda"):
        super().__init__()
        self.model = model
        self.t_fixed = t_fixed
        self.device = device

    def forward(self, x0_true, pixel_mask, history, deltas, temporal_mask):
        B = x0_true.shape[0]
        last_frame = extract_last_frame(history, temporal_mask)
        t_batch = torch.full((B,), self.t_fixed, device=self.device, dtype=torch.float)
        x0_pred = self.model(last_frame, history, deltas, temporal_mask, t_batch)

        sq = (x0_pred - x0_true) ** 2 * pixel_mask
        n_valid = pixel_mask.sum(dim=[1, 2, 3], keepdim=True).clamp(min=1)
        return (sq.sum(dim=[1, 2, 3], keepdim=True) / n_valid).mean()

    @torch.no_grad()
    def predict(self, history, deltas, temporal_mask):
        self.model.eval()
        B = history.shape[0]
        last_frame = extract_last_frame(history, temporal_mask)
        t_batch = torch.full((B,), self.t_fixed, device=self.device, dtype=torch.float)
        x0_pred = self.model(last_frame, history, deltas, temporal_mask, t_batch)
        self.model.train()
        return torch.clamp(x0_pred, -1, 1)


# =============================================================================
# IA-Nonlinear — DDPM α_bar schedule, x̂₀-predict, noise I_N
# =============================================================================

class IANonlinearDiffusion(nn.Module):
    """IA on a standard DDPM (cosine-ish via linear-β) noise schedule.

    Training: x_t = √ᾱₜ·I_N + √(1−ᾱₜ)·ε, with t uniform in [t_min, t_max).
    Loss: masked MSE between the model's direct x̂₀ output and I*.
    Inference: single forward pass with x_t computed the same way at t=t_infer.
    """

    def __init__(self, model, t_min=0, t_max=1000, n_steps=1000,
                 beta_start=1e-4, beta_end=0.02, device="cuda"):
        super().__init__()
        self.model = model
        self.t_min = t_min
        self.t_max = t_max
        self.device = device

        beta = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alpha_bar = torch.cumprod(1.0 - beta, dim=0)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    def forward(self, x0_true, pixel_mask, history, deltas, temporal_mask):
        B = x0_true.shape[0]
        t = torch.randint(self.t_min, self.t_max, (B,), device=self.device)
        last_frame = extract_last_frame(history, temporal_mask)
        noise = torch.randn_like(x0_true)
        sqrt_ab = self.sqrt_alpha_bar[t].view(B, 1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t].view(B, 1, 1, 1)
        x_t = sqrt_ab * last_frame + sqrt_1mab * noise
        x0_pred = self.model(x_t, history, deltas, temporal_mask, t.float())

        sq = (x0_pred - x0_true) ** 2 * pixel_mask
        n_valid = pixel_mask.sum(dim=[1, 2, 3], keepdim=True).clamp(min=1)
        return (sq.sum(dim=[1, 2, 3], keepdim=True) / n_valid).mean()

    @torch.no_grad()
    def x0hat_predict(self, history, deltas, temporal_mask, t_infer=200, seed=None):
        self.model.eval()
        B = history.shape[0]
        H, W = history.shape[-2:]
        last_frame = extract_last_frame(history, temporal_mask)
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        noise = torch.randn(B, 1, H, W, device=self.device)
        x_t = (self.sqrt_alpha_bar[t_infer] * last_frame
               + self.sqrt_one_minus_alpha_bar[t_infer] * noise)
        t_batch = torch.full((B,), t_infer, device=self.device, dtype=torch.float)
        x0_pred = self.model(x_t, history, deltas, temporal_mask, t_batch)
        self.model.train()
        return torch.clamp(x0_pred, -1, 1)


# =============================================================================
# IA-Linear — linear interpolation schedule, x̂₀-predict, noise I_N
# =============================================================================

class IALinearDiffusion(nn.Module):
    """IA on a linear interpolation schedule: x_t = (1−t)·I_N + t·ε, t ∈ [0,1].

    This is NOT flow matching — the model directly predicts x̂₀ in a
    single forward pass; there is no velocity field, no ODE solver.
    The schedule is simply a linear mix rather than the nonlinear
    α_bar schedule used by IA-Nonlinear.

    The continuous t is rescaled to [0, 999] before being fed to the
    backbone's SinusoidalPosEmb for compatibility.
    """

    def __init__(self, model, t_min=0.05, t_max=0.95, t_infer=0.20, device="cuda"):
        super().__init__()
        self.model = model
        self.t_min = t_min
        self.t_max = t_max
        self.t_infer = t_infer
        self.device = device

    def forward(self, x0_true, pixel_mask, history, deltas, temporal_mask):
        B = x0_true.shape[0]
        t = torch.rand(B, device=self.device) * (self.t_max - self.t_min) + self.t_min
        last_frame = extract_last_frame(history, temporal_mask)
        noise = torch.randn_like(x0_true)
        t_v = t.view(B, 1, 1, 1)
        x_t = (1.0 - t_v) * last_frame + t_v * noise
        t_embed = t * 999.0
        x0_pred = self.model(x_t, history, deltas, temporal_mask, t_embed)

        sq = (x0_pred - x0_true) ** 2 * pixel_mask
        n_valid = pixel_mask.sum(dim=[1, 2, 3], keepdim=True).clamp(min=1)
        return (sq.sum(dim=[1, 2, 3], keepdim=True) / n_valid).mean()

    @torch.no_grad()
    def x0hat_predict(self, history, deltas, temporal_mask, t_infer=None, seed=None):
        if t_infer is None:
            t_infer = self.t_infer
        if isinstance(t_infer, (int, float)) and t_infer > 1.0:
            t_infer = t_infer / 999.0
        self.model.eval()
        B = history.shape[0]
        H, W = history.shape[-2:]
        last_frame = extract_last_frame(history, temporal_mask)
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        noise = torch.randn(B, 1, H, W, device=self.device)
        x_noisy = (1.0 - t_infer) * last_frame + t_infer * noise
        t_batch = torch.full((B,), t_infer * 999.0, device=self.device, dtype=torch.float)
        x0_pred = self.model(x_noisy, history, deltas, temporal_mask, t_batch)
        self.model.train()
        return torch.clamp(x0_pred, -1, 1)


# =============================================================================
# Std-DDIM — noises TARGET (I*), v-predict, supports 50-step and 1-step inference
# =============================================================================

class StdVPredDiffusion(nn.Module):
    """Standard conditional DDIM diffusion with v-parameterization.

    Training noises the TARGET I*, not I_N — the key distributional
    mismatch that IA is designed to fix. Loss is MSE on the v-target
    `v = √ᾱ·ε − √(1−ᾱ)·I*`.

    Inference provides two entry points used in the paper:
      • `ddim_sample(n_steps=50)`: full 50-step DDIM from pure Gaussian noise.
      • `x0hat_predict(t_infer=200)`: 1-step x̂₀ readout from I_N-anchored noise.
    The 1-step path is included because it is the apples-to-apples
    single-forward-pass comparison against IA-Nonlinear / IA-Linear / TRU.
    """

    def __init__(self, model, n_steps=1000, beta_start=1e-4, beta_end=0.02,
                 device="cuda"):
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.device = device

        beta = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alpha_bar = torch.cumprod(1.0 - beta, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    def forward(self, x0_true, pixel_mask, history, deltas, temporal_mask):
        B = x0_true.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=self.device)
        noise = torch.randn_like(x0_true)
        sqrt_ab = self.sqrt_alpha_bar[t].view(B, 1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t].view(B, 1, 1, 1)
        x_t = sqrt_ab * x0_true + sqrt_1mab * noise
        v_true = sqrt_ab * noise - sqrt_1mab * x0_true
        v_pred = self.model(x_t, history, deltas, temporal_mask, t.float())

        sq = (v_pred - v_true) ** 2 * pixel_mask
        n_valid = pixel_mask.sum(dim=[1, 2, 3], keepdim=True).clamp(min=1)
        return (sq.sum(dim=[1, 2, 3], keepdim=True) / n_valid).mean()

    def _reshape(self, s, like):
        if s.dim() == 0:
            return s.view(1, 1, 1, 1)
        return s.view(-1, 1, 1, 1)

    def _v_to_x0(self, v_pred, x_t, t_idx):
        sqrt_ab = self._reshape(self.sqrt_alpha_bar[t_idx], x_t)
        sqrt_1mab = self._reshape(self.sqrt_one_minus_alpha_bar[t_idx], x_t)
        return sqrt_ab * x_t - sqrt_1mab * v_pred

    def _v_to_eps(self, v_pred, x_t, t_idx):
        sqrt_ab = self._reshape(self.sqrt_alpha_bar[t_idx], x_t)
        sqrt_1mab = self._reshape(self.sqrt_one_minus_alpha_bar[t_idx], x_t)
        return sqrt_1mab * x_t + sqrt_ab * v_pred

    @torch.no_grad()
    def ddim_sample(self, history, deltas, temporal_mask, n_steps=50, seed=None):
        """Full DDIM sampling from pure Gaussian noise (Std-DDIM-50step)."""
        self.model.eval()
        B = history.shape[0]
        H, W = history.shape[-2:]
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        x_t = torch.randn(B, 1, H, W, device=self.device)
        step_size = self.n_steps // n_steps
        timesteps = list(range(self.n_steps - 1, -1, -step_size))
        for i, t_now in enumerate(timesteps):
            t_batch = torch.full((B,), t_now, device=self.device, dtype=torch.float)
            v_pred = self.model(x_t, history, deltas, temporal_mask, t_batch)
            x0_pred = self._v_to_x0(v_pred, x_t, t_now).clamp(-1, 1)
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                eps_pred = self._v_to_eps(v_pred, x_t, t_now)
                x_t = (self.sqrt_alpha_bar[t_next] * x0_pred
                       + self.sqrt_one_minus_alpha_bar[t_next] * eps_pred)
            else:
                x_t = x0_pred
        self.model.train()
        return torch.clamp(x_t, -1, 1)

    @torch.no_grad()
    def x0hat_predict(self, history, deltas, temporal_mask, t_infer=200, seed=None):
        """1-step x̂₀ readout from I_N-anchored noise at t_infer (Std-DDIM-1step).

        Note: training noised I*, but inference starts from I_N + noise. This
        asymmetry is the distributional mismatch the paper diagnoses.
        """
        self.model.eval()
        B = history.shape[0]
        H, W = history.shape[-2:]
        last_frame = extract_last_frame(history, temporal_mask)
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        noise = torch.randn(B, 1, H, W, device=self.device)
        x_t = (self.sqrt_alpha_bar[t_infer] * last_frame
               + self.sqrt_one_minus_alpha_bar[t_infer] * noise)
        t_batch = torch.full((B,), t_infer, device=self.device, dtype=torch.float)
        v_pred = self.model(x_t, history, deltas, temporal_mask, t_batch)
        x0_pred = self._v_to_x0(v_pred, x_t, t_infer)
        self.model.train()
        return torch.clamp(x0_pred, -1, 1)


# =============================================================================
# Factory
# =============================================================================

METHOD_CHOICES = ("tru", "ia_nonlinear", "ia_linear", "std_ddim")


def build_wrapper(method, model, args, device):
    """Build the appropriate wrapper for the requested method.

    `args` is a parsed argparse.Namespace that may carry method-specific
    hyperparameters (t_min, t_max, t_fixed, t_infer, fm_t_min, ...).
    """
    if method == "tru":
        return TRUWrapper(
            model, t_fixed=getattr(args, "t_fixed", 200), device=str(device)
        ).to(device)
    if method == "ia_nonlinear":
        return IANonlinearDiffusion(
            model,
            t_min=getattr(args, "t_min", 0),
            t_max=getattr(args, "t_max", 1000),
            device=str(device),
        ).to(device)
    if method == "ia_linear":
        return IALinearDiffusion(
            model,
            t_min=getattr(args, "ia_linear_t_min", 0.05),
            t_max=getattr(args, "ia_linear_t_max", 0.95),
            t_infer=getattr(args, "ia_linear_t_infer", 0.20),
            device=str(device),
        ).to(device)
    if method == "std_ddim":
        return StdVPredDiffusion(model, device=str(device)).to(device)
    raise ValueError(f"Unknown method: {method!r} (choices: {METHOD_CHOICES})")

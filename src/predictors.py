import hashlib
from pathlib import Path

import numpy as np
import torch

from .backbone import MultiScaleTemporalUNet_v13
from .wrappers import IALinearDiffusion, IANonlinearDiffusion, StdVPredDiffusion, TRUWrapper


def eye_seed(seed, eye_id):
    token = hashlib.sha256(f"{seed}:{eye_id}".encode()).digest()
    return int.from_bytes(token[:4], "little")


def tensors(history, times, target_time, max_history, device):
    if max_history < 1 or not len(history) or len(history) != len(times):
        raise ValueError("History limit must be positive and images must match visit times.")
    images = list(history[-max_history:])
    deltas = list(target_time - np.asarray(times[-max_history:]))
    padding = max_history - len(images)
    shape = images[0].shape
    images = [np.zeros(shape, np.float32)] * padding + images
    deltas = [100.] * padding + deltas
    return (torch.from_numpy(np.stack(images)[None, :, None] * 2 - 1).float().to(device),
            torch.tensor([deltas], dtype=torch.float32, device=device),
            torch.tensor([[True] * padding + [False] * (max_history - padding)], device=device))


def checkpoint(path):
    if not path or not Path(path).is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = state.get("args", {})
    return state, config if isinstance(config, dict) else vars(config)


def load_core(path, method, device, max_history=None):
    state, config = checkpoint(path)
    expected = {"TRU": "tru", "IA_Nonlinear": "ia_nonlinear", "IA_Linear": "ia_linear",
                "Std_DDIM_1step": "std_ddim", "Std_DDIM_50step": "std_ddim"}[method]
    if config.get("method", expected) != expected:
        raise ValueError(f"Checkpoint method does not match {method}.")
    training_max_history = int(config.get("max_history", 6))
    max_history = training_max_history if max_history is None else max_history
    if max_history < 1:
        raise ValueError("max_history must be positive.")
    model = MultiScaleTemporalUNet_v13(max_history=max_history, history_dropout=0).to(device)
    model.load_state_dict(state["ema_model"], strict=True)
    model.eval()
    if method == "TRU":
        wrapper = TRUWrapper(model, t_fixed=config.get("t_fixed", 200), device=str(device))
    elif method == "IA_Nonlinear":
        wrapper = IANonlinearDiffusion(model, device=str(device))
    elif method == "IA_Linear":
        wrapper = IALinearDiffusion(model, device=str(device))
    else:
        wrapper = StdVPredDiffusion(model, device=str(device))

    @torch.inference_mode()
    def predict(history, times, target_time, seed):
        h, d, mask = tensors(history, times, target_time, max_history, device)
        if method == "TRU":
            output = wrapper.predict(h, d, mask)
        elif method == "Std_DDIM_50step":
            output = wrapper.ddim_sample(h, d, mask, n_steps=50, seed=seed)
        else:
            t = (config.get("ia_linear_t_infer", 0.20) if method == "IA_Linear"
                 else int(config.get("t_infer", 200)))
            output = wrapper.x0hat_predict(h, d, mask, t_infer=t, seed=seed)
        return output[0, 0].cpu().numpy() / 2 + 0.5
    predict.max_history = max_history
    predict.training_max_history = training_max_history
    return predict


def linear_spline(history, times, target_time, seed=None):
    if len(history) < 2:
        return history[-1].copy()
    interval = times[-1] - times[-2]
    if interval <= 0:
        raise ValueError("Nonpositive history interval.")
    factor = (target_time - times[-1]) / interval
    return np.clip(history[-1] + factor * (history[-1] - history[-2]), 0, 1)


def load_brlp(path, device, steps=50, samples=5):
    from external.models_brlp import (VAE2D, DiffusionUNet2D, ControlNet2D,
                                     ContinuousTimeEmbedding, DDIMScheduler,
                                     latent_average_stabilization)
    state, config = checkpoint(path)
    channels = int(config.get("latent_channels", 4))
    context_dim = int(config.get("context_dim", 256))
    vae = VAE2D(latent_channels=channels).to(device)
    unet = DiffusionUNet2D(in_channels=channels, out_channels=channels, context_dim=context_dim).to(device)
    control = ControlNet2D(in_channels=channels, cond_channels=channels + 1,
                          context_dim=context_dim).to(device)
    embedding = ContinuousTimeEmbedding(dim=context_dim).to(device)
    for model, key in [(vae, "vae"), (unet, "ema_unet"), (control, "ema_controlnet"),
                       (embedding, "ema_delta_embed")]:
        model.load_state_dict(state[key], strict=True)
        model.eval()
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.0015,
                              beta_end=0.0205, schedule="scaled_linear")

    @torch.inference_mode()
    def predict(history, times, target_time, seed):
        torch.manual_seed(seed)
        image = torch.from_numpy(history[-1][None, None] * 2 - 1).float().to(device)
        dt = torch.tensor([target_time - times[-1]], dtype=torch.float32, device=device)
        baseline = vae.encode_to_latent(image)
        condition = torch.cat([baseline, dt[:, None, None, None].expand(
            1, 1, *baseline.shape[2:])], dim=1)
        residual = latent_average_stabilization(
            unet, control, scheduler, condition, embedding(dt).unsqueeze(1),
            baseline.shape[1:], device, num_inference_steps=steps, m=samples)
        return vae.decode(baseline + residual).clamp(-1, 1)[0, 0].cpu().numpy() / 2 + 0.5
    return predict


def load_tadm(path, device, steps=5):
    from external.models_tadm import TADMUnet, ConditionEncoder, TADMDiffusion
    state, config = checkpoint(path)
    dim = int(config.get("dim", 64))
    unet = TADMUnet(dim=dim, out_dim=1, dim_mults=(1, 2, 4, 8), cond_dim=dim).to(device)
    condition = ConditionEncoder(in_channels=1, base_dim=dim, num_blocks=4).to(device)
    unet.load_state_dict(state["ema_unet"], strict=True)
    condition.load_state_dict(state["ema_cond_net"], strict=True)
    model = TADMDiffusion(unet, condition, timesteps=config.get("timesteps", 1000),
                          loss_type=config.get("loss_type", "l1"),
                          beta_schedule=config.get("beta_schedule", "linear"),
                          beta_end=config.get("beta_end", 0.02),
                          res_rescale=config.get("res_rescale", 2.0)).to(device).eval()

    @torch.inference_mode()
    def predict(history, times, target_time, seed):
        torch.manual_seed(seed)
        image = torch.from_numpy(history[-1][None, None] * 2 - 1).float().to(device)
        dt = torch.tensor([target_time - times[-1]], dtype=torch.float32, device=device)
        return model.sample_ddim(image, dt, num_steps=steps)[0, 0].cpu().numpy() / 2 + 0.5
    return predict


def load_imageflownet(path, repository, time_multiplier, device):
    import sys
    if repository is None or time_multiplier is None or time_multiplier <= 0:
        raise ValueError("ImageFlowNet requires its repository and training-derived time multiplier.")
    sys.path.insert(0, str(Path(repository).resolve() / "src"))
    from nn.imageflownet_ode import ImageFlowNetODE
    model = ImageFlowNetODE(device=device, num_filters=64, depth=5,
                             ode_location="all_connections", in_channels=1,
                             out_channels=1, contrastive=False).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False), strict=True)
    model.eval()

    @torch.inference_mode()
    def predict(history, times, target_time, seed):
        image = torch.from_numpy(history[-1][None, None] * 2 - 1).float().to(device)
        dt = torch.tensor([(target_time - times[-1]) * time_multiplier],
                          dtype=torch.float32, device=device)
        result = model(x=image, t=dt)
        return result.clamp(-1, 1)[0, 0].cpu().numpy() / 2 + 0.5
    return predict


def add_model_arguments(parser):
    parser.add_argument("--max_history", type=int,
                        help="TRU-family history limit; defaults to each checkpoint's training setting.")
    for name in ("tru", "std_ddim", "ia_nonlinear", "ia_linear", "brlp", "tadm", "imageflownet"):
        parser.add_argument("--ckpt_" + name)
    parser.add_argument("--ifn_repo")
    parser.add_argument("--ifn_time_multiplier", type=float)
    parser.add_argument("--tadm_steps", type=int, default=5)
    parser.add_argument("--brlp_steps", type=int, default=50)
    parser.add_argument("--brlp_samples", type=int, default=5)


def load_models(args, device, classical=True):
    if args.max_history is not None and args.max_history < 1:
        raise ValueError("max_history must be positive.")
    models = {}
    if classical:
        models["copy_last"] = lambda h, t, target, seed: h[-1].copy()
        models["linear_spline"] = linear_spline
    mapping = {"tru": ["TRU"], "std_ddim": ["Std_DDIM_50step", "Std_DDIM_1step"],
               "ia_nonlinear": ["IA_Nonlinear"], "ia_linear": ["IA_Linear"]}
    for key, names in mapping.items():
        path = getattr(args, "ckpt_" + key, None)
        if path:
            for name in names:
                models[name] = load_core(path, name, device, args.max_history)
    if args.ckpt_brlp:
        models["2D_BrLP"] = load_brlp(args.ckpt_brlp, device, args.brlp_steps, args.brlp_samples)
    if args.ckpt_tadm:
        models["TADM"] = load_tadm(args.ckpt_tadm, device, args.tadm_steps)
    if args.ckpt_imageflownet:
        models["ImageFlowNet"] = load_imageflownet(
            args.ckpt_imageflownet, args.ifn_repo, args.ifn_time_multiplier, device)
    return models

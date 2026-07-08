from __future__ import annotations

import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LATENT_FLOW_DIR = PROJECT_ROOT / "latent_flow"
for _p in (PROJECT_ROOT, LATENT_FLOW_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from model import GeoControlNet3D, UNet3DRectifiedFlow
from vae.model import AutoencoderKL_3D_DualHead


DEFAULT_MODEL_PATH = "dataset/model.npy"
DEFAULT_FAULT_PATH = "dataset/fault.npy"
DEFAULT_VAE_CKPT = "models/vae_3d_dualhead_checkpoint"
DEFAULT_FLOW_CKPT = "models/flow_3d_geo"
DEFAULT_CONTROL_CKPT = "models/flow_3d_geo_controlnet"
DEFAULT_LATENT_SCALE = 1.87


def project_path(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = project_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_checkpoint(path: str | os.PathLike[str], *, prefer_best: bool = False) -> Path:
    p = project_path(path)
    if p.is_file():
        return p
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {p}")
    names = ["best_model.pth", "latest.pth"] if prefer_best else ["latest.pth", "best_model.pth"]
    for name in names:
        candidate = p / name
        if candidate.exists():
            return candidate
    nested = sorted([x for x in p.rglob("*.pth") if x.is_file()] + [x for x in p.rglob("*.pt") if x.is_file()])
    if nested:
        return nested[0]
    raise FileNotFoundError(f"No .pth/.pt checkpoint found under: {p}")


def _normalize_volume(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim == 3:
        out = arr
    elif arr.ndim == 4:
        if arr.shape[0] <= 8 and arr.shape[0] <= arr.shape[-1]:
            out = arr[0]
        elif arr.shape[-1] <= 8:
            out = arr[..., 0]
        else:
            raise ValueError(f"{name} has unsupported shape {arr.shape}")
    else:
        raise ValueError(f"{name} has unsupported shape {arr.shape}")
    return np.asarray(out)


def load_model_fault_arrays(
    model_path: str | os.PathLike[str] = DEFAULT_MODEL_PATH,
    fault_path: str | os.PathLike[str] = DEFAULT_FAULT_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    model = _normalize_volume(np.load(project_path(model_path)), "model").astype(np.uint8, copy=False)
    fault = _normalize_volume(np.load(project_path(fault_path)), "fault").astype(np.uint8, copy=False)
    if model.shape != fault.shape:
        raise ValueError(f"model and fault shapes differ: {model.shape} vs {fault.shape}")
    return model, fault


def crop_pair(
    model: np.ndarray,
    fault: np.ndarray,
    *,
    crop_size: int = 128,
    seed: int = 42,
    origin: tuple[int, int, int] | None = None,
    physical_scale: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if model.shape != fault.shape:
        raise ValueError(f"model and fault shapes differ: {model.shape} vs {fault.shape}")
    x_size, y_size, z_size = model.shape
    if crop_size > min(model.shape):
        raise ValueError(f"crop_size={crop_size} exceeds volume shape={model.shape}")
    rng = np.random.default_rng(int(seed))
    if origin is None:
        x0 = int(rng.integers(0, x_size - crop_size + 1))
        y0 = int(rng.integers(0, y_size - crop_size + 1))
        z0 = int(rng.integers(0, z_size - crop_size + 1))
    else:
        x0, y0, z0 = [int(v) for v in origin]
        x0 = max(0, min(x0, x_size - crop_size))
        y0 = max(0, min(y0, y_size - crop_size))
        z0 = max(0, min(z0, z_size - crop_size))

    model_patch = model[x0:x0 + crop_size, y0:y0 + crop_size, z0:z0 + crop_size]
    fault_patch = fault[x0:x0 + crop_size, y0:y0 + crop_size, z0:z0 + crop_size]
    if z_size <= 1:
        coords_norm = 0.0
    else:
        z_center = float(z0) + 0.5 * float(crop_size - 1)
        coords_norm = max(0.0, min(1.0, z_center / float(z_size - 1)))
    meta = {
        "source_shape_xyz": [int(x_size), int(y_size), int(z_size)],
        "crop_origin_xyz": [int(x0), int(y0), int(z0)],
        "crop_size": int(crop_size),
        "physical_scale": float(physical_scale),
        "coords_norm": float(coords_norm),
    }
    return model_patch.copy(), fault_patch.copy(), meta


def patch_batch(model_patch: np.ndarray, fault_patch: np.ndarray, device: torch.device) -> torch.Tensor:
    data = np.stack([model_patch, fault_patch], axis=0)
    return torch.from_numpy(data).unsqueeze(0).to(device=device, dtype=torch.long)


def save_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _align_prefix_for_state_dict(
    state_dict: dict[str, torch.Tensor],
    model: torch.nn.Module,
    prefix: str = "_orig_mod.",
) -> dict[str, torch.Tensor]:
    model_is_compiled = hasattr(model, "_orig_mod")
    sd_is_compiled = any(k.startswith(prefix) for k in state_dict.keys())
    if model_is_compiled and not sd_is_compiled:
        return {prefix + k: v for k, v in state_dict.items()}
    if (not model_is_compiled) and sd_is_compiled:
        return {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}
    return state_dict


def _filter_state_dict_by_shape(
    state_dict: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    model_sd = model.state_dict()
    return {
        k: v
        for k, v in state_dict.items()
        if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)
    }


def load_vae(
    checkpoint: str | os.PathLike[str] = DEFAULT_VAE_CKPT,
    *,
    device: torch.device,
    use_ema: bool = True,
    prefer_best: bool = False,
) -> tuple[AutoencoderKL_3D_DualHead, dict[str, Any], Path]:
    ckpt_path = find_checkpoint(checkpoint, prefer_best=prefer_best)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = dict(ckpt.get("model_config", {}) or {})
    num_categories = int(model_cfg.get("num_semantic_classes", 8))
    num_edge_channels = int(model_cfg.get("num_edge_channels", 3))
    z_channels = int(model_cfg.get("z_channels", 16))
    base_ch = int(model_cfg.get("base_ch", 8))
    groups = int(model_cfg.get("groups", 4))
    dropout = float(model_cfg.get("dropout_prob", 0.1))
    input_shape = tuple(int(v) for v in model_cfg.get("input_shape", (128, 128, 128)))
    vae_variant = str(model_cfg.get("vae_variant", "dual_head"))

    vae = AutoencoderKL_3D_DualHead(
        num_semantic_classes=num_categories,
        num_edge_channels=num_edge_channels,
        z_channels=z_channels,
        input_shape=input_shape,
        base_ch=base_ch,
        groups=groups,
        dropout_prob=dropout,
        vae_variant=vae_variant,
    ).to(device)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = _align_prefix_for_state_dict(state_dict, vae)
    missing, unexpected = vae.load_state_dict(state_dict, strict=False)
    print(f"[VAE] loaded {ckpt_path} missing={len(missing)} unexpected={len(unexpected)}")

    if use_ema and isinstance(ckpt, dict) and ckpt.get("ema") is not None:
        ema_shadow = ckpt["ema"].get("shadow", None)
        if ema_shadow is not None:
            ema_shadow = _align_prefix_for_state_dict(ema_shadow, vae)
            missing_ema, unexpected_ema = vae.load_state_dict(ema_shadow, strict=False)
            print(f"[VAE] loaded EMA missing={len(missing_ema)} unexpected={len(unexpected_ema)}")

    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    cfg = {
        "num_categories": num_categories,
        "num_edge_channels": num_edge_channels,
        "z_channels": z_channels,
        "input_shape": input_shape,
        "base_ch": base_ch,
        "groups": groups,
        "dropout_prob": dropout,
        "vae_variant": vae_variant,
    }
    return vae, cfg, ckpt_path


def load_base_unet(
    checkpoint: str | os.PathLike[str] = DEFAULT_FLOW_CKPT,
    *,
    device: torch.device,
    prefer_best: bool = False,
) -> tuple[UNet3DRectifiedFlow, dict[str, Any], Path]:
    ckpt_path = find_checkpoint(checkpoint, prefer_best=prefer_best)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    sd = {str(k): v for k, v in sd.items()}

    latent_channels = int(sd.get("final_conv.weight").shape[0]) if "final_conv.weight" in sd else 16
    first_conv = sd.get("downs.0.0.block1.conv.weight")
    cond_channels = 17
    if first_conv is not None:
        cond_channels = int(first_conv.shape[1]) - latent_channels - 3
    time_weight = sd.get("time_mlp.0.weight")
    unet_dim = int(time_weight.shape[1]) if time_weight is not None else 128

    model = UNet3DRectifiedFlow(
        dim=unet_dim,
        dim_mults=(1, 2, 4),
        in_channels=latent_channels,
        out_channels=latent_channels,
        cond_channels=cond_channels,
        dropout=0.10,
        groups=8,
        use_coord=True,
        coord_channels=3,
        use_mid_attn=True,
        attn_heads=4,
        attn_dim_head=32,
        attn_down_levels=(0, 1, 2),
        rope_z_factor=2.0,
        use_scale_emb=True,
        use_log_scale=True,
        log_scale_eps=1e-6,
        scale_unknown_value=-1.0,
        use_depth_fourier=("depth_fourier.W" in sd),
        depth_fourier_dim=64,
        depth_unknown_value=-1.0,
    ).to(device)
    sd = _align_prefix_for_state_dict(sd, model)
    sd = _filter_state_dict_by_shape(sd, model)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[RF] loaded {ckpt_path} missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    cfg = {
        "latent_channels": latent_channels,
        "latent_spatial_dim": 16,
        "cond_channels": cond_channels,
        "cond_use_fault": cond_channels == latent_channels + 2,
        "unet_dim": unet_dim,
        "scale_unknown_value": -1.0,
        "depth_unknown_value": -1.0,
    }
    return model, cfg, ckpt_path


class SimpleControlUNet3D(nn.Module):
    def __init__(
        self,
        base_unet: UNet3DRectifiedFlow,
        *,
        control_in_channels: int = 13,
        crop_size: int = 128,
        groups: int = 8,
        scale_unknown_value: float = -1.0,
    ):
        super().__init__()
        self.time_embed = copy.deepcopy(base_unet.time_embed)
        self.time_mlp = copy.deepcopy(base_unet.time_mlp)
        self.scale_fourier = copy.deepcopy(getattr(base_unet, "scale_fourier", None))
        self.depth_fourier = copy.deepcopy(getattr(base_unet, "depth_fourier", None))
        self.scale_depth_mlp = copy.deepcopy(getattr(base_unet, "scale_depth_mlp", None))
        self.use_scale_emb = getattr(base_unet, "use_scale_emb", False)
        self.use_depth_fourier = getattr(base_unet, "use_depth_fourier", False)
        self.scale_unknown_value = float(scale_unknown_value)
        self.depth_unknown_value = float(getattr(base_unet, "depth_unknown_value", -1.0))
        self.use_log_scale = bool(getattr(base_unet, "use_log_scale", False))
        self.log_scale_eps = float(getattr(base_unet, "log_scale_eps", 1e-6))
        self.time_emb_dim = base_unet.time_emb_dim

        if self.scale_fourier is not None:
            scale_fourier_dim = int(self.scale_fourier.W.numel() * 2)
            self.unknown_scale_fourier = nn.Parameter(torch.zeros(1, scale_fourier_dim))
        else:
            self.unknown_scale_fourier = None
        if self.depth_fourier is not None:
            depth_fourier_dim = int(self.depth_fourier.W.numel() * 2)
            self.unknown_depth_fourier = nn.Parameter(torch.zeros(1, depth_fourier_dim))
        else:
            self.unknown_depth_fourier = None

        for module in (self.time_embed, self.time_mlp, self.scale_fourier, self.depth_fourier, self.scale_depth_mlp):
            if module is not None:
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False

        self.encoder = GeoControlNet3D(
            cond_in_channels=control_in_channels,
            unet=base_unet,
            full_res=int(crop_size),
            latent_res=int(crop_size) // 8,
            groups=int(groups),
            time_emb_dim=self.time_emb_dim,
        )

        self.down_zero_convs = nn.ModuleList()
        self.scale_mlps = nn.ModuleList()
        self.scale_bias_gates = nn.ParameterList()
        self.unknown_scale_biases = nn.ParameterList()
        for ch in base_unet.down_channels:
            conv = nn.Conv3d(ch, ch, kernel_size=1)
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
            self.down_zero_convs.append(conv)
            self.scale_mlps.append(nn.Linear(1, ch))
            self.scale_bias_gates.append(nn.Parameter(torch.zeros(1)))
            self.unknown_scale_biases.append(nn.Parameter(torch.zeros(1, ch, 1, 1, 1)))

        self.up_zero_convs = nn.ModuleList()
        for ch in reversed(base_unet.down_channels):
            conv = nn.Conv3d(ch, ch, kernel_size=1)
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
            self.up_zero_convs.append(conv)

        mid_ch = base_unet.down_channels[-1]
        self.mid_zero_conv = nn.Conv3d(mid_ch, mid_ch, kernel_size=1)
        nn.init.zeros_(self.mid_zero_conv.weight)
        nn.init.zeros_(self.mid_zero_conv.bias)

    def _build_shared_embedding(
        self,
        t: torch.Tensor,
        phys_scale: torch.Tensor | None = None,
        coords_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_embed(t))
        if not (
            self.use_scale_emb
            and phys_scale is not None
            and self.scale_fourier is not None
            and self.scale_depth_mlp is not None
        ):
            return t_emb

        if phys_scale.dim() == 1:
            s_scalar = phys_scale[:, None]
        elif phys_scale.dim() == 2:
            s_scalar = phys_scale if phys_scale.size(1) == 1 else phys_scale.norm(dim=1, keepdim=True)
        else:
            raise ValueError(f"Invalid phys_scale shape: {tuple(phys_scale.shape)}")

        unknown_mask = s_scalar == self.scale_unknown_value
        if self.use_log_scale:
            s_safe = torch.clamp(s_scalar, min=self.log_scale_eps)
            s_scalar = torch.where(unknown_mask, s_scalar, torch.log(s_safe))

        if coords_norm is None:
            z_scalar = torch.full_like(s_scalar, self.depth_unknown_value)
        elif coords_norm.dim() == 1:
            z_scalar = coords_norm[:, None]
        else:
            z_scalar = coords_norm[:, :1]

        s_fourier = self.scale_fourier(s_scalar)
        if self.unknown_scale_fourier is not None:
            mask_f = unknown_mask.to(s_fourier.dtype)
            s_fourier = s_fourier * (1.0 - mask_f) + self.unknown_scale_fourier * mask_f
        if self.use_depth_fourier and self.depth_fourier is not None:
            z_fourier = self.depth_fourier(z_scalar)
            if self.unknown_depth_fourier is not None:
                depth_unknown_mask = z_scalar == self.depth_unknown_value
                mask_f = depth_unknown_mask.to(z_fourier.dtype)
                z_fourier = z_fourier * (1.0 - mask_f) + self.unknown_depth_fourier * mask_f
            z_feat = z_fourier
        else:
            z_feat = z_scalar

        return t_emb + self.scale_depth_mlp(torch.cat([s_fourier, z_feat], dim=-1))

    def forward(
        self,
        control_cond_voxel: torch.Tensor,
        phys_scale: torch.Tensor,
        t: torch.Tensor,
        coords_norm: torch.Tensor | None = None,
    ):
        b = control_cond_voxel.shape[0]
        emb = self._build_shared_embedding(t, phys_scale=phys_scale, coords_norm=coords_norm)
        feats = self.encoder(cond_vox=control_cond_voxel, emb=emb)

        s = phys_scale.view(b, 1)
        unknown_mask = s == self.scale_unknown_value
        s_safe = torch.where(unknown_mask, torch.ones_like(s), s)
        s_log = torch.log(torch.clamp(s_safe, min=1e-6))

        control_down = []
        scaled_feats = []
        for level, (feat, conv, mlp) in enumerate(zip(feats, self.down_zero_convs, self.scale_mlps)):
            scale_bias = mlp(s_log).view(b, -1, 1, 1, 1)
            mask_5d = unknown_mask.view(b, 1, 1, 1, 1)
            unknown_bias = self.unknown_scale_biases[level].expand(b, -1, -1, -1, -1)
            scale_bias = torch.where(mask_5d, unknown_bias, scale_bias)
            gate = torch.tanh(self.scale_bias_gates[level]).clamp(min=0.0)
            feat_scaled = feat + scale_bias * gate
            scaled_feats.append(feat_scaled)
            control_down.append(conv(feat_scaled))

        control_mid = self.mid_zero_conv(scaled_feats[-1])
        control_up = [conv(feat) for feat, conv in zip(reversed(scaled_feats), self.up_zero_convs)]
        return control_down, control_mid, control_up


def load_control_unet(
    checkpoint: str | os.PathLike[str],
    *,
    device: torch.device,
    base_unet: UNet3DRectifiedFlow,
    prefer_best: bool = False,
    crop_size: int = 128,
) -> tuple[SimpleControlUNet3D, Path]:
    ckpt_path = find_checkpoint(checkpoint, prefer_best=prefer_best)
    model = SimpleControlUNet3D(
        base_unet,
        control_in_channels=13,
        crop_size=crop_size,
        groups=8,
        scale_unknown_value=-1.0,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    sd = _align_prefix_for_state_dict(sd, model)
    sd = _filter_state_dict_by_shape(sd, model)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ControlNet] loaded {ckpt_path} missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, ckpt_path


@torch.no_grad()
def decode_latents_to_voxels(
    latents: torch.Tensor,
    vae: AutoencoderKL_3D_DualHead,
    *,
    latent_scale: float = DEFAULT_LATENT_SCALE,
) -> torch.Tensor:
    z = latents / float(latent_scale) if float(latent_scale) != 0 else latents
    dec = vae.decode(z)
    logits = dec[0] if isinstance(dec, tuple) else dec
    return torch.argmax(logits, dim=1)


def make_full_unknown_condition(
    batch_size: int,
    *,
    latent_channels: int,
    latent_spatial_dim: int,
    cond_use_fault: bool,
    device: torch.device,
) -> torch.Tensor:
    z_cond = torch.zeros(
        batch_size,
        latent_channels,
        latent_spatial_dim,
        latent_spatial_dim,
        latent_spatial_dim,
        device=device,
    )
    mask = torch.ones(batch_size, 1, latent_spatial_dim, latent_spatial_dim, latent_spatial_dim, device=device)
    if cond_use_fault:
        fault = torch.zeros_like(mask)
        return torch.cat([z_cond, mask, fault], dim=1)
    return torch.cat([z_cond, mask], dim=1)


def make_phys_tensors(
    batch_size: int,
    *,
    scale: float,
    coords_norm: float | None,
    device: torch.device,
    unknown_scale: bool = False,
    unknown_depth: bool = False,
    scale_unknown_value: float = -1.0,
    depth_unknown_value: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_value = float(scale_unknown_value) if unknown_scale else float(scale)
    depth_value = float(depth_unknown_value) if unknown_depth else float(coords_norm if coords_norm is not None else 0.5)
    phys_scale = torch.full((batch_size, 1), scale_value, device=device, dtype=torch.float32)
    coords = torch.full((batch_size, 1), depth_value, device=device, dtype=torch.float32)
    return phys_scale, coords


def _align_control_dtype(control_down, control_mid, control_up, ref: torch.Tensor):
    if control_down is None:
        return None, None, None
    control_down = [x.to(dtype=ref.dtype) for x in control_down]
    control_mid = control_mid.to(dtype=ref.dtype)
    control_up = [x.to(dtype=ref.dtype) for x in control_up]
    return control_down, control_mid, control_up


@torch.no_grad()
def heun_sample_latents(
    base_unet: UNet3DRectifiedFlow,
    latents: torch.Tensor,
    *,
    cond: torch.Tensor,
    phys_scale: torch.Tensor,
    coords_norm: torch.Tensor,
    steps: int,
    control_unet: SimpleControlUNet3D | None = None,
    control_cond: torch.Tensor | None = None,
    control_strength: float = 1.0,
    use_amp: bool = True,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")
    device = latents.device
    amp_enabled = bool(use_amp and device.type == "cuda")
    dt = 1.0 / float(steps)

    def control_features(t_batch: torch.Tensor):
        if control_unet is None or control_cond is None:
            return None, None, None
        cd, cm, cu = control_unet(control_cond, phys_scale=phys_scale, t=t_batch, coords_norm=coords_norm)
        if float(control_strength) != 1.0:
            cd = [x * float(control_strength) for x in cd]
            cm = cm * float(control_strength)
            cu = [x * float(control_strength) for x in cu]
        return _align_control_dtype(cd, cm, cu, latents)

    for step_idx in range(int(steps)):
        t0 = float(step_idx) * dt
        t1 = float(step_idx + 1) * dt
        t_batch0 = torch.full((latents.size(0),), t0, device=device, dtype=torch.float32)
        t_batch1 = torch.full((latents.size(0),), t1, device=device, dtype=torch.float32)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            cd0, cm0, cu0 = control_features(t_batch0)
            v0 = base_unet(
                latents,
                t_batch0,
                cond=cond,
                phys_scale=phys_scale,
                coords_norm=coords_norm,
                control_down=cd0,
                control_mid=cm0,
                control_up=cu0,
            )
            x_pred = latents + dt * v0
            cd1, cm1, cu1 = control_features(t_batch1)
            v1 = base_unet(
                x_pred,
                t_batch1,
                cond=cond,
                phys_scale=phys_scale,
                coords_norm=coords_norm,
                control_down=cd1,
                control_mid=cm1,
                control_up=cu1,
            )
            latents = latents + 0.5 * dt * (v0 + v1)
    return latents


def build_borehole_control(
    semantic_patch: torch.Tensor,
    *,
    num_categories: int,
    num_boreholes: int,
    depth_ratio_range: tuple[float, float] = (0.6, 1.0),
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    if semantic_patch.dim() != 3:
        raise ValueError(f"Expected semantic_patch [D,H,W], got {tuple(semantic_patch.shape)}")
    device = semantic_patch.device
    d, h, w = [int(v) for v in semantic_patch.shape]
    sem_ctrl = torch.zeros(1, int(num_categories), d, h, w, device=device, dtype=torch.float32)
    valid = torch.zeros(1, 1, d, h, w, device=device, dtype=torch.float32)
    rng = np.random.default_rng(int(seed))
    records: list[dict[str, Any]] = []
    min_depth, max_depth = [float(v) for v in depth_ratio_range]
    if max_depth < min_depth:
        min_depth, max_depth = max_depth, min_depth

    for _ in range(int(num_boreholes)):
        x0 = int(rng.integers(0, d))
        y0 = int(rng.integers(0, h))
        depth_ratio = float(rng.uniform(min_depth, max_depth))
        drill_len = max(1, min(w, int(round(w * depth_ratio))))
        z_start = w - drill_len
        z_idx = torch.arange(z_start, w, device=device)
        x_idx = torch.full((drill_len,), x0, device=device, dtype=torch.long)
        y_idx = torch.full((drill_len,), y0, device=device, dtype=torch.long)
        classes = semantic_patch[x_idx, y_idx, z_idx].long()
        valid_cls = (classes >= 0) & (classes < int(num_categories))
        if valid_cls.any():
            x_valid = x_idx[valid_cls]
            y_valid = y_idx[valid_cls]
            z_valid = z_idx[valid_cls]
            cls_valid = classes[valid_cls]
            for cls in range(int(num_categories)):
                cls_mask = cls_valid == cls
                if cls_mask.any():
                    sem_ctrl[0, cls, x_valid[cls_mask], y_valid[cls_mask], z_valid[cls_mask]] = 1.0
                    valid[0, 0, x_valid[cls_mask], y_valid[cls_mask], z_valid[cls_mask]] = 1.0
        records.append(
            {
                "x": x0,
                "y": y0,
                "z_start": int(z_start),
                "z_end": int(w),
                "depth_ratio": depth_ratio,
                "valid_voxels": int(valid_cls.sum().item()),
            }
        )
    return sem_ctrl, valid, records


def build_control_tensor(
    sem_ctrl: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    b, _c, d, h, w = sem_ctrl.shape
    zeros = torch.zeros(b, 1, d, h, w, device=device, dtype=torch.float32)
    return torch.cat([sem_ctrl, valid_mask, zeros, zeros, zeros, zeros], dim=1)

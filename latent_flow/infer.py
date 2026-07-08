from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from latent_flow.infer_utils import (
    DEFAULT_CONTROL_CKPT,
    DEFAULT_FAULT_PATH,
    DEFAULT_FLOW_CKPT,
    DEFAULT_LATENT_SCALE,
    DEFAULT_MODEL_PATH,
    DEFAULT_VAE_CKPT,
    build_borehole_control,
    build_control_tensor,
    crop_pair,
    decode_latents_to_voxels,
    ensure_dir,
    heun_sample_latents,
    load_base_unet,
    load_control_unet,
    load_model_fault_arrays,
    load_vae,
    make_full_unknown_condition,
    make_phys_tensors,
    patch_batch,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run both unconditional RF generation and borehole-conditioned ControlNet generation."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Semantic model .npy path.")
    parser.add_argument("--fault", default=DEFAULT_FAULT_PATH, help="Fault mask .npy path.")
    parser.add_argument("--vae-ckpt", default=DEFAULT_VAE_CKPT, help="VAE checkpoint file or directory.")
    parser.add_argument("--flow-ckpt", default=DEFAULT_FLOW_CKPT, help="RF checkpoint file or directory.")
    parser.add_argument("--control-ckpt", default=DEFAULT_CONTROL_CKPT, help="ControlNet checkpoint file or directory.")
    parser.add_argument("--out-dir", default="outputs", help="Output root directory.")
    parser.add_argument("--rf-samples", type=int, default=4, help="Number of unconditional RF samples.")
    parser.add_argument("--control-samples", type=int, default=4, help="Number of conditional ControlNet samples.")
    parser.add_argument("--batch-size", type=int, default=4, help="Unconditional RF batch size.")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--num-boreholes", type=int, default=4)
    parser.add_argument("--depth-ratio-min", type=float, default=0.6)
    parser.add_argument("--depth-ratio-max", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=4.0, help="Physical scale value.")
    parser.add_argument("--latent-scale", type=float, default=DEFAULT_LATENT_SCALE)
    parser.add_argument("--control-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--origin", default=None, help="Optional x,y,z crop origin for the conditional demo.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prefer-best", action="store_true", help="Prefer best_model.pth over latest.pth.")
    parser.add_argument(
        "--known-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use --scale for unconditional RF instead of the checkpoint unknown scale token.",
    )
    parser.add_argument(
        "--known-depth",
        action="store_true",
        help="Use coords_norm=0.5 for unconditional RF instead of the checkpoint unknown depth token.",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA autocast.")
    return parser.parse_args()


def parse_origin(value: str | None) -> tuple[int, int, int] | None:
    if value is None or value.strip() == "":
        return None
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--origin must be formatted as x,y,z")
    return parts[0], parts[1], parts[2]


def seed_torch(seed: int, device: torch.device) -> None:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def make_condition_channel(model_patch: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    condition = np.zeros_like(model_patch, dtype=np.uint8)
    known = valid_mask > 0
    condition[known] = model_patch[known].astype(np.uint8) + 1
    return condition


def stack_output_channels(
    semantic: np.ndarray,
    fault: np.ndarray | None = None,
    condition: np.ndarray | None = None,
) -> np.ndarray:
    semantic_u8 = np.asarray(semantic, dtype=np.uint8)
    fault_u8 = np.zeros_like(semantic_u8, dtype=np.uint8) if fault is None else np.asarray(fault, dtype=np.uint8)
    condition_u8 = (
        np.zeros_like(semantic_u8, dtype=np.uint8)
        if condition is None
        else np.asarray(condition, dtype=np.uint8)
    )
    if fault_u8.shape != semantic_u8.shape:
        raise ValueError(f"fault shape {fault_u8.shape} does not match semantic shape {semantic_u8.shape}")
    if condition_u8.shape != semantic_u8.shape:
        raise ValueError(f"condition shape {condition_u8.shape} does not match semantic shape {semantic_u8.shape}")
    return np.stack([semantic_u8, fault_u8, condition_u8], axis=0)


@torch.no_grad()
def run_unconditional_rf(
    args: argparse.Namespace,
    *,
    device: torch.device,
    out_dir: Path,
    vae,
    vae_path: Path,
    base_unet,
    flow_cfg: dict,
    flow_path: Path,
) -> None:
    if args.rf_samples <= 0:
        raise ValueError("--rf-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    rf_dir = ensure_dir(out_dir / "rf_uncond")
    seed_torch(int(args.seed), device)

    saved = 0
    batch_index = 0
    while saved < int(args.rf_samples):
        cur_bs = min(int(args.batch_size), int(args.rf_samples) - saved)
        latents = torch.randn(
            cur_bs,
            int(flow_cfg["latent_channels"]),
            int(flow_cfg["latent_spatial_dim"]),
            int(flow_cfg["latent_spatial_dim"]),
            int(flow_cfg["latent_spatial_dim"]),
            device=device,
            dtype=torch.float32,
        )
        cond = make_full_unknown_condition(
            cur_bs,
            latent_channels=int(flow_cfg["latent_channels"]),
            latent_spatial_dim=int(flow_cfg["latent_spatial_dim"]),
            cond_use_fault=bool(flow_cfg["cond_use_fault"]),
            device=device,
        )
        phys_scale, coords_norm = make_phys_tensors(
            cur_bs,
            scale=float(args.scale),
            coords_norm=0.5,
            device=device,
            unknown_scale=not bool(args.known_scale),
            unknown_depth=not bool(args.known_depth),
            scale_unknown_value=float(flow_cfg["scale_unknown_value"]),
            depth_unknown_value=float(flow_cfg["depth_unknown_value"]),
        )
        latents = heun_sample_latents(
            base_unet,
            latents,
            cond=cond,
            phys_scale=phys_scale,
            coords_norm=coords_norm,
            steps=int(args.steps),
            use_amp=not args.no_amp,
        )
        voxels = decode_latents_to_voxels(
            latents,
            vae,
            latent_scale=float(args.latent_scale),
        ).detach().cpu().numpy().astype(np.uint8)

        for i in range(cur_bs):
            np.save(rf_dir / f"rf_sample_{saved + i:04d}.npy", stack_output_channels(voxels[i]))
        saved += cur_bs
        batch_index += 1
        print(f"[RF] batch {batch_index}: saved {saved}/{args.rf_samples}")

    save_json(
        rf_dir / "metadata.json",
        {
            "task": "rf_unconditional_generation",
            "vae_checkpoint": str(vae_path),
            "flow_checkpoint": str(flow_path),
            "num_samples": int(args.rf_samples),
            "steps": int(args.steps),
            "scale": float(args.scale),
            "unknown_scale": not bool(args.known_scale),
            "unknown_depth": not bool(args.known_depth),
            "latent_scale": float(args.latent_scale),
            "output_shape": "[3, X, Y, Z]",
            "output_channels": {
                "0": "generated semantic labels",
                "1": "fault mask; zeros for unconditional RF outputs",
                "2": "condition voxels; zeros for unconditional RF outputs",
            },
            "flow_config": flow_cfg,
        },
    )
    print(f"[RF] saved outputs to: {rf_dir}")


@torch.no_grad()
def run_controlnet(
    args: argparse.Namespace,
    *,
    device: torch.device,
    out_dir: Path,
    vae,
    vae_cfg: dict,
    vae_path: Path,
    base_unet,
    flow_cfg: dict,
    flow_path: Path,
) -> None:
    if args.control_samples <= 0:
        raise ValueError("--control-samples must be positive")
    if args.num_boreholes <= 0:
        raise ValueError("--num-boreholes must be positive")

    control_dir = ensure_dir(out_dir / "control_borehole")
    seed_torch(int(args.seed) + 1, device)

    model, fault = load_model_fault_arrays(args.model, args.fault)
    model_patch, fault_patch, crop_meta = crop_pair(
        model,
        fault,
        crop_size=int(args.crop_size),
        seed=int(args.seed),
        origin=parse_origin(args.origin),
        physical_scale=float(args.scale),
    )

    control_unet, control_path = load_control_unet(
        args.control_ckpt,
        device=device,
        base_unet=base_unet,
        prefer_best=bool(args.prefer_best),
        crop_size=int(args.crop_size),
    )

    batch = patch_batch(model_patch, fault_patch, device)
    gt_sem = batch[0, 0]
    sem_ctrl, valid_mask, boreholes = build_borehole_control(
        gt_sem,
        num_categories=int(vae_cfg["num_categories"]),
        num_boreholes=int(args.num_boreholes),
        depth_ratio_range=(float(args.depth_ratio_min), float(args.depth_ratio_max)),
        seed=int(args.seed) + 7919,
    )
    control_cond = build_control_tensor(sem_ctrl, valid_mask, device=device)

    cond = make_full_unknown_condition(
        int(args.control_samples),
        latent_channels=int(flow_cfg["latent_channels"]),
        latent_spatial_dim=int(flow_cfg["latent_spatial_dim"]),
        cond_use_fault=bool(flow_cfg["cond_use_fault"]),
        device=device,
    )
    control_cond = control_cond.repeat(int(args.control_samples), 1, 1, 1, 1)
    phys_scale, coords_norm = make_phys_tensors(
        int(args.control_samples),
        scale=float(args.scale),
        coords_norm=float(crop_meta["coords_norm"]),
        device=device,
        unknown_scale=False,
        unknown_depth=False,
        scale_unknown_value=float(flow_cfg["scale_unknown_value"]),
        depth_unknown_value=float(flow_cfg["depth_unknown_value"]),
    )

    latents = torch.randn(
        int(args.control_samples),
        int(flow_cfg["latent_channels"]),
        int(flow_cfg["latent_spatial_dim"]),
        int(flow_cfg["latent_spatial_dim"]),
        int(flow_cfg["latent_spatial_dim"]),
        device=device,
        dtype=torch.float32,
    )
    latents = heun_sample_latents(
        base_unet,
        latents,
        cond=cond,
        phys_scale=phys_scale,
        coords_norm=coords_norm,
        steps=int(args.steps),
        control_unet=control_unet,
        control_cond=control_cond,
        control_strength=float(args.control_strength),
        use_amp=not args.no_amp,
    )
    generated = decode_latents_to_voxels(
        latents,
        vae,
        latent_scale=float(args.latent_scale),
    ).detach().cpu().numpy().astype(np.uint8)

    valid_np = valid_mask[0, 0].detach().cpu().numpy().astype(np.uint8)
    condition_channel = make_condition_channel(model_patch, valid_np)

    np.save(control_dir / "gt_patch.npy", stack_output_channels(model_patch, fault_patch, condition_channel))
    for i, vol in enumerate(generated):
        np.save(control_dir / f"control_sample_{i:04d}.npy", stack_output_channels(vol, fault_patch, condition_channel))

    cond_acc = []
    valid_bool = valid_np > 0
    if valid_bool.any():
        for vol in generated:
            cond_acc.append(float((vol[valid_bool] == model_patch[valid_bool]).mean()))

    save_json(
        control_dir / "metadata.json",
        {
            "task": "controlnet_borehole_generation",
            "model_path": args.model,
            "fault_path": args.fault,
            "vae_checkpoint": str(vae_path),
            "flow_checkpoint": str(flow_path),
            "control_checkpoint": str(control_path),
            "crop": crop_meta,
            "num_samples": int(args.control_samples),
            "steps": int(args.steps),
            "num_boreholes": int(args.num_boreholes),
            "boreholes": boreholes,
            "valid_voxels": int(valid_np.sum()),
            "condition_accuracy": cond_acc,
            "scale": float(args.scale),
            "latent_scale": float(args.latent_scale),
            "control_strength": float(args.control_strength),
            "output_shape": "[3, X, Y, Z]",
            "output_channels": {
                "0": "semantic labels: generated labels for control_sample_*.npy, ground-truth labels for gt_patch.npy",
                "1": "fault mask from the cropped proxy model",
                "2": "condition voxels: 0 means unknown/unconditioned; values 1..8 encode semantic labels 0..7 at borehole voxels",
            },
            "flow_config": flow_cfg,
        },
    )

    print(f"[Control] saved outputs to: {control_dir}")
    print(f"[Control] valid_voxels={int(valid_np.sum())}")
    if cond_acc:
        print("[Control] condition_accuracy=" + ", ".join(f"{v:.4f}" for v in cond_acc))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = ensure_dir(args.out_dir)

    vae, vae_cfg, vae_path = load_vae(args.vae_ckpt, device=device, prefer_best=bool(args.prefer_best))
    base_unet, flow_cfg, flow_path = load_base_unet(
        args.flow_ckpt,
        device=device,
        prefer_best=bool(args.prefer_best),
    )

    run_unconditional_rf(
        args,
        device=device,
        out_dir=out_dir,
        vae=vae,
        vae_path=vae_path,
        base_unet=base_unet,
        flow_cfg=flow_cfg,
        flow_path=flow_path,
    )
    run_controlnet(
        args,
        device=device,
        out_dir=out_dir,
        vae=vae,
        vae_cfg=vae_cfg,
        vae_path=vae_path,
        base_unet=base_unet,
        flow_cfg=flow_cfg,
        flow_path=flow_path,
    )
    save_json(
        out_dir / "metadata.json",
        {
            "task": "combined_rf_and_controlnet_inference",
            "rf_output_dir": str(out_dir / "rf_uncond"),
            "control_output_dir": str(out_dir / "control_borehole"),
            "steps": int(args.steps),
            "scale": float(args.scale),
            "rf_samples": int(args.rf_samples),
            "control_samples": int(args.control_samples),
        },
    )
    print(f"[Infer] saved combined outputs to: {out_dir}")


if __name__ == "__main__":
    main()

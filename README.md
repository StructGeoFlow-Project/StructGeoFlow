# StructGeoFlow

StructGeoFlow is a structure-aware latent residual-control workflow for sparse-constrained 3D geological voxel generation. This repository provides the inference code and viewer for the Computers & Geosciences manuscript:

> StructGeoFlow: A structure-aware latent residual-control workflow for sparse-constrained 3D geological voxel generation

The release includes the model definitions needed to load the published checkpoints, a unified inference script, and a browser-based `.npy` voxel viewer.

## Demo Assets

Download the demo assets from the Hugging Face dataset repository, then place them under the repository root:

https://huggingface.co/datasets/snipervx/StructGeoFlow

```text
dataset/
  model.npy
  fault.npy
models/
  vae_3d_dualhead_checkpoint/
    latest.pth
  flow_3d_geo/
    latest.pth
  flow_3d_geo_controlnet/
    latest.pth
```

The asset repository also includes `manifest.json` and `checksums.sha256` for file verification.

The demo data use proxy geological volumes from the same regional modeling context as the study. The original commercial model data are not distributed with this release.

The demo data and pretrained checkpoints are licensed separately under CC BY-NC 4.0 for non-commercial research and educational use.

Expected inputs:

- `dataset/model.npy`: semantic geological labels, shape `[X, Y, Z]`.
- `dataset/fault.npy`: binary fault mask, shape `[X, Y, Z]`.
- Each axis should be at least `128` voxels.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA inference, install a PyTorch build that matches your local CUDA driver.

## Run Inference

```bash
python -m compileall -q vae latent_flow
python latent_flow/infer.py
```

By default, the script uses 50 Heun steps and physical scale `4.0`. It writes four unconditional RF samples to `outputs/rf_uncond/` and four ControlNet samples from one shared borehole condition to `outputs/control_borehole/`.

Generated samples use channel-first layout:

```text
[3, X, Y, Z]
```

- channel 0: semantic labels.
- channel 1: fault mask.
- channel 2: condition voxels. `0` is unknown; `1..8` encode semantic labels `0..7`.

For ControlNet outputs, `control_sample_*.npy` and `gt_patch.npy` already include the fault and condition channels.

## View Results

Open `viewer/npy_viewer.html` in a browser and select a generated `.npy` file. The viewer supports full-volume voxel rendering, slices, semantic colors, scalar colors, voxel-grid display, and channel switching.

The viewer loads Three.js from a CDN, so direct HTML use requires internet access.

## Repository Layout

- `latent_flow/`: RF and ControlNet model definitions, inference utilities, and `infer.py`.
- `vae/`: VAE model definition used by the released checkpoints.
- `viewer/`: standalone Three.js `.npy` voxel viewer.
- `docs/`: code availability notes.

## License

The source code in this repository is released under the MIT License. See `LICENSE`.

The demo data and pretrained model checkpoints hosted at https://huggingface.co/datasets/snipervx/StructGeoFlow are licensed separately under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). Commercial use of those assets requires prior written permission from the authors.

## Code Availability

Source code repository: https://github.com/StructGeoFlow-Project/StructGeoFlow

The journal-oriented code availability record is in `docs/computer_code_availability.md`.

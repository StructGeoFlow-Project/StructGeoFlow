# StructGeoFlow

StructGeoFlow is a structure-aware latent residual-control workflow for sparse-constrained 3D geological voxel generation. This demo release accompanies the Computers & Geosciences manuscript:

> StructGeoFlow: A structure-aware latent residual-control workflow for sparse-constrained 3D geological voxel generation

The repository includes the model definitions, a unified inference script, and a browser-based `.npy` voxel viewer.

## Quick Start

Download the [demo data and pretrained checkpoints](https://huggingface.co/datasets/snipervx/StructGeoFlow) and place them under the repository root:

```text
dataset/
  model.npy
  fault.npy
models/
  vae_3d_dualhead_checkpoint/latest.pth
  flow_3d_geo/latest.pth
  flow_3d_geo_controlnet/latest.pth
```

The input arrays have shape `[X, Y, Z]`, with each axis at least 128 voxels. `model.npy` contains semantic labels and `fault.npy` contains a binary fault mask. The demo uses proxy geological volumes developed from the same regional modeling context as the study; the source geological model remains proprietary.

Install the dependencies and run inference from the repository root:

```bash
python -m pip install -r requirements.txt
python latent_flow/infer.py
```

A CUDA-capable GPU is recommended. Install the PyTorch build appropriate for your CUDA environment when needed.

The default run uses 50 Heun steps and a physical scale of `4.0`. It creates four unconditional RF samples in `outputs/rf_uncond/` and four ControlNet samples from one borehole condition in `outputs/control_borehole/`.

## Output

Generated `.npy` files use channel-first layout `[3, X, Y, Z]`:

- channel 0: semantic labels
- channel 1: fault mask
- channel 2: condition voxels (`0` for unknown; `1..8` for semantic labels `0..7`)

ControlNet samples and `gt_patch.npy` include all three channels.

## Viewer

Open `viewer/npy_viewer.html` in a browser and select a generated `.npy` file. The viewer provides full-volume and slice views, channel switching, semantic and scalar color maps, and optional voxel-grid lines. It loads Three.js from a CDN and therefore requires internet access.

## License

The source code is released under the [MIT License](LICENSE). The demo data and pretrained checkpoints are released separately under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Availability

- Author: Renjun Qian ([qianrenjun@buaa.edu.cn](mailto:qianrenjun@buaa.edu.cn))
- Source code: https://github.com/StructGeoFlow-Project/StructGeoFlow
- Demo assets: https://huggingface.co/datasets/snipervx/StructGeoFlow
- Journal code-availability record: [docs/computer_code_availability.md](docs/computer_code_availability.md)

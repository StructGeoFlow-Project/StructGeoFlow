# Computer Code Availability

Name of code: StructGeoFlow.

Manuscript title: StructGeoFlow: A structure-aware latent residual-control workflow for sparse-constrained 3D geological voxel generation.

Developer and contact: Renjun Qian, qianrenjun@buaa.edu.cn.

Year first available: 2026.

Hardware required: Syntax-level verification runs on CPU. Demo inference with the released checkpoints is intended for CUDA GPUs.

Software required: Python 3.10 or later, PyTorch 2.1 or later, NumPy, and einops.

Program language: Python.

Program size: The released source code excludes internal model weights, generated logs, caches, and large benchmark volumes.

Source code repository: https://github.com/StructGeoFlow-Project/StructGeoFlow

Open-source license: MIT for the source code.

Demo data and checkpoint license: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). Commercial use of the demo data and pretrained checkpoints requires prior written permission from the authors.

Release version or tag: v0.1.0.

Included reproducibility assets: README, user instructions, inference source code, model architecture definitions, demo inference entry point, and voxel viewer.

Demo data and model weights: Not included in this minimal source repository. Local defaults expect demo data under `dataset/` and model files under `models/`. The released demo assets are available at https://huggingface.co/datasets/snipervx/StructGeoFlow under CC BY-NC 4.0.

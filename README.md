# SHARE-U: Semantic-aware Human Avatar Reconstruction with Efficient Gaussians via Uncertainty-weighted Contrastive Learning

This repository contains the implementation of SHARE-U, a semantic-aware Gaussian avatar reconstruction method for monocular human videos. The codebase builds on articulated Gaussian Splatting pipelines and includes training and rendering scripts for ZJU-MoCap-refine and MonoCap style datasets.

## Updates

- [06/2026] Training, rendering, SMPL, and SMPL-X utility code are included in this repository.

## Requirements

NVIDIA GPUs are required. We recommend using Anaconda to manage the Python environment.

```bash
conda create --name shareu python=3.8
conda activate shareu

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
pip install numpy scipy tqdm opencv-python matplotlib plyfile networkx scikit-image
```

Install PyTorch3D with the wheel or source build that matches your CUDA and PyTorch versions. The training code also expects the SMPL/SMPL-X utilities provided in `smpl/`, `smplx/`, and `utils/smplx/`.

## Set Up Dataset

Prepare datasets under the `data/` directory. The provided scripts expect the following layout:

```text
data/
|-- zju_mocap_refine/
|   |-- my_377/
|   |-- my_386/
|   |-- my_387/
|   |-- my_392/
|   |-- my_393/
|   `-- my_394/
`-- monocap/
    |-- lan_images620_1300/
    |-- marc_images35000_36200/
    |-- olek_images0812/
    `-- vlad_images1011/
```

For ZJU-MoCap-refine and MonoCap data preparation, follow the dataset setup convention used by Instant-NVR and GauHuman, then place each sequence at the paths above.

## Download SMPL and SMPL-X Models

Register and download the required body models from the official SMPL/SMPL-X sources. Put the model files under `assets/`:

```text
assets/
|-- SMPL_NEUTRAL.pkl
|-- SMPL_MALE.pkl
|-- SMPL_FEMALE.pkl
`-- models/
    `-- smplx/
        |-- SMPLX_NEUTRAL.npz
        |-- SMPLX_MALE.npz
        `-- SMPLX_FEMALE.npz
```

Only the gender/model files used by your experiment are required. The training scripts use `--smpl_type smpl --actor_gender neutral` by default.

## Training

Train on ZJU-MoCap-refine:

```bash
bash train_zju_mocap_refine.sh
```

Train on MonoCap:

```bash
bash train_monocap.sh
```

The scripts call `train_cl.py` with semantic-aware contrastive learning enabled through the SHARE-U training pipeline. Outputs are written under `output/`.

## Rendering and Evaluation

Render/evaluate ZJU-MoCap-refine checkpoints:

```bash
bash eval_zju_mocap_refine.sh
```

Render/evaluate MonoCap checkpoints:

```bash
bash eval_monocap.sh
```

You can also run `render.py` directly by passing the trained model directory:

```bash
python render.py -m output/zju_mocap_refine/my_377 --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration 1200 --skip_train
```

## Repository Structure

```text
arguments/              Command-line argument definitions
data/                   Dataset root
gaussian_renderer/      Gaussian rendering components
nets/                   SHARE-U network modules
scene/                  Gaussian model and scene utilities
smpl/                   SMPL helper code
smplx/                  SMPL-X helper code
submodules/             CUDA rasterizer and simple-knn dependencies
utils/                  Geometry, camera, SMPL-X, and training utilities
train_cl.py             Main SHARE-U training entry point
render.py               Rendering and evaluation entry point
```

## Citation

If you find this code useful for your research, please cite SHARE-U once the paper or preprint is available.

```bibtex
@article{shareu2026,
  title={SHARE-U: Semantic-aware Human Avatar Reconstruction with Efficient Gaussians via Uncertainty-weighted Contrastive Learning},
  author={},
  journal={},
  year={2026}
}
```

## License

This project is distributed under the license included in `LICENSE`. Please also respect the licenses and terms of the upstream Gaussian Splatting, SMPL, and SMPL-X resources used by this project.

## Acknowledgements

This project is built on source code and ideas from Gaussian Splatting, human avatar reconstruction, and articulated Gaussian Splatting projects including GauHuman.

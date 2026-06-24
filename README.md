# SHARE-U: Semantic-aware Human Avatar Reconstruction with Efficient Gaussians via Uncertainty-weighted Contrastive Learning

This repository contains the implementation of SHARE-U, a semantic-aware Gaussian avatar reconstruction method for monocular human videos. The codebase builds on articulated Gaussian Splatting pipelines and includes training and rendering scripts for ZJU-MoCap-refine and MonoCap style datasets.

## 📰 Updates

- [06/2026] Add training, rendering, and SMPL-X code.

## ⚙️ Requirements

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

## 📦 Set Up Dataset

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

For ZJU-MoCap-refine and MonoCap data preparation, follow the dataset setup convention used by [Instant-NVR](https://github.com/zju3dv/instant-nvr/blob/master/docs/install.md#set-up-datasets) and [GauHuman](https://github.com/skhu101/GauHuman), then place each sequence at the paths above.

## 🧍 Download SMPL and SMPL-X Models

Register and download the required body models from the official [SMPL](https://smpl.is.tue.mpg.de/) and [SMPL-X](https://smpl-x.is.tue.mpg.de/) websites. Put the model files under `assets/`:

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

## 🚆 Training

Train on ZJU-MoCap-refine:

```bash
bash train_zju_mocap_refine.sh
```

Train on MonoCap:

```bash
bash train_monocap.sh
```

The scripts call `train_cl.py` with semantic-aware contrastive learning enabled through the SHARE-U training pipeline. Outputs are written under `output/`.

## 🎥 Rendering and Evaluation

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

## 📚 Citation

If you find this code useful for your research, please cite SHARE-U.

```bibtex
@article{han2026shareu,
  title={SHARE-U: Semantic-aware Human Avatar Reconstruction with Efficient Gaussians via Uncertainty-weighted Contrastive Learning},
  author={Han, Zhisheng and Wu, Shiyao and Qiu, Jiayan and Ju, Yakun and Liu, Lu and Feng, Pengfei and Zhou, Huiyu and Jiang, Zheheng},
  journal={IEEE Transactions on Multimedia},
  pages={1--12},
  year={2026},
  doi={10.1109/TMM.2026.3685541}
}
```

## 📄 License

This project is distributed under the license included in `LICENSE`. Please also respect the licenses and terms of the upstream Gaussian Splatting, SMPL, and SMPL-X resources used by this project.

## 🙏 Acknowledgements

This project is built on source code and ideas from [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [GauHuman](https://github.com/skhu101/GauHuman). We thank the authors for releasing their excellent projects.

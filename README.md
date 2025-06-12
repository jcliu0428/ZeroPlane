# ZeroPlane
Official Implementation of **ZeroPlane** Towards In-the-wild 3D Plane Reconstruction from a Single Image (CVPR 2025 Highlight)

<a href="https://arxiv.org/pdf/2506.02493"><img src="https://img.shields.io/badge/arXiv-2506.02493-b31b1b" alt="arXiv"></a> <a href="https://github.com/jcliu0428/ZeroPlane"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a> <a href=""><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue"></a>


## 📖 Overview
![Zeroplane](imgs/teaser.png)
We present ZeroPlane, an end-to-end transformer-based framework on single-image 3D plane reconstruction and segmentation. Our model achieves state-of-the-art plane recovery quality across multiple indoor and outdoor dataset, especially on in-the-wild images. Besides, we construct a large-scale high-quality, high-resolution dataset across multiple benchmarks captured from various environments for future research on generalizable 3D plane reconstruction task.

## 🔨 Environment Installation
### 1. Git clone

```bash
git clone https://github.com/jcliu0428/ZeroPlane
```
### 2. Create a conda environment and install dependencies

```bash
# create conda and install all packages
conda env create -f environment.yml
conda activate zeroplane

# install pytorch under your GPU cuda version
pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# install detectron2
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# compile MSDeformAttn
cd ZeroPlane/modeling/pixel_decoder/ops
sh make.sh
```

### 3. Download pretrained checkpoints

```bash
# download full-model checkpoint
mkdir checkpoints && cd checkpoints
wget https://www.dropbox.com/scl/fi/87f8s5pxcps54ocnr6pxe/dust3r_encoder_released.pth?rlkey=c9eenh0dul15rlt6ucpf3yx67&st=fo19yrg0&dl=0 -O dust3r_encoder_released.pth
```

## 🔥 Getting Started
For a quick demo to infer our plane reconstruction on a specified image, please refer to this script:

```bash
python demo/demo.py \
    --config-file configs/ZeroPlaneNYUV2/dust3r_large_dpt_bs16_50ep.yaml \
    --input ./demo/0_d2_image.png \
    --out ./demo/nyu_demo \
    --opts \
    MODEL.WEIGHTS ./checkpoints/dust3r_encoder_released.pth \
    MODEL.MASK_FORMER.LEARN_NORMAL_CLS "True" \
    MODEL.MASK_FORMER.LEARN_OFFSET_CLS "True" \
    MODEL.MASK_FORMER.MIX_ANCHOR "True" \
    MODEL.MASK_FORMER.NORMAL_CLS_NUM 7 \
    MODEL.MASK_FORMER.PREDICT_GLOBAL_PIXEL_DEPTH "True" \
    MODEL.MASK_FORMER.PREDICT_GLOBAL_PIXEL_NORMAL "True" \
    MODEL.MASK_FORMER.WITH_PIXEL_NORMAL_ATTENTION "True" \
    MODEL.MASK_FORMER.WITH_PIXEL_DEPTH_ATTENTION "True" \
    MODEL.MASK_FORMER.SEPARATE_PIXEL_ATTENTION "True" \
```

## 🚀 Release Plan
* [x] Github created **[2025.03.24]**
* [x] Demo code and pretrained model release.
* [ ] Training and evaluation dataset.
* [ ] Training code release. 
* [ ] Gradio demo.

## 📜 Reference
If you find this project useful or insightful, please consider citing:
```
@inproceedings{liu2025towards,
  title={Towards In-the-wild 3D Plane Reconstruction from a Single Image},
  author={Liu, Jiachen and Yu, Rui and Chen, Sili and Huang, Sharon X and Guo, Hengkai},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={27027--27037},
  year={2025}
}
```

# ZeroPlane
[CVPR 2025] Towards In-the-wild 3D Plane Reconstruction from a Single Image

## 📖 Overview
![Zeroplane](imgs/teaser.png)
We present ZeroPlane, an end-to-end transformer-based framework on single-image 3D plane reconstruction and segmentation. Our model achieves state-of-the-art plane recovery quality across multiple indoor and outdoor dataset, especially on in-the-wild images. Besides, we construct a large-scale high-quality, high-resolution dataset across multiple benchmarks captured from various environments for future research on generalizable 3D plane reconstruction task.

## 🔨 Environment Installation
### 1. Git clone
```
git clone https://github.com/ant-research/PlanarSplatting.git --recursive 
```
### 2. Create a conda environment and install dependencies
```
# create conda and install all packages
conda env create -f environment.yml
conda activate zeroplane

# install pytorch under your cuda version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# compile MSDeformAttn
cd ZeroPlane/modeling/pixel_decoder/ops
sh make.sh
```

### 3. Download pretrained checkpoints
```
# download pretrained dust3r encoder
cd third_party/dust3r
mkdir checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# download full-model checkpoint
mkdir checkpoints && cd checkpoints
wget https://www.dropbox.com/scl/fi/87f8s5pxcps54ocnr6pxe/dust3r_encoder_released.pth?rlkey=c9eenh0dul15rlt6ucpf3yx67&st=fo19yrg0&dl=0
mv dust3r_encoder_released.pth?rlkey=c9eenh0dul15rlt6ucpf3yx67 dust3r_encoder_released.pth
```

## 🔥 Getting Started
For a quick demo to infer our plane reconstruction on a specified image, please refer to this script:
```
python demo/demo.py \
    --config-file configs/ZeroPlaneNYUV2/dust3r_large_dpt_bs16_50ep.yaml \
    --input ./demo/0_d2_image.png \
    --out ./demo/nyu_demo \
    --opts \
    MODEL.WEIGHTS ./checkpoints/dust3r_encoder_released.pth \
    OUTPUT_DIR wild_data_vis/final_dust3r/nyuv2_dataset_final_test_vis \
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

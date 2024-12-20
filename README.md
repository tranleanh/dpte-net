# DPTE-Net: Distilled Pooling Transformer Encoder for Efficient Realistic Image Dehazing

[![Weights](https://img.shields.io/badge/Weights-Hugging_Face-gold)](https://huggingface.co/tranleanh/dpte-net)
[![Preprint](https://img.shields.io/badge/Preprint-arXiv-red)](https://arxiv.org/abs/2412.14220)

The official implementation of the paper "Distilled Pooling Transformer Encoder for Efficient Realistic Image Dehazing".

Authors: [Le-Anh Tran](https://scholar.google.com/citations?user=WzcUE5YAAAAJ&hl=en), [Dong-Chul Park](https://scholar.google.com/citations?user=VZUH4sUAAAAJ&hl=en)

Journal: [Neural Computing and Applications](https://link.springer.com/journal/521) (Springer)

## Introduction

#### Framework diagram

<p align="center">
<img src="docs/dptenet.png" width="1000">
</p>

## Test

#### Create environment & install required packages
```
conda create -n dpteenv python=3.7
conda activate dpteenv
bash install_core_env.sh
```

#### Run test
```
python dehaze.py
```

#### Evaluate PSNR & SSIM
```
python eval_psnr_ssim.py
```

## Train

(will be updated)

## Citation

(will be updated)

LA Tran

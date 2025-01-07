# DPTE-Net: Distilled Pooling Transformer Encoder for Efficient Realistic Image Dehazing

[![Weights](https://img.shields.io/badge/Weights-Hugging_Face-gold)](https://huggingface.co/tranleanh/dpte-net)
[![Preprint](https://img.shields.io/badge/Preprint-arXiv-red)](https://arxiv.org/abs/2412.14220)
[![Journal](https://img.shields.io/badge/Article-Springer-blue)](https://link.springer.com/article/10.1007/s00521-024-10930-8)

The official implementation of the paper [Distilled Pooling Transformer Encoder for Efficient Realistic Image Dehazing](https://link.springer.com/article/10.1007/s00521-024-10930-8).

Authors: [Le-Anh Tran](https://scholar.google.com/citations?user=WzcUE5YAAAAJ&hl=en), [Dong-Chul Park](https://scholar.google.com/citations?user=VZUH4sUAAAAJ&hl=en)

Journal: [Neural Computing and Applications](https://link.springer.com/journal/521) (Springer)

## Introduction

#### Framework diagram

<p align="center">
<img src="docs/dptenet.png" width="1000">
</p>

## Test

- Create environment & install required packages
```
conda create -n dpteenv python=3.7
conda activate dpteenv
bash install_core_env.sh
```
- Download pre-trained weights from [Hugging Face](https://huggingface.co/tranleanh/dpte-net)
- Prepare test data
- Run test
```
python dehaze.py
```
- Evaluate PSNR & SSIM
```
python eval_psnr_ssim.py
```

## Train

- Create environment & install required packages
- Prepare dataset (a parent directory containing two sub-folders 'A' and 'B' like below):

```bashrc
.../path/to/data
            | A (containing hazy images)
            | B (containing clean images)
*** Note: a pair of hazy-clean images must have the same name
```
- Configure training parameters in [train.py](https://github.com/tranleanh/dpte-net/blob/main/train.py#L147)
- Train command
```
python train.py
```

## Citation

Please cite our paper if you use the data in this repo.
```bibtex
@article{tran2024distilled,
  title={Distilled Pooling Transformer Encoder for Efficient Realistic Image Dehazing},
  author={Tran, Le-Anh and Park, Dong-Chul},
  journal={arXiv preprint arXiv:2412.14220},
  year={2024}
}
```
Have fun.

LA Tran

# CBAM.PyTorch
Non-official implement of Paper：CBAM: Convolutional Block Attention Module

## Introduction
The codes are [PyTorch](https://pytorch.org/) re-implement version for paper: CBAM: Convolutional Block Attention Module

> Woo S, Park J, Lee J Y, et al. CBAM: Convolutional Block Attention Module[J]. 2018. [ECCV2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)

## Structure

The overview of CBAM. The module has two sequential sub-modules:
channel and spatial. The intermediate feature map is adaptively refined through
our module (CBAM) at every convolutional block of deep networks.

![1](imgs/01.png)

## Requirements
- Python3
- PyTorch 0.4.1
- tensorboardX (optional)
- torchnet
- pretrainedmodels (optional)

## Results
We just test four models in ImageNet-1K, both train set and val set are scaled to 256(minimal side), only use **Mirror** and **RandomResizeCrop** as training data augmentation, during validation, we use center crop to get 224x224 patch.

### ImageNet-1K

Models         | validation(Top-1) | validation(Top-5) |
-------------  | ----------------- | ----------------- |
ResNet50       | 74.26             | 91.91             |
ResNet50-CBAM  | 75.45             | 92.55             |
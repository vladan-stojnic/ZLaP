# Label Propagation for Zero-shot Classification with Vision-Language Models

This repository contains the code for the paper Vladan Stojnić, Yannis Kalantidis, Giorgos Tolias, ["Label Propagation for Zero-shot Classification with Vision-Language Models"](http://arxiv.org/abs/2404.04072), In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

## Setup

This code was implemented using Python 3.10.4 and the following dependencies:
```
torch==2.0.1
torchvision==0.15.2
cupy==11.4.0
faiss==1.7.3
numpy==1.22.3
```

## Features

Pre-extracted features used in this work can be downloaded from [here](http://ptak.felk.cvut.cz/personal/stojnvla/public/zlap_features.tar.gz).

## Running

The provided code can be run using

```
python zlap.py --help
usage: ZLaP [-h]
            [--dataset {imagenet,dtd,eurosat,fgvca,flowers,food101,pets,sun397,cars,caltech101,cifa10,cifar100,cub,ucf101,coco}]
            [--backbone {RN50_openai,ViT-B-16_openai,ViT-B-16_laion2b_s34b_b88k,ViT-H-14_laion2b_s32b_b79k,ViT-L-14-336_openai,ViT-L-14_openai,albef,blip,eva-clip-8b,eva-clip-18b}]
            [--k K] [--gamma GAMMA] [--alpha ALPHA]
            [--setup {transductive,inductive,sparse-inductive}]
            [--clf_type {text,proxy,cupl-text,cupl-proxy}]
```

- Example 1: to run ZLaP on top of CLIP in transductive setup on ImageNet use

```
python zlap.py --dataset imagenet --backbone RN50_openai --setup transductive --clf_type text --k 5 --gamma 5 --alpha 0.3
```

- Example 2: to run ZLaP on top of InMaP in inductive setup on CUB use

```
python zlap.py --dataset cub --backbone ViT-B-16_openai --setup inductive --clf_type proxy --k 10 --gamma 3 --alpha 0.3
```

- Example 3: to run ZLaP on top of CLIP for sparse inductive inference on DTD use

```
python zlap.py --dataset dtd --backbone RN50_openai --setup sparse-inductive --clf_type text --k 5 --gamma 5 --alpha 0.3
```

- Example 4: to run ZLaP with CuPL text prompts on top of CLIP on ImageNet use

```
python zlap.py --dataset imagenet --backbone ViT-B-16_openai --setup transductive --clf_type cupl-text --k 5 --gamma 5 --alpha 0.3
```

## Citation

```
@InProceedings{Stojnic_2024_CVPR,
    author    = {Stojni\'c, Vladan and Kalantidis, Yannis and Tolias, Giorgos},
    title     = {Label Propagation for Zero-shot Classification with Vision-Language Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2024}
}
```

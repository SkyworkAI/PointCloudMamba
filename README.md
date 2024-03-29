<div align="center">

# [Point Cloud Mamba: Point Cloud Learning via State Space Mode](https://arxiv.org/abs/2403.00762)
Tao Zhang, Xiangtai Li, Haobo Yuan, Shunping Ji, Shuicheng Yan

<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/pcm/pcm-idea.png" width="800"/>
</div>

## News
- All codes and weights are available.

## Features
- PCM introduces Mamba to point cloud analysis.
- PCM possesses the ability for global modeling while maintaining linear computational complexity.
- PCM outperforms PointNeXt on the ScanObjectNN, ModelNet40, and ShapeNetPart datasets.

## Install 
See [Installation Instructions](INSTALL.md).

## Getting Started
See [Preparing Datasets for PCM](data/README.md).

See [Getting Started with PCM](GETTING_STARTED.md).

## Demos
### ShapeNetPart
<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/pcm/pcm-demo.png" width="800"/>

## Performance
### 3-D Point Cloud Classification
<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/pcm/pcm-exp-1.png" width="600"/>

### 3-D Point Cloud Segmentation
<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/pcm/pcm-exp-2.png" width="500"/>


```BibTeX
@article{zhang2024pcm,
      title={Point Cloud Mamba: Point Cloud Learning via State Space Model}, 
      author={Tao Zhang and Xiangtai Li and Haobo Yuan and Shunping Ji and Shuicheng Yan},
      journal={arXiv preprint arXiv:2403.00762},
      year={2024}
}
```

## Acknowledgement

This repo is based on [PointNeXt](https://github.com/guochengqian/PointNeXt), 
[PointMLP](https://github.com/ma-xu/pointMLP-pytorch), [Mamba](https://github.com/state-spaces/mamba),
[Vim](https://github.com/hustvl/Vim), and [Pontcept](https://github.com/Pointcept/Pointcept).
Thanks for their excellent works.

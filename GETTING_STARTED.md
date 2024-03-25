# Getting Started with PCM

This document provides a brief intro of the usage of PCM.

## Pretrained weights

The pre-trained weights are available at [Hugging Face](https://huggingface.co/zhangtao-whu/PCM/tree/main) and [Baidu Pan](https://pan.baidu.com/s/18TBobF0owhE4BouJFlQVKA?pwd=nky0).

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">OA (ScanObjectNN)</th>
<th valign="bottom">mAcc (ScanObjectNN)</th>
<th valign="bottom">OA (ModelNet40)</th>
<th valign="bottom">mAcc (ModelNet40)</th>
<th valign="bottom">Ins. mIoU (ShapeNetPart)</th>
<th valign="bottom">Cls. mIoU (ShapeNetPart)</th>
<th valign="bottom">mIou (S3DIS)</th>
<th valign="bottom">OA (S3DIS)</th>
<!-- TABLE BODY -->

<!-- ROW: pcm -->
 <tr><td align="center">PCM</td>
<td align="center">88.0</td>
<td align="center">86.4</td>
<td align="center">93.1</td>
<td align="center">91.2</td>
<td align="center">87.3</td>
<td align="center">85.6</td>
<td align="center">62.8</td>
<td align="center">88.7</td>
</tr>

<!-- ROW: pcm-tiny -->
 <tr><td align="center">PCM-Tiny</td>
<td align="center">87.1</td>
<td align="center">85.2</td>
<td align="center">93.3</td>
<td align="center">90.5</td>
<td align="center">87.1</td>
<td align="center">85.2</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>

<tbody><table>

## Training and Testing

### ScanObjectNN
```
# train
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/scanobjectnn/PCM.yaml
# test
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/scanobjectnn/PCM.yaml  mode=test --pretrained_path /path/to/PCM.pth
```
### ModelNet40
```
# train
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/modelnet40ply2048/PCM.yaml
# test
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py --cfg cfgs/modelnet40ply2048/PCM.yaml mode=test --pretrained_path /path/to/PCM.pth
```
### ShapeNetPart
```
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/PCM.yaml
# test
CUDA_VISIBLE_DEVICES=0 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/PCM.yaml mode=test --pretrained_path /path/to/PCM.pth
```
### S3DIS
```
# train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/segmentation/main.py --cfg cfgs/s3dis/PCM.yaml
# test
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/PCM.yaml wandb.use_wandb=False mode=test --pretrained_path /path/to/PCM.pth
```

## Params, GFlops and Throughout
```
CUDA_VISIBLE_DEVICES=0 python examples/profile.py --cfg cfgs/scanobjectnn/PCM.yaml batch_size=128 num_points=1024 timing=True flops=True
```
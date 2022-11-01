# SL3D: Self-supervised-Self-labeled 3D Recognition

## Introduction
This repository holds the official implementation of SL3D framework described in the paper: [arXiv](https://arxiv.org/abs/2210.16810)

There are a lot of promising results in 3D recognition, including classification, object detection, and semantic segmentation. However, many of these results rely on manually collecting densely annotated real-world 3D data, which is highly time-consuming and expensive to obtain, limiting the scalability of 3D recognition tasks. Thus in this paper, we study unsupervised 3D recognition and propose a Self-supervised-Self-Labeled 3D Recognition (SL3D) framework. SL3D simultaneously solves two coupled objectives, i.e., clustering and learning feature representation to generate pseudo labeled data for unsupervised 3D recognition. SL3D is a generic framework and can be applied to solve different 3D recognition tasks, including classification, object detection, and semantic segmentation.

<p align="center">
<img src="figures/SL3D_illustration_JPG.jpg" width="80%" height="80%">
</p>

## Installation

### Requirements
* Linux (tested on Ubuntu 18.04)
* Python 3.7+
* PyTorch 1.10.1
* GPU(s) used for experiment: 4x Nvidia GeForce RTX 3090

Please run `conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch` to install PyTorch 1.10.1 and run `pip install -r requirements.txt` to install other required packages.

### Prepare datasets

* ScanNetv2 dataset link: https://github.com/ScanNet/ScanNet
* Modelnet40 dataset link: http://modelnet.cs.princeton.edu/ModelNet40.zip 


## Usage

### Preprocess (optional, only for object detection and Semantic segmentation tasks, ScanNetv2 dataset)
Reference: [link](https://github.com/facebookresearch/WyPR/blob/main/docs/RUNNING.md)
1. Shape detection
```shell
# Unzip cgal library
unzip 3rd_party.zip

cd preprocess/shape_det
mkdir build; cd build
cmake -DCGAL_DIR="$(realpath ../../3rd_party/cgal/)" -DCMAKE_BUILD_TYPE=Debug ../ 
make

# 1st: Convert data from *.ply into *.xyz which CGAL can use
#      You should open some *.xyz files in meshlab to make sure things are correct
# 2nd: Generate running scripts
# Note: you need to change the `data_path` to be the absolute path of output
python shape_det/generate_scripts.py

# Running
cd shape_det/build
# Use the generated *.sh files here to detect shapes
sh *.sh
# Results will be saved in *.txt files under shape_det/build/

# Pre-compute the adjancency matrix between detected shapes
python shape_det/preprocess.py
```
pre-computed detected shape: [link](https://www.dropbox.com/s/a8vrkya9wtayz3h/scannet_cgal_results.zip?dl=0)

2. Geometric Selective Search (GSS)
```shell
cd preprocess/gss

python selective_search_3d_run.py --split trainval --dataset scannet --datapath <scannet path> --cgal_path <cgal path> --seg_path <seg path>  

python selective_search_3d_ensamble.py
```

3. Prepare SL3D input data
```shell
cd preprocess/scannet
python get_scannet_object_unsup.py
```
precomputed unsupervised 3D proposals using gss: [link](https://www.dropbox.com/s/ahgbg5zehdlpb88/scannet_gss_unsup.zip?dl=0)

### Train SL3D and generate pseudo labels

```shell
cd self-label
python main_sl3d.py --dataset <scannet or modelnet40> --data_path <path to obj-level point cloud> --arch <point_transformer or pointnet2> --ncl <number of pseudo classes>
```

### Downstream tasks
Prepare dataset for downstream tasks, to align the number of pseudo classes with the actual class for each downstream tasks, please manually align the pseudo labels index with the actual class index in get_sl3d_psuedo_labels.py
```shell
# Create a symbolic-link to import SL3D models
ln -s /self-label/models/ /preprocess/scannet/models

cd preprocess/scannet

python get_sl3d_pseudo_labels.py
```

1. 3D Classification tasks
Please refer to [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) to setup the environment <br>
To train and evaluate the model:
```shell
cd downstream_tasks/cls
python main.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg
```

2. 3D Object Detection tasks
Please refer to [facebookresearch/votenet](https://github.com/facebookresearch/votenet) to setup the environment<br>
To train and evaluate the model:
```shell
cd downstream_tasks/det
python train.py --dataset scannet --log_dir log_scannet --num_point 40000

python eval.py --dataset scannet --checkpoint_path log_scannet/checkpoint.tar --dump_dir eval_scannet --num_point 40000 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
```

3. 3D Semantic Segmentation tasks
Please refer to [[daveredrum/Pointnet2.ScanNet](https://github.com/daveredrum/Pointnet2.ScanNet) to setup the environment <br>
To train and evaluate the model:
```shell
cd downstream_tasks/semseg
python scripts/train.py --use_color --use_normal --use_msg

python scripts/eval.py --folder <time_stamp>
```

## Changelog

* __09/29/2022__ Release the code *BETA


## Acknowledgement
* [yukimasano/self-label](https://github.com/yukimasano/self-label): Paper author and official code repo of [Self-labelling via simultaneous clustering and representation learning]
* [facebookresearch/WyPR](https://github.com/facebookresearch/WyPR): Paper author and official code repo of [Weakly-supervised Point Cloud Recognition]
* [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch): [Pytorch Implementation of PointNet and PointNet++]
* [facebookresearch/votenet](https://github.com/facebookresearch/votenet): Paper author and official code repo of [Deep Hough Voting for 3D Object Detection in Point Clouds]
* [daveredrum/Pointnet2.ScanNet](https://github.com/daveredrum/Pointnet2.ScanNet): [Pytorch implementation of PointNet++ Semantic Segmentation on ScanNet]
* [qq456cvb/Point-Transformers](https://github.com/qq456cvb/Point-Transformers): [Pytorch Implementation of Various Point Transformers]

## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.




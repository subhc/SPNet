## SPNet: Semantic Projection Network for Zero- and Few-Label Semantic Segmentation
##### Yongqin Xian<sup>\*</sup>, Subhabrata Choudhury<sup>\*</sup>, Yang He, Bernt Schiele, Zeynep Akata. In CVPR 2019
##### Abstract:
> <sup> Semantic segmentation is one of the most fundamental problems in computer vision and pixel-level labelling in this context is particularly expensive. Hence, there have been several attempts to reduce the annotation effort such as learning from image level labels and bounding box annotations. In this paper we take this one step further and focus on the challenging task of zero- and few-label learning of semantic segmentation. We define this task as image segmentation by assigning a label to every pixel even though either no labeled sample of that class was present during training, i.e. zero-label semantic segmentation, or only a few labeled samples were present, i.e. few-label semantic segmentation.Our goal is to transfer the knowledge from previously seen classes to novel classes. Our proposed semantic projection network (SPNet) achieves this goal by incorporating class-level semantic information into any net- work designed for semantic segmentation, in an end-to-end manner. We also propose a benchmark for this task on the challenging COCO-Stuff and PASCAL VOC12 datasets. Our model is effective in segmenting novel classes, i.e. alleviating expensive dense annotations, but also in adapting to novel classes without forgetting its prior knowledge, i.e. generalized zero- and few-label semantic segmentation.</sup>

##### Data Splits:
You can find the splits information in [data/datasets](/data/datasets) folder. Please see [splits.ipynb](splits.ipynb) for an example on how to use these files. 

##### Source Code:
Our code is built upon kazuto1011's superb [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch) codebase. Source code will be released soon. 

##### Initialization Model:
DeepLab initialization should be done using an ImageNet pretrained ResNet. You can use `ResNet-101-model.caffemodel` from [here](https://github.com/KaimingHe/deep-residual-networks#models). Use [this](https://github.com/kazuto1011/deeplab-pytorch#initial-weights) to convert it to a pytorch model.
##### Hyperparameters:
You can find the hyperparameters used for our paper in the [config](config) folder. 

##### Citation:
If you find this useful, please cite our work as follows:
```
@InProceedings{XianCVPR2019b,
author = {Xian, Yongqin and Choudhury, Subhabrata and He, Yang and Schiele, Bernt and Akata, Zeynep},
title = {Semantic Projection Network for Zero- and Few-Label Semantic Segmentation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```


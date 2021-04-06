# detection-fields-attributes


A [PyTorch](https://pytorch.org/) implementation of paper [Detecting 32 Pedestrian Attributes for Autonomous Vehicles](https://arxiv.org/abs/2012.02647) by Taylor Mordan, Matthieu Cord, Patrick Pérez and Alexandre Alahi.

#### Abstract
> Detecting 32 Pedestrian Attributes for Autonomous Vehicles
>
>Pedestrians are arguably one of the most safety-critical road users to consider for autonomous vehicles in urban areas.
>In this paper, we address the problem of jointly detecting pedestrians and recognizing 32 pedestrian attributes.
>These encompass visual appearance and behavior, and also include the forecasting of road crossing, which is a main safety concern.
>For this, we introduce a Multi-Task Learning (MTL) model relying on a composite field framework, which achieves both goals in an efficient way.
>Each field spatially locates pedestrian instances and aggregates attribute predictions over them.
>This formulation naturally leverages spatial context, making it well suited to low resolution scenarios such as autonomous driving.
>By increasing the number of attributes jointly learned, we highlight an issue related to the scales of gradients, which arises in MTL with numerous tasks.
>We solve it by normalizing the gradients coming from different objective functions when they join at the fork in the network architecture during the backward pass, referred to as fork-normalization.
>Experimental validation is performed on JAAD, a dataset providing numerous attributes for pedestrian analysis from autonomous vehicles, and shows competitive detection and attribute recognition results, as well as a more stable MTL training.

![detection_schema](imgs/detection_attributes.pdf)


## Installation

This project requires to install the dev branch of [OpenPifPaf](https://vita-epfl.github.io/openpifpaf/dev/intro.html) first.

Dataset [JAAD](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/) is used to train and evaluate the model.

This project has been tested with python==3.7 and pytorch==1.7.1.


## Interfaces  

This project is build as a plugin module for OpenPifPaf, and as such, should benefit from all the functionalities offered by it.

All the commands can be run through openpifpaf interface using subparsers, as described in its documentation.

For example, training on JAAD can be run with the command:
```
python3 -m openpifpaf.train \
  --dataset='jaad' --jaad-root-dir='<JAAD folder>' --jaad-training-set='train' --jaad-validation-set='val' \
  --epochs=5 \
  --batch-size=4 \
  --log-interval=10 --val-interval=1 \
  --lr=5e-4 --momentum=0.95 --lr-warm-up-start-epoch=-1 \
  --weight-decay=5e-4 \
  --jaad-pedestrian-attributes all \
  --basenet=fn-resnet50 \
  --pifpaf-pretraining \
  --jaad-head-upsample=2 \
  --fork-normalization-operation='power' --fork-normalization-duplicates=35 \
  --detection-bias-prior=0.01 \
  --attribute-regression-loss='l1' --attribute-focal-gamma=2 \
  --lambdas 7.0 7.0 7.0 7.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 \
  --auto-tune-mtl
```
or by selecting the appropriate arguments, if needed.


## Citation
If you use this project in your research, please cite the corresponding paper:
```text
@article{mordan2020detecting,
  title={Detecting 32 Pedestrian Attributes for Autonomous Vehicles},
  author={Mordan, Taylor and Cord, Matthieu and P{\'e}rez, Patrick and Alahi, Alexandre},
  journal={arXiv preprint arXiv:2012.02647},
  year={2020}
}
```

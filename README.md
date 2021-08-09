## Evaluating Weakly Supervised Object Localization Methods Right (CVPR 2020)

[Neurocomputing 2020 paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231220317331) 

Sabrina Narimene Benassou<sup>1*</sup>, Wuzhen Shi<sup>2*</sup>, Feng Jiang<sup>1</sup>
Sanghyuk Chun<sup>3</sup>, Zeynep Akata<sup>4</sup>, Hyunjung Shim<sup>1</sup>  
<sub>\* Equal contribution</sub>


<sup>1</sup> <sub>School of Computer Science and Technology, Harbin Institute of Technology</sub>  
<sup>2</sup> <sub>College of Electronics and Information Engineering, Shenzhen University</sub>


Weakly Supervised Object Localization is challenging due to the lack of bounding box annotations. Previous works tend to generate a Class Activation Map
(CAM) to localize the object. However, the CAM highlights only the most discirminative part of the object and does not highlight the whole object. To
address this problem, we propose an Entropy Guided Adversarial model (EGA model) to perform better localization of objects. EGA model uses adversarial
learning method to create adversarial examples, i.e., images where a perturbation is added. Treating adversarial examples as data augmentation regularize
our model as well as detect more discriminative visual pattern on the CAM. We further apply the Shannon entropy on the generated CAM to guide the
model during training. Minimizing the entropy loss forces the model to generate a high-confident CAM. The high-confident CAM detects the whole object
while excludes the background. Extensive experiments show that EGA model improves classification and localization performances on state-of-the-art bench-
marks. Ablation experiments also show that both the adversarial learning and the entropy loss contribute to the algorithm performance.
<img src="teaser.png" width="60%" height="60%" title="" alt="RubberDuck"></img>

__Overview of WSOL performances 2016-2019.__ Above image shows that recent improvements in WSOL are 
illusory due to (1) different amount of implicit full supervision through 
validation and (2) a fixed score-map threshold to generate object boxes. Under 
our evaluation protocol with the same validation set sizes and oracle threshold 
for each method, CAM is still the best. In fact, our few-shot learning baseline, 
i.e., using the validation supervision (10 samples/class) at training time, 
outperforms existing WSOL methods.

## Updates

- __9 Jul, 2020__: [Journal submission](https://arxiv.org/abs/2007.04178) available.
- __27 Mar, 2020__: [New WSOL evaluations](#5-library-of-wsol-methods) with `MaxBoxAccV2` are added.
- __28 Feb, 2020__: [New box evaluation](#improved-box-evaluation) (`MaxBoxAccV2`) is available.
- __22 Jan, 2020__: Initial upload.


## Table of contents 

* [1. Our dataset contribution](#1-our-dataset-contribution)
    + [The dataset splits](#the-dataset-splits)
* [2. Dataset downloading and license](#2-dataset-downloading-and-license)
    + [ImageNetV2](#imagenetv2)
    + [CUBV2](#cubv2)
    + [OpenImages30k](#openimages30k)
    + [Dataset statistics](#dataset-statistics)
    + [Licenses](#licenses)
* [3. Code dependencies](#3-code-dependencies)
* [4. WSOL evaluation](#4-wsol-evaluation)
    + [Prepare evaluation data](#prepare-evaluation-data)
    + [Prepare heatmaps to evaluate](#prepare-heatmaps-to-evaluate)
    + [Evaluate your heatmaps](#evaluate-your-heatmaps)
    + [Testing the evaluation code](#testing-the-evaluation-code)
* [5. Library of WSOL methods](#5-library-of-wsol-methods)
* [6. WSOL training and evaluation](#6-wsol-training-and-evaluation)
    + [Prepare train+eval datasets](#prepare-traineval-datasets)
        - [ImageNet](#imagenet)
        - [CUB](#cub)
        - [OpenImages](#openimages)
    + [Run train+eval](#run-traineval)
    + [Improved box evaluation](#improved-box-evaluation)
* [7. Code license](#7-code-license)
* [8. How to cite](#8-how-to-cite)

## 3. Code dependencies

To run the model, the scripts require only the following libraries: 
* [OpenCV](https://opencv.org/)
* [PyTorch](https://pytorch.org/)
* [munch](https://github.com/Infinidat/munch)

`pip freeze` returns the version information as below:
```
munch==2.5.0
numpy==1.18.1
opencv-python==4.1.2.30
Pillow==7.0.0
six==1.14.0
torch==1.4.0
torchvision==0.5.0
```

## 4. WSOL training and evaluation

Follow https://github.com/clovaai/wsolevaluation to prepare the datasets.

### Run train

Below is an example command line for the train+eval script.
```bash
python main_adv_ent.py --dataset_name OpenImages             --architecture vgg16               --wsol_method cam               --experiment_name OpenImages_vgg_ent_adv2               --pretrained TRUE                --num_val_sample_per_class 5                --large_feature_map False                --batch_size 1                --epochs 10                --lr 0.001               --lr_decay_frequency 3               --weight_decay 5.00E-04                --override_cache FALSE                --workers 4                --box_v2_metric False                --iou_threshold_list 50                --eval_checkpoint_type last  --epsilon 1   --k=1  
```

Below is an example command line for the eval script with the average of the scores from the softmax layer with 10 crops..
```bash
python main_adv_ent.py --dataset_name OpenImages             --architecture vgg16               --wsol_method cam               --experiment_name OpenImages_vgg_ent_adv2               --pretrained TRUE                --num_val_sample_per_class 5                --large_feature_map False                --batch_size 1                --epochs 10                --lr 0.001               --lr_decay_frequency 3               --weight_decay 5.00E-04                --override_cache FALSE                --workers 4                --box_v2_metric False                --iou_threshold_list 50                --eval_checkpoint_type last  --onlyTest True --resume True --tencrop True
```

See [config.py](config.py) for the full descriptions of the arguments, especially 
the method-specific hyperparameters.

## 5. How to cite

```
@article{EGA_model,
title = {Entropy guided adversarial model for weakly supervised object localization},
journal = {Neurocomputing},
volume = {429},
pages = {60-68},
year = {2021},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2020.11.006},
url = {https://www.sciencedirect.com/science/article/pii/S0925231220317331},
author = {Sabrina Narimene Benassou and Wuzhen Shi and Feng Jiang},
}
```


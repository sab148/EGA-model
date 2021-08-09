## Evaluating Weakly Supervised Object Localization Methods Right (CVPR 2020)

[Neurocomputing 2020 paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231220317331) 

Sabrina Narimene Benassou<sup>1</sup>, Wuzhen Shi<sup>2</sup>, Feng Jiang<sup>1</sup>


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
![a](https://user-images.githubusercontent.com/46344689/128652801-312661e1-e552-49f2-b2f7-4011b4c47d9f.png)

## 1. Code dependencies

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

## 2. WSOL training and evaluation

Follow https://github.com/clovaai/wsolevaluation to prepare the datasets.

### Run train

Below is an example command line for the train+eval script.
```bash
python main_adv_ent.py --dataset_name OpenImages             
                       --architecture vgg16              
                       --wsol_method cam               
                       --experiment_name OpenImages_vgg_ent_adv2               
                       --pretrained TRUE                
                       --num_val_sample_per_class 5                
                       --large_feature_map False               
                       --batch_size 1                
                       --epochs 10                
                       --lr 0.001               
                       --lr_decay_frequency 3               
                       --weight_decay 5.00E-04                
                       --override_cache FALSE                
                       --workers 4                
                       --box_v2_metric False                
                       --iou_threshold_list 50                
                       --eval_checkpoint_type last  
                       --epsilon 1   
                       --k=1  
```

Below is an example command line for the eval script with the average of the scores from the softmax layer with 10 crops.
```bash
python main_adv_ent.py --dataset_name OpenImages             
                       --architecture vgg16               
                       --wsol_method cam               
                       --experiment_name OpenImages_vgg_ent_adv2               
                       --pretrained TRUE                
                       --num_val_sample_per_class 5                
                       --large_feature_map False                
                       --batch_size 1                
                       --epochs 10                
                       --lr 0.001               
                       --lr_decay_frequency 3               
                       --weight_decay 5.00E-04                
                       --override_cache FALSE                
                       --workers 4                
                       --box_v2_metric False                
                       --iou_threshold_list 50                
                       --eval_checkpoint_type last  
                       --onlyTest True 
                       --resume True 
                       --tencrop True
```

See [config.py](config.py) for the full descriptions of the arguments, especially 
the method-specific hyperparameters.

## 3. How to cite

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


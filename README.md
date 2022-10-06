# PCA-based knowledge distillation towards lightweight and content-style balanced photorealistic style transfer models
Code for our CVPR 2022 paper 
[PCA-based knowledge distillation towards lightweight and content-style balanced photorealistic style transfer models](https://openaccess.thecvf.com/content/CVPR2022/html/Chiu_PCA-Based_Knowledge_Distillation_Towards_Lightweight_and_Content-Style_Balanced_Photorealistic_Style_CVPR_2022_paper.html)
![alt text](https://github.com/chiutaiyin/PCA-Knowledge-Distillation/blob/master/banner-pcakd.jpg)

## Advantages of our distilled models
- They achieve a better balance between content preservation and style transferral. 
Specifically, they transfer stronger style effects than [WCT2](https://github.com/clovaai/WCT2) and preserves better content than [PhotoWCT](https://github.com/NVIDIA/FastPhotoStyle) and [PhotoWCT2](https://github.com/chiutaiyin/PhotoWCT2).
- They are lightweight. The models distilled from VGG and MobileNet use only 283K and 73K parameters, respectively. 73K is only 0.72% of the number of parameters in WCT2, 10.12M.
- They run fast. Our VGG-distilled model runs around 10x, 10x, and 5x as fast as WCT2, PhotoWCT, and PhotoWCT2. Our MobileNet-distilled model is about 2x as fast as Our VGG-distilled model.
- They can support the 8K (7680 x 4320) resolution on a normal commercial GPU.
- To our best knowledge, this is the first distillation for photorealistic style transfer. The closest work to ours is [CKD](https://github.com/MingSun-Tse/Collaborative-Distillation) for artistic style transfer.

## Models and files
We apply our PCA distillation to two backbones: VGG and MobileNet. The corresponding files can be found in the folders ```VGG backbone``` and ```MobileNet backbone```.
For each backbone, we provide our trained parameters in the folder ```ckpts```, the demo of how to perform style transfer in the file ```demo.ipynb```, 
the training code in two files ```train_distilled_model.py``` and ```train_eigenbases.py```, and the distilled model structure in ```utils/lightweight_model.py```.

Note that ```train_distilled_model.py``` has to be run first to derive the global eignebases and ```train_eigenbases.py``` uses the eigenbases to distill models.


## Requirements 
- tensorflow v2.4.1 or above (we developed the models with tf-v2.4.1)

## Citation
If you find this repo useful, please cite our paper **PCA-based knowledge distillation towards lightweight and content-style balanced photorealistic style transfer models** published in CVPR 2022.

This codebase is largely extended from our previous work [**PhotoWCT2: Compact Autoencoder for Photorealistic Style Transfer Resulting from Blockwise Training and Skip Connections of High-Frequency Residuals**](https://github.com/chiutaiyin/PhotoWCT2) published in WACV2022. 
Please take a look at it if interested.

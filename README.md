
# 2D Multi-Mineral Segmentation of Rock SEM Images using Convolutional Neural Networks and Generative Adversarial Network

> This repository contains machine learning models written for the final project of the Skoltech Deep Learning course

>Tags: UNet, Linknet, ResUnet, inceptionv3, inceptionresnetv2, vgg16, vgg19, resnet18, efficientnetb3, efficientnetb4, backbone, DigitalRock, Rock, Pore, Segmentation, Neural Network, Deep Learning, Deep, Learning, grains, SEM, QEMSCAN, Segmentation Neural Network, Tensorflow, Keras, CNN, Convolutional Neural Network, GAN, Generative adversarial network
### Implemented by: 
* Vladislav Alekseev
* Victoria Dochkina
* Daniil Bragin
* Emre Özdemir

## Brief description of the project
Segmentation of images of rocks is a crucial step in almost any Digital Rock workflow. However, the QEMSCAN scanning method is a very time and money consuming approach. In this paper, we investigate an application of three popular Convolutional Neural Networks
(CNN) architectures: U-Net, LinkNet, ResUNet. We also applied the pix2pix - conditional Generative Adversarial Network (cGAN) for the segmentation of 2D microtomographic rock images.
Our dataset contains nine pairs of images. 2D images of rock surface obtained by scanning electron microscopy (SEM) and in one case QEMSCAN grayscale images are used as input for segmentation. Manually modified QEMSCAN images with mineral labels are used as ground truth labels. We have succeeded in building proper workflow, starting from image preprocessing and ending with inferencing the model results. We have found that U-Net (backbones: inceptionv3, efficientnetb4, inceptionresnetv2) and LinkNet (backbone: inceptionv3) performed better on this data.


## Guideline

in "lib" folder you can find all the necessary metrics, plot functions and defined models in "custom_modelname" .py files;

in "preprocessing" you can find all notebooks, connected with data processing and fixing pattern error in the QEMSCAN dataset;

"CNN_Segmentation_SEM.ipynb" - main notebook, reproducable in Google Colaboratory;

in "GAN" folder you can find Generative Adversarial Network implementation of segmentation model.

The Dataset is commercial, so it is not provided with the script.Though code is reproducable for any other dataset.

## Dataset description
To test the performance of segmentation algorithms for the purpose of Multi-Mineral Segmentation, we use images of sandstone samples obtained separately by SEM and QEMSCAN. Initial SEM data is 9 high-resolution images(88000×87000). QEMSCAN data is 9 colored low-resolution images of the same samples with each color associated with a mineral component or pore space (4700×4700 pixels). For the dataset we converted color-coded images to greyscale-coded images (classes coded with equidistant numbers from 0 to 255).
The total number of classes is 23 including pore/background category. Based on the fact that SEM doesn’t distinguish between several frequently occurring minerals we decided to combine all classes to 4:  Pores (0) Quartz (1),  Albite (2),  mixed group including mostly clays and accessory minerals (3).
Finally we got 4 main classes and presented each of them as a binary mask.

Both the SEM and QEMSCAN images were received from a company as a part of commercial contract. So, that is why the dataset is not provided with the script.



![Preprocessing the Dataset](https://i.imgur.com/jAMoOTJ.png)

<center> Figure 1. Initial preprocessing workflow </center>

## Main steps perfomed while doing the research:

1) Image preprocessing and identifying problems with initial data
2) Identifying and addressed class imbalances in several independent ways
3) Training 3 different convolution segmentation models with additional approaches: U-net and Linknet + backbones, ResUnet
4) Training GAN for image segmentation

## Experiment
In the experiment part training of the models listed in  tables in **Results** section was performed. Three different approaches to the training of each of the models were tested:

1. For all the backbones use weights trained on 2012 ILSVRC ImageNet dataset and train the whole model.
2. For all the backbones use weights trained on 2012 ILSVRC ImageNet dataset and freeze the encoder part in order to train only randomly initialized decoder and not to change weights of trained encoder with huge gradients during first steps of training. 
3. Randomly initialize encoder and decoder weights.

Results of the best approach for each of the models are presented in tables in **Results** section.

## Backbones

 One of the following backbones was used in the encoding part for U-Net and Linknet models:
 * inceptionv3
 * inceptionresnetv2
 * resnet18
 * vgg16
 * vgg19
 * efficientnetb3
 * efficientnetb4

## Data augmentation description:
* range of angles: from -15 to 15 degrees
* width shift range: 0.05 % in both directions
* height shift range: 0.05 % in both directions
* shear range: 50 degrees
* zoom range: 30%
* horizontal flipping
* vertical flipping

## 1.1 U-Net
The following schematically structure of U-Net was used:

<img src="https://github.com/ddvika/SEM_segmentation/blob/master/imgs/Unet.png" width="500" >

<center> Figure 2. U-Net architecture </center>


## 1.1 Linknet

<img src="https://github.com/ddvika/SEM_segmentation/blob/master/imgs/Linknet.png" width="500" >

<center> Figure 3. Linknet architecture </center>


## 1.1 ResUnet

<img src="https://github.com/ddvika/SEM_segmentation/blob/master/imgs/ResUnet.png" width="500" >

<center> Figure 4. ResUnet architecture </center>



#### Training details 
Due to the imbalance factor and specification of the task, it was decided to use combination of region-based and distribution-based losses like: Dice and Focal  loss functions.   Dice loss directly optimize the Dice coefficient which is the most commonly used segmentation evaluation metric, while Focal loss adapts the standard Cross Entropy to deal with extreme foreground-background class imbalance, where the weights of well-classified examples are reduced. Class weights were also assigned into Dice loss. The total final loss is presented by:


<img src="https://github.com/ddvika/SEM_segmentation/blob/master/imgs/loss.png" width="200" >

where DL is Dice Loss, FL - Focal Loss, and c - constant value.

The optimization method is chosen to be Adam with  learning rate scheduling. After the i-th run, learning rate is reduced with a cosine annealing for each batch as follows:

<img src="https://github.com/ddvika/SEM_segmentation/blob/master/imgs/cosine_lr.png" width="400" >

where η_min and η_max are ranges for the learning rate, T_cur accounts for how many epochs have been performed since the last restart.

## Results
### IoU scores for Convolutional models:

<img src="https://github.com/ddvika/SEM_segmentation/blob/master/imgs/128x128.png" width="500" >
<center> Table 1. Results for low-resolution 128x128 images  </center>

<img src="https://github.com/ddvika/SEM_segmentation/blob/master/imgs/256x256.png" width="500" >
<center> Table 2. Results for low-resolution 256x256 images </center>

<img src="https://github.com/ddvika/SEM_segmentation/blob/master/imgs/512x512.png" width="500" >
<center> Table 3. Results for high-resolution 512x512 images </center>


### Predictions obtained with U-Net + efficientnetb4 backbone

<img src="https://github.com/ddvika/SEM_segmentation/blob/master/imgs/ex2.jpg" width="1000" >

<img src="https://github.com/ddvika/SEM_segmentation/blob/master/imgs/ex4.jpg" width="1000" >

<center> Figure 5. U-Net + efficientnetb4 backbone prediction for 5 classes case </center>

### IoU scores for GAN:

<img src="https://github.com/ddvika/SEM_segmentation/blob/master/imgs/gan_512x512.png" width="500" >

<center> Table 4. Results for high-resolution 512x512 images </center>

### Predictions obtained with GAN

<img src="https://github.com/ddvika/SEM_segmentation/blob/master/imgs/ex_gan.png" width="1000" >

<center> Figure 6. GAN prediction for 5 classes case </center>

## Conclusion

* We have identified several problems related to dataset: class imbalance, image-mask inconsistencies and addressed them in preprocessing

* We have tested U-Net, Linknet and ResUnet models for image segmentations in several configurations listed in **Experiment** part. Also, we have implemented pix2pix segmentation with GAN model.


## Installation

#### Requirements:

* python >= 3.6
* matplotlib >= 3.1.1
* keras >= 2.2.0 or tensorflow >= 1.14
* setuptools >= 41.0.1
* numpy >= 1.16



```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

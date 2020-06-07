
# 2D Multi-Mineral Segmentation of Rock SEM Images using Convolutional Neural Networks and Generative Adversarial Network

> This repository contains machine learning models written for the final project of the Skoltech Machine Learning course

>Tags: UNet, Linknet, ResUnet, inceptionv3, inceptionresnetv2, vgg16, vgg19, resnet18, efficientnetb3, efficientnetb4, backbone, DigitalRock, Rock, Pore, Segmentation, Neural Network, Deep Learning, Deep, Learning, grains, SEM, QEMSCAN, Segmentation Neural Network, Tensorflow, Keras, CNN, Convolutional Neural Network, GAN, Generative adversarial network
### Implemented by: 
* Vladislav Alekseev
* Victoria Dochkina
* Daniil Bragin
* Emre Özdemir

## Annotation
Segmentation of images of rocks is a crucial step in almost any Digital Rock workflow. However,established segmentation methods introduce human bias. In this paper, we investigate an application of two popular convolutional neural networks(CNN) architectures: SegNet and U-Net for segmentation of 2D microtomographic rock images. Our dataset contains eight pairs of images. 2D images of rock surface obtained by scanning electronmicroscopy (SEM) are used as input for segmentation. Manually modified QEMSCAN images reused as a ground truth labels.

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
3) Trained 2 high-accuracy CNN-based models for image segmentation: U-Net and SegNet

## Experiment
After careful consideration of the original images we found periodic shifts on a QEMSCAN image diagnosed by broken contours of the grains and duplicated rows and columns of pixels.  As a consequence, all crops from the original QEMSCAN images for training and inference suffered from the slight shifts (3-5 pixels in one line depending on the direction) from the input.
By the sight of it this was caused by wrong merging of sub-images (patches) – essentially an error of the device and appeared through no fault of our own. All patches were merged using thesame algorithm, so all original labeled images have the same error that can potentially be fixed manually. After this discovery we’ve found recurring error pattern inall images and eliminated it by removing duplicating rowsand columns (60 pixels in vertical axes, and 36 pixels inhorizontal direction). Then we expanded images to restorethe  size.   We  applied  this  algorithm  for  all  QEMSCAN images and got new ground truth images.

## 1.1 U-Net
The following schematically structure of U-Net were used:

![UNet](https://i.imgur.com/VqMixFA.png)

<center> Figure 2. U-Net architecture </center>


#### U-Net description:
* Unet-like architecture with one of the following backbone:
**1
**2
* Batch-normalization after every convolutional layer
* Activation function: ReLU
* Maxpooling 2×2
* Dropout with rate 30%
* 963201 total trainable parameters
* 966145 total parameters


#### Data augmentation description:
* range of angles: from -15 to 15 degrees
* width shift range: 0.05 % in both directions
* height shift range: 0.05 % in both directions
* shear range: 50 degrees
* zoom range: from 80% to 120%


## 1.2 Prediction obtained with U-Net + efficientnetb4 backbone

![UNetResults](https://github.com/ddvika/SEM_segmentation/blob/master/imgs/ex1.jpg?raw=true)
![UNetResults](https://github.com/ddvika/SEM_segmentation/blob/master/imgs/ex2.jpg?raw=true)
![UNetResults](https://github.com/ddvika/SEM_segmentation/blob/master/imgs/ex3.jpg?raw=true)
![UNetResults](https://github.com/ddvika/SEM_segmentation/blob/master/imgs/ex4.jpg?raw=true)

<center> Figure 3. U-Net  + efficientnetb4 backbone prediction for 5 classes case </center>



#### Training details 
In both models the optimization method is chosen to be Adam (learning rate = 0.01, β1= 0.9, β2= 0.99). Batch size equals 3. Number of iterations during training is selected as follows.Every 100 iterations, we evaluate accuracy on holdout set(10% of patches for each rock sample) and stop the training,when the target metric stabilizes. In this experiment, target metric stabilized when the number of epochs reached about 80–100.

## Results

* We have identified several problems related to dataset: class imbalance, image-mask inconsistencies and addressed them in preprocessing

* We have tested U-Net and SegNet models for image segmentations in several configurations: with original images and with fixed images.  SegNet showed bad prediction resolution.

* We have found that U-Net performed better on thisdata,  possibly due to special architecture decision-skip connections.


## Future work

* After analyzing results we are planning to combine albite and quartz (the second and the third classes) because the texture is lost due to extreme compression,the original image was compressed more than 18 times,because we adapted the resolution of the image to theoutput QEMSCAN labels resolution.

* To increase a generalization capability, it is necessary to do data augmentation taking into account physical principles of image forming: model of noise, typical artefacts, parameters of reconstruction procedure, etc.

* To find a reasonable trade-off between the amount ofdata and time for model training

* To perform post-processing of the output of neural network can eliminate some abnormalities, for example, unconnected fragments of solids and overlapping fragments

## Installation

#### Requirements:

* python 3
* matplotlib >= 3.1.1
* keras >= 2.2.0 or tensorflow >= 1.14
* setuptools >= 41.0.1
* numpy >=1.16


## Comments


in "lib" folder you can find all the necessary metrics, plot functions and defined models in "custom_unet" and "custom_segnet" .py files.

in "pre-processing" you can find all notebooks, connected with data processing and fixing pattern error in the QEMSCAN dataset

"SEM_Images_segmentation.ipynb" - main notebook, reproducable in Google Colaboratory.


The Dataset is commercial, so it is not provided with the script.Though code is reproducable for any other dataset.


```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

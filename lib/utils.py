#define metrics
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread,imshow
from skimage.transform import resize
from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_img_mask_array(img_input_file, img_labels_file, dim = 256, n_channels = 1):
    
    img_input = Image.open(img_input_file)
    img_input = img_input.resize((dim, dim), Image.NEAREST)
    img_input = np.asarray(img_input)/255
    img_input = img_input[:,:,None]
    
    
    img_labels = Image.open(img_labels_file)
    img_labels = img_labels.resize((dim, dim), Image.NEAREST)
    img_labels = np.round(np.array(img_labels)/255, 1)
    img2 = img_input
    if n_channels>1:
      img2 = np.zeros( ( np.array(img_input).shape[0], np.array(img_input).shape[1], n_channels ) )
      img2[:,:,0] = img_input[:,:,0] # same value in each channel
      img2[:,:,1] = img_input[:,:,0]
      img2[:,:,2] = img_input[:,:,0]
    
    return img2, img_labels

def mask_to_rgba(mask, color="red"):

    h = mask.shape[0]
    w = mask.shape[1]
    zeros = np.zeros((h, w))
    ones = mask.reshape(h, w)
    return np.stack((ones, zeros, zeros, ones), axis=-1)


def plot_images(true_imgs, true_labels, y_pred = None, num_to_plot = 3, class_num = 0):
    for i in range(num_to_plot):
        fig, axs = plt.subplots(1,3, figsize = (20,20))
        axs[0].imshow(true_imgs[i], cmap = 'gray')
        axs[0].set_title('Input image',fontsize=15)
        axs[1].imshow(true_labels[i,:,:, class_num], cmap = 'gray')
        axs[1].set_title('True label',fontsize=15)
        if y_pred is None:
            axs[2].imshow(true_imgs[i], cmap = 'gray')
            axs[2].imshow(mask_to_rgba(true_labels[i,:,:, class_num]), alpha = 0.5)
            axs[2].set_title('Overlay',fontsize=15)
        else:
            axs[2].imshow(true_imgs[i], cmap = 'gray')
            axs[2].imshow(mask_to_rgba(y_pred[i,:,:, class_num]), alpha = 0.5)
        plt.show()
        
        
def plot__predicted_images(true_imgs, true_labels, y_pred = None, num_to_plot = 3, class_num = 0):
    for i in range(num_to_plot):
        fig, axs = plt.subplots(1,4, figsize = (20,20))
        axs[0].imshow(true_imgs[i], cmap = 'gray')
        axs[0].set_title('Input image',fontsize=15)
        axs[1].imshow(true_labels[i,:,:, class_num], cmap = 'gray')
        axs[1].set_title('True label',fontsize=15)
        if y_pred is None:
            axs[2].imshow(true_imgs[i], cmap = 'gray')
            axs[2].imshow(mask_to_rgba(true_labels[i,:,:, class_num]), alpha = 0.5)
            axs[2].set_title('Overlay',fontsize=15)
        else:
            axs[2].imshow(y_pred[i,:,:, class_num], cmap = 'gray')
            axs[2].set_title('Predicted label',fontsize=15)            
            
            axs[3].imshow(true_imgs[i], cmap = 'gray')
            axs[3].imshow(mask_to_rgba(y_pred[i,:,:, class_num]), alpha = 0.5)
            axs[3].set_title('Overlay',fontsize=15)
        plt.show()
        
        
def binarize_labels(masks, colormap):
    num_classes = len(colormap)
    target_shape = masks[0].shape[:2]+(num_classes,)
    encoded_images = []
    for i in range(len(masks)):
        img_decoded = masks[i]
        encoded_image = np.zeros(target_shape, dtype = np.uint8)
        for j in range(len(colormap)):
            encoded_image[:,:,j] = np.all(img_decoded.reshape( (-1,1) ) == colormap[j], axis=1).reshape(target_shape[:2])
        encoded_images.append(encoded_image)
    return np.array(encoded_images)



def augment_data(X_train,Y_train, batch_size=5, data_gen_args=dict()):

    X_data = ImageDataGenerator(**data_gen_args)
    Y_data = ImageDataGenerator(**data_gen_args)
    
    X_data.fit(X_train, augment=True, seed=0)
    Y_data.fit(Y_train, augment=True, seed=0)
    
    X_train_aug = X_data.flow(
        X_train, batch_size=batch_size, seed=0, shuffle=True
    )
    Y_train_aug = Y_data.flow(
        Y_train, batch_size=batch_size, seed=0, shuffle=True
    )

    generator = zip(X_train_aug, Y_train_aug)
    return (pair for pair in generator)


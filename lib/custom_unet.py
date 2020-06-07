

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.compat.v1.layers import conv2d_transpose

def conv2d_block(input_data,n_filters,kernel_size=3,
                 kernel_init = "he_normal",batchnorm=True):
    
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer = kernel_init,
               padding="same")(input_data)
    if batchnorm is True:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer = kernel_init,
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def custom_unet(input_img, n_filters=32, dropout=0.5, batchnorm=True,
                num_classes = 1, activation = 'sigmoid'):

    
    x1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=2, batchnorm=batchnorm)
    y1 = MaxPooling2D((2, 2)) (x1)
    y1 = Dropout(dropout*0.5)(y1)

    x2 = conv2d_block(y1, n_filters=n_filters*2, kernel_size=2, batchnorm=batchnorm)
    y2 = MaxPooling2D((2, 2)) (x2)
    y2 = Dropout(dropout)(y2)

    x3 = conv2d_block(y2, n_filters=n_filters*4, kernel_size=2, batchnorm=batchnorm)
    y3 = MaxPooling2D((2, 2)) (x3)
    y3 = Dropout(dropout)(y3)

    x4 = conv2d_block(y3, n_filters=n_filters*8, kernel_size=2, batchnorm=batchnorm)
    y4 = MaxPooling2D(pool_size=(2, 2)) (x4)
    y4 = Dropout(dropout)(y4)
    
    x5 = conv2d_block(y4, n_filters=n_filters*16, kernel_size=2, batchnorm=batchnorm)
    
    z6 = Conv2DTranspose(n_filters*8, (2, 2), strides=(2, 2), padding='same') (x5)
    z6 = concatenate([z6, x4])
    z6 = Dropout(dropout)(z6)
    x6 = conv2d_block(z6, n_filters=n_filters*8, kernel_size=2, batchnorm=batchnorm)

    z7 = Conv2DTranspose(n_filters*4, (2, 2), strides=(2, 2), padding='same') (x6)
    z7 = concatenate([z7, x3])
    z7 = Dropout(dropout)(z7)
    x7 = conv2d_block(z7, n_filters=n_filters*4, kernel_size=2, batchnorm=batchnorm)

    z8 = Conv2DTranspose(n_filters*2, (2, 2), strides=(2, 2), padding='same') (x7)
    z8 = concatenate([z8, x2])
    z8 = Dropout(dropout)(z8)
    x8 = conv2d_block(z8, n_filters=n_filters*2, kernel_size=2, batchnorm=batchnorm)

    z9 = Conv2DTranspose(n_filters*1, (2, 2), strides=(2, 2), padding='same') (x8)
    z9 = concatenate([z9, x1], axis=3)
    z9 = Dropout(dropout)(z9)
    x9 = conv2d_block(z9, n_filters=n_filters*1, kernel_size=2, batchnorm=batchnorm)
    
    outputs = Conv2D(num_classes, (1, 1), activation=activation) (x9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
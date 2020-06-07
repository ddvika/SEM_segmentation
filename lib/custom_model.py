
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

def get_custom_model(input_img, n_filters=16, kernel_size = 3, dropout=0.5,kernel_initializer = 'he_normal', activation = 'relu'):
    # contracting path
    c1 = Conv2D(filters=n_filters*1, kernel_size= kernel_size, kernel_initializer=kernel_initializer,
               padding="same")(input_img)
    c1 = Activation(activation)(c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    #p1 = Dropout(dropout*0.5)(p1)

    c2 = Conv2D(filters=n_filters*2, kernel_size= kernel_size, kernel_initializer=kernel_initializer,
               padding="same")(p1)
    c2 = Activation(activation)(c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    #p2 = Dropout(dropout)(p2)

    c3 = Conv2D(filters=n_filters*4, kernel_size= kernel_size, kernel_initializer=kernel_initializer,
               padding="same")(p2)
    c3 = Activation(activation)(c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    #p3 = Dropout(dropout)(p3)

    
    c5 = Conv2D(filters=n_filters*8, kernel_size= kernel_size, kernel_initializer=kernel_initializer,
                padding = "same")(p3)
    c5 = Activation(activation)(c5)
    
    # expansive path

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c5)
    #u7 = Dropout(dropout)(u7)
    c7 = Conv2D(filters=n_filters*4, kernel_size= kernel_size, kernel_initializer=kernel_initializer,
               padding="same")(u7)
    c7 = Activation(activation)(c7)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    #u8 = Dropout(dropout)(u8)
    c8 = Conv2D(filters=n_filters*2, kernel_size= kernel_size, kernel_initializer=kernel_initializer,
               padding="same")(u8)
    c8 = Activation(activation)(c8)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    #u9 = Dropout(dropout)(u9)
    c9 = Conv2D(filters=n_filters*1, kernel_size= kernel_size, kernel_initializer=kernel_initializer,
               padding="same")(u9)
    
    outputs = Conv2D(4, (1, 1), activation='softmax') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
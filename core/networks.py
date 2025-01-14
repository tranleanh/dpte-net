import tensorflow as tf
# tf.config.run_functions_eagerly(True)

from keras.layers import Input, UpSampling2D
from keras.layers import LeakyReLU
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.models import Model
from keras.layers import Multiply
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from keras.layers import Dropout, concatenate
from keras.layers import MaxPooling2D, AveragePooling2D



img_size = 512

image_shape = (img_size, img_size, 4)
image_d_shape = (img_size, img_size, 3)


# Define Num. of Channels:
num_filters_adapt = 512
num_filters_1st = 16


def ConvBlock(x, num_filter=32, k_size=3, act_type="mish"):
            
    x = Conv2D(num_filter, k_size, padding='same', kernel_initializer = 'he_normal')(x)
    
    if act_type=="mish": 
        softplus_x = Activation('softplus')(x)
        tanh_softplus_x = Activation('tanh')(softplus_x)
        x = multiply([x, tanh_softplus_x])

    elif act_type=="swish":
        sigmoid_x = Activation('sigmoid')(x)
        x = multiply([x, sigmoid_x])
        
    elif act_type=="leakyrelu": x = LeakyReLU(alpha=0.1)(x)
    elif act_type=="tanh": x = Activation('tanh')(x)
    
    return x


def gan_model(generator, discriminator):
    inputs = Input(shape=image_shape)
    generated_image, guided_fm = generator(inputs)
    outputs = discriminator(generated_image)
    model = Model(inputs=inputs, outputs=[generated_image, guided_fm, outputs])
    return model




def PoolingTransformerBlock(conv, emb_dim, mlp_ratio):
    
    # Route
    route = conv

    # Norm
    conv = BatchNormalization()(conv)

    # Pool
    conv = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(conv)

    # Add
    conv = Add()([conv, route])

    # Route
    route = conv

    # Norm
    conv = BatchNormalization()(conv)

    ## FC1
    conv = Conv2D(emb_dim*mlp_ratio, 1, padding='same', kernel_initializer='he_normal')(conv)
    
    # Swish Activation: Swish(x) = x*Sigmoid(x) #
    sigmoid_x = Activation('sigmoid')(conv)
    conv = multiply([conv, sigmoid_x])

    ## FC2
    conv = Conv2D(emb_dim, 1, padding='same', kernel_initializer='he_normal')(conv)

    # Add
    conv = Add()([conv, route])

    return conv



def SubPixel(x, scale):
    return tf.nn.depth_to_space(x, scale)



### DPTE_Net Model (Official)
def DPTE_Net():
    
    inputs = Input(image_shape)     # H
    
    conv1 = ConvBlock(inputs, int(num_filters_1st), 3,  act_type="swish")
    conv1 = PoolingTransformerBlock(conv1, int(num_filters_1st), 4)                        # H
    conv1 = PoolingTransformerBlock(conv1, int(num_filters_1st), 4)                        # H
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)                # H/2

    conv2 = ConvBlock(pool1, int(2*num_filters_1st), 3,  act_type="swish")
    conv2 = PoolingTransformerBlock(conv2, int(2*num_filters_1st), 4)                      # H/2
    conv2 = PoolingTransformerBlock(conv2, int(2*num_filters_1st), 4)                      # H/2
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)                # H/4
    
    conv3 = ConvBlock(pool2, int(4*num_filters_1st), 3,  act_type="swish")
    conv3 = PoolingTransformerBlock(conv3, int(4*num_filters_1st), 4)                      # H/4
    conv3 = PoolingTransformerBlock(conv3, int(4*num_filters_1st), 4)                      # H/4
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)                # H/8
    
    conv4 = ConvBlock(pool3, int(8*num_filters_1st), 3,  act_type="swish")
    conv4 = PoolingTransformerBlock(conv4, int(8*num_filters_1st), 4)                      # H/8
    conv4 = PoolingTransformerBlock(conv4, int(8*num_filters_1st), 4)                      # H/8
    drop4 = Dropout(0.5)(conv4) 
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)                # H/16
    
    conv5 = ConvBlock(pool4, int(16*num_filters_1st), 3,  act_type="swish")
    conv5 = PoolingTransformerBlock(conv5, int(16*num_filters_1st), 4)                      # H/16
    conv5 = PoolingTransformerBlock(conv5, int(16*num_filters_1st), 4)                      # H/16
    
    # SPP #
    conv5 = ConvBlock(conv5, int(8*num_filters_1st), 1, act_type="swish")

    # Guided Features
    guided_fm = PoolingTransformerBlock(conv5, int(8*num_filters_1st), 4)
    guided_fm = ConvBlock(guided_fm, int(num_filters_adapt), 1, act_type="leakyrelu")
    
    conv5 = concatenate([conv5,
                         MaxPooling2D(pool_size=(13, 13), strides=1, padding='same')(conv5),
                         MaxPooling2D(pool_size=(9, 9), strides=1, padding='same')(conv5),
                         MaxPooling2D(pool_size=(5, 5), strides=1, padding='same')(conv5)], axis = 3)

    conv5 = ConvBlock(conv5, int(16*num_filters_1st), 1, act_type="swish")     # H/16
    drop5 = Dropout(0.5)(conv5)

    up6 = ConvBlock((UpSampling2D(size = (2,2))(drop5)), int(8*num_filters_1st), 2, act_type="swish")
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = ConvBlock(merge6, int(8*num_filters_1st), 3,  act_type="swish")
    conv6 = ConvBlock(conv6, int(8*num_filters_1st), 3,  act_type="swish")
    conv6 = ConvBlock(conv6, int(8*num_filters_1st), 3,  act_type="swish")
    
    up7 = ConvBlock((UpSampling2D(size = (2,2))(conv6)), int(4*num_filters_1st), 2, act_type="swish")
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = ConvBlock(merge7, int(4*num_filters_1st), 3,  act_type="swish")
    conv7 = ConvBlock(conv7, int(4*num_filters_1st), 3,  act_type="swish")
    conv7 = ConvBlock(conv7, int(4*num_filters_1st), 3,  act_type="swish")
   
    up8 = ConvBlock((UpSampling2D(size = (2,2))(conv7)), int(2*num_filters_1st), 2, act_type="swish")
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = ConvBlock(merge8, int(2*num_filters_1st), 3,  act_type="swish")
    conv8 = ConvBlock(conv8, int(2*num_filters_1st), 3,  act_type="swish")
    conv8 = ConvBlock(conv8, int(2*num_filters_1st), 3,  act_type="swish")

    up9 = ConvBlock((UpSampling2D(size = (2,2))(conv8)), int(num_filters_1st), 2, act_type="swish")
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = ConvBlock(merge9, int(num_filters_1st), 3,  act_type="swish")
    conv9 = ConvBlock(conv9, int(num_filters_1st), 3,  act_type="swish")
    conv9 = ConvBlock(conv9, int(num_filters_1st), 3,  act_type="swish")
    
    conv10 = Conv2D(3, 1, activation = 'tanh')(conv9)

    model = Model(inputs=inputs, outputs=[conv10, guided_fm, conv10])
    return model



## Critic
def Critic():
    
    inputs = Input(shape=image_d_shape)
    
    conv1 = Conv2D(int(num_filters_1st), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(int(num_filters_1st), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(int(2*num_filters_1st), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(int(2*num_filters_1st), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(int(4*num_filters_1st), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(int(4*num_filters_1st), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(int(8*num_filters_1st), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(int(8*num_filters_1st), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(int(16*num_filters_1st), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(int(16*num_filters_1st), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    x = GlobalAveragePooling2D()(drop5)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x, name='Critic')
    return model





### EDN-GTM Teacher Model
def EDN_GTM():
       
    inputs = Input(image_shape)
    
    conv1 = ConvBlock(inputs, 64, 3,  act_type="swish")
    conv1 = ConvBlock(conv1, 64, 3,  act_type="swish")
    conv1 = ConvBlock(conv1, 64, 3,  act_type="swish")
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = ConvBlock(pool1, 128, 3,  act_type="swish")
    conv2 = ConvBlock(conv2, 128, 3,  act_type="swish")
    conv2 = ConvBlock(conv2, 128, 3,  act_type="swish")
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = ConvBlock(pool2, 256, 3,  act_type="swish")
    conv3 = ConvBlock(conv3, 256, 3,  act_type="swish")
    conv3 = ConvBlock(conv3, 256, 3,  act_type="swish")
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = ConvBlock(pool3, 512, 3,  act_type="swish")
    conv4 = ConvBlock(conv4, 512, 3,  act_type="swish")
    conv4 = ConvBlock(conv4, 512, 3,  act_type="swish")
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = ConvBlock(pool4, 1024, 3,  act_type="swish")
    conv5 = ConvBlock(conv5, 1024, 3,  act_type="swish")
    conv5 = ConvBlock(conv5, 1024, 3,  act_type="swish")
    
    # SPP #
    conv5_hint = ConvBlock(conv5, 512, 1, act_type="swish")
    
    conv5 = concatenate([conv5_hint,
                         MaxPooling2D(pool_size=(13, 13), strides=1, padding='same')(conv5_hint),
                         MaxPooling2D(pool_size=(9, 9), strides=1, padding='same')(conv5_hint),
                         MaxPooling2D(pool_size=(5, 5), strides=1, padding='same')(conv5_hint)], axis = 3)

    conv5 = ConvBlock(conv5, 1024, 1, act_type="swish")
    drop5 = Dropout(0.5)(conv5)

    up6 = ConvBlock((UpSampling2D(size = (2,2))(drop5)), 512, 2, act_type="swish")
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = ConvBlock(merge6, 512, 3,  act_type="swish")
    conv6 = ConvBlock(conv6, 512, 3,  act_type="swish")
    conv6 = ConvBlock(conv6, 512, 3,  act_type="swish")
    
    up7 = ConvBlock((UpSampling2D(size = (2,2))(conv6)), 256, 2, act_type="swish")
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = ConvBlock(merge7, 256, 3,  act_type="swish")
    conv7 = ConvBlock(conv7, 256, 3,  act_type="swish")
    conv7 = ConvBlock(conv7, 256, 3,  act_type="swish")
   
    up8 = ConvBlock((UpSampling2D(size = (2,2))(conv7)), 128, 2, act_type="swish")
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = ConvBlock(merge8, 128, 3,  act_type="swish")
    conv8 = ConvBlock(conv8, 128, 3,  act_type="swish")
    conv8 = ConvBlock(conv8, 128, 3,  act_type="swish")

    up9 = ConvBlock((UpSampling2D(size = (2,2))(conv8)), 64, 2, act_type="swish")
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = ConvBlock(merge9, 64, 3,  act_type="swish")
    conv9 = ConvBlock(conv9, 64, 3,  act_type="swish")
    conv9 = ConvBlock(conv9, 64, 3,  act_type="swish")
    
    conv10 = Conv2D(3, 1, activation = 'tanh')(conv9)

    model = Model(inputs=inputs, outputs=[conv10, conv5_hint])
    return model




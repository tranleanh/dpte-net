import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model

import numpy as np

from .networks import img_size


image_shape = (img_size, img_size, 3)



def l2_loss(y_true, y_pred):
    return K.mean((y_pred - y_true)**2)



def l2_loss_hint(y_true, y_pred):
    return K.mean((y_pred - y_true)**2)



def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))



def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)



def perceptual_and_l2_loss(y_true, y_pred):
    A = 1
    B = 50
    return A*perceptual_loss(y_true, y_pred) + B*l2_loss(y_true, y_pred)



def transmission_guided_loss(y_true, y_pred, guide):
    return K.mean(((y_pred - y_true)*guide)**2)



def perceptual_and_transmission_guided_loss(guide):
    A = 1
    B = 1
    def loss(y_true, y_pred):
        return A*perceptual_loss(y_true, y_pred) + B*transmission_guided_loss(y_true, y_pred, guide)
    return loss


def soft_perceptual_and_transmission_guided_loss(guide, slabel_decay):
    A = 1
    B = 1
    def loss(y_true, y_pred):
        return (A*perceptual_loss(y_true, y_pred) + B*transmission_guided_loss(y_true, y_pred, guide))*slabel_decay
    return loss


def soft_l2_loss_hint(slabel_decay):
    def loss(y_true, y_pred):
        return l2_loss_hint(y_true, y_pred)*slabel_decay
    return loss
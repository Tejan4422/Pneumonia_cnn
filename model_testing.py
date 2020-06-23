import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, image
from skimage import transform
from flask import Flask, redirect, url_for, request, render_template


cnn = keras.models.load_model('cnn_model.h5')


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('C:/Users/Tejan/.spyder-py3/work/pnuemonia/chest_xray/test',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_accuracy = cnn.evaluate(test_set, steps = 624)

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (64, 64, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

image = load('C:/Users/Tejan/.spyder-py3/work/pnuemonia/chest_xray/test/NORMAL/NORMAL2-IM-0329-0001.jpeg')
image1 = load('C:/Users/Tejan/.spyder-py3/work/pnuemonia/test_image.jpg')

cnn.predict(image)

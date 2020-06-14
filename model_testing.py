import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator, load_img, image
from sklearn.metrics import classification_report, confusion_matrix

cnn = keras.models.load_model('cnn_model.h5')


test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory('C:/Users/Tejan/.spyder-py3/work/pnuemonia/chest_xray/trail_1',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

img = image.load_img('C:/Users/Tejan/.spyder-py3/work/pnuemonia/chest_xray/trail_1/NORMAL2-IM-1427-0001.jpeg',
                     target_size = (64,64))

y_pred = cnn.predict_generator(test_set)

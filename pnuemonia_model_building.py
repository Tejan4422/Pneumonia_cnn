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
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.metrics import classification_report, confusion_matrix

mainDir = os.listdir('C:/Users/Tejan/.spyder-py3/work/pnuemonia/chest_xray')
print(mainDir)
train_folder = 'C:/Users/Tejan/.spyder-py3/work/pnuemonia/chest_xray/train'
test_folder = 'C:/Users/Tejan/.spyder-py3/work/pnuemonia/chest_xray/test'
val_folder = 'C:/Users/Tejan/.spyder-py3/work/pnuemonia/chest_xray/val'

os.listdir(train_folder)
train_n = train_folder+'/NORMAL/'
train_p = train_folder+'/PNEUMONIA/'

#Normal pic 
print(len(os.listdir(train_n)))
rand_norm= np.random.randint(0,len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ',norm_pic)

norm_pic_address = train_n+norm_pic

#Pneumonia
rand_p = np.random.randint(0,len(os.listdir(train_p)))

sic_pic =  os.listdir(train_p)[rand_norm]
sic_address = train_p+sic_pic
print('pneumonia picture title:', sic_pic)

# Load the images
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

#Let's plt these images
f = plt.figure(figsize= (10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')
plt.savefig('normal.png', dpi = 100)

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')
plt.savefig('pneumonia.png', dpi = 100)

cnn = Sequential()

cnn.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (64, 64, 3)))

cnn.add(MaxPooling2D(pool_size = (2,2)))

cnn.add(Conv2D(32, (3,3), activation = 'relu'))

cnn.add(MaxPooling2D(pool_size = (2,2)))

cnn.add(Flatten())

cnn.add(Dense(activation = 'relu', units = 128))
cnn.add(Dense(activation = 'sigmoid', units = 1))

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

num_of_test_samples = 600
batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/Users/Tejan/.spyder-py3/work/pnuemonia/chest_xray/train',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory('C:/Users/Tejan/.spyder-py3/work/pnuemonia/chest_xray/val',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('C:/Users/Tejan/.spyder-py3/work/pnuemonia/chest_xray/test',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
t1_set = test_datagen.flow_from_directory('C:/Users/Tejan/.spyder-py3/work/pnuemonia/chest_xray/trail',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

cnn.summary()

cnn_model = cnn.fit_generator(training_set,
                              steps_per_epoch = 163,
                              epochs = 10,
                              validation_data = validation_generator,
                              validation_steps = 624)

test_accuracy = cnn.evaluate(test_set, steps = 624)

y_pred = cnn.predict_generator(test_set, 100)
y_pred = np.argmax(y_pred, axis = 1)
y_t1 = cnn.predict_generator(t1_set)
y_t1 = np.argmax(y_t1, axis = 1)

# Accuracy 
plt.plot(cnn_model.history['accuracy'])
plt.plot(cnn_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.savefig('traning_val_accuracy.png', dpi = 100)
plt.show()
# Loss 
plt.plot(cnn_model.history['val_loss'])
plt.plot(cnn_model.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.savefig('loss.png', dpi = 100)
plt.show()

cnn.save('cnn_model.h5')
cn = keras.models.load_model('cnn_model.h5')




















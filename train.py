#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, GlobalAveragePooling2D    # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions                              # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img                                        # type: ignore
from tensorflow.keras.applications.resnet50 import ResNet50                                                          # type: ignore
from tensorflow.keras.preprocessing import image                                                                     # type: ignore
from tensorflow.keras.models import Sequential                                                                       # type: ignore
from tensorflow.keras.models import Model                                                                            # type: ignore
from keras.layers import Dropout
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import cv2
from PIL import Image
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
# In[ ]:


import pathlib

img_height, img_width = (384, 384)
batch_size = 32
train_data_dir = pathlib.Path(r'/mnt/c/Users/Carlos Magahis/thesis/images/train')
valid_data_dir = pathlib.Path(r'/mnt/c/Users/Carlos Magahis/thesis/images/train')
test_data_dir = pathlib.Path(r'/mnt/c/Users/Carlos Magahis/thesis/images/test')
# train_data_aug_dir = r"Dataset_Splitted1/train/aug"
# train_data_no_aug_dir = r"Dataset_Splitted1/train/no_aug"


# In[4]:


train_datagen = ImageDataGenerator(
                            preprocessing_function = preprocess_input,
                            #   shear_range = 0.2, # applies random shearing transformations up to 20% along horizontal/vertical axis
                          #  zoom_range = 0.2, # applies random zoom transformations by up to 20%
                             #  rotation_range = 60, # randomly rotate the images by up to 60 degrees
                            #  horizontal_flip = True, # randomly flip images horizontally
                            # vertical_flip=True,                 # Randomly flip images vertically
                           # brightness_range=[0.9, 1.1],        # Vary brightness between 90% to 110% of original
                              #  width_shift_range = 0.2,  # randomly shift images horizontally by up to 20%
                             #   heigh_shift_range = 0.2, # randomly shift images vertically by up to 20%
                               # fill_mode = 'nearest',  # specifies how to fill empty areas during transformation
                               validation_split = 0.2  # split 20% of training data for validation

   )



valid_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input
)



train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = 'categorical',
    subset = 'training')

valid_generator = train_datagen.flow_from_directory ( # kukuha siya ng 20% ng training for validation
    valid_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = 'categorical',
    subset = 'validation')

test_generator = test_datagen.flow_from_directory (
    test_data_dir,
    target_size = (img_height, img_width),
    batch_size = 2,
    class_mode = 'categorical')


# In[5]:


x,y = next(test_generator)
x.shape


# In[7]:

from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
"""
#transfer learning mode
old_model = tf.keras.models.load_model(r'/mnt/c/Users/Carlos Magahis/thesis/oldmodel15.h5')
base_model = ResNet50(include_top=False, weights= None)

for base_layer, old_layer in zip(base_model.layers[:-3], old_model.layers[:-3]):
    base_layer.trainable = False
    if base_layer.name == old_layer.name:
      base_layer.set_weights(old_layer.get_weights())
    else:
      print(f"Skipping weight transfer for layer: {base_layer.name}")
    if hasattr(base_layer, 'kernel_regularizer'):
      base_layer.kernel_regularizer = l1_l2(l1=0.01, l2=0.01)
"""
#Train from imagenet version
base_model = ResNet50(include_top=False, weights='imagenet')
for base_layer in base_model.layers[:-4]:
    base_layer.trainable = False
    if hasattr(base_layer, 'kernel_regularizer'):
      base_layer.kernel_regularizer =l1_l2(l1=0.01, l2=0.01)


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = 'relu', kernel_regularizer = l1_l2(l1 = 0.0000001, l2 = 0.0000001))(x)
x = Dropout(0.25)(x)


predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)



optimizer = Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy'], class_weight=class_weights)

history = model.fit(train_generator,
          epochs = 20,
          validation_data = valid_generator,
                   )


# In[8]:


# SAVING THE MODEL TO A FILE
model.save(r'/mnt/c/Users/Carlos Magahis/thesis/oldmodel9.keras')
model.save(r'/mnt/c/Users/Carlos Magahis/thesis/oldmodel19.h5')# this should be incremented
model.save_weights(r'/mnt/c/Users/Carlos Magahis/thesis/oldmodel19Weights.h5')


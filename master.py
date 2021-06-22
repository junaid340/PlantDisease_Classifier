# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 14:50:50 2020
@author: junaid
"""

#Importing libraries

from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
from keras.applications import DenseNet201
import numpy as np
import os
import glob as gb
import cv2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
#from sklearn.utils import class_weight
from tqdm import tqdm
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

#Image to Array Transfering

print("\n ----------------Loading Datasets---------------- \n")

def Dataset_loader(DIR, RESIZE):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    files = gb.glob(pathname = str (DIR + '/*.JPG'))
    for IMAGE_NAME in tqdm(files):
      img = read(IMAGE_NAME)
           
      img = cv2.resize(img, (RESIZE,RESIZE))
           
      IMG.append(np.array(img))
    return IMG

#Loading Datasets

pepper_dataset = np.array(Dataset_loader('master_dataset/pepper',150))  
potato_dataset = np.array(Dataset_loader('master_dataset/potato',150))
tomato_dataset = np.array(Dataset_loader('master_dataset/tomato',150))

image_list = np.concatenate((pepper_dataset, potato_dataset, tomato_dataset), axis = 0)

print("\n ----------------Datasets Loaded Sucessfully---------------- \n")

#Creating Labels from the dataset

path = 'master_dataset/'

code = {'pepper':0,'potato':1,'tomato':2}

def getcode(n):
      for x , y in code.items() : 
        if n == y : 
            return x

image_labels = []
for folder in  os.listdir(path) : 
    files = gb.glob(pathname= str( path + folder + '//*.JPG'))
    for file in files: 
        image_labels.append(code[folder])

image_labels = np.array(image_labels)

#Train/Test Split
print("\n ----------------Spliting Data into Train/Test---------------- \n")
x_train, x_test, y_train, y_test = train_test_split(image_list, image_labels, test_size = 0.2, random_state = 42)

BATCH_SIZE = 16

train_generator = ImageDataGenerator( rotation_range = 25, width_shift_range = 0.1,
                                     height_shift_range = 0.1, shear_range = 0.2,
                                     zoom_range = 0.2, horizontal_flip = True, fill_mode="nearest")

#Building Model

def build_model(backbone, lr = 1e-4):
  classifier = Sequential()
  classifier.add(backbone)
  classifier.add(layers.GlobalAveragePooling2D())
  classifier.add(layers.Dropout(0.5))
  classifier.add(layers.BatchNormalization())
  classifier.add(layers.Dense(3, activation = 'softmax'))
  
  classifier.compile(loss = 'sparse_categorical_crossentropy', 
                     optimizer=Adam(lr=lr),
                     metrics=['accuracy'])
  
  return classifier

resnet = DenseNet201(weights='imagenet',
                     include_top=False,
                     input_shape=(150,150,3),
                     classes = 3)

model = build_model(resnet, lr = 1e-4)
model.summary()

learn_control = ReduceLROnPlateau(monitor = 'val_loss', 
                                  patience = 5, verbose = 1, 
                                  factor = 0.2, min_lr = 1e-7)

#Training the Model

ckpt_path = './checkpoints'
ckpt_name = "Master_Model.hdf5"
os.makedirs(ckpt_path, exist_ok = True)
ckpt = ModelCheckpoint(ckpt_path + '/' + ckpt_name, monitor = 'val_loss', save_best_only = True, mode = 'min')

history = model.fit_generator(train_generator.flow(x_train, y_train, batch_size=BATCH_SIZE),
                              steps_per_epoch = len(x_train)//BATCH_SIZE,
                              epochs = 50,
                              validation_data=(x_test, y_test),
                              callbacks = [learn_control, ckpt], verbose = 1)

#Test on single Image
import matplotlib.pyplot as plt

label = {0:'pepper',1:'potato',2:'tomato'}

#load the test image here
#img_path = 'tomato_dataset/healthy/0a334ae6-bea3-4453-b200-85e082794d56___GH_HL Leaf 310.1.JPG'
#img_path = 'potato_dataset/late_blight/1a6fc494-81dd-4649-ad8d-a5a6e58a2aa7___RS_LB 2618.JPG'
img_path = 'pepper_dataset/bel_becterial_spot/3bf80a4f-7a2b-4cdf-a93f-dfa12ddb4128___NREC_B.Spot 9233.JPG'

#def test_model(img_path):
image = plt.imread(img_path)
test_image = cv2.resize(image, (150, 150))
#expand dimension according to the model's specs
test_image = np.expand_dims(test_image, axis = 0)
#get the predictions
result = model.predict(test_image)

pred = (result >= np.max(result))
  
if pred[0][0]:
  plt.title(label = '\n Predicted Class: ' + label[0])
  plt.imshow(image)  
elif pred[0][1]:
  plt.title(label = '\n Predicted Class: ' + label[1])
  plt.imshow(image)  
else:
  plt.title(label = '\n Predicted Class: ' + label[2])
  plt.imshow(image)  

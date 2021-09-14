from tensorflow import keras # модуль keras отвечает за процесс формирования структуры нейронной сети
from tensorflow.keras import layers # создает слои нс
from tensorflow.keras import Sequential # отвечает за модель нс
# типы слоев нс 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam # используемые оптимизаторы
import tensorflow_datasets as tfds # наборы данных tensorflow
import tensorflow as tf # непосредственно модуль tensorflow
from keras.utils.vis_utils import plot_model # модуль для визуализации нс  
import numpy as np # выполняет работу с массивами
# import logging
import matplotlib.pyplot as plt # средство для работы с графиками
import matplotlib.gridspec as gridspec # средство для работы с графиками
from heapq import merge
import glob 
import time
import os
from pets import *
N = 0
model = tf.keras.models.load_model('dogs_cats.h5')
img_path = "pets/"
types = ["*.jpg", "*.jpeg", "*.png"]
imgs = []
for t in types:
  itmp=glob.glob(img_path+t)
  if itmp!=[]:
    imgs = imgs + itmp
img = imgs
cols = 1
rows = len(img)
fig = plt.figure(figsize=((cols*5),(rows*5)))
grid = gridspec.GridSpec(nrows=rows, ncols=cols, figure=fig)

for i in range(cols*rows):
  fig.add_subplot(grid[i])
  res = dog_cat_predict(model, img[i])
  image=plt.imread(img[i])
  plt.title("File: '" + os.path.basename(img[i]) + "' Recognized: " + res)
  plt.axis(False)
  plt.imshow(image)


# Подключение библиотек
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

# Функция, которая создает модель нейронной сети
def dog_cat_model():
  model = Sequential() # задали модель нс
  # добавление скрытых слоев
  model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(units=128, activation='relu'))
  model.add(Dense(units=1, activation='sigmoid'))
  # создаем нс на основе разработанной архитектуры
  model.compile(optimizer=Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy'])
  return model # возвращаем созданную нс

# Функция запуска обучения нейронной сети
def dog_cat_train(model):
  (cat_train, cat_valid, cat_test), info = tfds.load('cats_vs_dogs', 
                                                     split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
                                                     with_info=True, as_supervised=True)

  # Функция подготовки обучающей выборки
  def pre_process_image(image, label):
    image = tf.cast(image, tf.float32)
    image = image/255.0
    image = tf.image.resize(image, (128, 128))
    return image, label

  BATCH_SIZE = 64
  SHUFFLE_BUFFER_SIZE = 1000
  train_batch = cat_train.map(pre_process_image).shuffle(SHUFFLE_BUFFER_SIZE).repeat().batch(BATCH_SIZE)
  validation_batch = cat_valid.map(pre_process_image).repeat().batch(BATCH_SIZE)

  t_start = time.time()
  model.fit(train_batch, steps_per_epoch=100, epochs=2,
            validation_data=validation_batch,
            validation_steps=10,
            callbacks=None)
  print("Training done, dT:", time.time() - t_start)


def dog_cat_predict(model, image_file):
  label_names = ["cat", "dog"]

  img = keras.preprocessing.image.load_img(image_file,
    target_size=(128, 128))
  img_arr = np.expand_dims(img, axis=0) / 255.0
  # result = model.predict_classes(img_arr)
  predict_x=model.predict(img_arr) 
  result=np.argmax(predict_x,axis=1)
  return label_names[result[0][0]]
  
model = dog_cat_model() # переменная model будет хранить нашу нс в памяти компьютера
print(model.summary()) # вывод информации о слоях и нейронах модели
dog_cat_train(model) # запуск процесса обучения
model.save('dogs_cats.h5') # сохранение модели

N = 0
model = tf.keras.models.load_model('dogs_cats.h5')
img_path = "drive/MyDrive/neuralnet/pets/"
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
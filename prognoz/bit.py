import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
# импорт данных
df = pd.read_csv('train.csv')

df.head()

#Выбираем поддиапазон данных
#Index 11856 указывает на конец 2013 года
df = pd.read_csv('train.csv', nrows = 11856)

#Создание обучающего и тестового наборов
#Индекс 10392 указывает на конец октября 2013 года
train=df[0:10392]
test=df[10392:]

# Агрегирование набора данных на ежедневной основе
df.Timestamp = pd.to_datetime(df.Datetime,format='%d-%m-%Y %H:%M') 
df.index = df.Timestamp 
df = df.resample('D').mean()
train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 
train.index = train.Timestamp 
train = train.resample('D').mean() 
test.Timestamp = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 
test.index = test.Timestamp 
test = test.resample('D').mean()

#Строим график
train.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14)
test.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14)
plt.show()

dd= np.asarray(df.Count)
y_hat = test.copy()
y_hat['naive'] = dd[len(dd)-1]
plt.figure(figsize=(12,8))
plt.plot(train.index, train['Count'], label='Train')
plt.plot(test.index,test['Count'], label='Test')
plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast')
plt.legend(loc='best')
plt.title("Naive Forecast")
plt.show()

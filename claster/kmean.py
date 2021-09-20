# Импортируем библиотеки
from sklearn import datasets
from sklearn.cluster import KMeans

random.seed(20)

iris_df = datasets.load_iris() # Загружаем набор данных
model = KMeans(n_clusters=3) # Описываем модель
model.fit(iris_df.data) # Проводим моделирование
predicted_label = model.predict([[7.2, 3.5, 0.8, 1.6]]) # Предсказание на единичном примере

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

# Загрузка данных из Excel
file_path = "D:\\Github\\artificial_intelligence_systems\\3_practice\\Mine_Dataset.xls"
df = pd.read_excel(file_path, sheet_name="Normalized_Data", engine="xlrd")

print(df.head())
df.info()

# Формирование данных для модели
X = df[['V', 'H', 'S']]
y = df['M']

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17, stratify=y)

# Обучение k-NN с K=3
K = 3
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели при K={K}: {accuracy:.4f}')

# Оптимизация количества соседей
k_list = list(range(1, 50))
cv_scores = []

for K in k_list:
    knn = KNeighborsClassifier(n_neighbors=K)
    scores = cross_val_score(knn, X, y, cv=9, scoring='accuracy')
    cv_scores.append(scores.mean())

# Ошибка классификации
MSE = [1 - x for x in cv_scores]

plt.figure(1)
plt.plot(k_list, MSE)
plt.xlabel('Количество соседей (K)')
plt.ylabel('Ошибка классификации (MSE)')
plt.title('Оптимизация K в k-NN')

# Выбор оптимального K
k_min = min(MSE)
all_k_min = [k_list[i] for i in range(len(MSE)) if MSE[i] <= k_min]
print('Оптимальные значения K:', all_k_min)

# Визуализация разделения классов (только по двум признакам V и H)
sb.pairplot(df, hue='M', vars=['V', 'H', 'S'], markers=["o", "s", "D"])
plt.show()

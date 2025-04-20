import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Загрузка данных
file_path = 'D:\\Github\\artificial_intelligence_systems\\9_practice\\Admission_Predict.csv'
dataset = pd.read_csv(file_path)

# Убираем лишние пробелы в названиях колонок
dataset.columns = dataset.columns.str.strip()

# Выбор признаков
X = dataset[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']].values

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Определение оптимального количества кластеров (метод локтя)
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Метод локтя')
plt.xlabel('Количество кластеров')
plt.ylabel('Сумма квадратов расстояний (Inertia)')
plt.grid(True)
plt.show()

# Кластеризация с 5 кластерами
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Визуализация кластеров по признакам GRE и CGPA
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[y_kmeans == 0, 0], X_scaled[y_kmeans == 0, 5], s=50, label='Cluster 1')
plt.scatter(X_scaled[y_kmeans == 1, 0], X_scaled[y_kmeans == 1, 5], s=50, label='Cluster 2')
plt.scatter(X_scaled[y_kmeans == 2, 0], X_scaled[y_kmeans == 2, 5], s=50, label='Cluster 3')
plt.scatter(X_scaled[y_kmeans == 3, 0], X_scaled[y_kmeans == 3, 5], s=50, label='Cluster 4')
plt.scatter(X_scaled[y_kmeans == 4, 0], X_scaled[y_kmeans == 4, 5], s=50, label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 5], s=200, c='black', label='Centroids')
plt.title('Кластеры поступающих')
plt.xlabel('GRE Score (стандартиз.)')
plt.ylabel('CGPA (стандартиз.)')
plt.legend()
plt.grid(True)
plt.show()

# === АППРОКСИМАЦИЯ ПРИ РАЗНЫХ СТЕПЕНЯХ ПОЛИНОМА ===
# Берем CGPA как X, Chance of Admit как y
X_poly_input = dataset[['CGPA']].values
y_output = dataset['Chance of Admit'].values

degrees = [1, 2, 3, 4, 10, 15]
plt.figure(figsize=(10, 6))

# Сортировка значений для красивого графика
sort_idx = np.argsort(X_poly_input[:, 0])
X_sorted = X_poly_input[sort_idx]
y_sorted = y_output[sort_idx]

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_poly_input)
    model = LinearRegression()
    model.fit(X_poly, y_output)

    y_pred = model.predict(X_poly)
    error = mean_squared_error(y_output, y_pred)

    # Строим аппроксимацию по отсортированным данным для плавной линии
    y_pred_sorted = model.predict(poly.transform(X_sorted))
    plt.plot(X_sorted, y_pred_sorted, label=f"Степень {degree}, MSE={error:.4f}")

plt.scatter(X_poly_input, y_output, color='black', s=15, label='Фактические данные')
plt.title('Полиномиальная аппроксимация для CGPA → Chance of Admit')
plt.xlabel('CGPA')
plt.ylabel('Chance of Admit')
plt.legend()
plt.grid(True)
plt.show()

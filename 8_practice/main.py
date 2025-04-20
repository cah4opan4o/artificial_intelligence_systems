import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Загрузка данных
file_path = 'D:\\Github\\artificial_intelligence_systems\\8_practice\\Admission_Predict.csv'
dataset = pd.read_csv(file_path)

# Убираем лишние пробелы в названиях колонок
dataset.columns = dataset.columns.str.strip()

# Выбор признаков
# X = dataset[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']].values
X = dataset[['GRE Score']].values
y = dataset['Chance of Admit'].values

# Линейная регрессия
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)
r2_lin = r2_score(y, y_pred_lin)

# Вывод R² линейной модели
print(f"Линейная регрессия: R² = {r2_lin:.4f}")

# Построение графика линейной регрессии
plt.figure(figsize=(6, 4))
plt.scatter(X, y, color='red', s=10)
plt.plot(X, y_pred_lin, color='blue')
plt.title('Линейная регрессия')
plt.xlabel('GRE Score')
plt.ylabel('Chance of Admit')
plt.grid(True)
plt.tight_layout()

# Полиномиальная регрессия для разных степеней
degrees = [1, 2, 3, 5, 10, 15]
plt.figure(figsize=(15, 10))

for i, d in enumerate(degrees, 1):
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    
    # Предсказания на более плотной сетке
    X_grid = np.linspace(min(X), max(X), 500).reshape(-1, 1)
    y_pred_grid = model.predict(poly.transform(X_grid))
    
    # Оценка R² на тренировочных данных
    y_pred_train = model.predict(X_poly)
    r2 = r2_score(y, y_pred_train)
    
    # График
    plt.subplot(2, 3, i)
    plt.scatter(X, y, color='red', s=10)
    plt.plot(X_grid, y_pred_grid, color='blue')
    plt.title(f'Полином степени {d} (R² = {r2:.4f})')
    plt.xlabel('GRE Score')
    plt.ylabel('Chance of Admit')
    plt.grid(True)

plt.tight_layout()
plt.show()

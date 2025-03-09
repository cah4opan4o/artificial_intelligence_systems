import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Путь к файлу
file_path = 'D:\\Github\\artificial_intelligence_systems\\5_practice\\diabetes_dataset.csv'
data = pd.read_csv(file_path)
data.head()

print("---------------------------------")

# Разделение признаков и целевой переменной
x = data.iloc[:, :16].values
y = data.iloc[:, -1].values
print("Матрица признаков")
print(x)
print("Зависимая переменная")
print(y)

print("---------------------------------")

# Инициализация и использование SimpleImputer для обработки пропусков
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x = imputer.fit_transform(x)  # Применение к всем признакам
print(x)

print("---------------------------------")
labelencoder_y = LabelEncoder()
print("Зависимая переменная до обработки")
print(y)
y = labelencoder_y.fit_transform(y)
print("Зависимая переменная после обработки")
print(y)
print("---------------------------------")
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import statsmodels.api as sm
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

# # Загрузка данных
# file_path = 'D:\\Github\\artificial_intelligence_systems\\6_practice\\Admission_Predict.csv'
# dataset = pd.read_csv(file_path)

# # Убираем лишние пробелы в названиях колонок
# dataset.columns = dataset.columns.str.strip()

# # Выбор признаков и целевой переменной
# X = dataset[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']].values
# y = dataset['Chance of Admit'].values

# # Добавляем столбец единиц (для учета свободного коэффициента в модели)
# X = np.append(arr=np.ones((X.shape[0], 1)).astype(int), values=X, axis=1)

# # Разделение на обучающую и тестовую выборки
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# # Обучение модели линейной регрессии
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)

# # Предсказания
# y_pred = regressor.predict(X_test)

# # Backward Elimination
# X_opt = X[:, :]
# model = sm.OLS(y, X_opt).fit()

# # Удаляем признаки с высоким p-value (> 0.05)
# while max(model.pvalues) > 0.05:
#     worst_feature = np.argmax(model.pvalues)  # Индекс наименее значимого признака
#     X_opt = np.delete(X_opt, worst_feature, axis=1)
#     model = sm.OLS(y, X_opt).fit()

# print("Окончательные коэффициенты модели:")
# print(model.summary())

# # Визуализация: фактические vs предсказанные значения
# plt.figure(figsize=(10, 5))
# plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
# plt.title('Фактические vs Предсказанные значения')
# plt.xlabel('Фактическая вероятность поступления')
# plt.ylabel('Предсказанная вероятность поступления')
# plt.grid(True)
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Загрузка данных
file_path = 'D:\\Github\\artificial_intelligence_systems\\6_practice\\Admission_Predict.csv'
dataset = pd.read_csv(file_path)

# Убираем лишние пробелы в названиях колонок
dataset.columns = dataset.columns.str.strip()

# Выбор признаков и целевой переменной
X = dataset[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']].values
y = dataset['Chance of Admit'].values

# Добавляем столбец единиц (для учета свободного коэффициента в модели)
X = np.append(arr=np.ones((X.shape[0], 1)).astype(int), values=X, axis=1)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Обучение модели линейной регрессии
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Предсказания
y_pred = regressor.predict(X_test)

# Backward Elimination
X_opt = X.copy()
cols = list(range(X_opt.shape[1]))  # Индексы признаков

while True:
    model = sm.OLS(y, X_opt).fit()
    p_values = model.pvalues
    print(model.summary())  # Вывод промежуточных результатов
    
    # Ищем индекс самого незначимого признака (с наибольшим p-value)
    max_p = max(p_values)
    if max_p > 0.05:  # Если p-value > 0.05, исключаем признак
        worst_feature = np.argmax(p_values)
        X_opt = np.delete(X_opt, worst_feature, axis=1)
        cols.pop(worst_feature)  # Удаляем соответствующий индекс признака
    else:
        break  # Если все признаки значимы, останавливаемся

print(f"Оставшиеся признаки: {cols}")
print(model.summary())  # Итоговая модель

# Визуализация: фактические vs предсказанные значения
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Фактические vs Предсказанные значения')
plt.xlabel('Фактическая вероятность поступления')
plt.ylabel('Предсказанная вероятность поступления')
plt.grid(True)
plt.show()

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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Загрузка данных
file_path = 'D:\\Github\\artificial_intelligence_systems\\7_practice\\Admission_Predict.csv'
dataset = pd.read_csv(file_path)

# Убираем лишние пробелы в названиях колонок
dataset.columns = dataset.columns.str.strip()

# Обработка пропущенных значений (если такие есть)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
dataset[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA']] = imputer.fit_transform(
    dataset[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA']]
)

# Пример обработки категориального признака (если бы он был строкой)
# Например, если бы был столбец 'University Name' с текстовыми значениями:
# labelencoder = LabelEncoder()
# dataset['University Name'] = labelencoder.fit_transform(dataset['University Name'])
# onehotencoder = ColumnTransformer(
#     [('one_hot_encoder', OneHotEncoder(), ['University Name'])],
#     remainder='passthrough'
# )
# dataset = onehotencoder.fit_transform(dataset)

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

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("Оценка модели на тестовой выборке:")
print(f"MAE  (средняя абсолютная ошибка): {mae:.4f}")
print(f"MSE  (среднеквадратичная ошибка): {mse:.4f}")
print(f"RMSE (корень из MSE):             {rmse:.4f}")
print(f"R²   (коэффициент детерминации):  {r2:.4f}")
print(f"MAPE (средняя ошибка в процентах): {mape:.2f}%")

# Backward Elimination
X_opt = X.copy()
cols = list(range(X_opt.shape[1]))  # Индексы признаков

while True:
    model = sm.OLS(y, X_opt).fit()
    p_values = model.pvalues
    print(model.summary())  # Вывод промежуточных результатов

    max_p = max(p_values)
    if max_p > 0.05:
        worst_feature = np.argmax(p_values)
        X_opt = np.delete(X_opt, worst_feature, axis=1)
        cols.pop(worst_feature)
    else:
        break

print(f"Оставшиеся признаки: {cols}")
print(model.summary())

# Визуализация
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Фактические vs Предсказанные значения')
plt.xlabel('Фактическая вероятность поступления')
plt.ylabel('Предсказанная вероятность поступления')
plt.grid(True)
plt.show()


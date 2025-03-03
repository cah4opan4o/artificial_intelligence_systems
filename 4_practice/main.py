import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # -----------------------------------------------------
    # Чтение данных
    file_path = 'D:\\Github\\artificial_intelligence_systems\\4_practice\\diabetes_dataset.csv'
    data = pd.read_csv(file_path)

    # Разделение признаков и целевой переменной
    dataX = data.iloc[:, 0:16]
    dataY = data['Outcome']
    print(dataX.head())
    print(dataY.head())

    # Матрица корреляции
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), cmap=plt.cm.Blues, annot=True, fmt='.2f')
    plt.title('Матрица Корреляции')

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        dataX, dataY, test_size=0.3, random_state=12
    )

    # Обучение дерева решений
    tree = DecisionTreeClassifier(max_depth=5, random_state=21, max_features=2)
    tree.fit(X_train, y_train)
    tree_pred = tree.predict(X_holdout)
    accur = accuracy_score(y_holdout, tree_pred)
    print('Accuracy =', accur)

    # Кросс-валидация по разной глубине дерева
    d_list = list(range(1, 20))
    cv_scores = []
    for d in d_list:
        tree = DecisionTreeClassifier(max_depth=d, random_state=21, max_features=2)
        scores = cross_val_score(tree, dataX, dataY, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

    # График ошибки MSE
    MSE = [1 - x for x in cv_scores]
    plt.figure(figsize=(8, 5))
    plt.plot(d_list, MSE, marker='o', linestyle='dashed')
    plt.xlabel('Max Depth')
    plt.ylabel('MSE')
    plt.title('Ошибка в зависимости от глубины дерева')

    # Поиск оптимальной глубины
    d_min = min(MSE)
    all_d_min = [d_list[i] for i in range(len(MSE)) if MSE[i] <= d_min]
    print("Optimal max depth:", all_d_min)

    # Подбор параметров с GridSearchCV
    dtc = DecisionTreeClassifier(max_depth=10, random_state=21, max_features=2)
    tree_params = {'max_depth': range(1, 20), 'max_features': range(1, 4)}
    tree_grid = GridSearchCV(dtc, tree_params, cv=10, verbose=True, n_jobs=-1)
    tree_grid.fit(dataX, dataY)

    print('\nBest parameters:', tree_grid.best_params_)
    print('Best cross-validation score:', tree_grid.best_score_)

    # Визуализация дерева решений
    export_graphviz(
        tree_grid.best_estimator_,
        feature_names=dataX.columns,
        class_names=[str(c) for c in dataY.unique()],
        out_file='diabet_tree.dot',
        filled=True, rounded=True
    )

    plt.show()

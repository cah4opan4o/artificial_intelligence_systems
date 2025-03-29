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
    sns.heatmap(dataX.corr(), cmap=plt.cm.Blues, annot=True, fmt='.2f')
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

    # -----------------------------------------------------
    dataX = data.drop(columns=['Outcome'])  # Все признаки
    dataY = data['Outcome']  # Целевая переменная

    plot_markers = ['r*', 'g^']
    answers = dataY.unique()
    razmer = 2
    f, places = plt.subplots(razmer, razmer, figsize=(16, 16))

    # Уменьшаем шаг сетки
    plot_step = 0.1

    # Ограничиваем диапазон значений (убираем выбросы)
    fmin = dataX.quantile(0.01) - 0.5
    fmax = dataX.quantile(0.99) + 0.5

    for i in range(razmer):
        for j in range(razmer):
            if i != j:
                # Ограничиваем количество точек в сетке
                xx, yy = np.meshgrid(
                    np.arange(fmin[i], fmax[i], plot_step, dtype=np.float32), 
                    np.arange(fmin[j], fmax[j], plot_step, dtype=np.float32)
                )

                # Обучаем модель
                model = DecisionTreeClassifier(max_depth=8, random_state=21, max_features=2)  # Уменьшаем max_depth
                model.fit(dataX.iloc[:, [i, j]], dataY)

                # Предсказываем классы (с меньшим потреблением памяти)
                p = model.predict(np.c_[xx.ravel(), yy.ravel()])
                p = p.reshape(xx.shape)

                places[i, j].contourf(xx, yy, p, cmap='Pastel1')

            # Отрисовка данных
            for id_answer in range(len(answers)):
                idx = np.where(dataY == answers[id_answer])[0]

                if i == j:
                    places[i, j].hist(dataX.iloc[idx, i], color=plot_markers[id_answer][0], histtype='step')
                else:
                    places[i, j].plot(dataX.iloc[idx, i], dataX.iloc[idx, j], plot_markers[id_answer], label=answers[id_answer], markersize=6)

            if j == 0:
                places[i, j].set_ylabel(dataX.columns[j])
            if i == 3:
                places[i, j].set_xlabel(dataX.columns[i])

    # Добавляем подписи для матрицы корреляции
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), cmap=plt.cm.Blues, annot=True, fmt='.2f')
    plt.title('Матрица Корреляции')
    plt.xticks(ticks=np.arange(len(dataX.columns)), labels=dataX.columns, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(dataX.columns)), labels=dataX.columns, rotation=0)
    plt.show()



    # -----------------------------------------------------
    # # Вычисляем корреляцию и выбираем 4 самых важных признака
    # correlation = data.corr()['Outcome'].drop('Outcome').abs()
    # top_features = correlation.sort_values(ascending=False).head(4).index.tolist()
    # print(f"Выбраны признаки: {top_features}")

    # # Готовим фигуру для отображения сетки графиков
    # fig, axes = plt.subplots(len(top_features), len(top_features), figsize=(12, 12))
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)

    # # Для каждой пары признаков строим график
    # for i, feature1 in enumerate(top_features):
    #     for j, feature2 in enumerate(top_features):
    #         ax = axes[i, j]
            
    #         if i == j:
    #             # Диагональные графики — это просто гистограммы распределения
    #             sns.histplot(data[feature1], ax=ax, kde=True, color='blue')
    #         else:
    #             # Обучаем модель на текущих двух признаках
    #             X = data[[feature1, feature2]]
    #             y = data['Outcome']
    #             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17, stratify=y)
                
    #             model = DecisionTreeClassifier(max_depth=5, random_state=21)
    #             model.fit(X_train, y_train)
                
    #             # Создаем сетку точек для предсказаний
    #             x_min, x_max = X.iloc[:, 0].min() - 0.1, X.iloc[:, 0].max() + 0.1
    #             y_min, y_max = X.iloc[:, 1].min() - 0.1, X.iloc[:, 1].max() + 0.1
    #             xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
    #                                 np.linspace(y_min, y_max, 200))
                
    #             Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    #             Z = Z.reshape(xx.shape)
                
    #             # Рисуем границы принятия решений
    #             ax.contourf(xx, yy, Z, alpha=0.3, cmap='Pastel1')
                
    #             # Добавляем точки выборки
    #             sns.scatterplot(x=X_train.iloc[:, 0], y=X_train.iloc[:, 1], hue=y_train, palette='Dark2', edgecolor='k', ax=ax)

    #         # Подписываем оси
    #         if j == 0:
    #             ax.set_ylabel(feature1)
    #         if i == len(top_features) - 1:
    #             ax.set_xlabel(feature2)

    # plt.show()
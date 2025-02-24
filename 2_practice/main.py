import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    # Загружаем данные из конкретного листа
    df = pd.read_excel("D:\\Github\\artificial_intelligence_systems\\2_practice\\land mine\\Mine_Dataset.xls", sheet_name="Normalized_Data", engine="xlrd")

    data = df.to_numpy()
    
    V = [] # Voltage
    H = [] # High
    S = [] # Soil type: 1 - dry and sandy, 2 - dry and humus, 3 - dry and limy
            # 4 - humid and sandy, 5 - humid and humus, 6 - humid and limy
    M = [] # Mine type: 1 - Null, 2 - Anti-Tank, 3 - Anti-personnel,
            #4 - Booby Trapped & anti-personnel, 5 - M14 Anti-personnel
    
    for dot in data:
        V.append(dot[0])
        H.append(dot[1])
        S.append(dot[2])
        M.append(dot[3])

    plt.figure(1)
    sns.heatmap(df.corr(), cmap=plt.cm.Blues)
    plt.title('Матрица Корреляции')
    
    plt.figure(2)
    plt.subplot(2,2,1)
    df["V"].hist(bins=100,density=True)
    plt.title("Voltage")
    plt.subplot(2,2,2)
    df["H"].hist(bins=50,density=True)
    plt.title("High")
    plt.subplot(2,2,3)
    df["S"].hist(bins=50,density=True)
    plt.title("Soil Type")
    plt.subplot(2,2,4)
    df["M"].hist(bins=50,density=True)
    plt.title("Mine Type")

    feats = ['V','H','S']
    sns.pairplot(df[feats + ['M']], hue='M')

  # Строим boxplot для каждого параметра
    plt.figure(4,figsize=(12, 6))
    sns.boxplot(data=df.iloc[:, :4])  # Первые 4 столбца (предполагаем, что они числовые)
    
    # Добавляем подписи
    plt.xlabel("Параметры")
    plt.ylabel("Значения")
    plt.title("Ящики с усами для параметров")
    plt.xticks(ticks=range(4), labels=["Voltage", "High", "Soil type", "Mine type"])
    
    plt.show()

    plt.show()
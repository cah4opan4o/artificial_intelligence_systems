import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns

if __name__ == "__main__":
    # Загружаем данные из конкретного листа
    df = pd.read_excel("D:\\Github\\artificial_intelligence_systems\\1_practice\\land mine\\Mine_Dataset.xls", sheet_name="Normalized_Data", engine="xlrd")

    # Конвертируем в numpy
    data = df.to_numpy()

    # print(data)
    df.head()
    df.info()
    
    
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
        
    # min scale
    print("-----------------------------------")
    print(min(V))
    print(min(H))
    print(min(S))
    
    # average scale
    print("-----------------------------------")
    print(np.mean(V))
    print(np.mean(H))
    print(np.mean(S))
    
    # max scale
    print("-----------------------------------")
    print(max(V))
    print(max(H))
    print(max(S))
    
    plt.figure(1)
    mine_Null = plt.plot(V[0:46]+V[225:248],H[0:46]+H[225:248],'ro',label='Null')
    Anti_Tank, = plt.plot(V[46:93]+V[249:271],H[46:93]+H[249:271],'g^',label='Anti Tank')
    Anti_personnel, = plt.plot(V[94:137]+V[272:293],H[94:137]+H[272:293],'bs',label='Anti personnel')
    Booby_Trapped = plt.plot(V[138:181]+V[294:315],H[138:181]+H[294:315],'k*',label='booby trapped')
    M14_Anti_personnel = plt.plot(V[182:224]+V[316:],H[182:224]+H[316:],'mD',label='M14 anti personnel')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Voltage')
    plt.ylabel('High')
    plt.title('Type of land mine')
    
    plt.figure(2)
    mine_Null = plt.plot(H[0:46]+H[225:248],S[0:46]+S[225:248],'ro',label='Null')
    Anti_Tank, = plt.plot(H[46:93]+H[249:271],S[46:93]+S[249:271],'g^',label='Anti Tank')
    Anti_personnel, = plt.plot(H[94:137]+H[272:293],S[94:137]+S[272:293],'bs',label='Anti personnel')
    Booby_Trapped = plt.plot(H[138:181]+H[294:315],S[138:181]+S[294:315],'k*',label='booby trapped')
    M14_Anti_personnel = plt.plot(H[182:224]+H[316:],S[182:224]+S[316:],'mD',label='M14 anti personnel')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('high')
    plt.ylabel('soil Type')
    plt.title('Type of land mine')
    
    plt.figure(3)
    mine_Null = plt.plot(V[0:46]+V[225:248],S[0:46]+S[225:248],'ro',label='Null')
    Anti_Tank, = plt.plot(V[46:93]+V[249:271],S[46:93]+S[249:271],'g^',label='Anti Tank')
    Anti_personnel, = plt.plot(V[94:137]+V[272:293],S[94:137]+S[272:293],'bs',label='Anti personnel')
    Booby_Trapped = plt.plot(V[138:181]+V[294:315],S[138:181]+S[294:315],'k*',label='booby trapped')
    M14_Anti_personnel = plt.plot(V[182:224]+V[316:],S[182:224]+S[316:],'mD',label='M14 anti personnel')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Voltage')
    plt.ylabel('soil Type')
    plt.title('Type of land mine')
    
    plt.figure(4)
    plt.subplot(3,2,1)
    plt.plot(H[0:46]+H[225:248],V[0:46]+V[225:248],'ro',label='Null')
    plt.subplot(3,2,2)
    plt.plot(H[46:93]+H[249:271],V[46:93]+V[249:271],'g^',label='Anti Tank')
    plt.subplot(3,2,3)
    plt.plot(H[94:137]+H[272:293],V[94:137]+V[272:293],'bs',label='Anti personnel')
    plt.subplot(3,2,4)
    plt.plot(H[138:181]+H[294:315],V[138:181]+V[294:315],'k*',label='booby trapped')
    plt.subplot(3,2,5)
    plt.plot(H[182:224]+H[316:],V[182:224]+V[316:],'mD',label='M14 anti personnel')
    plt.subplot(3,2,6)
    plt.plot(H[0:46]+H[225:248],V[0:46]+V[225:248],'ro',label='Null')
    plt.plot(H[46:93]+H[249:271],V[46:93]+V[249:271],'g^',label='Anti Tank')
    plt.plot(H[94:137]+H[272:293],V[94:137]+V[272:293],'bs',label='Anti personnel')
    plt.plot(H[138:181]+H[294:315],V[138:181]+V[294:315],'k*',label='booby trapped')
    plt.plot(H[182:224]+H[316:],V[182:224]+V[316:],'mD',label='M14 anti personnel')
    
    # Создаём диаграмму рассеяния
    plt.figure(5,figsize=(8, 6))
    scatter = plt.scatter(V, H, c=S, cmap='viridis', alpha=0.7, edgecolors='k')

    # Добавляем цветовую шкалу
    cbar = plt.colorbar(scatter)
    cbar.set_label("Тип почвы")

    plt.xlabel("Напряжение (V)")
    plt.ylabel("Высота (H)")
    plt.title("Диаграмма рассеяния напряжения и высоты с учетом типа почвы")
    plt.grid(True)
    
    # # Рассчитываем корреляционную матрицу
    # corr_matrix = df.corr()

    # # Визуализируем корреляционную матрицу
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    # plt.title("Корреляционная матрица")
    
    plt.figure(6)
    plt.subplot(2,2,1)
    df["V"].hist()
    plt.title("Voltage")
    plt.subplot(2,2,2)
    df["H"].hist()
    plt.title("High")
    plt.subplot(2,2,3)
    df["S"].hist()
    plt.title("Soil Type")
    plt.subplot(2,2,4)
    df["M"].hist()
    plt.title("Mine Type")
    
    plt.show()

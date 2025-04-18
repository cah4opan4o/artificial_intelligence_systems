import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
    print("----- MIN -----")
    print("V:",min(V))
    print("H:",min(H))
    print("S:",min(S))
    
    # average scale
    print("----- MEAN -----")
    print("V:",np.mean(V))
    print("H:",np.mean(H))
    print("S:",np.mean(S))
    
    # max scale
    print("----- MAX -----")
    print("V:",max(V))
    print("H:",max(H))
    print("S:",max(S))
    
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
    
    plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_excel("D:\\Github\\artificial_intelligence_systems\\3_practice\\Mine_Dataset.xls", sheet_name="Normalized_Data", engine="xlrd")
# d = pd.read_table("D:\\Github\\artificial_intelligence_systems\\3_practice\\Mine_Dataset.xls", delimiter=',')
# d.head()

data = df.to_numpy()
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

sb.pairplot(df,hue='M',markers=["o","s","D"])



plt.show()
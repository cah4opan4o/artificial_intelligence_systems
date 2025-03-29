import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

file_path = 'D:\\Github\\artificial_intelligence_systems\\6_practice\\diabetes_dataset.csv'
dataset = pd.read_csv(file_path)
print(dataset.head())

dataX = dataset.iloc[:,:1].values
dataY = dataset.iloc[:,-1].values
print('признаки\n',dataX[:5])
print('зависимая\n',dataY[:5])

x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=0)
print(x_train)
print('-----')
print(y_train)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print(y_pred)

plt.figure(1,figsize=(10,5))
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Experience (Training set)')
plt.ylabel('Salary')

plt.figure(2,figsize=(10,5))
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Expirience (Test set)')
plt.xlabel('Experience (Test set)')
plt.ylabel('Salary')

plt.show()
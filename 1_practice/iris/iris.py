import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

if __name__== "__main__":
    sepal_length = []
    sepal_width = []
    petal_length = []
    petal_width = []
    
    dt = np.dtype("f8, f8, f8, f8, U30")
    data = np.genfromtxt("D:\\Github\\Artificial_intelligence_systems\\1_practice\\iris\\iris.data", delimiter=",", dtype=dt)
    # data = np.genfromtxt("D:\\Github\\Artificial_intelligence_systems\\1_practice\\iris\\iris.data", delimiter=",",dtype=None)
    
    for dot in data:
        sepal_length.append(dot[0])
        sepal_width.append(dot[1])
        petal_length.append(dot[2])
        petal_width.append(dot[3])
    
    plt.figure(1)
    setosa, = plt.plot(sepal_length[:50], sepal_width[:50], 'ro', label = 'setosa')
    verticolot, = plt.plot(sepal_length[50:100], sepal_width[50:100], 'g^', label = 'vertocolor')
    virginica, = plt.plot(sepal_length[100:150], sepal_width[100:150], "bs", label = 'virginica')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    
    plt.figure(2)
    setosa, = plt.plot(sepal_length[:50], petal_length[:50], 'ro', label = 'setosa')
    verticolot, = plt.plot(sepal_length[50:100], petal_length[50:100], 'g^', label = 'vertocolor')
    virginica, = plt.plot(sepal_length[100:150], petal_length[100:150], "bs", label = 'virginica')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Sepal length')
    plt.ylabel('Petal lenght')
    
    plt.figure(3)
    setosa, = plt.plot(sepal_length[:50], petal_width[:50], 'ro', label = 'setosa')
    verticolot, = plt.plot(sepal_length[50:100], petal_width[50:100], 'g^', label = 'vertocolor')
    virginica, = plt.plot(sepal_length[100:150], petal_width[100:150], "bs", label = 'virginica')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Sepal length')
    plt.ylabel('Petal width')
    
    plt.show()
    
    # print(data)
    
    # print("Data type:",type(data))
    # print("Data size:",len(data))
    # print(data[:10])
    
    # print(data.shape)
    # print(type(data))
    # print(type(data[0]))
    # print(type(data[0][4]))
    # print(data[:10])
    
    
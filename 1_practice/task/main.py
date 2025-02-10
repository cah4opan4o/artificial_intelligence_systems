import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    area = []
    sensing_range = []
    transmission_range = []
    number_of_sensor = []
    number_of_barriers = []
    
    data = np.genfromtxt("D:\\Github\\Artificial_intelligence_systems\\1_practice\\task\\data.csv", delimiter=",", dtype=None, skiprows=1)
    
    for dot in data:
        area.append(dot[0])
        sensing_range.append(dot[1])
        transmission_range.append(dot[2])
        number_of_sensor.append(dot[3])
        number_of_barriers.append(dot[4])
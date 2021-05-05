#!/usr/bin/env python

import csv
import matplotlib.pyplot as plt

def readFile(path):
    with open(path, 'r') as csvfile:

        csvreader = csv.reader(csvfile)
        next(csvreader) #jump header line
        
        r_list = []
        dense_list = []
        for line in csvreader:
            r, dense = line[0].split(":") #split all contained items
            r_list.append(float(r))
            dense_list.append(float(dense))
    return r_list, dense_list

r, density = readFile("./P30_L335_t001_N100/densities.txt")

print(r, density)
plt.xlabel("radius")
plt.ylabel("densities")
plt.plot(r, density)
plt.grid()
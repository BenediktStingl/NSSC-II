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
            r_list.append(r)
            dense_list.append(dense)
    return r_list, dense_list

r, density = readFile("./densities.txt")

print(r, density)

plt.plot(r, density)
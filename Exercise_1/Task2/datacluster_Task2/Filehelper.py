#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import numpy as np

# First, lists of all returned files from the cluster were made, one for each resolution:

all_125 = ["jacobiMPI_125_10_n_1.log", "jacobiMPI_125_10_n_2.log","jacobiMPI_125_10_n_5.log",
           "jacobiMPI_125_10_n_10.log", "jacobiMPI_125_10_n_20.log", "jacobiMPI_125_10_n_40.log",
           "jacobiMPI_125_10_n_60.log", "jacobiMPI_125_10_n_80.log"]

all_500 = ["jacobiMPI_500_10_n_1.log", "jacobiMPI_500_10_n_2.log","jacobiMPI_500_10_n_5.log",
           "jacobiMPI_500_10_n_10.log", "jacobiMPI_500_10_n_20.log", "jacobiMPI_500_10_n_40.log",
           "jacobiMPI_500_10_n_60.log", "jacobiMPI_500_10_n_80.log"]

all_2000 = ["jacobiMPI_2000_10_n_1.log", "jacobiMPI_2000_10_n_2.log","jacobiMPI_2000_10_n_5.log",
           "jacobiMPI_2000_10_n_10.log", "jacobiMPI_2000_10_n_20.log", "jacobiMPI_2000_10_n_40.log",
           "jacobiMPI_2000_10_n_60.log", "jacobiMPI_2000_10_n_80.log"]

all_4000 = ["jacobiMPI_4000_10_n_1.log", "jacobiMPI_4000_10_n_2.log","jacobiMPI_4000_10_n_5.log",
           "jacobiMPI_4000_10_n_10.log", "jacobiMPI_4000_10_n_20.log", "jacobiMPI_4000_10_n_40.log",
           "jacobiMPI_4000_10_n_60.log", "jacobiMPI_4000_10_n_80.log"]



#defining function that searches file for value and adds it to list
def collector(file_list):
    """
    This function takes the list of files and searches for the line with the average runtime in each file.
    Extracting the average runtime in seconds for each file and adding it to a list. The list containing
    all average runtime values is then returned.
    """
    av_t = []
    for i in range(len(file_list)):
        with open(file_list[i], "r") as file_object:
            for line in file_object:
                if "Average Runtime" in line:
                    x = line
                    x = x.replace("Average Runtime =", "")
                    x = x.replace("seconds", "")
                    x = float(x)
                    av_t.append(x)     
    return av_t

#function that plots values from file:
def plotter_speed(av_t, res, line):
    """
    This function takes a list of average runtimes (av_t), the resolution (res) for which the runtimes were
    measured and the linetype (line) and plots the speedup against the corresponding number of processors. 
    For speedup the average runtimes were used to determine how much faster the program was using n processors
    compared to using just one. For this the av. runtime for one processor was divided by the av. runtime for
    n processors.
    """
    n_processors = [1,2,5,10, 20, 40, 60, 80]
    n_processors = np.array(n_processors)
    av_runtime = np.array(av_t)
    speedup = av_runtime[0]/ av_runtime
    plt.plot( n_processors, speedup, linestyle = line, label = res)
    return None
    

def plotter_effi(av_t, res, line):
    """
    Similar to to plotter_speed, the av. runtimes are here used to calculate the parallel efficiency.
    In this case the efficiency is expressed as the factor of measured speedup divided by ideal speedup.
    """
    n_processors = [1,2,5,10, 20, 40, 60, 80]
    p_arr= np.array(n_processors)
    n_processors= np.array(n_processors)
    av_runtime = np.array(av_t)
    speedup = av_runtime[0]/ av_runtime
    rspeedup = speedup/ p_arr
    plt.plot( n_processors, rspeedup, linestyle = line, label = res)





#The speedup plot for resolutions 125, 500, 2000 and 4000
plt.figure(0)
plotter_speed([1,1/2,1/5,1/10, 1/20, 1/40, 1/60, 1/80], "ideal", "--")
plotter_speed(collector(all_125), "res: 125", "-")
plotter_speed(collector(all_500), "res: 500", "-")
plotter_speed(collector(all_2000), "res: 2000", "-")
plotter_speed(collector(all_4000), "res: 4000", "-")
plt.xlabel("number of processors")
plt.ylabel("speedup factor")
plt.title("Speedup plot")
plt.grid()
plt.legend()
plt.show()
# For saving
#plt.savefig('Speedplot_ges.png')



#The efficiency plot for resolutions 125, 500, 2000 and 4000
plt.figure(1)
plotter_effi([1,1/2,1/5,1/10, 1/20, 1/40, 1/60, 1/80], "ideal", "--")
plotter_effi(collector(all_125), "res: 125", "-")
plotter_effi(collector(all_500), "res: 500", "-")
plotter_effi(collector(all_2000), "res: 2000", "-")
plotter_effi(collector(all_4000), "res: 4000", "-")
plt.xlabel("number of processors")
plt.ylabel("efficiency")
plt.title("Efficiency plot")
plt.legend()
plt.grid()
plt.show()
# For saving
#plt.savefig('Effiplot_ges.png')




# For comparison of Task 2 and Task 3:

#The data for Task 3 was collected using a similar script and the outfiles of Task 3, the generated Lists of 
# speedups were copied here:

res_125 = [ 1.      ,    1.90385517 , 4.18619575 , 7.96431852, 10.76771152, 14.56582514, 19.83731158, 31.6823583 ]
res_500 = [ 1.     ,     1.92538326 , 4.5311982,   8.69135078 ,13.69843082 ,27.12097998, 35.01287685, 46.10222314]
res_2000 = [ 1.     ,     1.90966271,  4.74329138 , 9.05711666, 18.39284036 ,31.32786892, 47.30662371 ,57.63240913]
res_4000 = [ 1. ,1.9087988  , 4.63150163,  8.74904129, 17.98772337, 32.79940382, 52.17733837, 67.91949285]
n_processors = [1,2,5,10, 20, 40, 60, 80]



#This data was then used to generate a common plot for the speedup in Task 2 and 3:
plt.figure(2)
plotter_speed([1,1/2,1/5,1/10, 1/20, 1/40, 1/60, 1/80], "ideal", "--")
plotter_speed(collector(all_125), "Task 2 res: 125", "-.")
plotter_speed(collector(all_500), "Task 2 res: 500", "-.")
plotter_speed(collector(all_2000), "Task 2 res: 2000", "-.")
plotter_speed(collector(all_4000), "Task 2 res: 4000", "-.")
plt.plot(n_processors, res_125, label = "Task 3 res: 125")
plt.plot(n_processors, res_500, label = "Task 3 res: 500")
plt.plot(n_processors, res_2000, label = "Task 3 res: 2000")
plt.plot(n_processors, res_4000, label = "Task 3 res: 4000")

plt.title("Speedup plot comparison")
plt.xlabel("number of processors")
plt.ylabel("speedup factor")
plt.legend()
plt.grid()
plt.show()
#To save plot as file:
#plt.savefig('Speedplot_compare.png')
    


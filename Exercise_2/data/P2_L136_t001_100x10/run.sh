#!/bin/bash

particles=2
L=13.6
time=0.01
N=100

python3 Task2.py $particles $L 1
cat ./initial.txt
python3 Task3.py ./initial.txt $time $N
cat ./trajectories.txt
python3 Task4.py ./trajectories.txt
cat ./energies.txt
cat ./densities.txt

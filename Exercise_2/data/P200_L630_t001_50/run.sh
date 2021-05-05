#!/bin/bash

particles=200
L=6.30
time=0.01
N=50

python3 Task2.py $particles $L 1
cat ./initial.txt
python3 Task3.py ./initial.txt $time $N
cat ./trajectories.txt
python3 Task4.py ./trajectories.txt
cat ./energies.txt
cat ./densities.txt

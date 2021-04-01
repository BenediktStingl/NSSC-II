# Makefile for Task 2

CXX = g++
MPICXX = mpic++
CXXFLAGS = -std=c++11 -Wall -pedantic -O3 # -ffast-math -march=native

all: jacobiMPI RunMPI

jacobiMPI: jacobiMPI.cpp mesh.hpp solver.hpp
	$(MPICXX) jacobiMPI.cpp -o jacobiMPI -lpthread $(CXXFLAGS)

RunMPI:
	mpirun -n 6 -oversubscribe ./jacobiMPI 1D 200 200

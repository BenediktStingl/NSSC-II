# requirements on ubuntu
# sudo apt-get build-essentials
# sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev

# required modules on cluster
# module load mpi/openmpi-x86_64 
# module load pmi/pmix-x86_64

CXX=g++
MPICXX?=mpic++
CXXFLAGS := $(CXXFLAGS) -std=c++14 -O3 -Wall -pedantic -march=native -ffast-math

jacobiMPI: Makefile jacobiMPI.cpp solver.hpp mesh.hpp
	$(MPICXX) jacobiMPI.cpp -o jacobiMPI -lpthread -DUSEMPI $(CXXFLAGS)

clean:
	rm jacobiMPI

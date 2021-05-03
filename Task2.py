#!/usr/bin/env python

import sys
import numpy.random as random
import scipy.optimize as optimize
import matplotlib.pyplot as plt

import jax
import jax.numpy as np
import jax.config

jax.config.update("jax_enable_x64", True)


def readInputArguments(argv):
	if len(argv) == 4:
		scipt, M, L, SIG = argv
		return int(M), float(L), float(SIG)
	else:
		print("Usage: ./Task2 M L SIG")
		exit()


def BC_control(pos_list, L, M):
    tmp_pos = []
    if len(np.shape(pos_list)) > 1:
        for p in pos_list:
            for pos in p:
                if pos > L:
                    tmp_pos.append(pos%L)
                elif pos < 0:
                    tmp_pos.append((L-(pos%L)))
                else:
                    tmp_pos.append(pos)
    else:
        for p in pos_list:
            if p > L:
                tmp_pos.append(p%L)
            elif p < 0:
                tmp_pos.append(L-(p%L))
            else:
                tmp_pos.append(p)
    return np.array(tmp_pos).reshape((M, 3))

def generateInitialCoords(M, L, dims):
	# create RNG with seed
	random.seed(42)
	# get random coordinates within the domain (floats)
	coords = L*random.rand(M, dims)
		
	print("potential random state: ", Epot(np.ndarray.flatten(coords), L))
	result = optimize.minimize(Epot, coords, L, method="CG")
	print("potential minimum: ", result.fun)

	return BC_control(result.x, L, M)


def minimumImage(delta, L):
    		return delta - L * np.round(delta / L)


@jax.jit
def Epot(coords, L):
    E_pot = 0.0
    coords = coords.flatten()
    for i in range(0, coords.size-3, 3):
        coords_i = coords[i:i+3]
        coords_j = coords[i+3:]
        diff = np.broadcast_to(coords_i, (np.shape(coords_j)[0]//3, 3)) - \
              np.array(coords_j).reshape((np.shape(coords_j)[0]//3, 3))
        diff = minimumImage(diff.flatten(), L)**2
        #diff[1:] = minimumImage(diff[:], L)**2
        #diff[2:] = minimumImage(diff[:], L)**2
        diff = diff.reshape((np.shape(coords_j)[0]//3, 3))
        r = np.sqrt(np.sum(diff, axis=1))
                
        E_pot += np.sum(4*( ( (2**(-1/6))/r )**12 - ( (2**(-1/6))/r )**6 ))
    return E_pot

"""
@jax.jit
def Epot(coords, L):

	E_pot = 0.0
	coords = coords.flatten()	
	for i in range(0, coords.size-1, 3):
		for j in range(i+3, coords.size, 3):
			r = np.sqrt( minimumImage(coords[i]-coords[j], L)**2 +
						 minimumImage(coords[i+1]-coords[j+1], L)**2 +
					 	 minimumImage(coords[i+2]-coords[j+2], L)**2 )
			# Lennard-Jones
			E_pot += 4*( ( (2**(-1/6))/r )**12 - ( (2**(-1/6))/r )**6 )
	return E_pot
"""

def calculateInitialForces(M, L, coords):
	gradient = jax.jit(jax.grad(Epot))
	forces = -gradient(coords.flatten(), L)

	forces = np.array(forces).reshape((M, 3))
	return forces


def generateInitialVelocities(M, SIG, dims):
	# create random velocities
	SIG_mat = SIG * np.eye(dims)
	mean = np.zeros(dims)
	vels = random.multivariate_normal(mean, SIG_mat, size=M)

	# calculate mean velocity
	mean_vels = np.mean(vels, axis=0)

	# update velocities
	vels = vels - mean_vels

	vels = np.array(vels).reshape((M, 3))
	return vels


def createFilestring(M, L, coords, vels, comment):
	filestring = f"{M}\n{comment}\n{L}\n"

	for k in range(0, coords.shape[0]):
		x, y, z = coords[k][0], coords[k][1], coords[k][2]
		vx, vy, vz = vels[k][0], vels[k][1], vels[k][2]
		filestring += f"{x} {y} {z} {vx} {vy} {vz}\n"

	return filestring


def writeFile(filestring, filename):
	# overwrites or creates a new file
	f = open(filename, "w")
	f.write(filestring)
	f.close()

	# test output to terminal
	print("\n", filestring)


if __name__ == "__main__":
	from time import perf_counter
	start = perf_counter()
	### input variables ###
	M, L, SIG = readInputArguments(sys.argv)
	dims = 3

	### find minimum potential and coordinates ###
	coords = generateInitialCoords(M, L, dims)

	### calculate forces ###
	forces = calculateInitialForces(M, L, coords)

	# check if sum of all forces for every axis is zero
	print(f"axis=x: forcesSum =", np.sum(forces[:,0]))
	print(f"axis=y: forcesSum =", np.sum(forces[:,1]))
	print(f"axis=z: forcesSum =", np.sum(forces[:,2]))

	### generate velocities ###
	vels = generateInitialVelocities(M, SIG, dims)

	### write results into file ###
	filestring = createFilestring(M, L, coords, vels, "Time step 0")
	writeFile(filestring, "initial.txt")
	end = perf_counter()
	print("execution time: ", end - start)


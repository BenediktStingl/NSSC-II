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
    """
    Reads input parameters from the command line.

    Parameters
    ----------
    argv : str
        Input parameter string containing the number of particles (M), the
        side length of the box under consideration (L) and the standard 
        deviation for the distribution of the initial velocities (SIG).

    Returns
    -------
    int, float, float
        Particle number, M, box side length, L, standard deviation, SIG.

    """
	if len(argv) == 4:
		scipt, M, L, SIG = argv
		return int(M), float(L), float(SIG)
	else:
		print("Usage: ./Task2 M L SIG")
		exit()


def BC_control(pos_list, L, M):
    """
    Applies Boundary Condisions, in case a particle leaves the box another one
    enters the box. Differentiates between multidimensional and 1D arrays.

    Parameters
    ----------
    pos_list : numy array 
        array of particle positions, can be 1D or multidimansional containing
        the xyz coordinates.
    L : float
        box side length.
    M : int
        Particle number.

    Returns
    -------
    numpy array
        Returns the corrected particle positions as a multidimensional array.

    """
    tmp_pos = []
    if len(np.shape(pos_list)) > 1: #if a multidimensional array
        for p in pos_list:
            for pos in p:
                if pos > L:
                    tmp_pos.append(pos%L)
                elif pos < 0:
                    tmp_pos.append((L-(pos%L)))
                else:
                    tmp_pos.append(pos)
    else: #if a 1D array
        for p in pos_list:
            if p > L:
                tmp_pos.append(p%L)
            elif p < 0:
                tmp_pos.append(L-(p%L))
            else:
                tmp_pos.append(p)
    return np.array(tmp_pos).reshape((M, 3))

def generateInitialCoords(M, L, dims):
    """
    Generates the initial random positions.

    Parameters
    ----------
    M : int
        Particle number.
    L : float
        box side length.
    dims : int
        number of dimensions considered. Used to be equal to two for initial
        testing.

    Returns
    -------
    numpy array
        initial random state.

    """
	# create RNG with seed for reproducable results for testing
	#random.seed(42) 
	# get random coordinates within the domain (floats)
	coords = L*random.rand(M, dims)
		
	print("potential random state: ", Epot(np.ndarray.flatten(coords), L))
	result = optimize.minimize(Epot, coords, L, method="CG")
	print("potential minimum: ", result.fun)

	return BC_control(result.x, L, M)


def minimumImage(delta, L):
    """
    applies the minimum image convention.

    Parameters
    ----------
    delta : float
        difference in one coordinate between two particles. See lecture slide 
        20.
    L : float
        box side length.

    Returns
    -------
    float
        corrected delta for minimum inmage convention.

    """
    		return delta - L * np.round(delta / L)


@jax.jit
def Epot(coords, L):
    """
    Determines the potential for the particle cluster.

    Parameters
    ----------
    coords : numpy array
        Array of particle coordinates.
    L : float
        box side length.

    Returns
    -------
    E_pot : float
        potential energy of the particles under consideration.

    """
    E_pot = 0.0
    coords = coords.flatten()
    for i in range(0, coords.size-3, 3):
        coords_i = coords[i:i+3]
        coords_j = coords[i+3:]
        diff = np.broadcast_to(coords_i, (np.shape(coords_j)[0]//3, 3)) - \
              np.array(coords_j).reshape((np.shape(coords_j)[0]//3, 3))
        diff = minimumImage(diff.flatten(), L)**2
        diff = diff.reshape((np.shape(coords_j)[0]//3, 3))
        r = np.sqrt(np.sum(diff, axis=1))
                
        E_pot += np.sum(4*( ( (2**(-1/6))/r )**12 - ( (2**(-1/6))/r )**6 ))
    return E_pot


def calculateInitialForces(M, L, coords):
    """
    Determines the forces acting on each particle.

    Parameters
    ----------
    M : int
        Particle number.
    L : float
        box side length.
    coords : numpy array
        Array of particle coordinates.

    Returns
    -------
    forces : numpy array
        Forces acting on each particle in each component.

    """
	gradient = jax.jit(jax.grad(Epot))
	forces = -gradient(coords.flatten(), L)

	forces = np.array(forces).reshape((M, 3))
	return forces


def generateInitialVelocities(M, SIG, dims):
    """
    Generates the initial random velocities for all particles.

    Parameters
    ----------
    
    M : int
        Particle number.
    SIG : float
        standard deviation.
    dims : int
        number of dimensions considered. Used to be equal to two for initial
        testing.

    Returns
    -------
    vels : numpy array
        initial velocity for each particle in each component.

    """
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
    """
    Generates the filestring which is then written to a file in the format
    requested in the tast description.

    Parameters
    ----------
    M : int
        Particle number.
    L : float
        box side length.
    coords : numpy array
        Array of particle coordinates.
    vels : numpy array
        initial velocity for each particle in each component.
    comment : string
        Any comment given.

    Returns
    -------
    filestring : string
        string which is then written to a file.

    """
	filestring = f"{M}\n{comment}\n{L}\n"

	for k in range(0, coords.shape[0]):
		x, y, z = coords[k][0], coords[k][1], coords[k][2]
		vx, vy, vz = vels[k][0], vels[k][1], vels[k][2]
		filestring += f"{x} {y} {z} {vx} {vy} {vz}\n"

	return filestring


def writeFile(filestring, filename):
    """
    Writes a given filestring to a file.

    Parameters
    ----------
    filestring : string
        string which is written to a file.
    filename : string
        name of the file (over)written.

    Returns
    -------
    None.

    """
	# overwrites or creates a new file
	f = open(filename, "w")
	f.write(filestring)
	f.close()

	# test output to terminal
	print("\n", filestring)


if __name__ == "__main__":
	from time import perf_counter #for time measurement
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


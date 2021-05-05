#!/usr/bin/env python

import Task2
from Task2 import BC_control

import sys
import numpy # necessary for updates

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
        Input parameter string containing the input file path (path), the
        time step (delta_t) and the number of iterations (N).

    Returns
    -------
    path : str float, int
        Input file path.
    delta_t : float    
        Time step.
    N : int
        Number of iterations.

    """
    if len(argv) == 4:
        scipt, path, delta_t, N = argv
        return path, float(delta_t), int(N)
    else:
        print("Usage: ./Task3 path delta_t N")
        exit()

    
def readFile(path):
    """
    Reads a file from the given input file path.

    Parameters
    ----------
    path : str
        Input file path.

    Returns
    -------
    M : int
        Number of particles.
    C : str
        Comment.
    L : float
        Box side length.
    coords : numpy array
        Array of initial particle coordinates.
    vels : numpy array
        Array of initial particle velocities.

    """
    f = open(path, "r")
    M = int(f.readline())
    C = f.readline()
    L = float(f.readline())

    coords = []
    vels = []
    for line in f:
        coords.append([float(x) for x in line.split()[0:3]])
        vels.append([float(x) for x in line.split()[3:6]])
    f.close()

    return M, C, L, np.array(coords), np.array(vels)


def calculateNewCoords(coords, vels, forces, M, L, delta_t):
    """
    Calculates the coordinates of the current time step.

    Parameters
    ----------
    coords : numpy array
        Array of the particle coordinates from the previous time step.
    vels : numpy array
        Array of the particle velocities from the previous time step.
    forces : numpy array
        Array of the particle forces from the current time step.
    M : int
        Number of particles.
    L : float
        Box side length.
    delta_t : float
        Time step.

    Returns
    -------
    coords : numpy array
        Array of the particle coordinates from the current time step.

    """
    newCoords = numpy.full_like(coords, 0.0) # numpy array allows updates
    newCoords[:][:] = coords[:][:] + \
                      vels[:][:] * delta_t + \
                      0.5 * forces[:][:] * (delta_t)**2
    return BC_control(newCoords, L, M)


def calculateNewVels(coords, vels, forces, M, L, delta_t):
    """
    Calculates the velocities of the current time step.

    Parameters
    ----------
    coords : numpy array
        Array of the particle coordinates from the current time step.
    vels : numpy array
        Array of the particle velocities from the previous time step.
    forces : numpy array
        Array of the particle forces from the current time step.
    M : int
        Number of particles.
    L : float
        Box side length.
    delta_t : float
        Time step.

    Returns
    -------
    newVels : numpy array
        Array of the particle velocities from the current time step.
    newForces : numpy array
        Array of the particle velocities from the next time step.

    """
    newForces = Task2.calculateInitialForces(M, L, coords)
    newVels = numpy.full_like(vels, 0.0) # numpy array allows updates
    newVels[:][:] = vels[:][:] + \
                    0.5 * (forces[:][:] + newForces[:][:]) * delta_t
    return newVels, newForces


def VerletAlgorithm(coords, vels, forces, M, delta_t, N):
    """
    Algorithm to integrate Newtons equations of motion. Writes the results 
    of the calulcations in the requested format into an output file.

    Parameters
    ----------
    coords : numpy array
        Array of the particle coordinates from the previous time step.
    vels : numpy array
        Array of the particle velocities from the previous time step.
    forces : numpy array
        Array of the particle forces from the previous time step.
    M : int
        Number of particles.
    delta_t : float
        Time step.
    N : int
        Number of iterations.

    Returns
    -------
    None.

    """
    filestring = Task2.createFilestring(M, L, coords, vels, "Time step 0")
    for k in range(1, N):
        coords = calculateNewCoords(coords, vels, forces, M, L, delta_t)
        vels, forces = calculateNewVels(coords, vels, forces, M, L, delta_t)
        filestring += Task2.createFilestring(M, L, coords, vels, f"Time step {k}")
        print(f"finished timestep {k}")

    Task2.writeFile(filestring, "trajectories.txt")


if __name__ == "__main__":
    from time import perf_counter
    start = perf_counter()

    path, delta_t, N = readInputArguments(sys.argv)
    M, C, L, coords, vels = readFile(path)

    forces = Task2.calculateInitialForces(M, L, coords)
    VerletAlgorithm(coords, vels, forces, M, delta_t, N)
    end = perf_counter()
    print("execution time: ", end - start)

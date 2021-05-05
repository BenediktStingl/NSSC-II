#!/usr/bin/env python

import sys
#For the epot function
from Task2 import Epot, writeFile, minimumImage
import numpy # necessary for updates

import jax
import jax.numpy as np
import jax.config

jax.config.update("jax_enable_x64", True)

def readInputArguments(argv):
    if len(argv) == 2:
        path = argv[1]
        return path
    else:
        print("Usage: ./Task4 path")
        exit()


def readFile(path):
    time_steps = []
    coords = []
    vels = []
    f = open(path, "r")

    for line in f:
        M = int(line)
        C = f.readline()
        #make list of timesteps names
        C = C.replace("\n", "")
        time_steps.append(C)
        L = float(f.readline())
        for i in range(M):
            coords_and_vels = f.readline()
            coords.append([float(x) for x in coords_and_vels.split()[0:3]])
            vels.append([float(x) for x in coords_and_vels.split()[3:6]])
    
    f.close()

    return M, C, L, np.array(coords).flatten(), np.array(vels).flatten(), time_steps

"""
def Get_Epots(coordinates, L):
    list_of_Epots = []
    for time_step in coordinates:
        list_of_Epots.append(Epot(time_step, L))
    
    return list_of_Epots
"""

@jax.jit
def Get_Epots(coordinates, L):
    list_of_Epots = []
    for j in range(len(time_steps)):
        arr_coords = coordinates[j*3*M:(j+1)*3*M]
        list_of_Epots.append(Epot(arr_coords, L))
    
    return list_of_Epots


def Get_Ekins(velocities, M, time_steps_len):
    list_of_Epots = []
    #For all timesteps
    for j in range(time_steps_len):
        arr_vels = velocities[j*3*M:(j+1)*3*M]**2
        vels_sum = 0.5 * np.sum(arr_vels)
        list_of_Epots.append(vels_sum)

    return list_of_Epots

"""
def Get_Ekins(velocities, M, time_steps):
    list_of_Epots = []
    #For all timesteps
    for j in range(len(time_steps)):
        sum = 0.
        #For all atoms
        for i in range(M):
            # |vi|
            for k in range(3):
                sum += velocities[j][i][k]**2
        list_of_Epots.append(sum/2)

    return list_of_Epots
"""

def createFile1(Epots, Ekins, time_steps, M, L):
    filestring = "Timestep:     Epot:   Ekin: \n"
    for k in range(len(time_steps)):
        timestep, E_pot, E_kin = time_steps[k], Epots[k], Ekins[k]
        filestring += f"{timestep}: {E_pot} {E_kin}\n"

    writeFile(filestring, "energies.txt")

def createFile2(r_list, av_vol_density, M, L):
    filestring = "r: average volumetric density: \n"
    for k in range(len(r_list)):
        r, avdensity = r_list[k], av_vol_density[k]
        filestring += f"{r}: {avdensity}\n"
    writeFile(filestring, "densities.txt")

# r list for file2:
def get_r(n_r , L):
    r_list = []
    step = L/(2*(n_r))
    for i in range(1, n_r+1):
        r = i*step
        r_list.append(r)
    return r_list


def get_av_vol_density(r_list, coordinates, M, time_steps):
    vol_densitys = []
    #Iteration over all rs in the list
    for r in r_list:
        #Because particle at center must be included
        inside_particles = 0.
        #Iteration over the last 3/4 of timesteps 
        twentyfive_percent = round(len(time_steps)/4)
        for i in range(twentyfive_percent, len(time_steps)):
            # choose first particle as center:
            center_coordinates = coordinates[i*3*M : i*3*M+3]
            other_coordinates = coordinates[i*3*M+3:(i+1)*3*M]
            diff = np.broadcast_to(center_coordinates, (np.shape(other_coordinates)[0]//3, 3)) - \
                   np.array(other_coordinates).reshape((np.shape(other_coordinates)[0]//3, 3))
            diff = minimumImage(diff.flatten(), L)**2
            diff = diff.reshape((np.shape(other_coordinates)[0]//3, 3))
            r_search = np.sqrt(np.sum(diff, axis=1))
            #print("jax: ", jax.numpy.less_equal(r_search, r))
            inside_particles += 1 + np.sum(jax.numpy.less_equal(r_search, r))
            #print("particles: ", inside_particles)

        average = inside_particles/(len(time_steps)-twentyfive_percent)
        volume = 4/3 *np.pi *r**3
        vol_densitys.append(average/volume)
        print(f"finished radius {r}")
    return(vol_densitys)




"""
def get_av_vol_density(r_list, coordinates, M):
    n_timesteps = len(coordinates)
    vol_densitys = []
    #Iteration over all rs in the list
    for r in r_list:
        #Because particle at center must be included
        inside_particles = 1.0
        #Iteration over the last 3/4 of timesteps 
        twentyfive_percent = round(len(coordinates[0])/4)
        for i in range(twentyfive_percent, len(coordinates)):
            # choose first particle as center:
            center_coordinates = [coordinates[i][0][0], coordinates[i][0][1], coordinates[i][0][2]]
            #Iteration over all other particles:
            for j in range(1,len(coordinates[0])):
                distance = 0.0
                #Iterations over x, y and z coordinate:
                for k in range(3):
                    distance += minimumImage((coordinates[i][j][k]-center_coordinates[k]),L)**2
                if distance < r**2:
                    inside_particles +=1

        average = inside_particles/(len(coordinates)-twentyfive_percent)
        volume = 4/3 *np.pi *r**3
        vol_densitys.append(average/volume)
    return(vol_densitys)
"""

if __name__ == "__main__":
    from time import perf_counter
    start = perf_counter()
    path = readInputArguments(sys.argv)
    M, C, L, coordinates, velocities, time_steps = readFile(path)
    #print(velocities)
    Epots = Get_Epots(coordinates, L)
    Ekins = Get_Ekins(velocities, M, np.shape(time_steps)[0])
    print("Epots: ", len(Epots))
    print("Ekins: ", len(Ekins))
    #print(len(time_steps))
    createFile1(Epots, Ekins, time_steps, M, L)
    rs = get_r(5, L)
    ava_volumetric_density = get_av_vol_density(rs, coordinates, M, time_steps)
    #print(len(rs))
    #print(len(ava_volumetric_density))
    #print(coordinates[0][0][0])
    createFile2(rs, ava_volumetric_density, M, L)

    #print("This is with r = 10",get_av_vol_density([10], coordinates, M))

    import matplotlib.pyplot as plt

    time_step_array = np.linspace(0, len(time_steps),len(time_steps))
    plt.plot(time_step_array, np.array(Epots)+np.array(Ekins))
    plt.savefig(f"Energies.png")
    end = perf_counter()
    print("execution time: ", end - start)

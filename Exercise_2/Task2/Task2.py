#!/usr/bin/env python

def LJ(r):
    sig = 2**(-1/6)
    return 4*((sig/r)**12 - (sig/r)**6)

def Epot(param):
    pot = 0.
    for i in range(0,len(param)-1, 3):
        for j in range(i+3,len(param), 3):
            r = np.sqrt((param[i]-param[j])**2+(param[i+1]-param[j+1])**2+(param[i+2]-param[j+2])**2)
            pot += LJ(r)
    return pot

def Epot2D(param):
    pot = 0.
    for i in range(0,len(param)-1, 2):
        for j in range(i+2,len(param), 2):
            r = np.sqrt((param[i]-param[j])**2+(param[i+1]-param[j+1])**2)
            pot += LJ(r)
    return pot


if __name__ == "__main__":

	import numpy as np
	import scipy.optimize as optimize
	import matplotlib.pyplot as plt

	#from jax import grad #uncomment for jax!

	### plot potential function ###
	"""
	x = np.arange(0.87, 3, 0.005)

	plt.plot(x, LJ(x))

	plt.grid()
	plt.box(False)
	plt.show()
	#--> local minimum at r=1
	"""

	### input variables ###
	dims = 3 #2D or 3D
	M = 2 #number of atoms
	L = M/0.8 #length of box
	SIG = 1 #standard deviation

	### create random coordinates ###
	### uncomment for integers as starting conditions ###
	#rng = np.random.default_rng(seed=42) #create RNG with seed
	#coords = rng.integers(low=0, high=L, size=(M,dims)) #get random coordinates within the domain (integers)

	### uncomment for floats as starting conditions ###
	np.random.seed(42) #create RNG with seed
	coords = L*np.random.rand(M,dims) #get random coordinates within the domain (floats)
		
	print("coords: \n", coords)
	if dims == 3:
		print("potential random state: ", Epot(np.ndarray.flatten(coords)), "\n")
		result = optimize.minimize(Epot, coords, method="CG")
		#derivative_fn = grad(Epot)                                  #uncomment for jax!
	elif dims == 2:
		print("potential random state: ", Epot2D(np.ndarray.flatten(coords)), "\n")
		result = optimize.minimize(Epot2D, coords, method="CG")
		#derivative_fn = grad(Epot2D)                                #uncomment for jax!
	else:
		print("RECOMMEND OTHER DIMENSIONAL INPUT")
		
	### find minimum potential and forces ###
	print("potential minimum: ", result.fun)
	print("new positions: ", result.x, "\n")
	#print("forces: ", derivative_fn(result.x))

	### create random velocities ###
	SIG_mat = SIG * np.eye(dims)
	mean = np.zeros(dims)
	vels = np.random.multivariate_normal(mean, SIG_mat, size=M)
	print("velocities: \n", vels)

	### calculate mean velocity ###
	mean_vels=np.mean(vels, axis=0)
	print("mean velocity: ", mean_vels)

	### update velocities ###
	vels = vels - mean_vels
	print("corrected velocities: \n", vels)

	### write results into file ###
	f = open("results.txt", "w") #overwrites or creates a new file

	filestring = f"""{M}
	This is a comment
	{L}\n"""

	### plot initial random state (only 2D) ###
	if dims == 2:
		for n, c in enumerate(coords):
		    if n == 0:
		        plt.scatter(c[0], c[1], color="k", label="rnd initial state")
		    else:
		        plt.scatter(c[0], c[1], color="k")
	else:
		pass

	for k in range(0, len(result.x), dims):
		if dims == 3:
		    x, y, z = result.x[k], result.x[k+1], result.x[k+2]
		    vx, vy, vz = vels[k//dims]  ### check if that works!
		    filestring += f"{x} {y} {z} {vx} {vy} {vz}\n"
		elif dims == 2:
		    x, y = result.x[k], result.x[k+1]
		    vx, vy = vels[k//dims]
		    filestring += f"{x} {y} {vx} {vy}\n"
		
		if k == 0 and dims == 2:
		    plt.scatter(x, y, color="r", label="minimum state") #label in plot
		elif k != 0 and dims == 2:
		    plt.scatter(x, y, color="r")
		else:
		    pass #only for dims neq 2

		
	f.write(filestring)
	f.close()

	if dims == 2:
		plt.legend()
		plt.show()
	else:
		pass

	print("\n", filestring)

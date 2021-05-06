import coilpy as cp
import numpy as np
import pickle
from reformat import fourier_to_cart
from pyevtk.hl import polyLinesToVTK

## plasma boundary file names
#plasma_input  = '../w7x_jf/w7x_jf.boundary'
#plasma_output = '../w7x_jf/w7x_jf_plasma_boundary'
#
## write the plasma boundary vtk file
#plasma = cp.FourSurf.read_focus_input(plasma_input)
#plasma.toVTK(plasma_output)

## coil file names
#coil_input  = "../data/ncsx/diagnos_mc_BFGS_optimization_seed_20210405103805.pickle"
#coil_output = "../data/ncsx/diagnos_mc_BFGS_optimization_seed_20210405103805.pickle"
#
## load the coils
#indata      = pickle.load(open(coil_input,"rb"))
#x0          = np.sqrt(1.5)*indata['x_best']
#n_modes     = 8+1 #indata['n_modes']
#n_coils     = 3 #indata['n_coils']
#
## convert coils to cartesian points
#n_time = 300
#C = fourier_to_cart(x0,n_modes,n_coils,n_time)
#
## write to VTK
#X = C[:,0,:].flatten()
#Y = C[:,1,:].flatten()
#Z = C[:,2,:].flatten()
#polyLinesToVTK(coil_output, X,Y,Z,n_time*np.ones(n_coils))

## use coilpy to plot coils
import matplotlib.pyplot as plt
coil = cp.Coil.read_makegrid('../ncsx/coils.c09r00_mc')
coil.plot(engine='pyplot', color=(0,0,1))
plt.show()

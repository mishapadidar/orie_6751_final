import numpy as np
import pickle
import sys
sys.path.append("../utils")
from reformat import fourier_to_cart
import matplotlib.pyplot as plt
from surface import FourSurf

# plasma boundary file names
plasma_input  = '../experiments/w7x_jf.boundary'
plasma = FourSurf.read_focus_input(plasma_input)

# load the coils
coil_input  = "../experiments/output/data_problem_0_batch_size_10_delta_0.05_psize_0.005_shift_0.002_20210519135733.pickle"
indata      = pickle.load(open(coil_input,"rb"))
x0          = indata['xopt']
#x0 = x0[:-1] # remove CVaR variable
n_modes     = indata['nfcoil']+1
n_coils     = indata['ncoils']

# convert coils to cartesian points
n_time = 300
# n_coils x dim x n_time
C = fourier_to_cart(x0,n_modes,n_coils,n_time)


fig = plt.figure()
ax = plt.axes(projection='3d')
for c in C:
  ax.plot3D(c[0],c[1],c[2], 'k')
plasma.plot3d(ax=ax,alpha=0.5)

plt.show()

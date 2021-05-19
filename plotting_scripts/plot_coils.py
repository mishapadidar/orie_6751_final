import numpy as np
import pickle
from reformat import fourier_to_cart
import matplotlib.pyplot as plt

# load the coils
coil_input  = "../experiments/output/data_problem_0_20210518161607.pickle"
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
  ax.plot3D(c[0],c[1],c[2], 'gray')
plt.show()

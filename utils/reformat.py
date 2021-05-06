import numpy as np

def fourier_to_cart(x,n_modes,n_coils,n_time=100):
  """
  Convert a coil in the vector fourier representation
  to cartesian representation.

  The variables x from FOCUS are organized first by coils, i.e. coil 1 information before coil 2 
  information. For each coil we organize the fourier coefficents by first the x cosines, 
  then x sines, y cosines, y sines, z cosines, z sines.

  return:
  C: 3D array; size (n_coils x 3 x n_time)
     Each nested 2D array contains an X, Y, Z row representing that coordinate of a coil
  """
  dim      = 3
  n_coeffs = 2*n_modes-1

  # reshape to get coil set in better format
  coils    = np.reshape(x,(n_coils,dim,n_coeffs))

  # coil time discretization
  time     = np.linspace(0,2*np.pi,n_time)

  # storage
  X = np.zeros((n_coils,dim,n_time))

  # cos and sin array
  coss = np.array([np.cos(n*time) for n in range(n_modes)])
  sins = np.array([np.sin(n*time) for n in range(1,n_modes)])
  r = np.vstack((coss,sins))

  for c in range(n_coils):
    # compute the fourier series for coils
    X[c] = np.copy(coils[c] @ r)

  return X

if __name__ == '__main__':
  import pickle
  d = pickle.load(open("../data/circular_coils.pickle","rb"))
  x = d['coils']
  n_coils = d['n_coils']
  n_modes = d['n_modes']
  C = fourier_to_cart(x,n_modes,n_coils)
  print(C)


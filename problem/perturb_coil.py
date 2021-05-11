import numpy as np
from uniform_rotations import generate_uniform_rotation

def perturb_coil(x,C,max_ang,max_shift,n_coils,nfcoil):
  """perturb a coil. 
  input 
  x: 1d array, fourier coeffs in focus format... no currents
  C: 2d covariance matrix for normal perturbations
  max_ang: float, maximum rotation of coil
  max_shift: array, upper bound on uniform translations
  n_coils: number of coils
  nfcoil: number of fourier coefficents per coil
  return fourier coeffs
  """
  dim = 3
  # apply the gaussian perturbation
  R = np.linalg.cholesky(C)
  p = x + R @ np.random.randn(len(x))
  # reshape to (ncoils x 3 dim x (2*nfcoil +1))
  p    = np.reshape(p,(n_coils,dim,2*nfcoil+1))
  # get the zeroth order terms
  zeroth = p[:,:,0]
  # apply the uniform rotation
  Q = generate_uniform_rotation(max_ang = max_ang)
  for ii in range(n_coils):
    p[ii] = np.copy(Q @ p[ii,:,:]) # batch matrix multiply
  # remove rotation from the 0th order terms
  p[:,:,0] = np.copy(zeroth)
  # apply the uniform translation
  p[:,:,0] += np.random.uniform(-max_shift,max_shift,size=np.shape(zeroth))
  # reshape
  p = np.reshape(p,x.shape)
  return p

if __name__ == "__main__":
  x = np.random.randn(195)
  n_coils = 5
  nfcoil  = 6
  n_modes = nfcoil+1
  perturbation_size = 0.002
  gp_lengthscale = 0.5
  from normal_covariances import compute_fourier_coefficient_covariance_for_FOCUS
  C = compute_fourier_coefficient_covariance_for_FOCUS(n_coils,n_modes,perturbation_size, gp_lengthscale,drop_pickle=False)
  max_ang = np.pi/12
  max_shift = 0.005
  y = perturb_coil(x,C,max_ang,max_shift,n_coils,nfcoil)
  #print(y)

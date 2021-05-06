import numpy as np
from sklearn.gaussian_process.kernels import ExpSineSquared
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle

def fourier_coeffs(kappa,n_modes = 10):
  """
  Compute Fourier Coefficients of kappa(t,s).
  Specifically we return the coefficients c_{n,n} and s_{n,n}
  for n < n_modes.
  
  input:
  kappa(s,t): function with 2d input
  a: length 2 array, lower bound
  b: length 2 array, upper bound
  n_modes: number of fourier modes

  return:
  C: 1d array of length n_modes, contains cosine fourier coeffs of kappa
  S: 1d array of length n_modes, contains sine fourier coeffs of kappa
  """
  assert n_modes >0 

  # storage Fourier modes
  C = np.zeros(n_modes)
  S = np.zeros(n_modes-1)

  # normalization for basis
  normalization = np.pi

  # zeroth mode cosine-cosine term
  f = lambda x,y: np.cos(0*x)*np.cos(0*y)*kappa(x,y)
  C[0],err=dblquad(f,0.0,2*np.pi,lambda x: 0.0,lambda x: 2*np.pi,epsabs=1e-9)
  #print(f"cos mode {0} err ={err}")

  for n in range(1,n_modes):
    # cosine-cosine term
    f = lambda x,y: np.cos(n*x)*np.cos(n*y)*kappa(x,y)
    C[n],err=dblquad(f,0.0,2*np.pi,lambda x: 0.0,lambda x: 2*np.pi,epsabs=1e-9)
    #print(f"cos mode {n} err ={err}")

    # sine-sine term
    f = lambda x,y: np.sin(n*x)*np.sin(n*y)*kappa(x,y)
    S[n-1],err = dblquad(f,0,2*np.pi,lambda x: 0.0,lambda x: 2*np.pi,epsabs=1e-9)
    #print(f"sin mode {n} err ={err}")

  C = C/normalization
  S = S/normalization

  return C,S

def all_fourier_coeffs(kappa,n_modes = 10):
  """
  Compute Fourier Coefficients of kappa(t,s).
  Specifically we return all fourier coefficients as a matrix
  [[cosine-cosine modes,cosine-sine modes],[sin-cosine modes, sine-sine modes]]
  for n < n_modes.

  We do not do any computations for the sin(0*t) mode b/c it is always zero and
  so we omit it. 
  
  input:
  kappa(s,t): function with 2d input
  a: length 2 array, lower bound
  b: length 2 array, upper bound
  n_modes: number of fourier modes

  return:
  F: (n+n-1) x (n+n-1) matrix of fourier coefficients
  """
  assert n_modes >0 

  # storage Fourier modes
  CC = np.zeros((n_modes,n_modes))
  CS = np.zeros((n_modes,n_modes-1))
  SS = np.zeros((n_modes-1,n_modes-1))
  F  = np.zeros((2*n_modes-1,2*n_modes-1))

  # normalization for basis
  normalization = (2*np.pi)**2

  for n1 in range(0,n_modes):
    for n2 in range(0,n_modes):
      #print("")
      if n1 == 0 and n2 == 0:
        kk =1
      elif n1 >0 and n2 == 0:
        kk = 2
      elif n1 == 0 and n2 > 0:
        kk = 2
      elif n1 > 0 and n2 >0:
        kk = 4 
      # cosine-cosine term
      f = lambda x,y: np.cos(n1*x)*np.cos(n2*y)*kappa(x,y)
      alpha,err=dblquad(f,0.0,2*np.pi,lambda x: 0.0,lambda x: 2*np.pi,epsabs=1e-9)
      CC[n1,n2] = alpha*kk
      #print(f"cos({n1}),cos({n2}): {CC[n1,n2]},  err ={err}")

      # cosine-sine term
      if n2 != 0: #skip sine zero mode
        f = lambda x,y: np.cos(n1*x)*np.sin(n2*y)*kappa(x,y)
        beta,err = dblquad(f,0,2*np.pi,lambda x: 0.0,lambda x: 2*np.pi,epsabs=1e-9)
        CS[n1,n2-1] = beta*kk
        #print(f"cos({n1}),sin({n2}): {CS[n1,n2-1]}, err ={err}")

      # sine-sine term
      if n1 != 0 and n2 != 0:  # skip 0 modes
        f = lambda x,y: np.sin(n1*x)*np.sin(n2*y)*kappa(x,y)
        gamma,err = dblquad(f,0,2*np.pi,lambda x: 0.0,lambda x: 2*np.pi,epsabs=1e-9)
        SS[n1-1,n2-1] = gamma*kk
        #print(f"sin({n1}),sin({n2}): {SS[n1-1,n2-1]}, err ={err}")


  # build F
  F[:n_modes,:n_modes] = CC
  F[:n_modes,n_modes:] = CS
  F[n_modes:,:n_modes] = CS.T
  F[n_modes:,n_modes:] = SS


  # normalize
  F = F/normalization

  return F

def generate_random_coils_with_gp(n_coils,n_time,h=1.0,p=2*np.pi,l=1.0):
  """
  Use the gp to generate random coils.

  n_coils: number of coils to generate
  n_time: number of points to discretize [0,2pi]
  h,p,l: hyperparameters for kernel
  """
  # kernel
  kernel = lambda x,y: h*np.exp((-2/l**2)*np.sin(np.pi*np.abs(x-y)/p)**2)

  # kernel matrix
  K = np.zeros((n_time,n_time))
  for i in range(n_time):
    K[i,:] = kernel(time[i],time)
  K +=1e-8*np.eye(n_time)
  Q = np.linalg.cholesky(K) 

  # sample coils
  C = np.zeros((n_coils,3,n_time))
  for i in range(n_coils):
    Z = (Q @ np.random.randn(n_time,3) ).T
    C[i] = deepcopy(Z)

  return C

def generate_perturbed_coils_with_fourier(n_coils,n_time,n_modes,Sigma):
  """
  Generate random coils via the random fourier series, i.e the fourier
  representation of the gp.

  n_coils: number of coils to generate
  n_time: number of points to discretize [0,2pi]
  Sigma: variance vector for the 2*n_modes -1 fourier coefficients of [x,y,z] 
  """
  dim     = 3   

  # build r(t); columns are time slices
  r     = np.zeros((2*n_modes-1,n_time))
  r[0] += 1.0
  for n in range(1,n_modes):
    r[n] = np.cos(n*time)
    r[n_modes + n-1] =   np.sin(n*time)

  # sample coils
  C = np.zeros((n_coils,dim,n_time))
  # cholesky of covariance
  Q = np.linalg.cholesky(Sigma)
  for i in range(n_coils):
    # pull from the distribution
    Z = Q @ np.random.randn(len(Sigma))  
    # reshape Z to look like A
    Z = np.reshape(Z,(dim,2*n_modes-1))
    # compute the coil set
    C[i] = deepcopy(Z @ r)

  return C
def plot_kernel(kernel,n_mesh=500):
  # plot kernel
  x_mesh = np.linspace(0,2*np.pi,n_mesh)
  plt.plot(x_mesh,[kernel(x_mesh[i],0.0) for i in range(n_mesh)])
  plt.title("kernel")
  plt.show()
  return None

def plot_fourier_kernel(kernel,F,n_time=100):
  n_modes =int((len(F)+1)/2)

  # build r(t); 
  r     = np.zeros(n_time)
  time  = np.linspace(0,2*np.pi,n_time)
  for ii in range(n_time):
    for n1 in range(0,n_modes):
      for n2 in range(0,n_modes):
        r[ii] += F[n1,n2]*np.cos(n1*time[ii])*np.cos(n2*0)
        if n2 !=0:
          r[ii] += F[n1,n_modes+n2-1]*np.cos(n1*time[ii])*np.sin(n2*0)
        if n1 !=0:
          r[ii] += F[n_modes+n1-1,n2]*np.sin(n1*time[ii])*np.cos(n2*0)
        if n1 !=0 and n2!=0:
          r[ii] += F[n_modes+n1-1,n_modes+n2-1]*np.sin(n1*time[ii])*np.sin(n2*0)
  plt.plot(time,r,label='fourier kernel')
  plt.plot(time,[kernel(time[i],0.0) for i in range(n_time)],label='kernel')
  plt.title("fourier kernel")
  plt.legend()
  plt.show()

def compute_fourier_coefficient_variance(n_modes,perturbation_size, lengthscale=0.5,drop_pickle=True):
  """
  Compute the variances for the fourier coefficients description of a single coil. 
  Return a 1D-array of variances for the cosine fourier coefficents and a 1D-array
  of variances for the sine coefficents. 

  input:
  n_modes: int,>0 number of fourier modes
  perturbation_size: float>0, desired size of perturbations (in meters)
  l: float >0, lengthscale for GP perturbations
  drop_pickle: bool, drop a pickle file with the result or not

  return C,S
  C: 1D-array, cosine fourier coefficient variances
  S: 1D-array, sine fourier coefficient variances
  """
  dim = 3

  # hyperparameters 
  h       = perturbation_size**2/dim
  l       = lengthscale
  p       = 2*np.pi
  kernel  = lambda x,y: h*np.exp(-2*np.sin(np.pi*np.sqrt((x-y)**2)/p)**2/l**2)
  # compute the variance for the fourier representation
  C,S       = fourier_coeffs(kernel,n_modes)

  if drop_pickle:
    # drop the pickle file
    outfilename = "fourier_coefficient_variances.pickle"
    outdata                     = {}
    outdata['cosine_variances'] = C
    outdata['sine_variances']   = S
    outdata['n_modes']          = n_modes
    outdata['h']                = h
    outdata['p']                = p
    outdata['l']                = l
    outdata['comment'] ="generated from Sine Squared Exponenetial kernel with hypers h,p,l"
    with open(outfilename,"wb") as of:
      pickle.dump(outdata,of)

  return C,S


def compute_fourier_coefficient_variance_for_FOCUS(n_coils,n_modes,perturbation_size, lengthscale,drop_pickle=True):
  """
  Compute the variances for the fourier coefficients of the x(t),y(t),z(t)
  description of n_coils coils. We assume that x(t), y(t) and z(t) are all identically distributed and
  that all of the coils are identically distributed. See the overleaf for more info


  input:
  n_coils: int>0, number of coils
  n_modes: int>0, number of fourier modes
  perturbation_size: float>0, desired size of perturbations (in meters)
  drop_pickle: bool, drop a pickle file with the result or not

  return:
  1d array of variances of the x,y, and z components of the coils. The array is organized
  first by coils, i.e. coil 1 information before coil 2 information. For each coil we organize
  the fourier coefficents by first the x cosines, then x sines, y cosines, y sines, z cosines, z sines.
  The zeroth order sine coefficients are omitted because sin(0*t) = 0 always.
  """
  # compute the variances for 1 coil's x(t)
  C,S = compute_fourier_coefficient_variance(n_modes,perturbation_size, lengthscale,drop_pickle)
  # fourier coeffs for one coil
  Z   = np.hstack((C,S,C,S,C,S))
  # repeat n_coils times
  var = np.zeros(0)
  for i in range(n_coils):
    var = np.hstack((var,Z))
  return var

def compute_fourier_coefficient_covariance_for_FOCUS(n_coils,n_modes,perturbation_size, lengthscale,drop_pickle=True):
  """
  compute the covariance matrix for the fourier coeffs and return it formatted
  for focus.
  """
  # compute the variances for 1 coil's x(t)
  dim = 3

  # hyperparameters 
  h       = perturbation_size**2/dim
  l       = lengthscale
  p       = 2*np.pi
  kernel  = lambda x,y: h*np.exp(-2*np.sin(np.pi*np.sqrt((x-y)**2)/p)**2/l**2)
  # compute the variance for the fourier representation
  F       = all_fourier_coeffs(kernel,n_modes)

  # fourier covariance for one coil
  Z = np.block([[F,np.zeros_like(F),np.zeros_like(F)],
    [np.zeros_like(F),F,np.zeros_like(F)],[np.zeros_like(F),np.zeros_like(F),F]])

  # repeat n_coils times
  cov = np.zeros(n_coils*np.array(np.shape(Z)))
  rr,cc = np.shape(Z)
  for i in range(n_coils):
    cov[i*rr:(i+1)*rr, i*cc:(i+1)*cc] = Z

  if drop_pickle:
    # drop the pickle file
    outfilename = f"fourier_coefficient_covariances_{n_coils}_coils_{n_modes}_modes.pickle"
    outdata                     = {}
    outdata['single_coil_fourier_coeffs']   = F
    outdata['all_coils_fourier_coeffs']     = cov
    outdata['n_modes']          = n_modes
    outdata['n_coils']          = n_coils
    outdata['h']                = h
    outdata['p']                = p
    outdata['l']                = l
    outdata['comment'] ="generated from Sine Squared Exponenetial kernel with hypers h,p,l"
    with open(outfilename,"wb") as of:
      pickle.dump(outdata,of)

  return cov







if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D

  # problem dimension
  dim    = 3
  # desired perturbation size
  perturbation_size   = 0.1

  # Make a circular coil
  n_time = 2000
  time   = np.linspace(0,2*np.pi,n_time)
  X      = np.array([np.cos(time),0.0*time,np.sin(time)])

  # generate coils with gp
  n_coils = 1
  # hyperparameters 
  h       = perturbation_size**2/dim
  l       = 0.5 # determines smallest perturbation width
  p       = 2*np.pi
  C       = generate_random_coils_with_gp(n_coils,n_time,h=h,p=p,l=l)

  # plot the coil
  fig = plt.figure()
  ax  = plt.axes(projection='3d')
  ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1);  
  ax.plot3D(X[0], X[1], X[2], linewidth=3,color='blue')
  for i in range(np.shape(C)[0]):
    Z = X + C[i]
    ax.plot3D(Z[0], Z[1], Z[2], linewidth=3,color='green')
  plt.title("Coils generated with samples from GP")
  plt.show()

  # compute the statistics for the GP perturbations
  print("Statistics on GP sampled coils")
  n_coils=100
  C = generate_random_coils_with_gp(n_coils,n_time,h=h,p=p,l=l)
  mean = np.mean(C)
  print(f"mean = {mean}")
  var = np.var(C)
  print(f"var  = {var}")
  # mean length perturbation
  mlp =np.mean(np.array([np.linalg.norm(C[i],axis=0) for i in range(n_coils)]))
  print(f"mean length perturbation = {mlp}")


  # compute the covariance for the fourier representation
  n_modes = 5
  kernel = lambda x,y: h*np.exp((-2/l**2)*np.sin(np.pi*np.abs(x-y)/p)**2)
  F = all_fourier_coeffs(kernel,n_modes)
  Sigma = np.block([[F,np.zeros_like(F),np.zeros_like(F)],
    [np.zeros_like(F),F,np.zeros_like(F)],[np.zeros_like(F),np.zeros_like(F),F]])

  # plot the fourier rep of the kernel
  plot_fourier_kernel(kernel,F,n_time=500)
  cov = compute_fourier_coefficient_covariance_for_FOCUS(5,3,0.1, 0.5,drop_pickle=True)
  # generate random coils
  n_coils = 1
  C = generate_perturbed_coils_with_fourier(n_coils,n_time,n_modes,Sigma)
  # plot the coil
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1);  
  ax.plot3D(X[0], X[1], X[2], linewidth=3,color='blue')
  for i in range(np.shape(C)[0]):
    Z = X + C[i]
    ax.plot3D(Z[0], Z[1], Z[2], linewidth=3,color='green')
  plt.title("Coils generated with random fourier coefficients")
  plt.show()

  # compute the statistics for the fourier sampled coils
  print("Statistics on Fourier sampled coils")
  n_coils=10000
  C = generate_perturbed_coils_with_fourier(n_coils,n_time,n_modes,Sigma)
  mean = np.mean(C)
  print(f"mean = {mean}")
  var = np.var(C)
  print(f"var  = {var}")
  # mean length perturbation
  mlp =np.mean(np.array([np.linalg.norm(C[i],axis=0) for i in range(n_coils)]))
  print(f"mean length perturbation = {mlp}")


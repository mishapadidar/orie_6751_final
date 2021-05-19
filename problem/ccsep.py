from sklearn.metrics.pairwise import euclidean_distances
from torch.autograd.gradcheck import zero_gradients
#from gpytorch.kernels import covar_dist
import torch as torch
import math

def smoothmin(x,alpha=-1.0):
  """
  Smooth version of the min function.
  x: vector of entries to take smooth min over
  alpha: float <0, as alpha decreases the smooth min converges
         to the min.
  """
  assert alpha < 0, "alpha must be negative for a minimum approximation"
  reuse = torch.exp(alpha*x)
  num = torch.sum(x*reuse)
  ret = num/torch.sum(reuse)
  return ret

def logSumExp(x, alpha = -1.0):
  """
  Log Sum Exponential Smooth minima function
  x: vector of entries to take smooth min over
  alpha: float <0, as alpha decreases the smooth min converges
         to the min.
  """
  assert alpha < 0, "alpha must be negative for a minimum approximation"
  #dydx  = 2*math.pi/n_seg

  # stabilize with the logSumExp Trick!
  shift = torch.min(x)
  ret = shift + torch.log(torch.sum(torch.exp(alpha*(x-shift))))/alpha
  return ret

def coil_coil_sep_grad(x,n_coils,**kwargs):
  kwargs['return_np'] = False # ensure f returns tensor
  if not torch.is_tensor(x):
    x = torch.tensor(x,requires_grad=True)
  y = coil_coil_sep(x,n_coils,**kwargs)
  jac = torch.zeros(len(y),len(x))
  for i in range(len(y)):
    zero_gradients(x)
    y[i].backward(retain_graph=True)
    jac[i] = x.grad
  return jac.detach().numpy()

def coil_coil_sep(x,n_coils,n_seg=100,alpha=-1.0,use_smoothmin=True,return_np=False,**kwargs):
  """Compute the coil-coil separation objective. 
  Computes the minimum squared distance between coils approximately. 
  itorch.t: x, focus variable of fourier coefficients. We assume
        that current is not a variable
  n_coils: int, number of coils
  n_seg: number of points to discretize a coil into
  alpha: float <0, as alpha decreases the smooth min converges
         to the min.

  return: 1D-array (n_coils-1,); 
          array of min distances between adjacent coils. min distance
          is computed using smooth-min.
  """
  if not torch.is_tensor(x):
    x = torch.tensor(x,requires_grad=True)

  # dim
  dim = len(x)
  D3  = 3
  # number of modes not including zeroeth mode
  n_modes = int((dim/n_coils/D3 - 1)/2) 
  # number of fourier modes per direction
  n_var_per_dir = 2*n_modes+1
  # one coil has 3*(2*n_modes +1) variables
  n_var_per_coil = 3*n_var_per_dir
  # discretize the coil
  segs = torch.linspace(0,2*math.pi,n_seg)
  # fourier cosine and sine vector
  r     = torch.zeros(n_var_per_dir,n_seg) 
  r[0] += 1.0
  for n in range(1,n_modes+1):
    r[n] = torch.cos(n*segs)
    r[n_modes + n] =   torch.sin(n*segs)
  r = r.double()

  # reshape 
  X = torch.reshape(x,(n_coils,D3,n_var_per_dir)) 

  # compute the coil set (n_coils,dim,n_seg)
  C = torch.matmul(X,r)

  # ret = torch.zeros(n_coils-1) # function values
  ret = torch.zeros(int(n_coils*(n_coils-1)/2)) # function values
  idx = 0
  for ii in range(n_coils-1):
    for jj in range(n_coils-1):
      if jj < ii:
        continue
      # compute pairwise distances between all points on the coils
      z  = C[ii].T @ C[ii+1] # cross term
      nx = torch.sum(C[ii]**2,dim=0).reshape(n_seg,1)
      ny = torch.sum(C[ii+1]**2,dim=0).reshape(1,n_seg)
      dist = nx - 2*z + ny
      if use_smoothmin:
        # compute smooth-min
        ret[idx] = logSumExp(dist, alpha,**kwargs)
      else:
        # use regular min
        ret[idx] = torch.min(dist)
      idx += 1

  if return_np is False:
    return ret
  else:
    return ret.detach().numpy()

def fdiff_jacobian(f,x0,h=1e-6,**kwargs):
  """
  Use central differences on the gradient to compute
  the hessian
  """
  h2   = h/2.0
  dim  = len(x0)
  Ep   = x0 + h2*np.eye(dim)
  Fp   = np.array([f(e,**kwargs) for e in Ep])
  Em   = x0 - h2*np.eye(dim)
  Fm   = np.array([f(e,**kwargs) for e in Em])
  jac = (Fp - Fm)/(h)
  return jac.T

if __name__ == "__main__":
  import pickle
  d = pickle.load(open("../experiments/output/baseline_20210519122852.pickle","rb"))
  x = d['xopt']
  n_coils = 3
  kwargs = {}
  kwargs['n_coils'] = n_coils
  kwargs['n_seg']   = 100
  kwargs['alpha']   = -1000
  # test function
  sep = coil_coil_sep(x,return_np=True,**kwargs)
  print(sep)

  # test finite difference
  from time import time
  import numpy as np
  def ff(x,**kwargs):
    return coil_coil_sep(x,return_np=True,**kwargs)
  t0 = time()
  diffjac     = fdiff_jacobian(ff,x,h=1e-4,**kwargs)
  print('finite difference time: ',time() - t0)
  print(diffjac)

  # test autodiff
  t0 = time()
  autojac = coil_coil_sep_grad(x,**kwargs)
  print("autodiff time: ",time() - t0)
  print(autojac)

  # compare
  print("")
  print(diffjac-autojac)


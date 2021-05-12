import numpy as np
import sys
from copy import copy
from time import time

def adam(grad, x0,max_iter=100,batch_size=1,eta=1.0,gamma=1.0,beta1=0.9,beta2=0.9,eps=1e-8,func=None,verbose=False):
  """
  ADAM with decreasing learning rate. Optimizes 
  A risk Neutral Objective E[ f(x+u)] by using a Monte Carlo
  Gradient esimate. 
  \\nabla E[f(x+u)] = E[\\nabla f(x+u)] 
  \\approx \\frac{1}{N}sum_{i=1}^N \\nabla f(x+u_i)   

  grad: function handle for a stochastic gradient
  x0: initial guess
  max_iter: maximum number of iterations
  batch_size: number of points used to approximate gradient
  eta: initial learning rate 
  eta: float > 0
  gamma: learning rate growth parameter; 
         large values shrink step size faster
  gamma: float > 0
  beta1: momentum parameter
  beta1: float in (0,1)
  beta2: momentum parameter; typically set equal to beta1
  beta2: float in (0,1)
  eps: ADAM epsilon for conditioning
  eps: small float > 0
  func: objective function handle

  return:
  x_k: 1D-array, the last iterate
  X: 2D-array, all points evaluated
  """
  if func is not None:
    f0 = func(x0)

  if verbose:
    print("")
    print("Adam Optimization")
    print(f"max_iter   = {max_iter}")
    print(f"batch_size = {batch_size}")
    print(f"eta        = {eta}")
    print(f"gamma      = {gamma}")
    print(f"beta1      = {beta1}")
    print(f"beta2      = {beta2}")
    print(f"eps        = {eps}")
    if func is not None:
      print(f"f(x0)      = {f0}")
    print("")
    sys.stdout.flush()

  # inital guess
  x_k  = x0;

  # first and second moments
  v_km1 = 0*x0;
  m_km1 = 0*x0;

  # powers of beta
  beta1_k = beta1
  beta2_k = beta2

  # dimension
  dim = len(x0)
  
  # storage
  X    = np.zeros((max_iter+1,dim));
  X[0] = copy(x0)

  # start timer
  t0 = time()
  
  # stop after number of iterations
  for kk in range(1,max_iter+1):

    # print stuff
    if verbose and kk>1:
      if func is not None:
        print(f"{kk}/{max_iter};f(x) = {f_kp1 :.10f}; {(time()-t0)/(kk-1)}sec per iteration")
      else:
        print(f"{kk}/{max_iter}; {(time()-t0)/(kk-1)}sec per iteration")
      sys.stdout.flush()
    
    # compute the gradient
    g_k = np.mean(np.array([grad(x_k) for ii in range(batch_size)]),axis=0)

    # compute moments
    m_k = beta1*m_km1 + (1-beta1)*g_k;
    v_k = beta2*v_km1 + (1-beta2)*(g_k**2);
    
    # shrink the moments
    mhat = m_k/(1-beta1_k);
    vhat = v_k/(1-beta2_k);

    # compute step using decreasing step size
    #x_kp1 = x_k -(eta/kk)*(mhat/(np.sqrt(vhat)+eps));
    x_kp1 = x_k -(eta/(1.0 + gamma*kk))*(mhat/(np.sqrt(vhat)+eps));

    if func is not None:
      # compute function value
      f_kp1 = func(x_kp1)
    
    # reset for next iteration
    x_k   = copy(x_kp1)
    m_km1 = copy(m_k)
    v_km1 = copy(v_k)
    beta1_k = beta1_k*beta1
    beta2_k = beta2_k*beta2
    
    # save w_k
    X[kk] = copy(x_k)

  if verbose:
    print("Done!")
    sys.stdout.flush()

  return x_kp1,X


if __name__ == '__main__':
  import matplotlib.pyplot as plt

  dim = 2
  x0  = 10*np.random.randn(dim)
  # function
  f      = lambda x: x @ x
  # gradient
  grad   = lambda x: 2*x 
  # hessian
  H      = 2.0*np.eye(dim)
  # covariance
  C   = 2.0*np.eye(dim)
  # risk function
  rn      = lambda x: x @ x + 0.5*np.trace(H @ C)
  # optimize
  x_best,X = adam(grad, x0,C,max_iter=1000,batch_size=5,eta=0.1,gamma=1.0,beta1=0.9,beta2=0.9,eps=1e-4,func=f,verbose=True)
  fX = [f(x) for x in X]
  rnX = [rn(x) for x in X]
  print(x_best)

  plt.plot(fX, linewidth=3, label='f')
  plt.plot(rnX, linewidth=3, label='risk neutral')
  plt.legend()
  plt.show()

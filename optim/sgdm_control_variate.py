import numpy as np
import sys
from copy import copy
from time import time

def sgdm_cv(grad, x0,C,max_iter=100,batch_size=2,eta=1.0,gamma=1.0,theta=0.9,func=None,verbose=False):
  """
  Momentum Stochastic Gradient with decreasing learning rate. Optimizes 
  A risk Neutral Objective E[ f(x+u)] by using a Monte Carlo
  Gradient esimate. 
  \\nabla E[f(x+u)] = E[\\nabla f(x+u)] 
  \\approx \\frac{1}{N}sum_{i=1}^N \\nabla f(x+u_i)   

  grad: returns gradient array
  x0: initial guess
  C: covariance matrix for perturbations
  max_iter: maximum number of iterations
  batch_size: number of points used to approximate gradient
  eta: initial learning rate;
       large values start iteration with a bigger step
  eta: float > 0
  gamma: learning rate growth parameter; 
         large values shrink step size faster
  gamma: float > 0
  theta: momentum parameter
  theta: float in (0,1)
  func: objective function handle


  return:
  x_k: 1D-array, the last iterate
  X: 2D-array, all points evaluated
  """
  if func is not None:
    f0 = func(x0)

  if verbose:
    print("")
    print("SGD Optimization")
    print(f"max_iter   = {max_iter}")
    print(f"batch_size = {batch_size}")
    print(f"eta        = {eta}")
    print(f"gamma      = {gamma}")
    print(f"theta      = {theta}")
    if func is not None:
      print(f"f(x0)      = {f0}")
    print("")
    sys.stdout.flush()

  # inital guess
  x_k  = x0;
  v_km1 = 0*x0;

  # dimension
  dim = len(x0)
  
  # storage
  X    = np.zeros((max_iter+1,dim));
  X[0] = copy(x0)

  # cholesky of covariance
  Q = np.linalg.cholesky(C)

  # BFGS Hessian
  H_k = np.eye(dim)

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
    
    # generate Gaussian Correlated perturbations
    x_batch = x_k + np.random.randn(batch_size,dim) @ Q.T

    # compute gradient of f
    nablaf = np.array([grad(x) for x in x_batch])

    # finite difference hessian
    #fk = func(x_k)
    #E  = np.eye(dim)
    #h  = 1e-6
    #H_k  = np.diag([(func(x_k+h*E[i])/h - 2*fk/h + func(x_k-h*E[i])/h)/h for i in range(dim)])

    # control variate
    mu_ghat  = grad(x_k) # mean of control variate
    ghat     = mu_ghat  + (x_batch - x_k) @ H_k.T
    
    # compute A
    cov_ghat    = H_k @ C @ H_k.T # covariance cov(ghat)
    cov_ghat_nf = np.cov(nablaf.T,(ghat-mu_ghat).T)
    cov_ghat_nf = cov_ghat_nf[dim:,:dim]

    #A  = -np.linalg.solve(H_k.T,np.linalg.solve(Q.T,np.linalg.solve(Q,np.linalg.solve(H_k,cov_ghat_nf))))
    # A  = -np.trace(cov_ghat_nf)/np.trace(cov_ghat)
    A = -np.diag(cov_ghat_nf)/np.diag(cov_ghat)

    # take sample average to get gradient estimator
    g_k = np.mean(nablaf + A*(ghat - mu_ghat), axis=0) 

    if verbose:
      print(f"    Variance Reduction:\
      {np.sum(np.var(nablaf + A*(ghat - mu_ghat),axis=0))/np.sum(np.var(nablaf,axis=0))}")
      sys.stdout.flush()

    # compute the step direction
    v_k = theta*v_km1 + g_k;

    # compute step using decreasing step size
    s_k   = -(eta/(1.0 + gamma*kk))*v_k; # 1/k step size decrease
    # s_k   = -(eta/(1.0 + gamma*np.sqrt(kk)))*v_k; # 1/sqrt(k) step size decrease
    x_kp1 = x_k + s_k

    if func is not None:
      # compute function value
      f_kp1 = func(x_kp1)

    # update BFGS Hessian
    y_k = grad(x_kp1) - mu_ghat
    r_k   = H_k @ s_k
    H_kp1 = H_k + np.outer(y_k,y_k)/(y_k @ s_k)\
            - np.outer(r_k, r_k)/(s_k@ r_k)
    H_k   = copy(H_kp1)
    
    # reset for next iteration
    x_k   = copy(x_kp1)
    v_km1 = copy(v_k);
    
    # save w_k
    X[kk] = copy(x_k)

  if verbose:
    print("Done!")
    sys.stdout.flush()

  return x_kp1,X


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  np.random.seed(0)
  dim = 3
  x0  = 10*np.random.randn(dim)
  # function
  f      = lambda x: x @ x
  # gradient
  grad   = lambda x: 2*x 
  # hessian
  H      = 2.0*np.eye(dim)
  # covariance
  C   = 2.0*np.eye(dim)
  
  # optimize
  x_best,X = sgdm_cv(grad, x0,C,max_iter=100,batch_size=100,\
             eta=0.01,gamma=1.0,theta=0.9,func =f, verbose=True)
  # risk function
  rn      = lambda x: x @ x + 0.5*np.trace(H @ C)

  fX = [f(x) for x in X]
  rnX = [rn(x) for x in X]
  print(x_best)

  plt.plot(fX, linewidth=3, label='f')
  plt.plot(rnX, linewidth=3, label='risk neutral')
  plt.legend()
  plt.show()

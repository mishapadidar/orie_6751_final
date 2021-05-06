import numpy as np
import sys
from copy import copy
from time import time

def adam_cv(grad, x0,C,grad_reg=None,max_iter=100,batch_size=1,eta=1.0,gamma=1.0,beta1=0.9,beta2=0.9,eps=1e-8,func=None,verbose=False):
  """
  ADAM with decreasing learning rate and control variates. Optimizes 
  A risk Neutral Objective with regularization E[ f(x+u)] + f_R(x) 
  by using a Monte Carlo Gradient esimate

  grad: gradient function handle
  x0: initial guess
  C: covariance matrix for perturbations
  grad_reg: gradient function handle for regularization functions r_R
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
    print("SGD Optimization")
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

    # control variate
    mu_ghat  = grad(x_k) # mean of control variate
    ghat     = mu_ghat  + (x_batch - x_k) @ H_k.T
    
    # compute A
    cov_ghat    = H_k @ C @ H_k.T # covariance cov(ghat)
    cov_ghat_nf = np.cov(nablaf.T,(ghat-mu_ghat).T)
    cov_ghat_nf = cov_ghat_nf[dim:,:dim]      
    A = -np.diag(cov_ghat_nf)/np.diag(cov_ghat)

    # take sample average to get gradient estimator
    g_k = np.mean(nablaf + A*(ghat - mu_ghat), axis=0) 

    # add in regularization gradients
    if grad_reg is not None:
      g_k = g_k + grad_reg(x_k)

    if verbose:
      print(f"    Variance Reduction:\
      {np.sum(np.var(nablaf + A*(ghat - mu_ghat),axis=0))/np.sum(np.var(nablaf,axis=0))}")
      sys.stdout.flush()

    # compute moments
    m_k = beta1*m_km1 + (1-beta1)*g_k;
    v_k = beta2*v_km1 + (1-beta2)*(g_k**2);
    
    # shrink the moments
    mhat = m_k/(1-beta1_k);
    vhat = v_k/(1-beta2_k);

    # compute step using decreasing step size
    s_k   = -(eta/(1.0 + gamma*kk))*(mhat/(np.sqrt(vhat)+eps));
    x_kp1 = x_k +s_k

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

  np.random.seed(0)
  dim = 2
  x0  = 10*np.random.randn(dim)
  # hessian
  #H      = 2.0*np.eye(dim)
  H      = np.random.randn(dim,dim)
  H      = H @ H.T
  # function
  f      = lambda x: 0.5*x @ H @ x
  # gradient
  grad   = lambda x: H @ x
  # l2 regularization
  lam   = 0.1
  f_reg = lambda x: lam*x @ x
  grad_reg = lambda x: 2*lam*x
  # covariance
  C   = 5.0*np.eye(dim)
  # risk function
  rn      = lambda x: x @ x + 0.5*np.trace(H @ C)
  from adam import adam
  y_best,Y = adam(grad, x0,C,max_iter=1000,batch_size=10,eta=2,beta1=0.9,beta2=0.9,eps=1e-7,func=f,verbose=True)
  # optimize
  x_best,X = adam_cv(grad, x0,C,grad_reg,max_iter=1000,batch_size=10,eta=2,beta1=0.9,beta2=0.9,eps=1e-7,func=f,verbose=True)


  fX = [f(x) for x in X]
  fY = [f(x) for x in Y]

  rnX = [rn(x) for x in X]
  print(x_best)

  plt.plot(fX, linewidth=3, label='adam cv')
  plt.plot(fY, linewidth=3, label='adam')
  # plt.plot(rnX, linewidth=3, label='risk neutral')
  plt.legend()
  plt.show()

import numpy as np
from sys import stdout

def BFGS(Loss,grad,x0,mu0 = 1e-6,max_iter=1000,gtol=1e-3,c_armijo=1e-9,mu_min=1e-14,mu_max=1e-6,verbose=True):
  """BFGS with Armijo Backtracking Linesearch.
  We use lazy matrix inversion rather than Woodbury which could be expensive
  in high dim.
  mu0: initial step size
  c_armijo: constant for determining sufficient decrease
  """
  dim = len(x0)

  # inital guess
  x_k = np.copy(x0)
  # initialize step size
  mu_k   = mu0
  # compute gradient
  g_k    = grad(x_k)
  # compute function value
  f_k    = Loss(x_k)
  # BFGS Hessian
  H_k = np.eye(dim)

  # storage
  X = np.zeros((1,dim))
  X[0] = np.copy(x0)
  fX = np.zeros(1)
  fX[0] = f_k

  # stop when gradient is flat (within tolerance)
  nn = 0
  while np.linalg.norm(g_k) > gtol and nn < max_iter:
    if verbose and nn%1 == 0:
      print(f"{nn})",f_k)
      stdout.flush()
    # double the step size to counter backtracking
    mu_k = min(2*mu_k,mu_max);
    
    # compute step
    d_k   = np.linalg.solve(H_k,g_k)
    x_kp1 = x_k - mu_k*d_k;
    f_kp1 = Loss(x_kp1);
    
    # gradient norm
    g_norm = np.linalg.norm(d_k)
    
    # backtracking to find step size
    while f_kp1 >= f_k - c_armijo*g_norm:
      # half our step size
      mu_k = mu_k /2 ;
      # take step
      x_kp1 = x_k -mu_k*d_k;
      # f_kp1
      f_kp1 = Loss(x_kp1);

      # break if mu is too small
      if mu_k <= mu_min:
        if verbose:
          print(f'Exiting BFGS: step size reached {mu_min}.')
          stdout.flush()
        return x_k, X, fX

    # update BFGS Hessian
    g_kp1 = grad(x_kp1)
    s_k   = x_kp1 - x_k
    y_k   = g_kp1 - g_k
    r_k   = H_k @ s_k
    H_kp1 = H_k + np.outer(y_k,y_k)/(y_k @ s_k)\
            - np.outer(r_k, r_k)/(s_k@ r_k)

    # reset for next iteration
    H_k   = np.copy(H_kp1)
    x_k   = np.copy(x_kp1)
    f_k   = f_kp1;
    
    # compute gradient
    g_k  = np.copy(g_kp1)

    # update iteration counter
    nn += 1
    X = np.copy(np.vstack((X,x_k)))
    fX = np.append(fX,f_k)

  return x_k,X,fX


if __name__ == '__main__':
  np.random.seed(0)
  f = lambda x: x @ x
  g = lambda x: 2*x
  dim = 2
  x0 = 10*np.random.randn(dim)
  xopt,X,fX = BFGS(f,g,x0,max_iter=200,gtol=1e-7)
  print(xopt)
  print(X)
  print(fX)

import numpy as np
from sys import stdout

def gradient_descent(Loss,grad,x0,mu0 = 1e-3,max_iter=1000,gtol=1e-3,c_armijo=1e-3,mu_min=1e-16,mu_max=1e6,verbose=True):
  # inital guess
  x_k = np.copy(x0)
  # initialize step size
  mu_k   = mu0
  # compute gradient
  g_k    = grad(x_k)
  # compute function value
  f_k    = Loss(x_k)

  # storage
  dim = len(x0)
  X = np.zeros((1,dim))
  X[0] = np.copy(x0)
  fX = np.zeros(1)
  fX[0] = f_k

  # stop when gradient is flat (within tolerance)
  nn = 0
  while np.linalg.norm(g_k) > gtol and nn < max_iter:
    if verbose and nn%1 == 0:
      print(nn,f_k,np.linalg.norm(g_k))
      stdout.flush()
    # double the step size to counter backtracking
    mu_k = min(2*mu_k,mu_max)
    
    # compute step 
    x_kp1 = x_k -mu_k*g_k;
    f_kp1 = Loss(x_kp1);
    
    # gradient norm
    g_norm = np.linalg.norm(g_k)
    
    # backtracking to find step size
    while f_kp1 >= f_k - c_armijo*g_norm:
      # half our step size
      mu_k = mu_k /2 ;
      # take step
      x_kp1 = x_k -mu_k*g_k;
      # f_kp1
      f_kp1 = Loss(x_kp1);

      # break if mu is too small
      if mu_k <= mu_min:
        if verbose:
          print(f'Exiting GD: step size reached {mu_min}.')
        stdout.flush()
        return x_k,X,fX

    # reset for next iteration
    x_k   = np.copy(x_kp1)
    f_k   = f_kp1;
    
    # compute gradient
    g_k  = grad(x_k);

    # update iteration counter
    nn += 1
    X = np.copy(np.vstack((X,x_k)))
    fX = np.append(fX,f_k)

  return x_k,X,fX


if __name__ == '__main__':
  f = lambda x: x @ x
  g = lambda x: 2*x
  dim = 2
  x0 = 10*np.random.randn(dim)
  xopt,X,fX = gradient_descent(f,g,x0,max_iter=200,gtol=1e-7,c_armijo = 0)
  print(xopt)
  print(X)


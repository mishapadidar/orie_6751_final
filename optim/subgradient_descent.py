import numpy as np
from sys import stdout

def subgradient_descent(Loss,grad,x0,mu0 = 1e-3,max_iter=1000,gtol=1e-3,verbose=True,lr_sched=None):
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
      print(f"{nn})",f_k)
      stdout.flush()

    # compute step 
    x_kp1 = x_k -mu_k*g_k;
    f_kp1 = Loss(x_kp1);
    
    # reset for next iteration
    x_k   = np.copy(x_kp1)
    f_k   = f_kp1;
    
    # compute gradient
    g_k  = grad(x_k);

    # update iteration counter
    nn += 1
    X = np.copy(np.vstack((X,x_k)))
    fX = np.append(fX,f_k)

    # reduce step size
    if lr_sched is not None:
      mu_k = mu0*lr_sched(nn)

  return x_k,X,fX


if __name__ == '__main__':
  f = lambda x: x @ x
  g = lambda x: 2*x
  dim = 2
  x0 = 10*np.random.randn(dim)
  xopt,X,fX = subgradient_descent(f,g,x0,max_iter=200,gtol=1e-7)
  print(xopt)
  print(X)


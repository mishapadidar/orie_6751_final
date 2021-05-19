import numpy as np
import pickle
from scipy.optimize import minimize
from datetime import datetime
import time
import sys
sys.path.append('../utils/')
sys.path.append('../problem/')
from get_input_settings import get_input_settings
from ccsep import coil_coil_sep_grad,coil_coil_sep
sys.path.append('../../FOCUS/python/')
from focuspy import FOCUSpy
import focus
from mpi4py import MPI

# MPI_INIT
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# set the random seed based off time
now     = datetime.now()
seed    = int("%d%.2d%.2d%.2d%.2d"%(now.month,now.day,now.hour,now.minute,now.second))
barcode = "%d%.2d%.2d%.2d%.2d%.2d"%(now.year,now.month,now.day,now.hour,now.minute,now.second)
np.random.seed(seed)

# run FOCUS
if rank==0:
    print("##### Begin FOCUS run with {:d} CPUs. #####".format(size))
    master = True
else:
    master = False
test = FOCUSpy(comm=comm, extension='w7x_jf', verbose=True)


# customize optimizers
test.prepare() 
x0 = focus.globals.xdof

# optimizer options
method = 'BFGS'
maxiter = 1000
gtol    = 1e-2
focus.globals.cg_maxiter = maxiter

if master:
    print('----------------------{:}-----------------------'.format(method))

focus.globals.iout = 0
test.time = time.time()
test.callback(x0)

# wrap coil sep functions
n_coils = 3
kwargs = {}
kwargs['n_coils']    = n_coils
kwargs['n_seg']      = 100 # coil discretization
kwargs['alpha']      = -100
kwargs['return_np']  = True
# constraint parameters
constraint_eps = 0.23**2 
lam  = 1e3 # initial lagrange multiplier
# number of iterative penalty solves
n_runs = 1

# coil-coil constriants
def ccsep(x,**kwargs):
  """
  constraint of form
  c(x)-eps >= 0 
  """
  return coil_coil_sep(x,**kwargs) - constraint_eps
def ccsep_grad(x,**kwargs):
  return coil_coil_sep_grad(x,**kwargs)

# new objective and gradient
def myObj(x):
  # penalty function
  F =  test.func(x) + lam*np.sum(np.minimum(coil_coil_sep(x,**kwargs) - constraint_eps,0.0)**2)
  #print('f(x) = ',F)
  #sys.stdout.flush()
  return F
def myGrad(x):
  # grad(f)
  f_g    = test.grad(x)
  # jac(min(constraint,0)**2)
  con_jac  = 2*np.minimum(coil_coil_sep(x,**kwargs) - constraint_eps,0.0) * coil_coil_sep_grad(x,**kwargs).T
  # sum them
  ret    = f_g + lam*np.sum(con_jac,axis=1)
  return ret

def callback(x):
  f0 = myObj(x)
  if master:
    print('f(x) = ',f0)
    sys.stdout.flush()

# optimize
for ii in range(n_runs):
  if ii >0 : 
    # increase lagrange multiplier
    lam = lam*10
  #res = minimize(test.func, x0, method=method, jac=test.grad, callback=test.callback, options={'gtol':gtol})
  res = minimize(myObj, x0, method=method, jac=myGrad, callback=callback, options={'gtol':gtol,'maxiter':maxiter})
  # get the results
  x_best    = res.x
  fX_best   = myObj(x_best) # make sure to reevaluate the function
  fvec_best = test.fvec(x_best, ['bnorm','ttlen','curv'])
  cX_best   = ccsep(x_best,**kwargs)
  if master:
    print(f"run {ii}/{n_runs} complete")
    print(f"fopt: {fX_best}")
    print(f"fopt: {fvec_best}")
    print(f"ccsep: {cX_best}")
  if np.all(cX_best >= 0.0):
    # finished... optima found
    break
  # prepare for next iteration
  x0 = np.copy(x_best)

test.callback(res.x)

# save the results
if master:
  d                      = {}
  d['xopt']            = x_best
  d['fXopt']           = fX_best
  d['method']            = method
  d['maxiter']           = maxiter
  d['seed']              = seed
  d['date']              = now
  d.update(get_input_settings(focus.globals))
  outfilename = f"./output/baseline_{barcode}.pickle"
  pickle.dump(d,open(outfilename,"wb"))
  print(f"Dumped {outfilename}")

sys.exit()


import numpy as np
import pickle
import sys
sys.path.append('../optim/')
from bfgs import BFGS
sys.path.append('../utils/')
from get_input_settings import get_input_settings
sys.path.append('../problem/')
from normal_covariances import compute_fourier_coefficient_covariance_for_FOCUS
from perturb_coil import perturb_coil
sys.path.append('../../FOCUS/python/')
from mpi4py import MPI
from focuspy import FOCUSpy
import focus

# load a pickle with the run params
args = sys.argv
param_filename = args[1]
run_params = pickle.load(open(param_filename,"rb"))
problem_num = run_params['problem_num']
seed = run_params['seed']
data_dir = run_params['data_dir']
data_filename = run_params['data_filename']
max_rotation = run_params['max_rotation']
max_shift = run_params['max_shift']
normal_pertubation_size= run_params['normal_pertubation_size']
gp_lengthscale = run_params['gp_lengthscale']
batch_size = run_params['batch_size']
max_iter = run_params['max_iter']
mu0 = run_params['mu0']
mu_min = run_params['mu_min']
mu0_max = run_params['mu_max']
gtol = run_params['gtol']
c_armijo = run_params['c_armijo']
alpha_pen = run_params['alpha_pen']
target_bnorm = run_params['target_bnorm']
delta = run_params['delta']

# MPI_INIT
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# set up FOCUS
if rank==0:
    print("##### Begin FOCUS run with {:d} CPUs. #####".format(size))
    master = True
    verbose = True
else:
    master = False
    verbose = False
test = FOCUSpy(comm=comm, extension='w7x_jf', verbose=True)
test.prepare() 
focus.globals.iout = 0

# set the seed
np.random.seed(seed)

# starting point
x0     = focus.globals.xdof

# perturbation covariance
n_coils = focus.globals.ncoils 
nfcoil  = focus.globals.nfcoil
n_modes = nfcoil + 1

# prepare the normal covariance matrix
cov_file = f"../problem/fourier_coefficient_covariances_{n_coils}_coils_\
{nfcoil}_modes_{normal_pertubation_size}_p_{gp_lengthscale}_l.pickle"
cov_data = pickle.load(open(cov_file,"rb"))
C = cov_data['all_coils_fourier_coeffs']

# set the objective
if problem_num == 0:
  """minimize_{x,t} CVaR_{1-delta}(bnorm)
  s.t. curvature constraint
  """
  # append the CVaR t parameter
  t0 = 1.0
  x0 = np.append(x0,t0)

  # CVaR objective
  def Obj(y):
    # split x and t
    t = y[-1]
    x = y[:-1]
    # evaluate the constraints
    f_pen = test.fvec(x,['curv'])
    # reset the seed so we get the same perturbations
    np.random.seed(seed)
    _X = [perturb_coil(x,C,max_rotation,max_shift,n_coils,nfcoil) for ii in range(batch_size)]
    # compute the CVaR
    _fX = np.array([test.fvec(_x,['bnorm']) for _x in _X])
    f_stoch = np.mean(np.maximum(_fX - t,0.0))/delta + t
    # make penalty obj
    F = f_stoch + f_pen
    return F

  def Grad(y):
    # split x and t
    t = y[-1]
    x = y[:-1]
    # evaluate the constraint gradients
    g_pen = test.gradvec(x,['curv']) 
    # reset the seed so we get the same perturbations
    np.random.seed(seed)
    _X = [perturb_coil(x,C,max_rotation,max_shift,n_coils,nfcoil) for ii in range(batch_size)]
    # compute the CVaR gradient
    _fX = np.array([test.fvec(_x,['bnorm']) for _x in _X])
    _gX = np.array([test.gradvec(_x,['bnorm']) for _x in _X])
    g_stoch = np.mean(np.diag(_fX-t > 0) @ _gX,axis=0)/delta
    # make penalty obj
    dgdx = g_stoch + g_pen
    # now get the t-derivatives
    dgdt = 1 - np.mean(_fX-t > 0)/delta
    # stack
    g = np.append(dgdx,dgdt)
    return g

print(Obj(x0))
print(Grad(x0))
quit()
# optimize w/ BFGS
xopt,X,fX = BFGS(Obj,Grad,x0,mu0 = mu0,max_iter=max_iter,
  gtol=gtol,c_armijo=c_armijo,mu_min=mu_min,mu_max=mu_max)

# get the function values along the trajectory
fXopt          = fX[-1]
if master:
  print(f"fopt: {fXopt}")

if master:
  # save the results
  d                    = {}
  d['X']               = X
  d['fX']              = fX
  d['xopt']            = xopt
  d['fXopt']           = fXopt
  d.update(run_params)
  d.update(get_input_settings(focus.globals))
  import os
  if os.path.exists(data_dir) is False:
    os.mkdir(data_dir)
  pickle.dump(d,open(data_filename,"wb"))
  print(f"Dumped {data_filename}")

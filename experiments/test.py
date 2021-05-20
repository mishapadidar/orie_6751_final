import numpy as np
import pickle
import sys
sys.path.append('../optim/')
from bfgs import BFGS
from gradient_descent import gradient_descent
from subgradient_descent import subgradient_descent
sys.path.append('../utils/')
from get_input_settings import get_input_settings
sys.path.append('../problem/')
from normal_covariances import compute_fourier_coefficient_covariance_for_FOCUS
from perturb_coil import perturb_coil
import ccsep,cpsep
sys.path.append('../../FOCUS/python/')
from mpi4py import MPI
from focuspy import FOCUSpy
import focus
import multiprocessing

# load a pickle with the run params
args = sys.argv
param_filename = args[1]
run_params = pickle.load(open(param_filename,"rb"))
problem_num = run_params['problem_num']
seed = run_params['seed']
data_dir = run_params['data_dir']
data_filename = run_params['data_filename']
max_shift = run_params['max_shift']
normal_pertubation_size= run_params['normal_pertubation_size']
gp_lengthscale = run_params['gp_lengthscale']
batch_size = run_params['batch_size']
max_iter = run_params['max_iter']
mu0 = run_params['mu0']
mu_min = run_params['mu_min']
mu_max = run_params['mu_max']
gtol = run_params['gtol']
c_armijo = run_params['c_armijo']
alpha_pen = run_params['alpha_pen']
target_bnorm = run_params['target_bnorm']
delta = run_params['delta']
lr_sched = run_params['lr_sched']
lr_gamma = run_params['lr_gamma']
lr_benchmarks = run_params['lr_benchmarks']
ccsep_eps = run_params['ccsep_eps']
cpsep_eps = run_params['cpsep_eps']
x0_file = run_params['x0_file']
optimizer = run_params['optimizer']
assert optimizer in ['GD',"SubGD"]

if lr_sched == "MultiStepLR":
  def lr_sched(step):
    a = np.sum(np.array(lr_benchmarks) < step)
    # lr_gamma should be in (0,1)
    return (lr_gamma)**a
elif lr_sched == "LambdaLR":
  lr_sched = lambda step: 1./(1+lr_gamma*np.sqrt(step))
else:
  lr_sched = None

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


"""
===============================================

Prepare Perturbatons, x0, objectives, constraints

"""


# starting point
if x0_file is not None:
  d = pickle.load(open(x0_file,"rb"))
  x0 = d['xopt']
else:
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

# determine the max rotation: solve r*theta = max_shift
max_rotation = max_shift/focus.globals.init_radius

# define coil separation penalty
cckwargs = {}
cckwargs['n_coils']    = n_coils
cckwargs['n_seg']      = 100 # coil discretization
cckwargs['alpha']      = -100
cckwargs['return_np']  = True
# coil-plasma separation
cpkwargs = {}
cpkwargs['n_coils'] = n_coils
cpkwargs['n_seg']   = 100
cpkwargs['alpha']   = -100
cpkwargs['ntor']    = 200
cpkwargs['npol']    = 200 # should be equal to ntor
cpkwargs['return_np']  = True
cpkwargs['plasma_file'] = '../experiments/w7x_jf.boundary'

def constraints(x):
    f_pen  = test.fvec(x,['ttlen'])[0]
    f_pen += np.sum(np.minimum(ccsep.coil_coil_sep(x,**cckwargs) - ccsep_eps,0.0)**2)
    f_pen += np.sum(np.minimum(cpsep.cpsep(x,**cpkwargs) - cpsep_eps,0.0)**2)
    return f_pen
def constraints_grad(x):
    g_pen   = test.gvec(x,['ttlen'])[0]
    ccgrad  = 2*np.minimum(ccsep.coil_coil_sep(x,**cckwargs) - ccsep_eps,0.0) * ccsep.coil_coil_sep_grad(x,**cckwargs).T
    g_pen  += np.sum(ccgrad,axis=1)
    cpgrad  = 2*np.minimum(cpsep.cpsep(x,**cpkwargs) - cpsep_eps,0.0) * cpsep.cpsep_grad(x,**cpkwargs).T
    g_pen  += np.sum(cpgrad,axis=1)
    return g_pen

#  for pool evaluation of stochastic objective
def pool_bnorm(_x): 
  return test.fvec(_x,['bnorm'])[0]
def pool_bnorm_grad(_x):
    return test.gvec(_x,['bnorm'])[0]


"""
===============================================

Define the problems

"""

# set the objective
if problem_num == 0:
  """minimize_{x,t} CVaR_{1-delta}(bnorm)
  """
  # append the CVaR t parameter
  t0 = 0.0
  x0 = np.append(x0,t0)

  # CVaR objective
  def Obj(y):
    # split x and t
    t = y[-1]
    x = y[:-1]
    ## evaluate the constraints
    f_pen = constraints(x)
    # reset the seed so we get the same perturbations
    np.random.seed(seed)
    _X = [perturb_coil(x,C,max_rotation,max_shift,n_coils,nfcoil) for ii in range(batch_size)]

    # compute the CVaR
    with multiprocessing.Pool(4) as pool:
      _fX = np.array(pool.map(pool_bnorm, _X))
    # _fX = np.array([test.fvec(_x,['bnorm'])[0] for _x in _X])

    f_stoch = np.mean(np.maximum(_fX - t,0.0))/delta + t
    # make penalty obj
    F = f_stoch + alpha_pen*f_pen
    if verbose:
      print(f_stoch,f_pen)
    return F

  def Grad(y):
    ## split x and t
    t = y[-1]
    x = y[:-1]
    # evaluate the constraint gradients
    g_pen = constraints_grad(x)
    # reset the seed so we get the same perturbations
    np.random.seed(seed)
    _X = [perturb_coil(x,C,max_rotation,max_shift,n_coils,nfcoil) for ii in range(batch_size)]

    # compute the CVaR gradient
    with multiprocessing.Pool(4) as pool:
      _fX = np.array(pool.map(pool_bnorm, _X))
      _gX = np.array(pool.map(pool_bnorm_grad, _X))
    # _fX = np.array([test.fvec(_x,['bnorm'])[0] for _x in _X])
    # _gX = np.array([test.gvec(_x,['bnorm'])[0] for _x in _X])

    g_stoch = np.mean((_fX-t > 0.0) * _gX.T,axis=1)/delta
    # make penalty obj
    dgdx = g_stoch + alpha_pen*g_pen
    # now get the t-derivatives
    dgdt = 1 - np.mean(_fX-t > 0)/delta
    # stack
    g = np.append(dgdx,dgdt)

    return g

elif problem_num == 1:
  """Risk Neutral Problem"""

  def Obj(x):
    ## evaluate the constraints
    f_pen = constraints(x)
    # reset the seed so we get the same perturbations
    np.random.seed(seed)
    _X = [perturb_coil(x,C,max_rotation,max_shift,n_coils,nfcoil) for ii in range(batch_size)]

    # compute the Expectation
    with multiprocessing.Pool(4) as pool:
      _fX = np.array(pool.map(pool_bnorm, _X))
    f_stoch = np.mean(_fX)
    # f_stoch = test.fvec(x,['bnorm'])[0]

    # make penalty obj
    F = f_stoch + alpha_pen*f_pen
    if verbose:
      print(F,f_stoch,f_pen)
    return F

  def Grad(x):
    # evaluate the constraint gradients
    g_pen = constraints_grad(x)
    # reset the seed so we get the same perturbations
    np.random.seed(seed)
    _X = [perturb_coil(x,C,max_rotation,max_shift,n_coils,nfcoil) for ii in range(batch_size)]

    # sample average the gradient
    with multiprocessing.Pool(4) as pool:
      _gX = np.array(pool.map(pool_bnorm_grad, _X))
    g_stoch = np.mean(_gX,axis=0)
    # g_stoch = test.gvec(x,['bnorm'])[0]

    # make penalty obj
    G = g_stoch + alpha_pen*g_pen

    return G

elif problem_num == 2:
  """Maximize probability of low field error"""

  # TODO: define the problem here

elif problem_num == 3:
  """VaR with Chernoff Approximation"""

  # TODO: define the problem here

elif problem_num == 4:
  """Maximize probability with Chernoff Approximation"""

  # TODO: define the problem here


"""
===============================================

Now run the optimization

"""

f0 = Obj(x0)
if verbose:
  print(f"Starting from Objective value of {f0}")
  sys.stdout.flush()

# optimize
if optimizer == "GD":
  # xopt,X,fX = gradient_descent(Obj,Grad,x0,mu0 = mu0,max_iter=max_iter,
  #       gtol=gtol,c_armijo=c_armijo,mu_min=mu_min,mu_max=mu_max,verbose=verbose)
  xopt,X,fX = BFGS(Obj,Grad,x0,mu0 = mu0,max_iter=max_iter,
        gtol=gtol,c_armijo=c_armijo,mu_min=mu_min,mu_max=mu_max,verbose=verbose)
elif optimizer== "SubGD":
  xopt,X,fX = subgradient_descent(Obj,Grad,x0,mu0 = mu0,max_iter=max_iter,
        gtol=gtol,lr_sched=lr_sched,verbose=verbose)

# get the function values along the trajectory
fXopt          = fX[-1]
gopt = Grad(xopt)
if master:
  print(f"fopt: {fXopt}")
  print(f"gopt: {gopt}")
  print(f"norm(grad): {np.linalg.norm(gopt)}")

if master:
  # save the results
  d                    = {}
  d['X']               = X
  d['fX']              = fX
  if problem_num == 0: # remove the CVar variable
    d['xopt']        = xopt[:-1]
    d['VaRopt']      = xopt[-1]
  else:
    d['xopt']        = xopt
  d['fXopt']           = fXopt
  d.update(run_params)
  d.update(get_input_settings(focus.globals))
  import os
  if os.path.exists(data_dir) is False:
    os.mkdir(data_dir)
  pickle.dump(d,open(data_filename,"wb"))
  print(f"Dumped {data_filename}")

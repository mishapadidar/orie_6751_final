import numpy as np
import pickle
import sys
sys.path.append('../optim/')
from adam import adam
#from adam_control_variate import adam_cv
sys.path.append('../utils/')
sys.path.append('../problem/')
from normal_covariances import compute_fourier_coefficient_covariance_for_FOCUS
from perturb_coil import perturb_coil
from get_input_settings import get_input_settings
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
lr_eta = run_params['lr_eta']
lr_gamma = run_params['lr_gamma']
eps = run_params['eps']
beta1 = run_params['beta1']             
beta2 = run_params['beta2']  
batch_size = run_params['batch_size']
max_iter = run_params['max_iter']

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
  # Markov to maximize probability of low error
  def stoch_Obj(x):
    y = perturb_coil(x,C,max_rotation,max_shift,n_coils,nfcoil)
    # TODO: write objective function
    return test.func(y) #+ ccsep(x)
  def stoch_grad(x):
    y = perturb_coil(x,C,max_rotation,max_shift,n_coils,nfcoil)
    # TODO: write gradient function
    return test.grad(y) #+ ccsep_grad(x)

# optimize
x_best,X = adam(stoch_grad, x0,max_iter=max_iter,batch_size=batch_size,
           eta=lr_eta,gamma=lr_gamma,beta1=beta1,beta2=beta2,
           eps=1e-8,func=stoch_Obj,verbose=verbose)
# get the function values along the trajectory
if master:
  print("")
  print("Computing function values")
fX               = np.array([stoch_Obj(x) for x in X])
fX_best          = fX[-1]
if master:
  print(f"fopt: {fX_best}")

if master:
  # save the results
  d                      = {}
  d['X']                 = X
  d['fX']                = fX
  d['x_best']            = x_best
  d['fX_best']           = fX_best
  d.update(run_params)
  d.update(get_input_settings(focus.globals))
  import os
  if os.path.exists(data_dir) is False:
    os.mkdir(data_dir)
  pickle.dump(d,open(data_filename,"wb"))
  print(f"Dumped {data_filename}")

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


# load the starting point
args = sys.argv
param_filename = args[1]
run_params = pickle.load(open(param_filename,"rb"))
xopt_file = run_params['xopt_file']
data_dir = run_params['data_dir']
data_filename = run_params['data_filename']
seed = run_params['seed']
n_evals = run_params['n_evals']
# starting point
indata = pickle.load(open(xopt_file,"rb"))
x0 = indata['xopt']

# perturbation info
max_shift = indata['max_shift']
normal_pertubation_size = indata['normal_pertubation_size']
gp_lengthscale = indata['gp_lengthscale']
# max_shift = 0.005
# normal_pertubation_size = 0.002
# gp_lengthscale = 0.5

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
test = FOCUSpy(comm=comm, extension='../experiments/w7x_jf', verbose=True)
focus.globals.input_surf = '../experiments/w7x_jf.boundary'
test.prepare() 
focus.globals.iout = 0

# set the seed
np.random.seed(seed)



"""
========================================

Define perturbations

"""

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

#  for pool evaluation of stochastic objective
def pool_bnorm(_x): 
  return test.fvec(_x,['bnorm'])[0]
def pool_bnorm_grad(_x):
    return test.gvec(_x,['bnorm'])[0]

"""
===========================

Do the evals

"""

# generate perturbations
X = [perturb_coil(x0,C,max_rotation,max_shift,n_coils,nfcoil) for ii in range(n_evals)]

# compute the function values
with multiprocessing.Pool(4) as pool:
  fX = np.array(pool.map(pool_bnorm, X))

if master:
  # save the results
  d                = {}
  d['X_samples']   = X
  d['fX_samples']  = fX
  d.update(indata)
  d.update(get_input_settings(focus.globals))
  import os
  if os.path.exists(data_dir) is False:
    os.mkdir(data_dir)
  pickle.dump(d,open(data_filename,"wb"))
  print(f"Dumped {data_filename}")

import numpy as np
import multiprocessing
import glob
import pickle
import sys
from datetime import datetime
sys.path.append('../utils/')
from get_input_settings import get_input_settings
sys.path.append('../problem/')
from normal_covariances import compute_fourier_coefficient_covariance_for_FOCUS
from perturb_coil import perturb_coil
sys.path.append('../../FOCUS/python/')
from mpi4py import MPI
from focuspy import FOCUSpy
import focus
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


# save data directory
data_dir = "./output/"
n_evals  = 1000
max_shift = 0.002
normal_perturbation_size = 0.002
gp_lengthscale = 0.5

filelist = glob.glob("../experiments/output/*.pickle")
#filelist = ["../experiments/output/baseline_20210520135037.pickle"]
for infile in filelist:

  # starting point
  indata = pickle.load(open(infile,"rb"))
  x0 = indata['xopt']
  
  # perturbation info
  indata['sample_max_shift'] = max_shift
  indata['sample_normal_perturbation_size'] = normal_perturbation_size
  indata['sample_gp_lengthscale'] = gp_lengthscale

  # set the seed
  now     = datetime.now()
  seed    = int("%d%.2d%.2d%.2d%.2d"%(now.month,now.day,now.hour,now.minute,now.second))
  np.random.seed(seed)

  # save filename
  base_name = infile.split("/")[-1]
  # base_name = base_name.split(".")[0]
  data_filename  = data_dir +"samples_" + base_name #+ ".pickle"
  
  
  
  """
  ========================================
  
  Define perturbations
  
  """
  
  # perturbation covariance
  n_coils = focus.globals.ncoils 
  nfcoil  = focus.globals.nfcoil
  n_modes = nfcoil + 1
  
  # prepare the normal covariance matrix
  
  cov_file = f"../problem/fourier_coefficient_covariances_{n_coils}_coils_"+\
  f"{nfcoil}_modes_{normal_perturbation_size}_p_{gp_lengthscale}_l.pickle"
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

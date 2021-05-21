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
n_evals  = 2000
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

  # set the seed for mpi
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

  # compute information
  mean = np.mean(fX)
  std  = np.std(fX)
  med  = np.median(fX)
  var99 = np.percentile(fX,99)
  var95 = np.percentile(fX,95)
  cvar95 = (1/.05)*np.mean(fX*(fX >=var95))
  cvar99 = (1/.01)*np.mean(fX*(fX >=var99))
  mx   = np.max(fX)
  mn   = np.min(fX)
  bn   = test.fvec(x0,['bnorm'])[0]
  
  
  if master:
    # save the results
    d                = {}
    d['X_samples']   = X
    d['fX_samples']  = fX
    d['mean']        = mean 
    d['std']         = std  
    d['med']         = med  
    d['var99']       = var99
    d['var95']       = var95    
    d['cvar95']      = cvar95
    d['cvar99']      = cvar99
    d['max']         = mx   
    d['min']         = mn   
    d['bnorm']       = bn

    d.update(indata)
    d.update(get_input_settings(focus.globals))
    import os
    if os.path.exists(data_dir) is False:
      os.mkdir(data_dir)
    pickle.dump(d,open(data_filename,"wb"))
    print(f"Dumped {data_filename}")

import os
import pickle
import subprocess
from datetime import datetime
import numpy as np

# flags
write_sbatch =True
submit       =False

# write a pickle file with the run info
run_params_dir = "./param_files/"
if os.path.exists(run_params_dir) is False:
  os.mkdir(run_params_dir)
run_params = {}
mpi_nodes                       = 1
run_params['problem_num']       = 0
run_params['batch_size']        = 10
run_params['max_iter']          = 10
run_params['mu0']               = 1e-6
run_params['mu_min']            = 1e-16
run_params['mu_max']            = 1e6
run_params['gtol']              = 1e-2
run_params['c_armijo']          = 0.0
run_params['target_bnorm']      = 1e-4 # TODO: Look to JF for an appropriate value
run_params['delta']             = 0.05 
run_params['alpha_pen']         = 10
run_params['normal_pertubation_size'] = 0.002  # 2mm,5mm,10mm
run_params['gp_lengthscale']    = 0.5
run_params['max_shift']         = 0.005 # 2mm,5mm,10mm
# angle set so it rotates a 1.5m coil by 2mm, 5mm, 10mm
# 2mm = pi/1570, 5mm=pi/628, 10mm=pi/313
run_params['max_rotation']      = np.pi/1570 
run_params['lr_sched']          = "MultiStepLR"
run_params['lr_gamma']          = 0.1
run_params['lr_benchmarks']     = [100,200]
run_params['data_dir']          = "./output/" # data save location

# seed and date
now     = datetime.now()
seed    = int("%d%.2d%.2d%.2d%.2d"%(now.month,now.day,now.hour,now.minute,now.second))
barcode = "%d%.2d%.2d%.2d%.2d%.2d"%(now.year,now.month,now.day,now.hour,now.minute,now.second)
run_params['date']  = now
run_params['seed']  = seed
# file name
base_name = f"problem_{run_params['problem_num']}_{barcode}"
data_filename  = run_params['data_dir'] + "data_" + base_name + ".pickle"
run_params['data_filename'] = data_filename
param_filename = run_params_dir + "params_" +base_name + ".pickle"
pickle.dump(run_params,open(param_filename,'wb'))
print(f"Dumped param file: {param_filename}")

if write_sbatch:
  # write a slurm batch script
  slurm_dir  = "./slurm_scripts/"
  if os.path.exists(slurm_dir) is False:
    os.mkdir(slurm_dir)
  slurm_name = slurm_dir + base_name + ".sub"
  #slurm_name = base_name + ".sub"
  f = open(slurm_name,"w")
  f.write(f"#!/bin/bash\n")
  f.write(f"#SBATCH -J focus_prob_{run_params['problem_num']}\n")
  f.write(f"#SBATCH -o ./slurm_output/job_%j.out\n")
  f.write(f"#SBATCH -e ./slurm_output/job_%j.err\n")
  f.write(f"#SBATCH --get-user-env\n")
  f.write(f"#SBATCH -N 1\n")
  f.write(f"#SBATCH -n 1\n")
  f.write(f"#SBATCH --mem=15000\n")
  f.write(f"#SBATCH -t 168:00:00\n")
  f.write(f"#SBATCH --partition=default_partition\n")
  f.write(f"#SBATCH --nodes={mpi_nodes}\n")
  f.write(f"#SBATCH --ntasks={mpi_nodes}\n")
  f.write(f"#SBATCH --tasks-per-node=1\n")
  f.write(f"#SBATCH --cpus-per-task=1\n")
  f.write(f"mpiexec -n {mpi_nodes} python {param_filename}\n")
  print(f"Dumped slurm file: {slurm_name}")
    
  # write the shell submission script
  submit_name = slurm_dir + 'slurm_submit.sh'
  f = open(submit_name,"w")
  f.write(f"#!/bin/bash\n")
  f.write(f"sbatch --requeue {slurm_name}")
  f.close()
  print(f"Dumped bash script: {submit_name}")
  
  if submit:
    # submit the script
    #bash_command = f"sbatch {slurm_name}"
    bash_command = f"bash {submit_name}"
    subprocess.run(bash_command.split(" "))

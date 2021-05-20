import os
import pickle
import subprocess
from datetime import datetime
import numpy as np
import glob

# flags
write_sbatch =True
submit       =False

# run parameters
run_params = {}
# for the run script
run_params_dir = "./param_files/"
if os.path.exists(run_params_dir) is False:
  os.mkdir(run_params_dir)

#filelist =["../experiments/output/baseline_20210519202815.pickle"]
filelist = glob.glob("../experiments/output/data_*.pickle")
for infile in filelist:

  mpi_nodes = 1
  # starting point
  run_params['xopt_file']           = infile
  run_params['data_dir']          = "./output/"
  run_params['n_evals']           = 10

  # seed and date
  now     = datetime.now()
  seed    = int("%d%.2d%.2d%.2d%.2d"%(now.month,now.day,now.hour,now.minute,now.second))
  barcode = "%d%.2d%.2d%.2d%.2d%.2d"%(now.year,now.month,now.day,now.hour,now.minute,now.second)
  run_params['date']  = now
  run_params['seed']  = seed

  # get the file name with no directory or extensions
  base_name = infile.split("/")[-1]
  base_name = base_name.split(".")[0]
  # where to save data
  data_filename  = run_params['data_dir'] +"samples_" + base_name + ".pickle"
  # params for the post processing
  run_params['data_filename'] = data_filename
  param_filename = run_params_dir + "post_params_" +base_name + ".pickle"
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
    f.write(f"#SBATCH -J post\n")
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
    f.write(f"mpiexec -n {mpi_nodes} python gen_distribution_data.py {param_filename}\n")
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

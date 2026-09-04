#!/bin/bash
#SBATCH -J ondemand/sys/myjobs/basic_python_serial
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --job-name=generate_BLRAN_model
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=pas3353
scontrol show job $SLURM_JOBID

# Move to the directory where the job was submitted from
cd $SLURM_SUBMIT_DIR

module load miniconda3/24.1.2-py310
source activate jax-fem-env
export PYTHONPATH="/users/PAS3353/peterfrazier/.local/bin:$PYTHONPATH"
python3 combine_data.py
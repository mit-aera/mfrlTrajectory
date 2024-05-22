#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive

source ~/.bashrc
conda activate s2s_py310
srun python gendata_mfrl_traj.py -bs 50 -bss 50 -dmin 14 -dmax 14

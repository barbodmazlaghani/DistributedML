#!/bin/bash
#SBATCH --job-name=e_parallel_1n2c
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=partition
#SBATCH --output=e_parallel_1n2c.out
echo "Jobs are started ..."
srun --mpi=pmix_v4 python3 e_sim_b.py

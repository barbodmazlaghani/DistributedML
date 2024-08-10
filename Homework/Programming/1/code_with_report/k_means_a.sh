#!/bin/bash
#SBATCH --job-name=k_serial_1n1c
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=partition
#SBATCH --output=k_serial_1n1c.out
echo "Jobs are started ..."
srun --mpi=pmix_v4 python3 k_means_a.py

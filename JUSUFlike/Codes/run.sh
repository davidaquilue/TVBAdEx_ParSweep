#!/bin/bash -x
#SBATCH --account=icei-hbp-2022-0005
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=128
#SBATCH --time=07:00:00
#SBATCH --output=output_%j.out  
#SBATCH --error=error_%j.er     
#SBATCH --mail-user=daquilue99@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=batch
#SBATCH --job-name=chunks_5

source activateTVB
srun python3 HPC_sim.py

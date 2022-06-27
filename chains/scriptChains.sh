#!/bin/bash
#SBATCH --job-name=opendustChains
#SBATCH --error=errChains
#SBATCH --output=outputChains
#SBATCH --time=23:00:00            
#SBATCH --ntasks=1                
#SBATCH --nodes=1                  
#SBATCH --gpus=6
#SBATCH --cpus-per-task=1         
#SBATCH  --constraint="type_e"


module load CUDA/11.4

time python test.py



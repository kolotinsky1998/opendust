#!/bin/bash
#SBATCH --job-name=opendustPiel2017Fig6     
#SBATCH --error=errPiel2017Fig6       
#SBATCH --output=outputPiel2017Fig6  
#SBATCH --time=23:00:00            
#SBATCH --ntasks=1                
#SBATCH --nodes=1                  
#SBATCH --gpus=2                 
#SBATCH --cpus-per-task=1         
#SBATCH  --constraint="type_a|type_b|type_c"


module load CUDA/11.4

python launchPiel2017Figure6.py



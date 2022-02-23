#!/bin/bash
#SBATCH --job-name=opendustPiel2017Fig5     
#SBATCH --error=errPiel2017Fig5          
#SBATCH --output=outputPiel2017Fig5       
#SBATCH --time=01:00:00            
#SBATCH --ntasks=1                
#SBATCH --nodes=1                  
#SBATCH --gpus=1                   
#SBATCH --cpus-per-task=1         
#SBATCH  --constraint="type_a|type_b|type_c|type_e"


module load CUDA/11.4

python launchPiel2017Figure5.py



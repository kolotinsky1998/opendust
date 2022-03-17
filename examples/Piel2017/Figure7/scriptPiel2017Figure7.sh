#!/bin/bash
#SBATCH --job-name=opendustPiel2017Fig7     
#SBATCH --error=errPiel2017Fig7        
#SBATCH --output=outputPiel2017Fig7    
#SBATCH --time=23:00:00            
#SBATCH --ntasks=1                
#SBATCH --nodes=1                  
#SBATCH --gpus=8                 
#SBATCH --cpus-per-task=1         
#SBATCH  --constraint="type_e"


module load CUDA/11.4

python launchPiel2017Figure7.py



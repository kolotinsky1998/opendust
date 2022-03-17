#!/bin/bash
#SBATCH --job-name=opendustPatacchini2008Figure3      
#SBATCH --error=errPatacchini2008Figure3        
#SBATCH --output=outputPatacchini2008Figure3     
#SBATCH --time=23:00:00            
#SBATCH --ntasks=1                
#SBATCH --nodes=1                  
#SBATCH --gpus=3           
#SBATCH --cpus-per-task=1         
#SBATCH  --constraint="type_a|type_b|type_c"


module load CUDA/11.4

time python launchPatacchini2008Figure3.py



#!/bin/bash
#SBATCH --job-name=opendustPatacchini2008   
#SBATCH --error=errPatacchini2008             
#SBATCH --output=outputPatacchini2008       
#SBATCH --time=24:00:00            
#SBATCH --ntasks=1                
#SBATCH --nodes=1                  
#SBATCH --gpus=8                
#SBATCH --cpus-per-task=1         
#SBATCH  --constraint="type_e"


module load CUDA/11.4
touch force.txt
touch charge.txt

python launchPatacchini2008Figure3.py



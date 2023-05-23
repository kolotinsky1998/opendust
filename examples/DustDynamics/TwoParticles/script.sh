#!/bin/bash
#SBATCH --job-name=opendust_dynamics 
#SBATCH --error=error_opendust_dynamics
#SBATCH --output=output_opendust_dynamics
#SBATCH --time=10-24:00:00            
#SBATCH --ntasks=1                
#SBATCH --nodes=1                  
#SBATCH --gpus=2
#SBATCH --cpus-per-task=1         

python -u launch.py ./data
                          

                          



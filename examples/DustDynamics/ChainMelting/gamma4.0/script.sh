#!/bin/bash
#SBATCH --job-name=gamma_4
#SBATCH --error=error_opendust_ChainStability
#SBATCH --output=output_opendust_ChainStability
#SBATCH --time=30-00:00:00            
#SBATCH --ntasks=1                
#SBATCH --nodes=1                  
#SBATCH --gpus=2
#SBATCH --cpus-per-task=1      
#SBATCH -A proj_1371   

python -u launch.py ./data 4.0
                          

                          



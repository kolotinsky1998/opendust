#!/bin/bash
#SBATCH --job-name=opendust_ChainWithHole
#SBATCH --error=error_opendust_ChainWithHole
#SBATCH --output=output_opendust_ChainWithHole
#SBATCH --time=30-00:00:00            
#SBATCH --ntasks=1                
#SBATCH --nodes=1                  
#SBATCH --gpus=4
#SBATCH --cpus-per-task=1      
#SBATCH -A proj_1371   

python -u launch.py ./data
                          

                          



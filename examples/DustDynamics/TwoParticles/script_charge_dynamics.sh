#!/bin/bash
#SBATCH --job-name=opendust_charge_dynamics
#SBATCH --error=err_charge_dynamics
#SBATCH --output=output_charge_dynamics
#SBATCH --time=24:00:00            
#SBATCH --ntasks=1                
#SBATCH --nodes=1                  
#SBATCH --gpus=2
#SBATCH --cpus-per-task=1         

python launch.py ./data_dynamics
                          

                          



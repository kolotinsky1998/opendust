#!/bin/bash
#SBATCH --job-name=opendustCharge_classic    
#SBATCH --error=erropendustCharge_classic
#SBATCH --output=outputopendustCharge_classic
#SBATCH --time=24:00:00            
#SBATCH --ntasks=1                
#SBATCH --nodes=1                  
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1         

python launch_classic_charging.py ./data_classic_charging
                          

                          



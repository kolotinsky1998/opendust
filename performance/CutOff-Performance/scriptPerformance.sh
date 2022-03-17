#!/bin/bash
#SBATCH --job-name=opendustPerformanceTest     
#SBATCH --error=errPerformanceTest
#SBATCH --output=outputPerformanceTest
#SBATCH --time=05:00:00            
#SBATCH --ntasks=1                
#SBATCH --nodes=1                  
#SBATCH --gpus=1        
#SBATCH --cpus-per-task=1         
#SBATCH  --constraint="type_a"

touch time.txt

for var in 32
do
time python benchmark.py $var
done                              

                          



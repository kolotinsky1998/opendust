#!/bin/bash
#SBATCH --job-name=opendustPerformanceTest     
#SBATCH --error=errPerformanceTest19        
#SBATCH --output=outputPerformanceTest19
#SBATCH --time=05:00:00            
#SBATCH --ntasks=1                
#SBATCH --nodes=1                  
#SBATCH --gpus=1                   
#SBATCH --cpus-per-task=1         
#SBATCH  --constraint="type_a|type_b|type_c"

touch time.txt
for var in 15 16 17 18 
do
time python benchmark.py $var
done                              



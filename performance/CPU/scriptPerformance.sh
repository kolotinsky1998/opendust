#!/bin/bash
#SBATCH --job-name=opendustPerformanceTestCPU
#SBATCH --error=errPerformanceTest   
#SBATCH --output=outputPerformanceTest
#SBATCH --time=23:00:00            
#SBATCH --ntasks=1                
#SBATCH --nodes=1              
#SBATCH --gpus=1              
#SBATCH --cpus-per-task=1
#SBATCH  --constraint="type_a|type_b"

touch time.txt
for var in 15 16 17 18
do
time python benchmark.py $var
done




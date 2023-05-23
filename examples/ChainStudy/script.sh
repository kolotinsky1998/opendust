#!/bin/bash
#SBATCH --job-name=opendustChainStudy
#SBATCH --error=errChainStudy
#SBATCH --output=outputChainStudy
#SBATCH --time=3-00:00     
#SBATCH --ntasks=1                
#SBATCH --nodes=1                  
#SBATCH --gpus=2
#SBATCH --constraint="type_a|type_b|type_c"
#SBATCH -A proj_1293

dir=/home/avtimofeev/opendust/examples/ChainStudy/data
mkdir -p ${dir}
mkdir -p ${dir}/csv
#python -u zero_iter.py ${dir}
python -u launch.py  ${dir}

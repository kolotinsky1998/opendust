for dir in gamma1.0 gamma2.0 gamma3.0 gamma4.0 gamma5.0 gamma6.0 gamma7.0 gamma8.0 gamma9.0 gamma10.0 
do
cd $dir
sbatch script.sh
cd ..
done

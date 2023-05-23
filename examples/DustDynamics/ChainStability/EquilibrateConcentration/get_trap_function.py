import sys
sys.path.insert(0,'/home/avtimofeev/opendust/tools')
from fit import fit
import numpy as np

z = np.array([
-0.003224806746736638,	
-0.0027427480062450784,		
-0.0022773119809428835,	
-0.0018783668163981446,	
-0.0014794216518534062,	
-0.001030608341740575,	
-0.0005485496012490158,	
])

E = np.array([
1952.9145917470426,	
2257.91003224094,
2313.069154502875,	
2525.050644076086,	
2480.1848927644396,	
2496.689215446066,	
2499.769890017505,
])

H = 0.008311357594682057
r_D_e = 0.0016622715189364113
z_min = -H/2.0 + 0.5*r_D_e
z_max = H/2.0
E_0, alpha, beta = fit(z, E, z_min, z_max)

print("E_0 = {}".format(E_0))
print("alpha = {}".format(alpha))
print("beta = {}".format(beta))


import jax.numpy as jnp
from jax import random
from jaxfmm import *

print("Generating sample points and charges.")
N = 2**15
key = random.key(124)
pts = random.uniform(key,(N,3))
chrgs = random.uniform(key,N)

print("Generating FMM hierarchy.")
tree_info = gen_hierarchy(pts)

print("Compiling and computing FMM potential.")
pot_FMM = eval_potential(chrgs, **tree_info)

print("Compiling and computing analytic potential.")
pot_dir = eval_potential_direct(pts, chrgs)

print("FMM normwise relative error: %.2e"%(jnp.linalg.norm(pot_dir-pot_FMM)/jnp.linalg.norm(pot_dir)))

### if dev dependencies (jaxfmm[dev]) are installed, debug information can be printed as follows:
# from jaxfmm.debug_helpers import *
# tree_info = gen_hierarchy(pts, debug_info = True)
# gen_stats(**tree_info, print_stats = True)

### additionally, vtk visualizations can be generated as follows:
# gen_hierarchy_vtk(**tree_info)
# gen_wellsep_vtk(0, **tree_info)
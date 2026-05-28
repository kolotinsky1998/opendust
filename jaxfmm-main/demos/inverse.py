import jax
import jax.numpy as jnp
from jaxfmm import *
import optimistix as optx

### NOTE: this example requires optimistix

print("Assembling grid of charges.")
nside, sidelen = 20, 4*jnp.pi
pts = (jnp.mgrid[:nside,:nside,:nside].T/(nside-1) * sidelen - sidelen/2).reshape((-1,3))
chrgs = 0.01*jnp.ones(pts.shape[0])

print("Generating hierarchy and desired potential.")
tree_info = gen_hierarchy(pts)
desired_pot = jnp.sin(jnp.linalg.norm(pts,axis=-1))
norm = jnp.linalg.norm(desired_pot)

def loss(chrgs, args):
    loss = jnp.linalg.norm(desired_pot-eval_potential(chrgs, **tree_info))/norm
    return loss, loss

@jax.jit
def optimize(chrgs):
    solver = optx.LBFGS(history_length=10, rtol=1e-6, atol=1e-6)
    sol = optx.minimise(loss,solver,chrgs, has_aux=True, throw=False, max_steps=1000)
    return sol.value, sol.stats['num_steps'], sol.aux

print("Compiling (might take a while) + running minimizer.")
res = optimize(chrgs)
print("Final loss (%i iterations): %.2e"%(res[1],res[2]))
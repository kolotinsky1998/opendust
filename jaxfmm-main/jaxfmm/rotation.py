import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.special import factorial as fac
from jaxfmm.basis import mpl_idx, inv_mpl_idx

@jax.jit
def cart_to_sph(rvec):
    r"""
    Convert cartesian to spherical coordinates.
    """
    xy = rvec[...,0]**2 + rvec[...,1]**2
    r = jnp.sqrt(xy + rvec[...,2]**2)
    theta = jnp.arctan2(jnp.sqrt(xy),rvec[...,2])
    phi = jnp.arctan2(rvec[...,1],rvec[...,0])
    return r, theta, phi

def precompute_combs(p):
    r"""
    Precompute unique Jacobi polynomial parameters that appear in Wigner rotation matrices up to expansion order p.
    Returns the unique combinations and their positions in the rotation matrices.
    """
    unqs, invs, unq_num = [], [], [0]

    for j in range(p+1):
        rng = jnp.arange(-j,j+1)    # m and k
        mu = jnp.abs(rng[:,None]-rng[None,:]).flatten()
        nu = jnp.abs(rng[:,None]+rng[None,:]).flatten()
        s = j - (mu+nu)//2
        comb = jnp.stack((s,mu,nu),axis=0)
        unq, inv = jnp.unique(comb,axis=1,return_inverse=True)
        unqs.append(unq)
        invs.append(inv)
        unq_num.append(unq_num[j] + unq.shape[1])

    for j in range(p+1):
        invs[j] = invs[j].reshape((2*j+1,2*j+1))[j:,:]
    return unqs, invs

def cut(i, p):
    r"""
    Number of Jacobi polynomial parameter combinations with s==i for expansion order p.
    """
    return 2*(p-i)+1

def cut_cum(i, p):
    r"""
    Number of Jacobi polynomial parameter combinations with s<=i for expansion order p.
    """
    return (p+1)**2 - (p-i)**2

@partial(jax.jit, static_argnames=['j'])
def fast_jacobi(j, mu, nu, x):
    r"""
     Compute Jacobi polynomials P^(mu,nu)_s(x) with a recursion relation 
    (adapted from https://doi.org/10.1103/PhysRevC.106.054320).
    """
    res = jnp.ones((*x.shape,mu.shape[0])) # res[...,:cut_cum(0,j)] = 1
    mu = mu[None,...]
    nu = nu[None,...]
    x = x[...,None]

    if(j>0):
        mu = mu[...,cut(0,j):]
        nu = nu[...,cut(0,j):]
        res1 = res[...,cut_cum(0,j):]

        res = res.at[...,cut_cum(0,j):].set(0.5*(2*(mu+1)+(mu+nu+2)*(x-1)))
    
    for i in range(2,j+1):
        mu = mu[...,cut(i-1,j):]
        nu = nu[...,cut(i-1,j):]
        res2 = res1[...,cut(i-1,j):]
        res1 = res[...,cut_cum(i-1,j):]

        xn = i - 1
        two_nab = 2*xn + mu + nu
        a1 = 2*(xn+1) * (xn+mu+nu+1) * two_nab
        a2 = (two_nab+1) * (mu**2 - nu**2)
        a3 = two_nab * (two_nab+1) * (two_nab+2)
        a4 = 2*(xn+mu) * (xn+nu) * (two_nab+2)
        res = res.at[...,cut_cum(i-1,j):].set(((a2+a3*x)*res1 - a4*res2)/a1)
    return res

def factorialfac(s, mu, nu):
    return jnp.sqrt(fac(s)*fac(s+mu+nu) / (fac(s+mu)*fac(s+nu)))

@partial(jax.jit, static_argnames=['p'])
def get_polar_rot_coeff(p, unqs, theta):
    r"""
    Compute all unique Wigner rotation matrix coefficients up until expansion order p.
    """
    basemat = fast_jacobi(p, unqs[1], unqs[2], jnp.cos(theta))
    basemat *= factorialfac(unqs[0], unqs[1], unqs[2])[None,...] * \
               jnp.sin(theta/2)[...,None]**unqs[1][None,...] * \
               jnp.cos(theta/2)[...,None]**unqs[2][None,...]
    return basemat

def Nfac(k, l, m):
    return jnp.sqrt(fac(l-k)*fac(l+k) / (fac(l-m)*fac(l+m)))

def eps(m, k):
    return (-1)**((m-k) * (k < m))

@jax.jit
def rot_polar(unqs, invs, theta, coeffs, inv=False, loc=False): # TODO: speed this up
    r"""
    Rotate multipole/local coefficients around the y-axis.
    """
    p = invs[-1].shape[0] - 1
    k = jnp.arange(-p,p+1)
    for j in range(p+1):
        ### assemble rotation matrix
        coeff_j = get_polar_rot_coeff(j, unqs[j], theta)
        real_part = coeff_j[...,invs[j][:,j:]]
        imag_part = coeff_j[...,invs[j][:,:j]]

        fact = (eps(k[None,p-j:p+j+1], k[p:p+j+1,None]) *                # missing factor from the recursion
                Nfac(k[p:p+j+1,None],j,k[None,p-j:p+j+1])**(2*loc-1) *   # normalization of the real basis
                (2*inv-1)**(k[None,p-j:p+j+1] + k[p:p+j+1,None]))        # inverse rotation (theta -> -theta)
        fact = fact.at[...,:j].mul(((-1)**jnp.arange(j,2*j))[None,:])    # invert signs for m<0
        real_part *= fact[None, :, j:]
        imag_part *= fact[None, :, :j]

        ### real update
        tmp = real_part.at[...,1:].add(jnp.flip(imag_part,axis=-1))
        coeffs = coeffs.at[...,mpl_idx(0,j):mpl_idx(j,j)+1].set(jnp.einsum("jmk, jk -> jm",tmp,coeffs[...,mpl_idx(0,j):mpl_idx(j,j)+1]))
        
        ### imag update
        tmp = jnp.flip(real_part[...,1:,1:],axis=(-2,-1)) - jnp.flip(imag_part[...,1:,:],axis=(-2))    # flipped to match the real basis ordering
        coeffs = coeffs.at[...,mpl_idx(-j,j):mpl_idx(0,j)].set(jnp.einsum("jmk, jk -> jm",tmp,coeffs[...,mpl_idx(-j,j):mpl_idx(0,j)]))
    return coeffs

@jax.jit
def rot_azimuth(coeffs, phi, inv=False, loc=False):
    r"""
    Rotate multipole/local coefficients around the z-axis.
    """
    ms, ns = inv_mpl_idx(jnp.arange(coeffs.shape[-1]))
    flipidx = mpl_idx(-ms,ns)
    sgns = (1-2*inv)*(1-2*loc)
    mphi = ms[None,:]*phi[...,None]
    coeffs = coeffs * jnp.cos(mphi) - sgns*coeffs[...,flipidx] * jnp.sin(mphi)   # NOTE: we intentionally swap the sign for sin
    return coeffs
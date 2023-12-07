import jax
from jax import numpy as jnp
from jax import random
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

@partial(jax.jit, static_argnums=(4,))
def kernel_scan(key, xold, eps, theta, prob):

    subkey1, subkey2 = jax.random.split(key)
    a = eps*jax.random.normal(subkey1, shape = xold.shape, dtype = np.float64)
    x = xold + a
    R = prob(x, theta)/prob(xold, theta)
    e = jax.random.uniform(subkey2, dtype=np.float64)

    xnew = jnp.where(R > e, x, xold)
    
    return xnew, subkey2, R > e

@partial(jax.jit, static_argnums=(4,))
def kernel_scan_NN(key, xold, eps, theta, prob):

    subkey1, subkey2 = jax.random.split(key)
    a = eps*jax.random.normal(subkey1, shape = xold.shape, dtype = np.float64)
    x = xold + a
    R = prob(x.reshape(1,-1), theta)/prob(xold.reshape(1,-1), theta)
    e = jax.random.uniform(subkey2, dtype=np.float64)

    xnew = jnp.where(R > e, x, xold)
    
    return xnew, subkey2, R > e


def sample(key, N, n_particles, dim, n_sweeps, eps, theta, prob):

    key, subkey = jax.random.split(key)
    #x0 = jax.random.uniform(key, shape=(n_particles*dim,), minval=-0.05, maxval=0.05, dtype=np.float64)
    x0 = jax.random.normal(key, shape=(n_particles*dim,), dtype = np.float64)

    def loop_scan(carry, i):

        xold, key, accept = carry
        xnew, key, cond = kernel_scan(key, xold, eps, theta, prob)
        accept += cond 
        return (xnew, key, accept), xnew
    
    steps = jnp.arange(0,N,1)
    accept = 0.0
    (xnew, key, accept), res = jax.lax.scan(loop_scan, (x0, subkey, accept), steps)
    res = res[2000::n_sweeps]

    return res, accept/N



def sample_NN(key, N, n_particles, dim, n_sweeps, eps, theta, prob):

    key, subkey = jax.random.split(key)
    #x0 = jax.random.uniform(key, shape=(n_particles*dim,), minval=-0.05, maxval=0.05, dtype=np.float64)
    x0 = jax.random.normal(key, shape=(n_particles*dim,), dtype = np.float64)
    
    def loop_scan(carry, i):

        xold, key, accept = carry
        xnew, key, cond = kernel_scan_NN(key, xold, eps, theta, prob)
        accept += cond 
        return (xnew, key, accept), xnew
    
    steps = jnp.arange(0,N,1)
    accept = 0.0
    (xnew, key, accept), res = jax.lax.scan(loop_scan, (x0, subkey, accept), steps)
    res = res[2000::n_sweeps]

    return res, accept/N


sample_mapped = jax.vmap(sample, in_axes=(0, None, None, None, None, None, None, None), out_axes=(0))
sample_mapped_NN = jax.vmap(sample, in_axes=(0, None, None, None, None, None, None, None), out_axes=(0))
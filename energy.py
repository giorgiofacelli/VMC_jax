import jax
from jax import numpy as jnp
from jax.tree_util import tree_map

def Ekin(x, theta, logpsi):

    basis = jnp.eye(x.shape[0])
    logpsi_x = lambda x: logpsi(x, theta)
    dlogpsi = lambda y: jax.vjp(logpsi_x, y)[1](1.)[0]
    ddlogpsi = lambda v: jax.jvp(dlogpsi, (x,), (v.reshape(x.shape),))
    Temp = jax.vmap(ddlogpsi, in_axes=(0))(basis)[1]
    Temp = Temp.reshape(Temp.shape[0],Temp.shape[-1])
    temp = jnp.diag(Temp)
    E = -jnp.sum(temp+dlogpsi(x)**2)/2.

    return E

def Ekin_NN(x, theta, logpsi):

    basis = jnp.eye(x.shape[0])
    x = x.reshape(1,-1)
    logpsi_x = lambda x: logpsi(x, theta)
    dlogpsi = lambda y: jax.vjp(logpsi_x, y)[1](1.)[0]
    ddlogpsi = lambda v: jax.jvp(dlogpsi, (x,), (v.reshape(x.shape),))
    Temp = jax.vmap(ddlogpsi, in_axes=(0))(basis)[1]
    Temp = Temp.reshape(Temp.shape[0],Temp.shape[-1])
    temp = jnp.diag(Temp)
    E = -jnp.sum(temp+dlogpsi(x)**2)/2.
    
    return E

def pot(x):
    return 0.5*jnp.linalg.norm(x, axis=-1)**2

Ekin_mapped = jax.vmap(Ekin, in_axes=(0, None, None), out_axes = 0)
Ekin_NN_mapped = jax.vmap(Ekin_NN, in_axes=(0, None, None), out_axes = 0)
pot_mapped = jax.vmap(pot, in_axes = 0, out_axes = 0)

def tot_energy(x, theta, logpsi):

    E = Ekin_mapped(x, theta, logpsi) + pot_mapped(x)

    return E

def tot_energy_NN(x, theta, logpsi):

    E = Ekin_NN_mapped(x, theta, logpsi) + pot_mapped(x)

    return E


def grad_energy(x, theta0, E, logpsi):

    logpsi_theta = lambda t: logpsi(x, t)
    dlogpsi = lambda theta: jax.vjp(logpsi_theta, theta)[1]
    deltaE = E-jnp.mean(E)
    temp = dlogpsi(theta0)(deltaE/x.shape[0])[0]
    G_k = tree_map(lambda x: 2.0*jnp.real(jnp.conj(x)), temp)

    return G_k

grad_energy_mapped = jax.vmap(grad_energy, in_axes=(0, None, None, None), out_axes=0)
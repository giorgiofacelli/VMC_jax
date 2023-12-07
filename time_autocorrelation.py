import jax
from jax import numpy as jnp
from energy import tot_energy, tot_energy_NN

def auto_correlation(key, sampler, eps, theta, logpsi):

    t_int = []

    for e in eps:

        key = jax.random.split(key[0], num = key.shape[0])
        X, _  = sampler(key, e)
        E_avg = jnp.zeros(X.shape[0])
        E_loc = jnp.zeros(shape = (X.shape[0], X.shape[1]))

        for i in range(X.shape[0]):
            E_loc = E_loc.at[i,:].set(tot_energy(X[i,:,:], theta, logpsi))
            E_avg = E_avg.at[i].set(jnp.mean(E_loc[i,:]))

        var_Eloc = jnp.var(E_loc)
        var_Eavg = jnp.var(E_avg)
        t_int.append(X.shape[1]*var_Eavg/(2.0*var_Eloc))

    return t_int



def auto_correlation_NN(key, sampler, eps, theta, logpsi):

    t_int = []

    for e in eps:

        key = jax.random.split(key[0], num = key.shape[0])
        X, _  = sampler(key, e)
        E_avg = jnp.zeros(X.shape[0])
        E_loc = jnp.zeros(shape = (X.shape[0], X.shape[1]))

        for i in range(X.shape[0]):
            E_loc = E_loc.at[i,:].set(tot_energy_NN(X[i,:,:], theta, logpsi))
            E_avg = E_avg.at[i].set(jnp.mean(E_loc[i,:]))

        var_Eloc = jnp.var(E_loc)
        var_Eavg = jnp.var(E_avg)
        t_int.append(X.shape[1]*var_Eavg/(2.0*var_Eloc))

    return t_int
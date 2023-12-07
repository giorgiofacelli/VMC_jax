import jax
from jax import numpy as jnp
from functools import partial
#from sampling import sample, sample_NN
from energy import tot_energy, tot_energy_NN, grad_energy
from jax.tree_util import tree_map
import optax


@partial(jax.jit, static_argnums=(1, 2, 6,))
def update_theta_gradient(key, sampler, sdim, theta0, grad, lr, logpsi):

    theta0 = tree_map(lambda x,y: x - lr*y, theta0, grad)
    X, a = sampler(key, theta0)
    X = X.reshape(-1,sdim)
    E = tot_energy(X, theta0, logpsi)
    grad = grad_energy(X, theta0, E, logpsi)

    return theta0, E, grad, jnp.mean(a)


def minimise_energy(key, sampler, sdim, theta, N_iter, lr, logpsi):
    
    key, subkey = jax.random.split(key)
    As = jnp.zeros(N_iter)
    X, a = sampler(key, theta)
    X = X.reshape(-1,sdim)
    E = tot_energy(X, theta, logpsi)
    grad = grad_energy(X, theta, E, logpsi)
    sigma = jnp.zeros(N_iter)

    def body_scan(carry, i):
        theta0, dE, As, sigma, subkey = carry
        key, subkey = jax.random.split(subkey)
        theta0, E, dE, a  =  update_theta_gradient(key, sampler, sdim, theta0, dE, lr, logpsi)
        As = As.at[i].set(a)
        sigma = sigma.at[i].set(jnp.std(E))
        return (theta0, dE, As, sigma, subkey), jnp.mean(E) 

    steps = jnp.arange(0, N_iter, 1)
    (theta_final, dE_final, As, sigma, key), Es = jax.lax.scan(body_scan, (theta, grad, As, sigma, subkey), steps)
    
    return theta_final, dE_final, Es, As, sigma


def minimise_energy_mapped(key, sampler, sdim, theta, N_iter, lr, logpsi):

    As = jnp.zeros(N_iter)
    X, a = sampler(key, theta)
    X = X.reshape(-1,sdim)
    E = tot_energy(X, theta, logpsi)
    grad = grad_energy(X, theta, E, logpsi)
    sigma = jnp.zeros(N_iter)

    def body_scan(carry, i):
        theta0, dE, As, sigma, key = carry
        key = jax.random.split(key[0], num = key.shape[0])
        theta0, E, dE, a  =  update_theta_gradient(key, sampler, sdim, theta0, dE, lr, logpsi)
        As = As.at[i].set(a)
        sigma = sigma.at[i].set(jnp.std(E))
        return (theta0, dE, As, sigma, key), jnp.mean(E) 

    steps = jnp.arange(0, N_iter, 1)
    (theta_final, dE_final, As, sigma, key), Es = jax.lax.scan(body_scan, (theta, grad, As, sigma, key), steps)
    
    return theta_final, dE_final, Es, As, sigma





@partial(jax.jit, static_argnums=(2,4,6,))
def update_theta_gradient_NN(key, opt_state, optimizer, grad, sampler, params, logpsi):

    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    X, a = sampler(key, params)
    X = X.reshape(-1,X.shape[-1])
    E = tot_energy_NN(X, params, logpsi)
    grad = grad_energy(X, params, E, logpsi)

    return params, opt_state, grad, jnp.mean(a), E


def minimise_energy_NN(key, sampler, params, logpsi, T, lr, optim_func):

    #key = jax.random.split(key[0], num=key.shape[0])
    #key, subkey = jax.random.split(key)
    As = jnp.zeros(T)
    sigma = jnp.zeros(T)
    optimizer = optim_func(learning_rate=lr)
    opt_state = optimizer.init(params)
    X, a = sampler(key, params)
    X = X.reshape(-1,X.shape[-1])
    E = tot_energy_NN(X, params, logpsi)
    grad = grad_energy(X, params, E, logpsi)

    def body_scan(carry, i):

        params, opt_state, grad, As, sigma, key = carry
        key, subkey = jax.random.split(key)
        #key = jax.random.split(key[0], num=key.shape[0])
        params, opt_state, grad, a, E = update_theta_gradient_NN(key, opt_state, optimizer, grad, sampler, params, logpsi)
        As = As.at[i].set(a)
        sigma = sigma.at[i].set(jnp.std(E))

        return (params, opt_state, grad, As, sigma, subkey), jnp.mean(E)

    steps = jnp.arange(0, T, 1)
    (params, opt_state, grad, As, sigma, key), Es = jax.lax.scan(body_scan, (params, opt_state, grad, As, sigma, key), steps)

    return params, opt_state, Es, As, sigma



def minimise_energy_mapped_NN(key, sampler, params, logpsi, T, lr, optim_func):

    #key = jax.random.split(key[0], num=key.shape[0])
    #key, subkey = jax.random.split(key)
    As = jnp.zeros(T)
    sigma = jnp.zeros(T)
    optimizer = optim_func(learning_rate=lr)
    opt_state = optimizer.init(params)
    X, a = sampler(key, params)
    X = X.reshape(-1,X.shape[-1])
    E = tot_energy_NN(X, params, logpsi)
    grad = grad_energy(X, params, E, logpsi)

    def body_scan(carry, i):

        params, opt_state, grad, As, sigma, key = carry
        key = jax.random.split(key[0], num=key.shape[0])
        params, opt_state, grad, a, E = update_theta_gradient_NN(key, opt_state, optimizer, grad, sampler, params, logpsi)
        As = As.at[i].set(a)
        sigma = sigma.at[i].set(jnp.std(E))

        return (params, opt_state, grad, As, sigma, key), jnp.mean(E)

    steps = jnp.arange(0, T, 1)
    (params, opt_state, grad, As, sigma, key), Es = jax.lax.scan(body_scan, (params, opt_state, grad, As, sigma, key), steps)

    return params, opt_state, Es, As, sigma
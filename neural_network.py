import jax
from jax import numpy as jnp
from flax import linen as nn
from typing import Tuple, Callable
import numpy as np
from netket.utils.types import NNInitFunc, DType
from jax.nn.initializers import (
    zeros,
    ones,
    lecun_normal,
    normal
)


class FullModel(nn.Module):
    
    layers_phi: int = 2
    layers_rho: int = 2
    width_phi: Tuple = (16,16)
    width_rho: Tuple = (16,1)
    sdim: int = 3
    activation: Callable = nn.activation.gelu
    initfunc: NNInitFunc = lecun_normal()

    @nn.compact
    def __call__(self, x):

        N = x.shape[-1]//self.sdim
        x = x.reshape(-1,N,self.sdim)
        
        #first sequence of activations
        for i in range(self.layers_phi):
            x = nn.Dense(features=self.width_phi[i], kernel_init=self.initfunc, param_dtype=np.float64)(x)
            if i == self.layers_phi-1:
                break
            
            x = self.activation(x) 
            
        x = jnp.sum(x, axis=-2)

        #second sequence of activations
        for i in range(self.layers_rho):
            x = nn.Dense(features=self.width_rho[i], kernel_init=self.initfunc, param_dtype=np.float64)(x)
            if i == self.layers_rho-1:
                break
            x = self.activation(x) 
        
        if x.shape[0] == 1:
            return x.reshape(-1)[0]

        return x.reshape(-1)

        #return jnp.sum(x, axis=-1)


class FullModel2(nn.Module):
    layers_phi: int = 2
    width_phi: Tuple = (16,1)
    sdim: int = 1
    activation: Callable = nn.activation.gelu
    initfunc: NNInitFunc = lecun_normal()

    @nn.compact
    def __call__(self, x):

        N = x.shape[-1]//self.sdim
        x = x.reshape(-1,N,self.sdim)
        
        #apply layers
        for i in range(self.layers_phi):
            x = nn.Dense(features=self.width_phi[i], kernel_init=self.initfunc, param_dtype=np.float64)(x)
            if i == self.layers_phi-1:
                break
            x = self.activation(x)
        
        x = jnp.sum(x, axis=-2)
         
        if x.shape[0] == 1:
            return x.reshape(-1)[0]

        return x.reshape(-1)
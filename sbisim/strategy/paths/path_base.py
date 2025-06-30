from abc import ABC, abstractmethod

import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap

class PathBase(ABC):
    def __init__(self):
        pass

    def snr(self, t):
        return 2 * (jnp.log(self.alpha(t)) - jnp.log(self.sigma(t)))

    def grad_snr(self, t):
        return vmap(grad(self.snr))(t)

    @abstractmethod
    def alpha(self, t):
        pass

    @abstractmethod
    def sigma(self, t):
        pass

    def grad_sigma(self, t):
        return vmap(grad(self.sigma))(t)

    def grad_alpha(self, t):
        return vmap(grad(self.alpha))(t)

    @abstractmethod
    def conditional_sample(self, key: jr.PRNGKey, theta1: jnp.ndarray, t: jnp.ndarray):
        pass

    @abstractmethod
    def conditional_vector_field(self, t, theta, theta1):
        pass
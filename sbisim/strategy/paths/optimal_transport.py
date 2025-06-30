from .path_base import PathBase

import jax.random as jr
import jax.numpy as jnp

class OptimalTransportPaths(PathBase):
    def __init__(self, sigma_min=1e-4):
        super().__init__()
        self.sigma_min = sigma_min

    def alpha(self, t):
        return t

    def mu(self, t, theta):
        return jnp.einsum("a,ab->ab", t, theta)

    def sigma(self, t):
        return 1 - (1-self.sigma_min) * t

    def conditional_sample(self, key: jr.PRNGKey, theta1: jnp.ndarray, t: jnp.ndarray):
        mu = self.mu(t, theta1)
        sigma = self.sigma(t)
        z = jr.normal(key, shape=theta1.shape)
        key = jr.split(key, 2)[0]
        return jnp.einsum("ab,a->ab", z, sigma) + mu, key

    def conditional_vector_field(self, t, theta, theta1):
        return jnp.einsum("ab,a->ab", (theta1 - (1-self.sigma_min) * theta),(1 / (1-(1-self.sigma_min)*t)))



from .path_base import PathBase

import jax.random as jr
import jax.numpy as jnp

class VPPaths(PathBase):

    def __init__(self, beta_min = 0.1, beta_max = 20):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max

    def T(self, t):
        return t * self.beta_min + 0.5 * (t ** 2) * (self.beta_max-self.beta_min)

    def alpha(self, t):
        return jnp.exp(-0.5 * self.T(1-t))

    def mu(self, t, theta):
        return theta

    def sigma(self, t):
        return jnp.sqrt(1 - self.alpha(t) ** 2)

    def conditional_sample(self, key: jr.PRNGKey, theta1: jnp.ndarray, t: jnp.ndarray):
        pass

    def conditional_vector_field(self, t, theta, theta1):
        pass



class VEPaths(PathBase):
    def __init__(self, sigma_max: float, sigma_min: float):
        super().__init__()
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

    def alpha(self, t):
        return jnp.ones_like(t, dtype=jnp.float32)

    def mu(self, t, theta):
        return theta

    def sigma(self, t):
        return self.sigma_max * jnp.power((self.sigma_min / self.sigma_max), t)

    def conditional_sample(self, key: jr.PRNGKey, theta1: jnp.ndarray, t: jnp.ndarray):
        mu = self.mu(t, theta1)
        sigma = self.sigma(t, theta1)
        z = jr.normal(key, shape=theta1.shape)
        key = jr.split(key, 2)[0]
        return jnp.einsum("ab,a->ab", z, sigma) + mu, key

    def conditional_vector_field(self, t, theta, theta1):
        sigma_t = self.sigma(t, theta)
        sigma_t = jnp.maximum(sigma_t, self.sigma_min)
        return jnp.einsum("ab,a->ab", (theta1 - theta), (1 / sigma_t))


class ExponentialDecay(PathBase):

    def __init__(self, alpha=10.0):
        super().__init__()
        self.alpha = alpha

    def alpha(self, t):
        return self.alpha

    def mu(self, t, theta):
        return theta

    def sigma(self, t, theta):
        return jnp.exp(-self.alpha * t)

    def conditional_sample(self, key: jr.PRNGKey, theta1: jnp.ndarray, t: jnp.ndarray):
        mu = self.mu(t, theta1)
        sigma = self.sigma(t, theta1)
        z = jr.normal(key, shape=theta1.shape)
        key = jr.split(key, 2)[0]
        return jnp.einsum("ab,a->ab", z, sigma) + mu, key

    def conditional_vector_field(self, t, theta, theta1):
        return self.alpha * (theta1 - theta)
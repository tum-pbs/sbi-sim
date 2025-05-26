from typing import Tuple

from flows.flowbase import FlowBlock
import jax.nn.initializers
import jax.numpy as jnp
import jax.random as jr

class ActNorm(FlowBlock):
    """
    ActNorm layer.

    [Kingma and Dhariwal, 2018.]
    """

    def setup(self):
        self.mu = self.param('mu', jax.nn.initializers.zeros, (self.dim_flow,))
        self.log_sigma = self.param('log_sigma', jax.nn.initializers.zeros, (self.dim_flow,))

    def forward_(self, x: jnp.ndarray, rng: jr.PRNGKey,
                 ldj: jnp.ndarray,
                 conditioning: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:
        z = x * jnp.exp(self.log_sigma) + self.mu
        log_det = jnp.sum(self.log_sigma)
        return z, ldj + log_det.reshape(-1), rng

    def inverse_(self, z: jnp.ndarray, rng: jr.PRNGKey,
                 ldj: jnp.ndarray,
                 conditioning: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:
        x = (z - self.mu) / jnp.exp(self.log_sigma)
        log_det = -jnp.sum(self.log_sigma)
        return x, ldj + log_det.reshape(-1), rng
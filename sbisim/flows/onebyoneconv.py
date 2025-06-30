from typing import Tuple

import jax.random as jr

from flows.flowbase import FlowBlock
import jax.scipy.linalg
import jax.numpy as jnp

class OneByOneConv(FlowBlock):
    """
    Invertible 1x1 convolution.

    [Kingma and Dhariwal, 2018.]
    """

    seed: int = 0

    def setup(self):

        self.rng = jr.PRNGKey(self.seed)

        W, _ = jax.scipy.linalg.qr(jr.normal(self.rng, (self.dim_flow, self.dim_flow)))
        P, L, U = jax.scipy.linalg.lu(W)

        self.P = self.param('P', lambda *args: P, (self.dim_flow, self.dim_flow))
        self.L = self.param('L', lambda *args: L, (self.dim_flow, self.dim_flow))
        self.S = self.param('S', lambda *args: jnp.diag(U), (self.dim_flow, self.dim_flow))
        self.U = self.param('U', lambda *args: jnp.triu(U, k=1), (self.dim_flow, self.dim_flow))
        self.W_inv = None

        if not self.W_inv:
            L = jnp.tril(self.L, k = -1) + \
                jnp.diag(jnp.ones(self.dim_flow))
            U = jnp.triu(self.U, k = 1)
            W = self.P @ L @ (U + jnp.diag(self.S))
            self.W_inv = jnp.linalg.inv(W)

    def forward_(self, x: jnp.ndarray, rng: jr.PRNGKey,
                 ldj: jnp.ndarray,
                 conditioning: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:
        L = jnp.tril(self.L, k = -1) + jnp.diag(jnp.ones(self.dim_flow))
        U = jnp.triu(self.U, k = 1)
        z = x @ self.P @ L @ (U + jnp.diag(self.S))
        log_det = jnp.sum(jnp.log(jnp.abs(self.S)))
        return z, ldj + log_det.reshape(-1), rng

    def inverse_(self, z: jnp.ndarray, rng: jr.PRNGKey,
                 ldj: jnp.ndarray,
                 conditioning: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:

        x = z @ self.W_inv
        log_det = - jnp.sum(jnp.log(jnp.abs(self.S)))
        return x, ldj + log_det.reshape(-1), rng
from typing import Sequence, Tuple

from flows.layers.actnorm import ActNorm
from flows.flowbase import FlowBase, NormalizingFlow
from flows.models.fcnn import FCNN

import flax.linen as nn

from flows.onebyoneconv import OneByOneConv
from flows.utils import uniform_min_max_init

import jax.numpy as jnp
import jax.random as jr

class MAF(FlowBase):
    """
    Masked auto-regressive flow.

    [Papamakarios et al. 2018]
    """

    hidden_dim : int = 8
    base_network = FCNN

    def setup(self):

        self.layers : Sequence[nn.Module] = []
        self.initial_param = self.param('initial_param',
                                        uniform_min_max_init(-jnp.sqrt(0.5), jnp.sqrt(0.5)),
                                        (2,))
        for i in range(1, self.dim_flow):
            self.layers += (self.base_network(i + self.dim_conditioning, 2, self.hidden_dim),)

    def forward_(self, x: jnp.ndarray, rng: jr.PRNGKey,
                 ldj: jnp.ndarray,
                 conditioning: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:
        z = jnp.zeros_like(x)
        log_det = jnp.zeros(z.shape[0])
        for i in range(self.dim_flow):
            if i == 0:
                mu, alpha = self.initial_param[0], self.initial_param[1]
            else:
                layer_input = jnp.concatenate([x[:, :i], conditioning], axis=-1)
                out = self.layers[i - 1](layer_input)
                mu, alpha = out[:, 0], out[:, 1]
            z = z.at[:, i].set((x[:, i] - mu) / jnp.exp(alpha))
            log_det -= alpha
        return jnp.flip(z, axis=1), ldj + log_det.reshape(-1), rng

    def inverse_(self, z: jnp.ndarray, rng: jr.PRNGKey,
                 ldj: jnp.ndarray,
                 conditioning: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:
        x = jnp.zeros_like(z)
        log_det = jnp.zeros(z.shape[0])
        z = jnp.flip(z, axis=1)
        for i in range(self.dim_flow):
            if i == 0:
                mu, alpha = self.initial_param[0], self.initial_param[1]
            else:
                layer_input = jnp.concatenate([x[:, :i], conditioning], axis=-1)
                out = self.layers[i - 1](layer_input)
                mu, alpha = out[:, 0], out[:, 1]
            x = x.at[:, i].set(mu + jnp.exp(alpha) * z[:, i])
            log_det += alpha
        return x, ldj + log_det.reshape(-1), rng

class StackedMAF(NormalizingFlow):
    """
    Stacked masked auto-regressive flow.
    [Papamakarios et al. 2018]
    """

    steps: int = 6
    hidden_dim: int = 256
    use_actnorm: bool = False
    use_onebyoneconv: bool = False
    seed: int = 0

    def setup(self):

        self.flows: Sequence[FlowBase] = []

        seed = self.seed

        for i in range(self.steps):

            self.flows += (MAF(self.dim_flow, self.dim_conditioning,
                               self.import_samples,
                               hidden_dim=self.hidden_dim),)

            if self.use_onebyoneconv:
                self.flows += (OneByOneConv(self.dim_flow, self.dim_conditioning,
                                            seed),)
                seed += 1

            if self.use_actnorm:
                self.flows += (ActNorm(self.dim_flow, self.dim_conditioning),)


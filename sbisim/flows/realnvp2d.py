from typing import Optional, Tuple, Sequence

import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn

from flows.models.fcnn import FCNN
from flows.flowbase import FlowBlock, NormalizingFlow

kernel_init_fn = nn.initializers.glorot_normal
bias_init_fn = nn.initializers.normal

def sample_n01(rng, N):
  D = 2
  return jr.normal(rng, (N, D))
def log_prob_n01(x):
  return jnp.sum(-jnp.square(x)/2 - jnp.log(jnp.sqrt(2*jnp.pi)),axis=-1)


class NVPBlock2D(FlowBlock):

    hidden_dim: int
    flip: bool = False

    def setup(self):
        self.f = FCNN(int(self.dim_flow // 2) + self.dim_conditioning,
                      self.dim_flow, self.hidden_dim)

    def shift_and_log_scale_fn(self, x1):
        s = self.f(x1)
        shift, log_scale = jnp.split(s, 2, axis=1)
        return shift, log_scale

    def forward_(self, x: jnp.ndarray, rng: jr.PRNGKey,
               ldj: jnp.ndarray,
               conditioning: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:

        d = x.shape[-1] // 2
        x1, x2 = x[:, :d], x[:, d:]
        if self.flip:
            x2, x1 = x1, x2

        fcnn_input = jnp.concatenate([x1, conditioning], axis=-1)

        shift, log_scale = self.shift_and_log_scale_fn(fcnn_input)
        y2 = x2 * jnp.exp(log_scale) + shift
        if self.flip:
            x1, y2 = y2, x1
        z = jnp.concatenate([x1, y2], axis=-1)
        return z, ldj + log_scale.reshape(-1), rng

    def forward(self, x: jnp.ndarray, rng: jr.PRNGKey,
                ldj: Optional[jnp.ndarray] = None,
                conditioning: Optional[jnp.ndarray] = None) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:

        if ldj is None:
            ldj = jnp.zeros(x.shape[0])

        if conditioning is None:
            conditioning = jnp.zeros((x.shape[0], self.dim_conditioning))

        return self.forward_(x, rng, ldj, conditioning)

    def inverse_(self, z: jnp.ndarray, rng: jr.PRNGKey,
               ldj: jnp.ndarray,
               conditioning: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:

        d = z.shape[-1] // 2
        y1, y2 = z[:, :d], z[:, d:]
        if self.flip:
            y1, y2 = y2, y1

        fcnn_input = jnp.concatenate([y1, conditioning], axis=-1)

        shift, log_scale = self.shift_and_log_scale_fn(fcnn_input)
        x2 = (y2 - shift) * jnp.exp(-log_scale)
        if self.flip:
            y1, x2 = x2, y1
        x = jnp.concatenate([y1, x2], axis=-1)
        return x, ldj - log_scale.reshape(-1), rng


    def inverse(self, z: jnp.ndarray, rng: jr.PRNGKey,
                ldj: Optional[jnp.ndarray] = None,
                conditioning: Optional[jnp.ndarray] = None) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:

        if ldj is None:
            ldj = jnp.zeros(z.shape[0])

        if conditioning is None:
            conditioning = jnp.zeros((z.shape[0], self.dim_conditioning))

        return self.inverse_(z, rng, ldj, conditioning)


class RealNVP2D(NormalizingFlow):

    steps: int = 6
    hidden_dim: int = 256

    def setup(self):

        self.flows: Sequence[FlowBlock] = []

        flip = False

        for i in range(self.steps):
            self.flows += (NVPBlock2D(self.dim_flow, self.dim_conditioning,
                                      self.hidden_dim, flip=flip),)
            flip = not flip


from functools import partial
from typing import Tuple, Any, Dict, Union, List

import jax.random as jr
import jax.numpy as jnp
import jax
from jax.lax import stop_gradient
from jaxtyping import PyTree
from .paths import sample_log_std, sample_time

from .strategy import Strategy

from jax import jit, value_and_grad

from utils import instantiate_from_config, generate_apply_rngs

from abc import ABC

from flax import linen as nn

class DDPM(Strategy, ABC):

    start_time: float = 0.0
    end_time: float = 1.0

    def __init__(self, dim_flow: int, model: Dict, **kwargs):
        super().__init__()

        self.dim_flow = dim_flow
        self.model: nn.Module = instantiate_from_config(model)

        self.T = 1000
        self.beta_0 = 0.0001
        self.beta_T = 0.02

        self.betas = jnp.asarray(self._betas(beta_1=self.beta_0, beta_T=self.beta_T, T=self.T))
        self.alphas = jnp.asarray(self._alphas(self.betas))
        self.alpha_bars = jnp.asarray(self._alpha_bars(self.alphas))

        self.opt = None
        self.initialized = False

    def get_flow_dimension(self) -> int:
        return self.dim_flow

    # based on https://github.com/andylolu2/jax-diffusion/blob/main/jax_diffusion/diffusion.py

    @property
    def steps(self) -> int:
        return self.T

    def timesteps(self, steps: int):
        timesteps = jnp.linspace(0, self.steps, steps + 1)
        timesteps = jnp.rint(timesteps).astype(jnp.int32)
        return timesteps[::-1]

    @partial(jax.jit, static_argnums=(0,))
    def forward_sample(self, x_0: jnp.ndarray, rng: jr.PRNGKey):
        """See algorithm 1 in https://arxiv.org/pdf/2006.11239.pdf"""
        rng1, rng2 = jr.split(rng)
        t = jr.randint(rng1, (len(x_0), 1), 0, self.steps)
        x_t, eps = self.sample_q(x_0, t, rng2)
        t = t.astype(x_t.dtype)
        return x_t, t, eps

    def sample_q(self, x_0: jnp.ndarray, t: jnp.ndarray, rng: jr.PRNGKey):
        """Samples x_t given x_0 by the q(x_t|x_0) formula."""
        # (bs, 1)
        alpha_t_bar = self.alpha_bars[t]

        eps = jr.normal(rng, shape=x_0.shape, dtype=x_0.dtype)
        x_t = (alpha_t_bar ** 0.5) * x_0 + ((1 - alpha_t_bar) ** 0.5) * eps
        return x_t, eps

    @classmethod
    def _betas(cls, beta_1: float, beta_T: float, T: int) -> jnp.ndarray:
        return jnp.linspace(beta_1, beta_T, T, dtype=jnp.float32)

    @classmethod
    def _alphas(cls, betas) -> jnp.ndarray:
        return 1 - betas

    @classmethod
    def _alpha_bars(cls, alphas) -> jnp.ndarray:
        return jnp.cumprod(alphas)

    @staticmethod
    def expand_t(t: int, x: jnp.ndarray):
        return jnp.full((len(x), 1), t, dtype=x.dtype)

    def loss_fn(self, params: PyTree, rng: jr.PRNGKey, batch: PyTree) \
            -> Tuple[jnp.ndarray, jr.PRNGKey]:

        data = batch["parameters"]

        x_t, t, eps = self.forward_sample(data, rng)
        rng = jr.split(rng)[0]

        rng_apply, rng = generate_apply_rngs(rng)

        pred = self.model.apply({'params': params},
                                             t / self.T, x_t,
                                             None)

        loss = jnp.mean((pred - eps) ** 2, axis=1)

        return jnp.mean(loss), rng

    def setup(self, opt, example_data: PyTree, key: jr.PRNGKey, batch_size: int) -> Tuple[PyTree, jr.PRNGKey]:

        self.opt = opt
        self.batch_size = batch_size

        rng, init_rng, model_rng = jr.split(key, 3)

        t_in = jnp.zeros((batch_size, 1))

        params_rng, dropout_rng, drop_path_rng, dropout_rng = jr.split(init_rng, 4)

        init_dict = {'params': model_rng, 'drop_path': init_rng, 'dropout': dropout_rng}
        init_model = self.model.init(init_dict, t_in, example_data["parameters"],
                                     None)

        self.opt.init(init_model['params'])

        self.initialized = True

        return init_model, rng

    @partial(jit, static_argnums=(0,))
    def train_step(self, i: int, opt_state: PyTree, rng: jr.PRNGKey, logs: Dict[str, Any],
                   batch: PyTree) -> Tuple[PyTree, jr.PRNGKey, Dict[str, Any]]:

        (loss, rng), grads = value_and_grad(self.loss_fn, has_aux=True)(
            self.opt.get_params_from_state(opt_state), rng, batch)

        opt_state = self.opt.update(i, opt_state, grads)

        logs["train/loss"] = jnp.mean(loss)

        return opt_state, rng, logs

    @partial(jit, static_argnums=(0, 5,))
    def eval_step(self, params: PyTree, rng: jr.PRNGKey, logs: Dict[str, Any],
                  batch: PyTree, testing: bool) -> Tuple[jr.PRNGKey, Dict[str, Any]]:

        loss, rng = self.loss_fn(params, rng, batch)

        logs["val/loss"] = jnp.mean(loss)

        return rng, logs

    @partial(jit, static_argnums=(0,))
    def _forward(self, params: PyTree, x: jnp.ndarray, rng: jr.PRNGKey, *args, **kwargs) \
            -> Tuple[PyTree, jr.PRNGKey]:

        return self.sampler.forward(
            x, self.scaled_model, {'params': params}, rng, *args, **kwargs
        )

    @partial(jit, static_argnums=(0,))
    def _compute_likelihood(self, params: PyTree, x: jnp.ndarray, rng: jr.PRNGKey, *args, **kwargs) \
            -> Tuple[PyTree, jr.PRNGKey]:

        return self.sampler.compute_likelihood(
            x, self.scaled_model, {'params': params}, rng, *args, **kwargs
        )

    @partial(jit, static_argnums=(0, 2))
    def _sample(self, params: PyTree, num_samples: int, rng: jr.PRNGKey,
                *args, **kwargs) -> Tuple[PyTree, jr.PRNGKey]:

        return self.sampler.sample(
            num_samples, self.dim_flow, self.scaled_model, {'params': params},
            rng, *args, **kwargs
        )

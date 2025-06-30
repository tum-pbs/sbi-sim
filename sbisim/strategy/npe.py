from functools import partial
from typing import Tuple, Any, Dict

import flax.linen as nn
import jax.random as jr
import jax.numpy as jnp
from jaxtyping import PyTree
from omegaconf import OmegaConf

from jax import jit, value_and_grad

from ..utils import instantiate_from_config
from ..strategy import Strategy
from ..flows import FlowBase

from abc import ABC


class NeuralPosteriorEstimation(Strategy, ABC):

    def __init__(self, model: OmegaConf):
        super().__init__()

        self.model: FlowBase = instantiate_from_config(model)

        self.opt = None
        self.initialized = False

    def get_flow_dimension(self) -> int:
        return self.model.dim_flow

    def loss_fn(self, params: PyTree, rng: jr.PRNGKey, batch: PyTree) \
            -> Tuple[jnp.ndarray, jr.PRNGKey]:

        nll, rng = self.model.apply({'params': params}, batch["parameters"], rng,
                                    batch["weighting"], batch["conditioning"], testing=False, bpd=False)

        return jnp.mean(nll), rng

    def setup(self, opt, example_data: PyTree, key: jr.PRNGKey, batch_size: int) -> Tuple[PyTree, jr.PRNGKey]:
        self.opt = opt

        self.batch_size = batch_size

        rng, init_rng, model_rng = jr.split(key, 3)

        init_jit = jit(self.model.init)

        init_ = init_jit(init_rng, example_data["parameters"], model_rng,
                                example_data["weighting"], example_data["conditioning"])

        self.opt.init(init_['params'])

        self.initialized = True

        return init_, rng

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
    def _forward(self, params: PyTree, x: jnp.ndarray, rng: jr.PRNGKey, *args, **kwargs) -> Tuple[PyTree, jr.PRNGKey]:

        out = self.model.apply({'params': params}, x, rng, method=self.model.__class__.forward,
                             *args, **kwargs)

        return {'samples': out[0], 'likelihood': out[1]}, out[2]

    @partial(jit, static_argnums=(0,))
    def _compute_likelihood(self, params: PyTree, x: jnp.ndarray, rng: jr.PRNGKey, *args, **kwargs) -> Tuple[PyTree, jr.PRNGKey]:

        out = self.model.apply({'params': params}, x, rng, method=self.model.compute_likelihood,
                               *args, **kwargs)

        return {'likelihood': out[0]}, out[1]

    @partial(jit, static_argnums=(0,2))
    def _sample(self, params: PyTree, num_samples: int, rng: jr.PRNGKey, *args, **kwargs) -> Tuple[PyTree, jr.PRNGKey]:

        out = self.model.apply({'params': params}, num_samples, rng, method=self.model.sample,
                                    *args, **kwargs)

        return {'samples': out[0], 'likelihood': out[1]}, out[2]
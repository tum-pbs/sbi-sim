from functools import partial
from typing import Tuple, Any, Dict, Union

import jax.random as jr
import jax.numpy as jnp
from jaxtyping import PyTree
from omegaconf import OmegaConf

from ..flows.cnf import ContinuousNormalizingFlow
from .improved_inference import BaseSampler
from .paths import sample_t
from .paths.diffusion import VEPaths, ExponentialDecay, VPPaths
from .paths.optimal_transport import OptimalTransportPaths
from .paths.path_base import PathBase
from .strategy import Strategy

from jax import jit, value_and_grad

from ..utils import instantiate_from_config

from abc import ABC

from flax import linen as nn

def get_paths(paths: Dict) -> PathBase:

    name = paths["name"]
    config = paths["params"]

    if name == "optimal_transport":
        return OptimalTransportPaths(**config)
    elif name == "ve_diffusion":
        return VEPaths(**config)
    elif name == "vp_diffusion":
        return VPPaths(**config)
    elif name == "exponential_decay":
        return ExponentialDecay(**config)
    else:
        raise ValueError(f"Flow matching path configuration {name} not recognized")

class FlowMatching(Strategy, ABC):

    import_samples: bool = 10

    def __init__(self, dim_flow: int, dim_conditioning: Union[int, Tuple[int]], model: OmegaConf, paths: Dict,
                 sampler: Dict):
        super().__init__()

        self.dim_flow = dim_flow
        self.dim_conditioning = dim_conditioning

        self.model: nn.Module = instantiate_from_config(model)

        self.sampler: BaseSampler = instantiate_from_config(sampler)

        self.paths = get_paths(paths)
        self.opt = None
        self.initialized = False

    def get_flow_dimension(self) -> int:
        return self.dim_flow

    def loss_fn(self, params: PyTree, rng: jr.PRNGKey, batch: PyTree) \
            -> Tuple[jnp.ndarray, jr.PRNGKey]:

        parameters = batch["parameters"]

        t, rng = sample_t(rng, parameters.shape[0])

        conditional_path, rng = self.paths.conditional_sample(rng, parameters, t)

        vector_field = self.paths.conditional_vector_field(t, conditional_path, parameters)

        regressed_field = self.model.apply({'params': params},
                                                 jnp.expand_dims(t, axis=1), conditional_path,
                                                 batch["conditioning"])

        loss = jnp.mean(jnp.sum((vector_field - regressed_field) ** 2, axis=1))

        return loss, rng

    def setup(self, opt, example_data: PyTree, key: jr.PRNGKey, batch_size: int) -> Tuple[PyTree, jr.PRNGKey]:

        self.opt = opt
        self.batch_size = batch_size

        rng, init_rng, model_rng = jr.split(key, 3)

        t, _ = sample_t(rng, example_data["parameters"].shape[0])

        init_model = self.model.init(init_rng, t, example_data["parameters"],
                                     example_data["conditioning"])

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
            x, self.model, {'params': params}, rng, *args, **kwargs
        )

    @partial(jit, static_argnums=(0,))
    def _compute_likelihood(self, params: PyTree, x: jnp.ndarray, rng: jr.PRNGKey, *args, **kwargs) \
            -> Tuple[PyTree, jr.PRNGKey]:
        return self.sampler.compute_likelihood(
            x, self.model, {'params': params}, rng, *args, **kwargs
        )

    @partial(jit, static_argnums=(0, 2))
    def _sample(self, params: PyTree, num_samples: int, rng: jr.PRNGKey,
                *args, **kwargs) -> Tuple[PyTree, jr.PRNGKey]:
        return self.sampler.sample(
            num_samples, self.dim_flow, self.model, {'params': params},
            rng, *args, **kwargs
        )

class CNFWrapper(ContinuousNormalizingFlow):
    solver_name: str = 'dopri5'
    mode: str = 'exact'
    init_stepsize: int = 0.01
    stepsize_controller_name: str = 'constant'
    model_config: OmegaConf = None

    def setup(self):

        super().setup()
        self.model = instantiate_from_config(self.model_config)
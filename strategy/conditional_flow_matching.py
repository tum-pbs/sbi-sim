from functools import partial
from typing import Tuple, Any, Dict, Union, List

import jax.random as jr
import jax.numpy as jnp
from jax.lax import stop_gradient
from jaxtyping import PyTree

from .improved_inference import BaseSampler
from .paths import sample_log_std, sample_time
from .paths.path_base import PathBase
from .strategy import Strategy
from .distributions.base_distribution import BaseDistribution

from jax import jit, value_and_grad

from utils import instantiate_from_config, generate_apply_rngs

from abc import ABC

from flax import linen as nn

def get_weighting_fun(name: str):

    def identity(t, weighting):
        return weighting

    def one_weighting(t, weighting):
        return jnp.ones_like(t)

    if name == "identity":
        return identity
    elif name == "one":
        return one_weighting
    else:
        raise ValueError(f"Unknown weighting function: {name}")

class ConditionalFlowMatching(Strategy, ABC):

    import_samples: bool = 10
    max_weighting: float = 1e3
    start_time: float = 0.0
    end_time: float = 1.0

    def get_loss_fn(self, loss_type: str):

        if loss_type == "x1":
            return self.x_loss_fm_ot_fn
        elif loss_type == "flow":
            return self.flow_loss_fn
        elif loss_type == "lipman":
            return self.lipman_loss_fn
        elif loss_type == "fm_ot":
            return self.fm_ot_loss
        elif loss_type == "self_cond":
            # TODO ?
            return self.self_cond_loss_fn
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def get_scaled_model(self, scaling_type: str):
        if scaling_type == 'x1':
            return ScaledModelFromX(model=self.model, path=self.path)
        elif scaling_type == 'none':
            return ScaledModelWrapper(model=self.model, path=self.path)
        else:
            raise ValueError(f"Unknown scaling type: {scaling_type}")

    def __init__(self, dim_flow: int, dim_conditioning: Union[int, Tuple[int]], model: Dict,
                 schedule: Dict, sampler: Dict, prior: Dict, weighting: str = "identity", loss_type: str = "x1",
                 scaling_type: str = 'x1', bandwidth: float = 0.5,
                 time_alpha: float = 0.0, **kwargs):
        super().__init__()

        self.dim_flow = dim_flow
        self.dim_conditioning = dim_conditioning
        self.model: nn.Module = instantiate_from_config(model)

        self.bandwidth = bandwidth
        self.time_alpha = time_alpha

        self.loss_type = loss_type
        self.loss_fn = self.get_loss_fn(loss_type)
        self.path = instantiate_from_config(schedule)
        self.scaling_type = scaling_type
        self.scaled_model = self.get_scaled_model(scaling_type)
        self.prior: BaseDistribution = instantiate_from_config(prior)
        self.sampler: BaseSampler = instantiate_from_config(sampler)
        self.weighting_function = get_weighting_fun(weighting)

        self.opt = None
        self.initialized = False

    def get_flow_dimension(self) -> int:
        return self.dim_flow

    def loss_fn(self, params: PyTree, rng: jr.PRNGKey, batch: PyTree) \
            -> Tuple[jnp.ndarray, jr.PRNGKey]:

        return self.get_loss_fn(self.loss_type)(params, rng, batch)

    def x_loss_fn(self, params: PyTree, rng: jr.PRNGKey, batch: PyTree) \
            -> Tuple[jnp.ndarray, jr.PRNGKey]:

        # sample time
        t, rng = sample_time(rng, batch["parameters"].shape[0], t0=self.start_time, t1=self.end_time,
                             alpha=self.time_alpha)

        x_0, rng = self.prior.sample(rng, batch["parameters"].shape[0])

        mu_t = (jnp.einsum('ab,a->ab', batch["parameters"], t) +
                jnp.einsum('ab,a->ab', x_0, 1 - t))

        x = mu_t + self.bandwidth * jr.normal(rng, shape=batch["parameters"].shape)
        rng = jr.split(rng)[0]

        rng_apply, rng = generate_apply_rngs(rng)

        x1_prediction = self.model.apply({'params': params},
                                           jnp.expand_dims(t, axis=1), x,
                                           batch["conditioning"], rngs=rng_apply)

        # flow_prediction = jnp.einsum('ab,a->ab', x1_prediction - x, 1 / (1 - t))
        # loss = jnp.sum((flow_prediction - (batch['parameters'] - x_0)) ** 2, axis=1)

        loss = jnp.sum((x1_prediction - batch['parameters']) ** 2, axis=1)

        weighting = jnp.ones(batch["parameters"].shape[0])
        weighting = self.weighting_function(t, weighting)
        loss = jnp.multiply(weighting, loss)

        return jnp.mean(loss), rng

    def x_loss_fm_ot_fn(self, params: PyTree, rng: jr.PRNGKey, batch: PyTree) \
            -> Tuple[jnp.ndarray, jr.PRNGKey]:

        sigma_min = 0.0001

        # sample time
        t, rng = sample_time(rng, batch["parameters"].shape[0],
                             t0=self.start_time, t1=self.end_time - 0.001,
                             alpha=self.time_alpha, eps=0.0)

        # sample noise
        epsilon = jr.normal(rng, shape=batch["parameters"].shape)
        noise = jnp.einsum("ab,a->ab", epsilon, (1 - (1 - sigma_min) * t))

        # get x_t
        mu_x = jnp.einsum('a, ab->ab', t, batch["parameters"])
        x_t = mu_x + noise

        rng_apply, rng = generate_apply_rngs(rng)

        x1_prediction = self.model.apply({'params': params},
                                           jnp.expand_dims(t, axis=1), x_t,
                                           batch["conditioning"], rngs=rng_apply)

        # flow_prediction = jnp.einsum('ab,a->ab', x1_prediction - x, 1 / (1 - t))
        # loss = jnp.sum((flow_prediction - (batch['parameters'] - x_0)) ** 2, axis=1)

        loss = jnp.mean((x1_prediction - batch['parameters']) ** 2, axis=1)

        loss = jnp.multiply(loss, 1 / (1 - t))

        weighting = jnp.ones(batch["parameters"].shape[0])
        weighting = self.weighting_function(t, weighting)
        loss = jnp.multiply(weighting, loss)

        return jnp.mean(loss), rng

    def lipman_loss_fn(self, params: PyTree, rng: jr.PRNGKey, batch: PyTree) \
            -> Tuple[jnp.ndarray, jr.PRNGKey]:

        # sample time
        t, rng = sample_time(rng, batch["parameters"].shape[0],
                             t0=self.start_time, t1=self.end_time,
                             alpha=self.time_alpha, eps=0.0)

        # get all relevant quantities: x_t = alpha_t * x_1 + sigma_t * epsilon
        alpha = self.path.alpha(t)
        grad_alpha = self.path.grad_alpha(t)
        sigma = self.path.sigma(t)
        grad_sigma = self.path.grad_sigma(t)

        # sample noise
        epsilon = jr.normal(rng, shape=batch["parameters"].shape)
        noise = jnp.einsum("ab,a->ab", epsilon, sigma)

        # get x_t
        mu_x = jnp.einsum('a, ab->ab', alpha, batch["parameters"])
        x = mu_x + noise

        rng_apply, rng = generate_apply_rngs(rng)

        # prediction of flow
        flow_prediction = self.model.apply({'params': params},
                                           jnp.expand_dims(t, axis=1), x,
                                           batch["conditioning"], rngs=rng_apply)

        grad_mu_x = jnp.einsum('ab,a->ab', batch["parameters"], grad_alpha)

        w = jnp.multiply(grad_sigma, 1 / sigma)

        u_t = jnp.einsum("ab,a->ab", (x - mu_x), w) + grad_mu_x

        # not sure if we better use jnp.mean or jnp.sum here
        loss = jnp.mean((u_t - flow_prediction) ** 2, axis=1)

        weighting = self.weighting_function(t, jnp.ones_like(t))

        weighted_loss = jnp.multiply(weighting, loss)

        return jnp.mean(weighted_loss), rng

    def fm_ot_loss(self, params: PyTree, rng: jr.PRNGKey, batch: PyTree) \
            -> Tuple[jnp.ndarray, jr.PRNGKey]:

        sigma_min = 0.0001

        # sample time
        t, rng = sample_time(rng, batch["parameters"].shape[0],
                             t0=self.start_time, t1=self.end_time,
                             alpha=self.time_alpha, eps=0.0)

        # sample noise
        epsilon = jr.normal(rng, shape=batch["parameters"].shape)
        noise = jnp.einsum("ab,a->ab", epsilon, (1 - (1-sigma_min) * t))

        # get x_t
        mu_x = jnp.einsum('a, ab->ab', t, batch["parameters"])
        x_t = mu_x + noise

        rng_apply, rng = generate_apply_rngs(rng)

        # prediction of flow
        flow_prediction = self.model.apply({'params': params},
                                           jnp.expand_dims(t, axis=1), x_t,
                                           batch["conditioning"], rngs=rng_apply)

        target = batch['parameters'] - (1-sigma_min) * epsilon

        loss = jnp.mean((target - flow_prediction) ** 2, axis=1)

        return jnp.mean(loss), rng

    def flow_loss_fn(self, params: PyTree, rng: jr.PRNGKey, batch: PyTree) \
            -> Tuple[jnp.ndarray, jr.PRNGKey]:

        # sample time
        t, rng = sample_time(rng, batch["parameters"].shape[0], t0=self.start_time, t1=self.end_time,
                             alpha=self.time_alpha)

        x_0, rng = self.prior.sample(rng, batch["parameters"].shape[0])

        mu_t = (jnp.einsum('ab,a->ab', batch["parameters"], t) +
                jnp.einsum('ab,a->ab', x_0, 1 - t))

        x = mu_t + self.bandwidth * jr.normal(rng, shape=batch["parameters"].shape)

        rng_apply, rng = generate_apply_rngs(rng)

        flow_prediction = self.model.apply({'params': params},
                                         jnp.expand_dims(t, axis=1), x,
                                         batch["conditioning"], rngs=rng_apply)

        loss = jnp.sum((flow_prediction - (batch['parameters'] - x_0)) ** 2, axis=1)

        weighting = self.weighting_function(t, jnp.ones_like(t))

        weighted_loss = jnp.multiply(weighting, loss)

        return jnp.mean(weighted_loss), rng

    def setup(self, opt, example_data: PyTree, key: jr.PRNGKey, batch_size: int) -> Tuple[PyTree, jr.PRNGKey]:

        self.opt = opt
        self.batch_size = batch_size

        rng, init_rng, model_rng = jr.split(key, 3)

        t, _ = sample_time(rng, example_data["parameters"].shape[0],
                           t0=self.start_time, t1=self.end_time, alpha=self.time_alpha)

        t_in = jnp.expand_dims(t, axis=1)

        params_rng, dropout_rng, drop_path_rng, dropout_rng = jr.split(init_rng, 4)

        init_dict = {'params': model_rng, 'drop_path': init_rng, 'dropout': dropout_rng}
        init_model = self.model.init(init_dict, t_in, example_data["parameters"],
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

class ScaledModelFromX:

    def __init__(self, model, path):
        self.model = model
        self.path = path

    def apply(self, params, t, *args, **kwargs):

        x = args[0]

        x1 = self.model.apply(params, t, *args, **kwargs)

        drift = - jnp.einsum('ab,a->ab', x - x1, 1 / (1 - t))

        return drift

    def forward_inference(self, params, t, *args, **kwargs):

        x = args[0][0]

        if len(t.shape) < 2:
            t = jnp.expand_dims(t, axis=1)

        # TODO check this equation
        flow, x_pred = self.model.apply(params, t, *args, **kwargs, method=self.model.forward_inference)

        flow = jnp.einsum('ab,a->ab', flow - x, 1 / (1 - t[:,0]))

        return flow, x_pred



class ScaledModelWrapper:

    def __init__(self, model, path):
        self.model = model
        self.path = path

    def apply(self, params, t, *args, **kwargs):

        if len(t.shape) < 2:
            t = jnp.expand_dims(t, axis=1)

        return self.model.apply(params, t, *args, **kwargs)

    def forward_flow(self, params, t, *args, **kwargs):

        if len(t.shape) < 2:
            t = jnp.expand_dims(t, axis=1)

        return self.model.apply(params, t, *args, **kwargs, method=self.model.forward_flow)

    def forward(self, params, t, *args, **kwargs):

        if len(t.shape) < 2:
            t = jnp.expand_dims(t, axis=1)

        return self.apply(params, t, *args, **kwargs)

    def forward_inference(self, params, t, *args, **kwargs):

        if len(t.shape) < 2:
            t = jnp.expand_dims(t, axis=1)

        return self.apply(params, t, *args, **kwargs, method=self.model.forward_inference)


class SelfConditionedWrapper(nn.Module):

    model: Dict
    active: bool = True
    scaling_type: str = 'x1'

    def collate_condition(self, *args):
        return jnp.concatenate(args, axis=-1)

    def setup(self):
        self.network: nn.Module = instantiate_from_config(self.model)

    def forward_inference(self, t, x_self_conditioned, conditioning, *args, **kwargs):

        x, x_self = x_self_conditioned

        conditioning = self.collate_condition(conditioning, x_self)

        flow = self.network(t, x, conditioning, *args, **kwargs)

        if self.active:

            if self.scaling_type == 'x1':
                x_forward = flow
            else:
                x_forward = x + jnp.einsum('ab,a->ab', flow, 1 - t[:, 0])

        else:

            x_forward = jnp.zeros_like(x)

        return flow, x_forward

    def impl(self, t, x, conditioning, *args, train=True, **kwargs):


        if self.active:

            # get flow with zero conditioning
            stub = jnp.zeros(x.shape)
            conditioning_1 = self.collate_condition(conditioning, stub)
            flow = self.network(t, x, conditioning_1, *args, **kwargs)

            # predict x1 and stop gradient
            if self.scaling_type == 'x1':
                x_forward = flow
            else:
                x_forward = x + jnp.einsum('ab,a->ab', flow, 1 - t[:, 0])

            x_forward = stop_gradient(x_forward)

            if train:
                # randomly drop it
                dropout = self.make_rng('dropout')
                active = jr.uniform(dropout, shape=(t.shape[0],)) > 0.5
                x_forward = x_forward * active[:, None]

        else:

            x_forward = jnp.zeros_like(x)

        # get new conditioning based on estimated x1
        conditioning_2 = self.collate_condition(conditioning, x_forward)

        return self.network(t, x, conditioning_2, *args, **kwargs)

    def __call__(self, t, x, conditioning, **kwargs):
        return self.impl(t, x, conditioning, **kwargs)

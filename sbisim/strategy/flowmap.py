from functools import partial
from pathlib import Path
from typing import Tuple, Any, Dict, Union

import jax.random as jr
import jax.numpy as jnp
import jax.lax
from jaxtyping import PyTree
from omegaconf import OmegaConf
import numpy as np

from .flow_matching import get_paths
from .improved_inference import BaseSampler
from .paths import sample_t, sample_time

from .strategy import Strategy

from jax import jit, value_and_grad

from ..utils import instantiate_from_config, restore_checkpoint

from abc import ABC

from flax import linen as nn

model_config = {
    'class_dropout_prob': 0.1,
    'num_classes': 1000,
    'denoise_timesteps': 128,
    'cfg_scale': 4.0,
    'bootstrap_cfg': 0,
    'bootstrap_every': 8,
    'bootstrap_dt_bias': 0,
}

FLAGS = {
    'batch_size': 64,
    'epoch': -1,
    'max_epochs': -1,
    'model': model_config,
}

def get_target_fn(method):

    if method=='flow_matching':
        from .flowmaps.flow_matching import get_targets
    elif method=='consistency_distillation':
        from .flowmaps.consistency_distillation import get_targets
    elif method=='consistency_training':
        from .flowmaps.consistency_training import get_targets
    elif method=='livereflow':
        from .flowmaps.livereflow import get_targets
    elif method=='progressive':
        from .flowmaps.progressive import get_targets
    elif method=='shortcut':
        from .flowmaps.shortcut import get_targets
    else:
        raise ValueError('Unknown method: {}'.format(method))

    return get_targets

def load_teacher_params(teacher_params_weight_file):
    if teacher_params_weight_file is None:
        print('No teacher weight file provided.')
        return None
    else:
        print(f'Loading teacher weight file: {teacher_params_weight_file}')
        weight_file = Path(teacher_params_weight_file)
        checkpoint = restore_checkpoint(weight_file, type='best_val')
        return checkpoint['state'][0]


class FlowMap(Strategy, ABC):
    import_samples: bool = 10

    def __init__(self, dim_flow: int,
                 dim_conditioning: Union[int, Tuple[int]],
                 model: OmegaConf,
                 sampler: Dict,
                 method: str = 'flow_matching',
                 teacher_params_weight_file: str = None,
                 max_epochs: int = -1,
                 ):
        super().__init__()

        self.dim_flow = dim_flow
        self.dim_conditioning = dim_conditioning

        self.model: nn.Module = instantiate_from_config(model)

        self.sampler: BaseSampler = instantiate_from_config(sampler)

        self.method = method
        self.get_targets = get_target_fn(method)
        self.teacher_params = load_teacher_params(
            teacher_params_weight_file
        )

        self.opt = None
        self.initialized = False
        self.FLAGS = FLAGS
        self.FLAGS['max_epochs'] = max_epochs

    def get_flow_dimension(self) -> int:
        return self.dim_flow



    def get_param_dict(self, opt_state, epoch: int, max_epochs: int) -> PyTree:

        param_dict = {}

        model_ema_params = self.opt.get_ema_params_from_state(opt_state)
        model_params = self.opt.get_params_from_state(opt_state)
        teacher_params = None

        if self.method in ['consistency_training', 'shortcut', 'livereflow']:
            teacher_params = self.opt.get_ema_params_from_state(opt_state)
        elif self.method in ['consistency_distillation']:
            teacher_params = self.teacher_params
        elif self.method in ['progressive']:
            num_sections = jnp.log2(self.FLAGS['model']['denoise_timesteps']).astype(jnp.int32)

            teacher_params = jax.lax.cond(
                epoch % (max_epochs // num_sections) == 0,
                # lambda _: self.opt.get_ema_params_from_state(opt_state),
                lambda _: self.opt.get_params_from_state(opt_state),
                lambda _: self.opt.get_teacher_weights(opt_state),
                operand=None
            )
        elif self.method in ['flow_matching']:
            pass
        else:
            raise ValueError('Unknown method: {}'.format(self.method))

        param_dict['teacher_params'] = teacher_params
        param_dict['model_params'] = model_params
        param_dict['model_ema_params'] = model_ema_params

        return param_dict

    def loss_fn(self, model_params: PyTree, params_dict: PyTree,
                      rng: jr.PRNGKey, batch: PyTree, logs: PyTree) \
            -> Tuple[jnp.ndarray, jr.PRNGKey]:

        force_t = -1
        force_dt = -1

        x_t, v_t, t, y, dt_base, labels, info = self.get_targets(self.FLAGS, rng, self.model, params_dict,
                                                            batch, force_t, force_dt, epoch=logs['epoch'])
        rng, _ = jr.split(rng, 2)

        regressed_field = self.model.apply({'params': model_params},
                                           x_t, y, t, dt_base)

        loss = jnp.mean(jnp.mean((v_t - regressed_field) ** 2, axis=1))

        return loss, (rng, logs)

    def setup(self, opt, example_data: PyTree, key: jr.PRNGKey, batch_size: int) -> Tuple[PyTree, jr.PRNGKey]:
        self.opt = opt
        self.batch_size = batch_size

        rng, init_rng, model_rng = jr.split(key, 3)

        t, _ = sample_t(rng, example_data["parameters"].shape[0])
        t = jnp.expand_dims(t, 1)
        d = t

        init_model = self.model.init(init_rng, example_data["parameters"],
                                     example_data["conditioning"], t, d)

        self.opt.init(init_model['params'])

        self.initialized = True

        return init_model, rng

    @partial(jit, static_argnums=(0,))
    def train_step(self, i: int, opt_state: PyTree, rng: jr.PRNGKey, logs: Dict[str, Any],
                   batch: PyTree) -> Tuple[PyTree, jr.PRNGKey, Dict[str, Any]]:

        param_dict = self.get_param_dict(opt_state, logs['epoch'], logs['max_epochs'])
        self.opt.set_teacher_weights(opt_state, param_dict['teacher_params'])

        (loss, (rng, logs)), grads = value_and_grad(self.loss_fn, has_aux=True)(
            param_dict['model_params'], param_dict, rng, batch, logs)

        opt_state = self.opt.update(i, opt_state, grads)

        logs["train/loss"] = jnp.mean(loss)

        return opt_state, rng, logs

    @partial(jit, static_argnums=(0, 5,))
    def eval_step(self, opt_state: PyTree, rng: jr.PRNGKey, logs: Dict[str, Any],
                  batch: PyTree, testing: bool) -> Tuple[jr.PRNGKey, Dict[str, Any]]:

        param_dict = self.get_param_dict(opt_state, logs['epoch'], logs['max_epochs'])

        loss, (rng, logs) = self.loss_fn(param_dict['model_params'], param_dict, rng, batch, logs)

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

    @partial(jit, static_argnums=(0, 2), static_argnames=("num_steps",))
    def _sample(self, params: PyTree, num_samples: int, rng: jr.PRNGKey,
                *args, **kwargs) -> Tuple[PyTree, jr.PRNGKey]:
        return self.sampler.sample(
            num_samples, self.dim_flow, self.model, {'params': params},
            rng, *args, **kwargs
        )

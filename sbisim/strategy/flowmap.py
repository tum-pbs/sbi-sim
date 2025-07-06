from functools import partial
from typing import Tuple, Any, Dict, Union

import jax.random as jr
import jax.numpy as jnp
from jaxtyping import PyTree
from omegaconf import OmegaConf
import numpy as np

from .flow_matching import get_paths
from .improved_inference import BaseSampler
from .paths import sample_t

from .strategy import Strategy

from jax import jit, value_and_grad

from ..utils import instantiate_from_config

from abc import ABC

from flax import linen as nn

model_config = dict({
    'lr': 0.0001,
    'beta1': 0.9,
    'beta2': 0.999,
    'weight_decay': 0.1,
    'use_cosine': 0,
    'warmup': 0,
    'dropout': 0.0,
    'hidden_size': 64,  # change this!
    'patch_size': 8,  # change this!
    'depth': 2,  # change this!
    'num_heads': 2,  # change this!
    'mlp_ratio': 1,  # change this!
    'class_dropout_prob': 0.1,
    'num_classes': 1000,
    'denoise_timesteps': 128,
    'cfg_scale': 4.0,
    'target_update_rate': 0.999,
    'use_ema': 0,
    'use_stable_vae': 1,
    'sharding': 'dp',  # dp or fsdp.
    't_sampling': 'discrete-dt',
    'dt_sampling': 'uniform',
    'bootstrap_cfg': 0,
    'bootstrap_every': 8,
    'bootstrap_ema': 1,
    'bootstrap_dt_bias': 0,
    'train_type': 'shortcut'  # or naive.
})

FLAGS = OmegaConf.create({
    'batch_size': 64,
    'model': model_config,
})


def get_targets(FLAGS, key, model, params, batch,
                force_t=-1, force_dt=-1):
    label_key, time_key, noise_key = jr.split(key, 3)
    info = {}

    # 1) =========== Sample dt. ============
    bootstrap_batchsize = FLAGS.batch_size // FLAGS.model['bootstrap_every']
    log2_sections = np.log2(FLAGS.model['denoise_timesteps']).astype(np.int32)
    if FLAGS.model['bootstrap_dt_bias'] == 0:
        dt_base = jnp.repeat(log2_sections - 1 - jnp.arange(log2_sections), bootstrap_batchsize // log2_sections)
        dt_base = jnp.concatenate([dt_base, jnp.zeros(bootstrap_batchsize - dt_base.shape[0], )])
        num_dt_cfg = bootstrap_batchsize // log2_sections
    else:
        dt_base = jnp.repeat(log2_sections - 1 - jnp.arange(log2_sections - 2),
                             (bootstrap_batchsize // 2) // log2_sections)
        dt_base = jnp.concatenate([dt_base, jnp.ones(bootstrap_batchsize // 4), jnp.zeros(bootstrap_batchsize // 4)])
        dt_base = jnp.concatenate([dt_base, jnp.zeros(bootstrap_batchsize - dt_base.shape[0], )])
        num_dt_cfg = (bootstrap_batchsize // 2) // log2_sections
    force_dt_vec = jnp.ones(bootstrap_batchsize, dtype=jnp.float32) * force_dt
    dt_base = jnp.where(force_dt_vec != -1, force_dt_vec, dt_base)
    dt = 1 / (2 ** (dt_base))  # [1, 1/2, 1/4, 1/8, 1/16, 1/32]
    dt_base_bootstrap = dt_base + 1
    dt_bootstrap = dt / 2

    # 2) =========== Sample t. ============
    dt_sections = jnp.power(2, dt_base)  # [1, 2, 4, 8, 16, 32]
    t = jr.randint(time_key, (bootstrap_batchsize,), minval=0, maxval=dt_sections).astype(jnp.float32)
    t = t / dt_sections  # Between 0 and 1.
    force_t_vec = jnp.ones(bootstrap_batchsize, dtype=jnp.float32) * force_t
    t = jnp.where(force_t_vec != -1, force_t_vec, t)
    t_full = t[:, None, None, None]

    # 3) =========== Generate Bootstrap Targets ============

    x_samples, y_samples = batch['parameters'], batch['conditioning']
    labels = jnp.zeros(shape=(bootstrap_batchsize,), dtype=jnp.int32)

    x_1 = x_samples[:bootstrap_batchsize]
    x_0 = jr.normal(noise_key, x_1.shape)
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    bst_labels = labels[:bootstrap_batchsize]

    # call_model_fn = train_state.call_model if FLAGS.model['bootstrap_ema'] == 0 else train_state.call_model_ema

    if not FLAGS.model['bootstrap_cfg']:
        # v_b1 = call_model_fn(x_t, t, dt_base_bootstrap, bst_labels, train=False)
        v_b1 = model.apply({'params': params}, x_t, y_samples[:bootstrap_batchsize], t, dt_base_bootstrap,
                            train=False)
        t2 = t + dt_bootstrap
        x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
        x_t2 = jnp.clip(x_t2, -4, 4)
        # v_b2 = call_model_fn(x_t2, t2, dt_base_bootstrap, bst_labels, train=False)
        v_b2 = model.apply({'params': params}, x_t2, y_samples[:bootstrap_batchsize], t2, dt_base_bootstrap,
                            train=False)
        v_target = (v_b1 + v_b2) / 2
    else:
        x_t_extra = jnp.concatenate([x_t, x_t[:num_dt_cfg]], axis=0)
        t_extra = jnp.concatenate([t, t[:num_dt_cfg]], axis=0)
        dt_base_extra = jnp.concatenate([dt_base_bootstrap, dt_base_bootstrap[:num_dt_cfg]], axis=0)
        labels_extra = jnp.concatenate([bst_labels, jnp.ones(num_dt_cfg, dtype=jnp.int32) * FLAGS.model['num_classes']],
                                       axis=0)
        v_b1_raw = call_model_fn(x_t_extra, t_extra, dt_base_extra, labels_extra, train=False)
        v_b_cond = v_b1_raw[:x_1.shape[0]]
        v_b_uncond = v_b1_raw[x_1.shape[0]:]
        v_cfg = v_b_uncond + FLAGS.model['cfg_scale'] * (v_b_cond[:num_dt_cfg] - v_b_uncond)
        v_b1 = jnp.concatenate([v_cfg, v_b_cond[num_dt_cfg:]], axis=0)

        t2 = t + dt_bootstrap
        x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
        x_t2 = jnp.clip(x_t2, -4, 4)
        x_t2_extra = jnp.concatenate([x_t2, x_t2[:num_dt_cfg]], axis=0)
        t2_extra = jnp.concatenate([t2, t2[:num_dt_cfg]], axis=0)
        v_b2_raw = call_model_fn(x_t2_extra, t2_extra, dt_base_extra, labels_extra, train=False)
        v_b2_cond = v_b2_raw[:x_1.shape[0]]
        v_b2_uncond = v_b2_raw[x_1.shape[0]:]
        v_b2_cfg = v_b2_uncond + FLAGS.model['cfg_scale'] * (v_b2_cond[:num_dt_cfg] - v_b2_uncond)
        v_b2 = jnp.concatenate([v_b2_cfg, v_b2_cond[num_dt_cfg:]], axis=0)
        v_target = (v_b1 + v_b2) / 2

    v_target = jnp.clip(v_target, -4, 4)
    bst_v = v_target
    bst_dt = dt_base
    bst_t = t
    bst_xt = x_t
    bst_l = bst_labels

    # 4) =========== Generate Flow-Matching Targets ============

    labels_dropout = jr.bernoulli(label_key, FLAGS.model['class_dropout_prob'], (labels.shape[0],))
    labels_dropped = jnp.where(labels_dropout, FLAGS.model['num_classes'], labels)
    info['dropped_ratio'] = jnp.mean(labels_dropped == FLAGS.model['num_classes'])

    # Sample t.
    t = jr.randint(time_key, (x_samples.shape[0],), minval=0, maxval=FLAGS.model['denoise_timesteps']).astype(
        jnp.float32)
    t /= FLAGS.model['denoise_timesteps']
    force_t_vec = jnp.ones(x_samples.shape[0], dtype=jnp.float32) * force_t
    t = jnp.where(force_t_vec != -1, force_t_vec, t)  # If force_t is not -1, then use force_t.
    t_full = t[:, None, None, None]  # [batch, 1, 1, 1]

    # Sample flow pairs x_t, v_t.
    x_0 = jr.normal(noise_key, x_samples.shape)
    x_1 = x_samples
    x_t = x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    v_t = v_t = x_1 - (1 - 1e-5) * x_0
    dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
    dt_base = jnp.ones(x_samples.shape[0], dtype=jnp.int32) * dt_flow

    # ==== 5) Merge Flow+Bootstrap ====
    bst_size = FLAGS.batch_size // FLAGS.model['bootstrap_every']
    bst_size_data = FLAGS.batch_size - bst_size
    x_t = jnp.concatenate([bst_xt, x_t[:bst_size_data]], axis=0)
    t = jnp.concatenate([bst_t, t[:bst_size_data]], axis=0)
    dt_base = jnp.concatenate([bst_dt, dt_base[:bst_size_data]], axis=0)
    v_t = jnp.concatenate([bst_v, v_t[:bst_size_data]], axis=0)
    labels_dropped = jnp.concatenate([bst_l, labels_dropped[:bst_size_data]], axis=0)
    info['bootstrap_ratio'] = jnp.mean(dt_base != dt_flow)

    info['v_magnitude_bootstrap'] = jnp.sqrt(jnp.mean(jnp.square(bst_v)))
    info['v_magnitude_b1'] = jnp.sqrt(jnp.mean(jnp.square(v_b1)))
    info['v_magnitude_b2'] = jnp.sqrt(jnp.mean(jnp.square(v_b2)))

    return x_t, v_t, t, y_samples, dt_base, labels_dropped, info


class FlowMap(Strategy, ABC):
    import_samples: bool = 10

    def __init__(self, dim_flow: int,
                 dim_conditioning: Union[int, Tuple[int]],
                 model: OmegaConf,
                 paths: Dict,
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

        force_t = -1
        force_dt = -1

        params_ema = self.opt.ema.get_params()

        x_t, v_t, t, y, dt_base, labels, info = get_targets(FLAGS, rng, self.model, params_ema,
                                                            batch, force_t, force_dt)

        regressed_field = self.model.apply({'params': params},
                                           t, dt_base, x_t, y)

        loss = jnp.mean(jnp.sum((v_t - regressed_field) ** 2, axis=1))

        return loss, rng

    def setup(self, opt, example_data: PyTree, key: jr.PRNGKey, batch_size: int) -> Tuple[PyTree, jr.PRNGKey]:
        self.opt = opt
        self.batch_size = batch_size

        rng, init_rng, model_rng = jr.split(key, 3)

        t, _ = sample_t(rng, example_data["parameters"].shape[0])
        t = jnp.expand_dims(t, 1)
        d = t

        init_model = self.model.init(init_rng, t, d, example_data["parameters"],
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

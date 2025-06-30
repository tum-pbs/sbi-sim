from typing import Tuple

import jax.random as jr
import jax.numpy as jnp

def sample_t(rng:jr.PRNGKey, dim: int = 1, t0: float = 0.0, t1: float = 1.0) -> Tuple[jnp.ndarray, jr.PRNGKey]:

    t = t0 + (t1 - t0) * jr.uniform(rng, (dim,))
    rng = jr.split(rng)[0]
    return t, rng

def sample_time(rng: jr.PRNGKey, num: int = 1, t0: float = 0.0,
                t1: float = 1.0, eps: float = 2e-5, alpha: float = 0.0) -> Tuple[jnp.ndarray, jr.PRNGKey]:

    t0 += eps
    t1 -= eps

    times = jnp.linspace(t0, t1, num+1)[:num]

    uniform = jr.uniform(rng, (num,), minval=0, maxval=((t1 - t0) / num))

    rng = jr.split(rng)[0]

    time_samples = times + uniform

    time_samples = jnp.power(time_samples, 1 / (1 + alpha))

    return time_samples, rng

def sample_log_std(rng: jr.PRNGKey, num: int = 1, min_log_std: float = -4, max_log_std: float = 1) \
        -> Tuple[jnp.ndarray, jr.PRNGKey]:

    intervals = jnp.linspace(min_log_std, max_log_std, num+1)[:num]

    uniform = jr.uniform(rng, (num,), minval=0, maxval=((max_log_std - min_log_std) / num))

    rng = jr.split(rng)[0]

    log_stds = intervals + uniform

    return log_stds, rng
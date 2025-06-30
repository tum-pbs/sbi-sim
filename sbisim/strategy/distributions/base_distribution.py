from abc import abstractmethod, ABC
from typing import Tuple

import jax.random as jr
import jax.numpy as jnp

class BaseDistribution(ABC):

    @abstractmethod
    def sample(self, rng: jr.PRNGKey, shape: Tuple[int]) -> Tuple[jnp.ndarray, jr.PRNGKey]:
        pass

    @abstractmethod
    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        pass

class GaussianBase(BaseDistribution):

    def __init__(self, shape, mean: jnp.ndarray = 0, std: jnp.ndarray = 1):
        self.mean = mean
        self.std = std
        self.shape = tuple(shape)

    def sample(self, rng: jr.PRNGKey, num_samples: int) -> Tuple[jnp.ndarray, jr.PRNGKey]:
        out = jr.normal(rng, (num_samples,) + self.shape) * self.std + self.mean
        rng = jr.split(rng)[0]
        return out, rng

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * jnp.sum(jnp.log(2 * jnp.pi * self.std**2) + (x - self.mean)**2 / (2 * self.std**2))

class UniformBase(BaseDistribution):

    def __init__(self, shape, low: jnp.ndarray = 0, high: jnp.ndarray = 1):
        self.low = low
        self.high = high
        self.shape = tuple(shape)

    def sample(self, rng: jr.PRNGKey, num_samples: int) -> Tuple[jnp.ndarray, jr.PRNGKey]:
        out = jr.uniform(rng, (num_samples,) + self.shape) * (self.high - self.low) + self.low
        rng = jr.split(rng)[0]
        return out, rng

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.where((x >= self.low) & (x <= self.high), -jnp.log(self.high - self.low), -jnp.inf)

def sample_log_normal(rng: jr.PRNGKey, mean: jnp.ndarray = 0, std: jnp.ndarray = 1) -> Tuple[jnp.ndarray, jr.PRNGKey]:
    out = jnp.exp(jr.normal(rng, mean.shape) * std + mean)
    rng = jr.split(rng)[0]
    return out, rng

class LogNormalBase(BaseDistribution):

    def __init__(self, shape, mean: jnp.ndarray = 0, std: jnp.ndarray = 1):
        self.mean = mean
        self.std = std
        self.shape = shape

    def sample(self, rng: jr.PRNGKey, num_samples: int) -> Tuple[jnp.ndarray, jr.PRNGKey]:
        out = jnp.exp(jr.normal(rng, (num_samples,) + self.shape) * self.std + self.mean)
        rng = jr.split(rng)[0]
        return out, rng

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * jnp.sum(jnp.log(2 * jnp.pi * self.std**2) + (jnp.log(x) - self.mean)**2 / (2 * self.std**2))
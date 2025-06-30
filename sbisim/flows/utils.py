from jax.numpy import float32
from flax.linen.initializers import uniform
def uniform_min_max_init(min_, max_):
    return lambda key, shape, dtype=float32: (
            uniform(scale=max_ - min_)(key, shape, dtype) + min_)

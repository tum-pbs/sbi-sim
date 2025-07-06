import jax.numpy as jnp
from jax import tree_util

class EMA:
    def __init__(self, decay: float):
        self.decay = decay
        self.ema_params = None

    def init(self, params):
        self.ema_params = tree_util.tree_map(lambda x: jnp.array(x), params)

    def update(self, i, params):
        if self.ema_params is None:
            self.init(params)
        else:
            self.ema_params = tree_util.tree_map(
                lambda ema, p: self.decay * ema + (1.0 - self.decay) * p,
                self.ema_params, params
            )

    def get_params(self):
        return self.ema_params
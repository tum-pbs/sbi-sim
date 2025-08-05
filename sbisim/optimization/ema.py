import jax.numpy as jnp
from jax import tree_util

class EMA:
    def __init__(self, decay: float):
        self.decay = decay

    def update(self, i, opt_state):

        params = opt_state.params
        ema_params = opt_state.ema_params
        update = tree_util.tree_map(
            lambda ema, p: self.decay * ema + (1.0 - self.decay) * p,
            ema_params, params
        )
        opt_state.ema_params = update
        return opt_state

    def get_params(self, opt_state):
        return opt_state.ema_params
import dataclasses
from abc import ABC, abstractmethod
from typing import Callable, Union

from flax.core import FrozenDict
from jax import tree_util
import jax.numpy as jnp
from jax._src.tree_util import register_pytree_node_class
from jaxtyping import PyTree
from omegaconf import OmegaConf

from ..utils import instantiate_from_config, instantiate_from_config_


class OptimizerWrapper(ABC):
    lr_schedule: Union[int, Callable[[int], float]]

    def __init__(self, lr_schedule):

        if not callable(lr_schedule):
            self.lr_schedule = lambda _: lr_schedule
        else:
            self.lr_schedule = lr_schedule

    @abstractmethod
    def init(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, i, state, grads):
        pass

    @abstractmethod
    def get_params(self, state):
        pass


@register_pytree_node_class
class OptState:

    params: PyTree
    ema_params: PyTree
    teacher_weights: PyTree
    state: PyTree

    def __init__(self, params, ema_params, state, teacher_weights):
        self.params = params
        self.ema_params = ema_params
        self.teacher_weights = teacher_weights
        self.state = state

    def tree_flatten(self):
        children = (self.params, self.ema_params, self.state, self.teacher_weights)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

class Optimization:

    def __init__(self, config):

        self.scheduler = instantiate_from_config(config.scheduler)

        self.optimizer = instantiate_from_config_(config.optimizer.target, lr_schedule=self.scheduler,
                                                  **config.get("params", dict()))

        self.ema_active = True
        self.ema = None
        if "ema" in config:
            self.ema = instantiate_from_config(config.ema)
        else:
            self.ema_active = False

        self.state = None
        self.params = None

    def init(self, params):

        self.state = self.optimizer.init(params)
        self.state = tree_util.tree_map(lambda x: jnp.array(x), self.state)

    def set_state(self, state):
        if isinstance(state, OptState):
            self.state = state

        else:

            # https://github.com/google-deepmind/optax/discussions/180
            def restore_optimizer_state(opt_state, restored):
                """Restore optimizer state from loaded checkpoint (or .msgpack file)."""
                return tree_util.tree_unflatten(
                    tree_util.tree_structure(opt_state), tree_util.tree_leaves(restored)
                )

            self.state = restore_optimizer_state(self.state, state)

        self.state = tree_util.tree_map(lambda x: jnp.array(x), self.state)

    def get_state(self):
        return self.state

    def get_learning_rate(self, i):
        return self.scheduler(i)

    def get_params(self):
        return self.optimizer.get_params(self.state)

    def get_params_from_state(self, state):
        return self.optimizer.get_params(state)

    def get_ema_params_from_state(self, state):
        return self.ema.get_params(state)

    def update_scheduler(self, i, logs):

        if hasattr(self.scheduler, "update"):
            self.scheduler.update(logs)
        self.state = self.optimizer.update_learning_rate(self.state, self.scheduler(i))

    def set_teacher_weights(self, opt_state, teacher_weights):
        """Set teacher weights in the optimizer state."""
        if isinstance(opt_state, OptState):
            opt_state.teacher_weights = teacher_weights
        else:
            raise ValueError("opt_state must be an instance of OptState")

    def get_teacher_weights(self, opt_state):
        """Get teacher weights from the optimizer state."""
        if isinstance(opt_state, OptState):
            return opt_state.teacher_weights
        else:
            raise ValueError("opt_state must be an instance of OptState")

    def update(self, i, opt_state, grads):

        opt_state = self.optimizer.update(i, opt_state, grads)

        if self.ema_active and self.ema is not None:
            opt_state = self.ema.update(i, opt_state)

        return opt_state

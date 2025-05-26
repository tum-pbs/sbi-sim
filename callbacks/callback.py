from abc import ABC, abstractmethod
from typing import Iterable, Tuple

import jax.random as jr

from flows.flowbase import FlowBase


class Callback(ABC):

    name: str

    @abstractmethod
    def __call__(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs) \
            -> Tuple[dict, jr.PRNGKey]:
        pass

    def on_test(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return logs, rng

    def on_train_begin(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return logs, rng

    def on_train_end(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return logs, rng

    def on_epoch_begin(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return logs, rng

    def on_epoch_end(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return logs, rng
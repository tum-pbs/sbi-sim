from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .callback import Callback
from ..data.benchmarks.generator_dataloader import GeneratorDataloader
from ..metrics.c2st import c2st

# from ..metrics.mmd import mmd

from ..strategy import Strategy

import jax.random as jr
import jax.numpy as jnp

class C2ST(Callback):

    name: str = 'c2st'
    num_observations: int = 10
    num_posterior_samples: int = 10000
    batch_size: int = 128
    metrics: List[str] = ['c2st']

    def __init__(self, batch_size: int = 512, metrics: List[str] = None):
        super().__init__()
        self.batch_size = batch_size
        if metrics is not None:
            self.metrics = metrics

    def __call__(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return self._call(logs, rng, *args, **kwargs)

    def _call(self, logs: dict, rng: jr.PRNGKey, strategy: Strategy,
              train_loader: GeneratorDataloader, val_loader: GeneratorDataloader, *args, **kwargs) \
                -> Tuple[dict, jr.PRNGKey]:

        scores = {metric: 0.0 for metric in self.metrics}

        p = tqdm(range(1, self.num_observations+1))

        for observation_idx in p:

            observation, true_theta, reference_posterior = train_loader.get_observation(observation_idx)

            observation = jnp.array(observation).repeat(self.num_posterior_samples, axis=0)
            posterior_samples, _ = strategy.sample(self.num_posterior_samples, rng,
                                                   conditioning=observation, batch_size=self.batch_size)

            if 'c2st' in self.metrics:
                score_idx = c2st(reference_posterior, posterior_samples["samples"])
                scores['c2st'] += score_idx
                p.set_description(f'Observation {observation_idx} C2ST: {score_idx:.3f}')

            if 'mmd' in self.metrics:
                score_mmd = mmd(np.array(reference_posterior), np.array(posterior_samples["samples"]))
                scores['mmd'] += score_mmd
                print(f'Observation {observation_idx} MMD: {score_mmd:.3f}')

        for metric in self.metrics:
            scores[metric] = scores[metric] / self.num_observations
            logs[metric] = scores[metric]

        return logs, rng

    def on_test(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return self.__call__(logs, rng, *args, **kwargs)
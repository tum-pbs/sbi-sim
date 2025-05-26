from typing import Tuple, List

import wandb

from callbacks.callback import Callback

import jax.random as jr

from data.benchmarks.generator_dataloader import GeneratorDataloader
from strategy import Strategy

import jax.numpy as jnp

from matplotlib import pyplot as plt

class BenchmarkScatterPlot(Callback):

    name: str = 'benchmark_scatter_plot'
    save_every: int = 10

    observation_idx: List[int] = [1,2]
    num_total_samples: int = 10000
    batch_size: int = 128

    def __init__(self, save_every: int = 10):
        super().__init__()
        self.save_every = save_every

    def __call__(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return self._call(logs, rng, *args, **kwargs)

    def _call(self, logs: dict, rng: jr.PRNGKey, strategy: Strategy,
              train_loader: GeneratorDataloader, val_loader: GeneratorDataloader, *args, **kwargs) \
            -> Tuple[dict, jr.PRNGKey]:

        figure_list = []

        for id in self.observation_idx:

            observation, true_theta, reference_posterior = train_loader.get_observation(id)

            observation = jnp.array(observation).repeat(self.num_total_samples, axis=0)
            posterior_samples, _ = strategy.sample(self.num_total_samples, rng, conditioning=observation,
                                                   batch_size=self.batch_size)

            fig = plt.figure(figsize=(10, 10))
            plt.scatter(reference_posterior[:, 0], reference_posterior[:, 1], label="Reference Posterior", alpha=0.2)
            plt.scatter(posterior_samples["samples"][:, 0], posterior_samples["samples"][:, 1],
                        label="Posterior Samples", alpha=0.2)

            plt.legend()
            plt.xlabel("Parameter 1")
            plt.ylabel("Parameter 2")

            plt.title(f"Observation {id}")

            figure_list.append(fig)

        logs['posteriors'] = [wandb.Image(fig) for fig in figure_list]

        for fig in figure_list:
            plt.close(fig)

        return logs, rng

    def on_train_begin(self, *args, **kwargs):
        return self.__call__(*args, init=True, **kwargs)

    def on_epoch_end(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):

        if "epoch" in logs and logs["epoch"] % self.save_every == 0:
            return self.__call__(logs, rng, *args, **kwargs)
        else:
            return logs, rng

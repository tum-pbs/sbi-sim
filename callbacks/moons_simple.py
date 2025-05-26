from typing import Iterable, Tuple

import jax.random as jr
import wandb
from jax import jit
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from callbacks.callback import Callback
from flows.flowbase import FlowBase
import jax.numpy as jnp
import matplotlib.pyplot as plt

from strategy import Strategy
from torch.utils.data import DataLoader

class SimpleMoonDatasetPlots(Callback):
    name: str = 'simple_moon_dataset_plots'
    save_every: int = 10

    num_total_samples: int = 2500
    batch_size: int = 128

    def __call__(self, logs: dict, rng: jr.PRNGKey, model: FlowBase,
                 train_loader: Iterable, *args, init=False, **kwargs) -> Tuple[dict, jr.PRNGKey]:

        if init:

            gt_data = []
            for i, batch in enumerate(train_loader):
                gt_data.extend(list(batch["parameters"]))
                if len(gt_data) >= self.num_total_samples:
                    break
            gt_data = jnp.array(gt_data[:self.num_total_samples])

            figure, ax = plt.subplots(1, 1, figsize=(5, 5))

            ax.scatter(gt_data[:, 0], gt_data[:, 1], s=1, label='ground truth')
            ax.set_xlim([-3, 3])
            ax.set_ylim([-3, 3])

            logs['ground truth'] = wandb.Image(figure)


        samples_left = self.num_total_samples
        samples_list = []
        samples_prob_list = []
        for _ in tqdm(range(jnp.int32(jnp.ceil(self.num_total_samples / self.batch_size)))):
            samples, probability, rng = model.sample(min(self.batch_size, samples_left), rng)
            samples_list.append(samples)
            samples_prob_list.append(probability)
            samples_left -= self.batch_size

        samples = jnp.concatenate(samples_list, axis=0)
        samples_prob = jnp.concatenate(samples_prob_list, axis=0)

        figure, ax = plt.subplots(1, 1, figsize=(5, 5))
        im = ax.scatter(samples[:, 0], samples[:, 1], s=1, c=samples_prob, label='', vmin=-10, vmax=-1)
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        logs['posterior samples'] = wandb.Image(figure)
        plt.close(figure)

        # density estimation
        #############################################################################

        likelihood_fn = jit(model.compute_likelihood_)

        resolution = 400
        X = jnp.linspace(-3, 3, resolution)
        Y = jnp.linspace(-3, 3, resolution)
        X, Y = jnp.meshgrid(X, Y)
        Z = jnp.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)

        cells_left = Z.shape[0]
        cells_value = []

        for i in tqdm(range(jnp.int32(jnp.ceil(cells_left / self.batch_size)))):
            x = Z[i * self.batch_size:(i + 1) * self.batch_size]

            weighting = jnp.ones((x.shape[0],))
            conditioning = jnp.zeros((x.shape[0], model.dim_conditioning))

            cell_value, rng = likelihood_fn(x, rng, weighting=weighting, conditioning=conditioning)
            cells_value.append(cell_value)
            cells_left -= self.batch_size

        cells_value = jnp.concatenate(cells_value, axis=0)
        cells_value = cells_value.reshape(resolution, resolution)

        figure, ax = plt.subplots(1, 1, figsize=(5, 5))

        # ax.contourf(X, Y, cells_value, cmap='Blues')

        # cut off very unlikely densities for better visualization
        cells_value = jnp.minimum(cells_value, 10.0)
        im = ax.imshow(- cells_value, extent=[-3, 3, -3, 3], vmin=-10, vmax=-1, origin='lower')

        # ax.set_xlim([-3, 3])
        # ax.set_ylim([-3, 3])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        logs['density estimation'] = wandb.Image(figure)
        plt.close(figure)

        return logs, rng

    def on_train_begin(self, *args, **kwargs):
        return self.__call__(*args, init=True, **kwargs)

    def on_epoch_end(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):

        if "epoch" in logs and logs["epoch"] % self.save_every == 0:
            return self.__call__(logs, rng, *args, **kwargs)
        else:
            return logs, rng


class SimpleMoonDatasetPlotsConditional(Callback):
    name: str = 'simple_moon_dataset_plots_conditional'
    save_every: int = 10
    num_total_samples: int = 2500
    batch_size: int = 128

    def __init__(self, save_every: int = 10):
        self.save_every = save_every

    def __call__(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return self._call(logs, rng, *args, **kwargs)

    def _call(self, logs: dict, rng: jr.PRNGKey, strategy: Strategy,
              train_loader: DataLoader, val_loader: DataLoader, init=True, *args, **kwargs) \
            -> Tuple[dict, jr.PRNGKey]:

        # Initial sampling without jit compile to combine flax and jax transform
        # Otherwise we get this error: https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.JaxTransformError
        _ = strategy.sample(num_samples=1, rng=rng)

        if init:

            gt_data = []
            for i, batch in enumerate(train_loader):
                gt_data.extend(list(batch["parameters"]))
                if len(gt_data) >= self.num_total_samples:
                    break
            gt_data = jnp.array(gt_data[:self.num_total_samples])

            figure, ax = plt.subplots(1, 1, figsize=(5, 5))

            ax.scatter(gt_data[:, 0], gt_data[:, 1], s=1, label='ground truth')
            ax.set_xlim([-3, 3])
            ax.set_ylim([-3, 3])

            logs['ground truth'] = wandb.Image(figure)

        # posterior sampling for both y == 0 and y == 1

        sample_fn = jit(strategy.sample, static_argnums=(0,))

        figure, axes = plt.subplots(1, 2, figsize=(10, 5))

        samples_left = self.num_total_samples
        samples_list = []
        samples_prob_list = []
        for _ in tqdm(range(jnp.int32(jnp.ceil(self.num_total_samples / self.batch_size)))):
            samples_iteration = min(self.batch_size, samples_left)

            y = jnp.ones((samples_iteration, 1))

            samples, rng = sample_fn(min(self.batch_size, samples_left), rng,
                                                  conditioning=y)
            samples_list.append(samples["samples"])
            samples_prob_list.append(samples["likelihood"])
            samples_left -= self.batch_size

        samples = jnp.concatenate(samples_list, axis=0)
        samples_prob = jnp.concatenate(samples_prob_list, axis=0)

        im = axes[0].scatter(samples[:, 0], samples[:, 1], s=1, c=samples_prob, label='', vmin=-10, vmax=-1)
        axes[0].set_xlim([-3, 3])
        axes[0].set_ylim([-3, 3])

        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        samples_left = self.num_total_samples
        samples_list = []
        samples_prob_list = []
        for _ in tqdm(range(jnp.int32(jnp.ceil(self.num_total_samples / self.batch_size)))):
            samples_iteration = min(self.batch_size, samples_left)

            y = jnp.zeros((samples_iteration, 1))

            samples, rng = sample_fn(min(self.batch_size, samples_left), rng, conditioning=y)
            samples_list.append(samples["samples"])
            samples_prob_list.append(samples["likelihood"])
            samples_left -= self.batch_size

        samples = jnp.concatenate(samples_list, axis=0)
        samples_prob = jnp.concatenate(samples_prob_list, axis=0)

        im = axes[1].scatter(samples[:, 0], samples[:, 1], s=1, c=samples_prob, label='', vmin=-10, vmax=-1)
        axes[1].set_xlim([-3, 3])
        axes[1].set_ylim([-3, 3])

        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        logs['posterior samples'] = wandb.Image(figure)
        plt.close(figure)

        # density estimation
        #############################################################################

        likelihood_fn = jit(strategy.compute_likelihood)

        resolution = 400
        X = jnp.linspace(-3, 3, resolution)
        Y = jnp.linspace(-3, 3, resolution)
        X, Y = jnp.meshgrid(X, Y)
        Z = jnp.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)

        figure, axes = plt.subplots(1, 2, figsize=(10, 5))

        cells_left = Z.shape[0]
        cells_value = []

        for i in tqdm(range(jnp.int32(jnp.ceil(cells_left / self.batch_size)))):
            x = Z[i * self.batch_size:(i + 1) * self.batch_size]

            weighting = jnp.ones((x.shape[0],))
            y = jnp.zeros((x.shape[0], 1))

            cell_value, rng = likelihood_fn(x, rng, weighting=weighting, conditioning=y)
            cells_value.append(cell_value["likelihood"])
            cells_left -= self.batch_size

        cells_value = jnp.concatenate(cells_value, axis=0)
        cells_value = cells_value.reshape(resolution, resolution)

        # ax.contourf(X, Y, cells_value, cmap='Blues')

        # cut off very unlikely densities for better visualization
        cells_value = jnp.minimum(cells_value, 10.0)
        im = axes[0].imshow(- cells_value, extent=[-3, 3, -3, 3], vmin=-10, vmax=-1, origin='lower')

        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        cells_left = Z.shape[0]
        cells_value = []

        for i in tqdm(range(jnp.int32(jnp.ceil(cells_left / self.batch_size)))):
            x = Z[i * self.batch_size:(i + 1) * self.batch_size]

            weighting = jnp.ones((x.shape[0],))
            y = jnp.ones((x.shape[0], 1))

            cell_value, rng = likelihood_fn(x, rng, weighting=weighting, conditioning=y)
            cells_value.append(cell_value["likelihood"])
            cells_left -= self.batch_size

        cells_value = jnp.concatenate(cells_value, axis=0)
        cells_value = cells_value.reshape(resolution, resolution)

        # ax.contourf(X, Y, cells_value, cmap='Blues')

        # cut off very unlikely densities for better visualization
        cells_value = jnp.minimum(cells_value, 10.0)
        im = axes[1].imshow(- cells_value, extent=[-3, 3, -3, 3], vmin=-10, vmax=-1, origin='lower')


        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        logs['density estimation'] = wandb.Image(figure)
        plt.close(figure)

        return logs, rng

    def on_train_begin(self, *args, **kwargs):
        return self.__call__(*args, init=True, **kwargs)

    def on_epoch_end(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):

        if "epoch" in logs and logs["epoch"] % self.save_every == 0:
            return self.__call__(logs, rng, *args, **kwargs)
        else:
            return logs, rng

from typing import Tuple

import wandb
from matplotlib.patches import Rectangle, Circle

from .callback import Callback

import jax.random as jr

from ..data.density_estimation.golden_ratio import fibonacci_ratio, get_centers

from ..strategy import Strategy
from matplotlib import pyplot as plt
import jax.numpy as jnp
from tqdm import tqdm
import seaborn as sns


class DataLoader:
    pass


class TwoDimensionalPlot(Callback):

    name: str = '2d_plot'
    save_every: int = 10
    num_samples: int = 3000
    num_samples_gt: int = 3000

    def __init__(self, save_every: int = 10, num_samples: int = 3000, savedir: str = None):
        super(self.__class__, self).__init__()

        self.save_every = save_every
        self.num_samples = num_samples
        self.savedir = savedir


    def __call__(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return self._call(logs, rng, *args, **kwargs)

    def _call(self, logs: dict, rng: jr.PRNGKey, strategy: Strategy,
              train_loader: DataLoader, val_loader: DataLoader, init=True, *args, **kwargs) \
                -> Tuple[dict, jr.PRNGKey]:

        if init:

            # plot dataset
            fig, ax = plt.subplots()

            x = []
            y = []

            for batch in tqdm(iter(val_loader)):
                parameters = batch["parameters"]
                x_elem, y_elem = parameters[:, 0], parameters[:, 1]
                x.extend(list(x_elem))
                y.extend(list(y_elem))

                if len(x) > self.num_samples_gt:
                    break

            ax.scatter(x[:self.num_samples_gt], y[:self.num_samples_gt])

            if self.savedir is not None:
                plt.savefig(f"{self.savedir}/pictures/samples_gt.png")

            logs[f'samples_gt'] = wandb.Image(fig)

            plt.close(fig)

        sample_fn = strategy.sample

        conditioning = jnp.zeros((self.num_samples, 0))

        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 4))

        for i, steps in enumerate([1, 2, 4, 8, 16]):

            samples, rng = sample_fn(self.num_samples, rng, z_init=None, conditioning=conditioning,
                                     num_steps=steps)

            X = samples['samples']

            x, y = X[:, 0], X[:, 1]

            axes[i].scatter(x, y)
            axes[i].set_title(f"steps={steps}")

        epoch = logs.get("epoch", 0)

        if self.savedir is not None:

            import os
            # create directory if it does not exist
            os.makedirs(f"{self.savedir}/pictures", exist_ok=True)

            plt.savefig(f"{self.savedir}/pictures/samples_{epoch}.pdf")

        logs[f'samples'] = wandb.Image(fig)

        plt.close(fig)

        return logs, rng

    def on_train_begin(self, *args, **kwargs):
        return self.__call__(*args, init=True, **kwargs)

    def on_test(self, *args, **kwargs):
        return self.__call__(*args, init=False, **kwargs)

    def on_epoch_end(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):

        if "epoch" in logs and logs["epoch"] % self.save_every == 0:
            return self.__call__(logs, rng, *args, **kwargs)
        else:
            return logs, rng


class GoldenRatioPlot(Callback):

    name: str = 'golden_ratio_plot'
    save_every: int = 10
    num_samples: int = 10000
    num_samples_gt: int = 10000

    def plot_golden_ratio(self, ax, centers, fibonacci, samples):

        min_x = 10,
        max_x = -10,
        min_y = 10,
        max_y = -10

        for i, center in enumerate(centers):
            min_x = min(min_x, center[0] - fibonacci[i] / 2)
            max_x = max(max_x, center[0] + fibonacci[i] / 2)
            min_y = min(min_y, center[1] - fibonacci[i] / 2)
            max_y = max(max_y, center[1] + fibonacci[i] / 2)
            ax.add_patch(
                Rectangle((center[0] - fibonacci[i] / 2, center[1] - fibonacci[i] / 2), fibonacci[i], fibonacci[i],
                          facecolor="none", ec='k', lw=1.5))
            ax.add_patch(Circle(center, fibonacci[i] / 2, facecolor="none", ec='k', lw=1))

        ax.set_aspect('equal')

        x, y = zip(*centers)

        sns.scatterplot(x=samples[:, 0], y=samples[:, 1], s=2, marker='o', alpha=0.5)
        sns.scatterplot(x=x, y=y, marker='x', c='r', s=20)

        # ax remove border
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

        # ax remove ticks
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])

        return ax

    def __init__(self, save_every: int = 10, num_samples: int = 10000, num_modes: int = 8, std_factor=0.1):
        super(self.__class__, self).__init__()

        self.std_factor = std_factor
        self.fibonacci = fibonacci_ratio(num_modes)
        self.centers = get_centers(self.fibonacci)
        self.sigma_level = 4
        self.save_every = save_every
        self.num_samples = num_samples


    def __call__(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return self._call(logs, rng, *args, **kwargs)

    def _call(self, logs: dict, rng: jr.PRNGKey, strategy: Strategy,
              train_loader: DataLoader, val_loader: DataLoader, init=False, *args, **kwargs) \
                -> Tuple[dict, jr.PRNGKey]:

        if init:

            x = []
            y = []

            for batch in tqdm(iter(val_loader)):
                parameters = batch["parameters"]
                x_elem, y_elem = parameters[:, 0], parameters[:, 1]
                x.extend(list(x_elem))
                y.extend(list(y_elem))

                if len(x) > self.num_samples_gt:
                    break

            with sns.axes_style("ticks"):

                # plot dataset
                fig, ax = plt.subplots()

                x = jnp.array(x[:self.num_samples_gt])
                y = jnp.array(y[:self.num_samples_gt])

                samples = jnp.column_stack((x, y))
                ax = self.plot_golden_ratio(ax, self.centers, self.fibonacci, samples)

                logs[f'samples_gt'] = wandb.Image(fig)

                plt.close(fig)

        sample_fn = strategy.sample

        conditioning = jnp.zeros((self.num_samples, 0))
        samples, rng = sample_fn(self.num_samples, rng, z_init=None, conditioning=conditioning)

        X = samples['samples']

        x, y = X[:, 0], X[:, 1]

        with sns.axes_style("ticks"):

            fig, ax = plt.subplots()
            samples = jnp.column_stack((x, y))
            ax = self.plot_golden_ratio(ax, self.centers, self.fibonacci, samples)

            logs[f'samples'] = wandb.Image(fig)

            plt.close(fig)

        # check in which center the points are

        checkmark = jnp.zeros((len(samples), len(self.centers)))

        for i, center in enumerate(self.centers):

            x_dist = jnp.abs(samples[:, 0] - center[0])
            y_dist = jnp.abs(samples[:, 1] - center[1])

            checkmark = checkmark.at[:, i].set((jnp.sqrt(x_dist ** 2 + y_dist ** 2) <
                                  self.sigma_level * (self.fibonacci[i] / 2) * self.std_factor))


        num_samples_per_center = jnp.sum(checkmark, axis=0)

        fraction_per_center = (num_samples_per_center / self.num_samples)
        precision = jnp.max(jnp.abs(fraction_per_center - 1 / len(self.centers)))
        recall = 1 - jnp.sum(fraction_per_center)

        logs[f'precision'] = precision
        logs[f'recall'] = recall

        return logs, rng

    def on_train_begin(self, *args, **kwargs):
        return self.__call__(*args, init=True, **kwargs)

    def on_epoch_end(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):

        if "epoch" in logs and logs["epoch"] % self.save_every == 0:
            return self.__call__(logs, rng, *args, **kwargs)
        else:
            return logs, rng
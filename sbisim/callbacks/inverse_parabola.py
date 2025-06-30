from typing import Tuple

from callbacks.callback import Callback
# from ..data.dataloader import DataLoader
import jax.random as jr
import jax.numpy as jnp

from ..strategy import Strategy
from matplotlib import pyplot as plt

class InverseParabolaPlot(Callback):

    name: str = 'inverse_parabola_plot'
    save_every: int = 10
    time_points: int = 32
    num_points_per_time: int = 6

    def __init__(self, save_every: int = 10, time_points: int = 32):
        super(self.__class__, self).__init__()

        self.save_every = save_every
        self.time_points = time_points


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

            for batch in iter(val_loader):
                parameters = batch["parameters"]

                conditioning = batch["conditioning"]
                x.extend(list(conditioning[0]))
                y.extend(list(parameters[0]))

            ax.scatter(x, y)

            logs[f'inverse_parabola_gt'] = fig

            plt.close(fig)


        times = jnp.linspace(-1, 1, self.time_points)
        times = times.repeat(self.num_points_per_time, axis=0)
        times = times[:, None]

        sample_fn = strategy.sample
        samples, rng = sample_fn(times.shape[0], rng, z_init=None, conditioning=times)

        x = samples['samples']

        fig, ax = plt.subplots()
        ax.scatter(times, x)

        logs[f'inverse_parabola'] = fig

        plt.close(fig)

        return logs, rng

    def on_train_begin(self, *args, **kwargs):
        return self.__call__(*args, init=True, **kwargs)

    def on_epoch_end(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):

        if "epoch" in logs and logs["epoch"] % self.save_every == 0:
            return self.__call__(logs, rng, *args, **kwargs)
        else:
            return logs, rng
import wandb
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..flows.parameterflow import ParameterFlow

import numpy as np
# from torch.utils.data import DataLoader

import jax.random as jr
from tqdm import tqdm

def logger(every=5, num_total_samples=10000, batch_size=1024):

    def callback(logs, rng : jr.PRNGKey, model : ParameterFlow, dataset : DataLoader):

        if logs['epoch'] == 0:

            gt_data = []
            for i, (x, _, _) in enumerate(dataset):
                gt_data.extend(list(x))
                if len(gt_data) >= num_total_samples:
                    break
            gt_data = np.array(gt_data[:num_total_samples])

            figure, ax = plt.subplots(1, 1, figsize=(10, 5))

            ax.scatter(gt_data[:, 0], gt_data[:, 1], s=1, label='ground truth')

            logs['ground truth'] = figure


        if logs['epoch'] % every == 0:

            # posterior samples
            #############################################################################

            samples_left = num_total_samples
            samples_list = []
            for i in tqdm(range(np.int32(np.ceil(num_total_samples / batch_size)))):
                samples_list.append(model.sample(min(batch_size, samples_left), rng)[0])
                samples_left -= batch_size
                rng = jr.split(rng)[0]

            samples = np.concatenate(samples_list, axis=0)

            figure, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.scatter(samples[:, 0], samples[:, 1], s=1, label='')
            ax.set_xlim([-3, 3])
            ax.set_ylim([-3, 3])

            logs['posterior samples'] = wandb.Image(figure)

            # density estimation
            #############################################################################

            resolution = 400
            X = np.linspace(-3, 3, resolution)
            Y = np.linspace(-3, 3, resolution)
            X, Y = np.meshgrid(X, Y)
            Z = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)

            cells_left = Z.shape[0]
            cells_value = []
            for i in tqdm(range(np.int32(np.ceil(cells_left / batch_size)))):
                x = Z[i*batch_size:(i+1)*batch_size]
                data = [x, np.ones((x.shape[0],)), None]
                cells_value.append(model.compute_conditional_likelihood(data, rng)[0])
                rng, _ = jr.split(rng)
                cells_left -= batch_size

            cells_value = np.concatenate(cells_value, axis=0)
            cells_value = cells_value.reshape(resolution, resolution)

            figure, ax = plt.subplots(1, 1, figsize=(5, 5))

            # ax.contourf(X, Y, cells_value, cmap='Blues')
            im = ax.imshow(cells_value, extent=[-3, 3, -3, 3], cmap='Blues', origin='lower')
            # ax.set_xlim([-3, 3])
            # ax.set_ylim([-3, 3])

            # add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)


            logs['density estimation'] = wandb.Image(figure)

        return logs


    return callback
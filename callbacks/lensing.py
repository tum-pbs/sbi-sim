from typing import Tuple, List, Dict

import jax.random as jr
import wandb
import corner
import numpy as np

from scipy import stats
from jax import vmap, jit

from callbacks.callback import Callback

import jax.numpy as jnp
import matplotlib.pyplot as plt

from lensing_simulation import get_simu_images
from simulations import SimulationTargets, LensingSetup
from source.data.dataloader import DataLoader
from strategy import Strategy
from utils import instantiate_from_config

index_to_label = {
    0: r'$\theta_E$',
    1: r'Source $e_1$',
    2: r'Source $e_2$',
    3: r'Source $x$',
    4: r'Source $y$',
    5: r'$\gamma_1$',
    6: r'$\gamma_2$',
    7: r'$\alpha_0$',
    8: r'$\delta_0$',
    9: r'$A$',
    10: r'$R_s$',
    11: r'$n$',
    12: r'Lens $e_1$',
    13: r'Lens $e_2$',
    14: r'Lens $x$',
    15: r'Lens $y$',
}

def wrapper_expand(x, num_samples):

    def _expand(r):
        return jnp.repeat(r[None], num_samples, axis=0)

    # check if x is ndarray
    if isinstance(x, jnp.ndarray):
        return _expand(x)
    elif isinstance(x, list) or isinstance(x, tuple):
        return [_expand(xi) for xi in x]
    elif isinstance(x, dict):
        return {k: _expand(v) for k, v in x.items()}
    else:
        raise ValueError(f"Type {type(x)} not supported")


def lens_plot(observation_pred, observation, nll, noise_std):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(observation_pred, cmap='bone_r')
    ax[0].set_title('Predicted')

    ax[1].imshow(observation, cmap='bone_r')
    ax[1].set_title('Ground Truth')

    im = ax[2].imshow(observation_pred - observation, cmap='bwr',
                      vmin=-7 * noise_std,
                      vmax=7 * noise_std)

    ax[2].set_title('Difference')

    # set colorbar for ax[2]
    cbar = plt.colorbar(im, ax=ax[2])

    # set ticks based on self.noise

    cbar.set_ticks([-7 * noise_std, 0,
                    7 * noise_std])

    cbar.set_ticklabels([r'-7$\sigma$', '0', r'7$\sigma$'])

    # add log likelihood to title
    ax[0].set_title(f'log likelihood: {nll:.5f}')

    return fig


class LensParameterLogProb(Callback):

    name: str = 'lens_parameter_logprob'
    on_train_start: bool = False

    def __init__(self, simulation: Dict, num_total_samples: int = 1000, save_every: int = 10, on_train_start: bool = False):

        super(self.__class__, self).__init__()

        self.on_train_start = on_train_start

        self.num_total_samples = num_total_samples

        self.simulation_model: LensingSetup = instantiate_from_config(simulation["setup"])

        self.targets: SimulationTargets = (
            instantiate_from_config(simulation["target"]))

        self.target_mask = self.targets.get_target_mask()

        self.save_every = save_every

        self.forward_fn = vmap(self.simulation_model.forward, in_axes=(0, None, None, None))
        self.compute_log_likelihood_per_pixel_fn = vmap(self.simulation_model.compute_log_likelihood_per_pixel, in_axes=(0, 0))

    def __call__(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return self._call(logs, rng, *args, **kwargs)

    def _call(self, logs: dict, rng: jr.PRNGKey, strategy: Strategy,
              train_loader: DataLoader, val_loader: DataLoader, init=True, *args, **kwargs) \
            -> Tuple[dict, jr.PRNGKey]:

        sample_fn = strategy.sample

        print(f'{self.name}: Sampling {self.num_total_samples} samples for log prob evaluation [validation set]')

        sample = 0
        log_prob_list = []

        for batch in iter(val_loader):

            parameters = batch["parameters"]
            weighting = batch["weighting"]
            conditioning = batch["conditioning"]

            batch_size = parameters.shape[0]
            samples, rng = sample_fn(batch_size, rng, z_init=None, conditioning=conditioning)

            x = samples['samples']

            parameters_predicted = self.targets.assemble(conditioning[1], x)
            observation_predicted = self.forward_fn(
                parameters_predicted, rng, False, True)
            log_probs = self.compute_log_likelihood_per_pixel_fn(
                observation_predicted, conditioning[0])
            log_prob_list.append(log_probs)

            sample += batch_size

            if sample >= self.num_total_samples:
                break

        log_prob = jnp.concatenate(log_prob_list, axis=0)
        log_prob = jnp.mean(log_prob[:self.num_total_samples])

        logs['log_prob_val'] = log_prob

        print(f'{self.name}: Sampling {self.num_total_samples} samples for log prob evaluation [training set]')

        sample = 0
        log_prob_list = []

        for batch in iter(val_loader):

            parameters = batch["parameters"]
            weighting = batch["weighting"]
            conditioning = batch["conditioning"]

            batch_size = parameters.shape[0]
            samples, rng = sample_fn(batch_size, rng, z_init=None, conditioning=conditioning)

            x = samples['samples']

            parameters_predicted = self.targets.assemble(conditioning[1], x)
            observation_predicted = self.forward_fn(
                parameters_predicted, rng, False, True)
            log_probs = self.compute_log_likelihood_per_pixel_fn(
                observation_predicted, conditioning[0])
            log_prob_list.append(log_probs)

            sample += batch_size

            if sample >= self.num_total_samples:
                break

        log_prob = jnp.concatenate(log_prob_list, axis=0)
        log_prob = jnp.mean(log_prob[:self.num_total_samples])

        logs['log_prob_train'] = log_prob

        return logs, rng

    def on_train_begin(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):

        if self.on_train_start:
            return self.__call__(logs, rng, *args, init=True, **kwargs)
        else:
            return logs, rng


    def on_epoch_end(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):

        if "epoch" in logs and logs["epoch"] % self.save_every == 0:
            return self.__call__(logs, rng, *args, **kwargs)
        else:
            return logs, rng


class LensParameterPlots(Callback):

    name: str = 'lens_parameter_dataset_plots'
    on_train_start: bool = False

    def __init__(self, simulation: Dict, save_every: int = 10, num_posteriors: int = 2,
                 num_total_samples: int = 1000, max_number_features: int = 4, on_train_start: bool = False):

        super(self.__class__, self).__init__()

        self.on_train_start = on_train_start
        self.num_posteriors = num_posteriors
        self.num_total_samples = num_total_samples
        self.max_number_features = max_number_features

        self.simulation_model = instantiate_from_config(simulation["setup"])

        self.targets: SimulationTargets = (
            instantiate_from_config(simulation["target"]))

        self.target_mask = self.targets.get_target_mask()

        self.save_every = save_every

    def _make_mode_plot(self, x, params_gt, observation):

        modes = stats.mode(x, axis=0, keepdims=True)
        mode = modes[0]

        rng = jr.PRNGKey(0)
        prediction_full = self.targets.assemble(params_gt, mode[0])
        observation_pred = self.simulation_model.forward(prediction_full, rng, False, True)

        nll = self.simulation_model.compute_log_likelihood_per_pixel(observation_pred, observation)

        return lens_plot(observation_pred, observation[0], nll, self.simulation_model.get_noise_std())

    def _make_corner_plot(self, x, y, margin=0.15, figsize=(15, 15)):

        features_predicted = x[:, :self.max_number_features]
        feature_names = self.targets.get_target_names()
        y = y[:self.max_number_features]

        features_predicted = np.array(features_predicted)
        features_predicted = features_predicted[~np.isnan(features_predicted).any(axis=1)]

        ranges = [(elem-margin, elem+margin) for elem in y]

        ndim = y.shape[0]

        figure = corner.corner(features_predicted, # range=ranges,
                               labels=feature_names, truths=y,
                               show_titles=True, title_fmt='.3f')

        axes = figure.axes

        for i in range(ndim):
            for j in range(min(ndim, i+1)):
                ax = axes[i*ndim+j]
                ax.set_xlim(ranges[j])
                if j < i:
                    ax.set_ylim(ranges[i])



        return figure

    def __call__(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return self._call(logs, rng, *args, **kwargs)

    def _call(self, logs: dict, rng: jr.PRNGKey, strategy: Strategy,
              train_loader: DataLoader, val_loader: DataLoader, init=True, *args, **kwargs) \
            -> Tuple[dict, jr.PRNGKey]:

        # VAL LOADER

        figure_list_corner = []
        figure_list_mode = []

        sample = 0
        sample_fn = strategy.sample

        print(f'{self.name}: Generating {self.num_posteriors} posterior distribution(s) for validation set')

        for batch in iter(val_loader):

            parameters = batch["parameters"]
            weighting = batch["weighting"]
            conditioning = batch["conditioning"]

            if isinstance(conditioning, tuple):
                # convert tuple of lists to list of tuples
                conditioning = list(zip(*conditioning))

            for p, w, c in zip(parameters, weighting, conditioning):

                if sample >= self.num_posteriors:
                    break

                c = wrapper_expand(c, self.num_total_samples)

                samples, rng = sample_fn(self.num_total_samples, rng, z_init=None, conditioning=c)

                x = samples['samples']

                figure_list_corner.append(self._make_corner_plot(x, p))

                figure_list_mode.append(self._make_mode_plot(x, c[1][0], c[0][0]))

                sample += 1

            if sample >= self.num_posteriors:
                break

        logs['posterior val'] = [wandb.Image(fig) for fig in figure_list_corner]
        logs['mode val'] = [wandb.Image(fig) for fig in figure_list_mode]


        for fig in figure_list_corner:
            plt.close(fig)
        for fig in figure_list_mode:
            plt.close(fig)

        # TRAIN LOADER

        figure_list_corner = []
        figure_list_mode = []
        sample = 0

        print(f'{self.name}: Generating {self.num_posteriors} posterior distribution(s) for training set')

        for batch in iter(train_loader):

            parameters = batch["parameters"]
            weighting = batch["weighting"]
            conditioning = batch["conditioning"]

            if isinstance(conditioning, tuple):
                # convert tuple of lists to list of tuples
                conditioning = list(zip(*conditioning))

            for p, w, c in zip(parameters, weighting, conditioning):

                if sample >= self.num_posteriors:
                    break

                c = wrapper_expand(c, self.num_total_samples)

                samples, rng = sample_fn(self.num_total_samples, rng, z_init=None, conditioning=c)

                x = samples['samples']

                figure_list_corner.append(self._make_corner_plot(x, p))

                figure_list_mode.append(self._make_mode_plot(x, c[1][0], c[0][0]))

                sample += 1

            if sample >= self.num_posteriors:
                break

        logs['posterior train'] = [wandb.Image(fig) for fig in figure_list_corner]
        logs['mode train'] = [wandb.Image(fig) for fig in figure_list_mode]

        for fig in figure_list_corner:
            plt.close(fig)
        for fig in figure_list_mode:
            plt.close(fig)

        return logs, rng

    def on_train_begin(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):

        if self.on_train_start:
            return self.__call__(logs, rng, *args, init=True, **kwargs)
        else:
            return logs, rng

    def on_epoch_end(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):

        if "epoch" in logs and logs["epoch"] % self.save_every == 0:
            return self.__call__(logs, rng, *args, **kwargs)
        else:
            return logs, rng


def assemble_lens_parameters(parameters, mode, features):
    res = parameters.at[jnp.array(features)].set(mode.reshape(-1))
    return res


class LensParameterMode(Callback):
    features: List[int]

    name: str = 'lens_parameter_dataset_mode'
    save_every: int = 10

    num_total_samples: int = 25
    batch_size: int = 4

    on_train_start: bool = False

    def compute_log_likelihood_per_pixel(self, x, image):
        return - (jnp.sum((x - image) ** 2) / (self.noise ** 2)) / jnp.prod(jnp.array(image.shape))

    def __init__(self, features: List[int], save_every: int = 10, noise: float = 0.005, on_train_start: bool = False):
        super(self.__class__, self).__init__()


        self.on_train_start = on_train_start
        self.features = features
        self.save_every = save_every
        self.num_observations = 3
        self.num_samples = 5 # TODO increase
        self.simu_fn = get_simu_images()
        self.noise = noise

    def comparison_image(self, rng, strategy, loader, key):

        data = loader[key]

        c = wrapper_expand(data['conditioning'], self.num_samples)

        sample, rng = strategy.sample(self.num_samples, rng, z_init=None,
                                 conditioning=c)

        x = sample['samples']

        modes = stats.mode(x, axis=0, keepdims=True)
        mode = modes[0]

        params_pred = assemble_lens_parameters(data['parameters'], mode, data['features'])
        params_gt = data['parameters']

        image_pred = self.simu_fn(jnp.array(params_pred)[None])
        image_gt = self.simu_fn(jnp.array(params_gt)[None])

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(image_pred[0], cmap='bone')
        ax[0].set_title('Predicted')

        ax[1].imshow(image_gt[0], cmap='bone')
        ax[1].set_title('Ground Truth')

        im = ax[2].imshow(image_pred[0] - image_gt[0], cmap='bwr', vmin=-7 * self.noise, vmax=7 * self.noise)
        ax[2].set_title('Difference')

        # set colorbar for ax[2]
        cbar = plt.colorbar(im, ax=ax[2])

        # set ticks based on self.noise
        cbar.set_ticks([-7 * self.noise, 0, 7 * self.noise])
        cbar.set_ticklabels([r'-7$\sigma$', '0', r'7$\sigma$'])

        log_likelihood = self.compute_log_likelihood_per_pixel(image_pred, image_gt)

        # add log likelihood to title
        ax[0].set_title(f'Predicted - log likelihood: {log_likelihood:.5f}')

        return rng, fig

    def __call__(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return self._call(logs, rng, *args, **kwargs)

    def _call(self, logs: dict, rng: jr.PRNGKey, strategy: Strategy,
              train_loader: DataLoader, val_loader: DataLoader, init=True, *args, **kwargs) \
                -> Tuple[dict, jr.PRNGKey]:

        # TRAIN LOADER

        figure_list = []

        for i in range(self.num_observations):
            rng, fig = self.comparison_image(rng, strategy, train_loader, i)

            figure_list.append(fig)

        logs['train_comparison_mode'] = [wandb.Image(fig) for fig in figure_list]

        for fig in figure_list:
            plt.close(fig)

        # VAL LOADER

        figure_list = []

        for i in range(self.num_observations):
            rng, fig = self.comparison_image(rng, strategy, val_loader, i)

            figure_list.append(fig)

        logs['val_comparison_mode'] = [wandb.Image(fig) for fig in figure_list]

        for fig in figure_list:
            plt.close(fig)

        return logs, rng

    def on_train_begin(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):

        if self.on_train_start:
            return self.__call__(logs, rng, *args, init=True, **kwargs)
        else:
            return logs, rng

    def on_epoch_end(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):

        if "epoch" in logs and logs["epoch"] % self.save_every == 0:
            return self.__call__(logs, rng, *args, **kwargs)
        else:
            return logs, rng

class LensParameterSimulation(Callback):

    name: str = 'lens_parameter_simulation'
    save_every: int = 10
    on_train_start: bool = False


    def compute_log_likelihood_per_pixel(self, x, image):
        return - (jnp.sum((x - image) ** 2) / (self.noise ** 2)) / jnp.prod(jnp.array(image.shape))

    def __init__(self, simulation: Dict, save_every: int = 10, noise: float = 0.005, on_train_start: bool = False):
        super(self.__class__, self).__init__()


        self.on_train_start = on_train_start
        self.simulation_model = instantiate_from_config(simulation["setup"])

        self.targets: SimulationTargets = (
            instantiate_from_config(simulation["target"]))

        self.target_mask = self.targets.get_target_mask()

        self.save_every = save_every
        self.num_observations = 2
        self.num_samples = 3
        self.simu_fn = get_simu_images()
        self.noise = noise

    def comparison_image(self, rng, strategy, loader, key):

        data = loader[key]

        c = wrapper_expand(data['conditioning'], self.num_samples)

        sample, rng = strategy.sample(self.num_samples, rng, z_init=None,
                                 conditioning=c)

        x = sample['samples']

        figure_list = []

        for i in range(self.num_samples):

            params_gt = data['conditioning'][1]
            observation = data['conditioning'][0]

            prediction_full = self.targets.assemble(params_gt, x[i])
            observation_pred = self.simulation_model.forward(prediction_full, rng, False, True)

            nll = self.simulation_model.compute_log_likelihood_per_pixel(observation_pred, observation)

            figure_list.append(lens_plot(observation_pred, observation[0], nll, self.simulation_model.get_noise_std()))

        return rng, figure_list

    def __call__(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):
        return self._call(logs, rng, *args, **kwargs)

    def _call(self, logs: dict, rng: jr.PRNGKey, strategy: Strategy,
              train_loader: DataLoader, val_loader: DataLoader, init=True, *args, **kwargs) \
                -> Tuple[dict, jr.PRNGKey]:


        print(f'{self.name}: Generating {self.num_samples} sample(s) for training set')

        for i in range(self.num_observations):
            rng, figure_list = self.comparison_image(rng, strategy, train_loader, i)
            logs[f'train_comparison_sample_{i}'] = [wandb.Image(fig) for fig in figure_list]
            for fig in figure_list:
                plt.close(fig)

        print(f'{self.name}: Generating {self.num_samples} sample(s) for validation set')

        for i in range(self.num_observations):
            rng, figure_list = self.comparison_image(rng, strategy, val_loader, i)
            logs[f'val_comparison_sample_{i}'] = [wandb.Image(fig) for fig in figure_list]
            for fig in figure_list:
                plt.close(fig)

        return logs, rng

    def on_train_begin(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):

        if self.on_train_start:
            return self.__call__(logs, rng, *args, init=True, **kwargs)
        else:
            return logs, rng

    def on_epoch_end(self, logs: dict, rng: jr.PRNGKey, *args, **kwargs):

        if "epoch" in logs and logs["epoch"] % self.save_every == 0:
            return self.__call__(logs, rng, *args, **kwargs)
        else:
            return logs, rng

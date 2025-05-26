from abc import ABC, abstractmethod
from math import ceil
from pathlib import Path
from typing import Iterable, Tuple

import jax.random as jr
import jax.numpy as jnp

import pandas as pd


class GeneratorDataloader(Iterable):
    num_samples: int
    rng: jr.PRNGKey
    num_steps: int
    batch_size: int
    X: jnp.ndarray
    y: jnp.ndarray
    normalize: bool

    def __iter__(self):
        self.current_step = 1

        if not self.replacement:
            perm = jr.permutation(self.rng, self.X.shape[0])
            self.X = self.X[perm]
            self.y = self.y[perm]
            self.rng = jr.split(self.rng)[0]

        return self

    def load_file(self, file):

        data = pd.read_csv(file)

        return jnp.array(data.values)

    def get_observation(self, idx):

        base_dir = Path(self.DATA_ROOT).joinpath(self.dataset).joinpath(f'num_observation_{idx}')

        if not base_dir.exists():
            raise FileNotFoundError(f"Directory {base_dir} does not exist")

        observation = self.load_file(base_dir.joinpath(f"observation.csv"))

        true_theta = self.load_file(base_dir.joinpath(f"true_parameters.csv"))

        reference_posterior = self.load_file(base_dir.joinpath(f"reference_posterior_samples.csv"))

        if self.normalize:
            true_theta = (true_theta - self.mean_X) / self.std_X
            reference_posterior = (reference_posterior - self.mean_X) / self.std_X
            observation = (observation - self.mean_y) / self.std_y

        return observation, true_theta, reference_posterior

    def __init__(self, dataset: str, DATA_ROOT: str, num_samples: int, split: Tuple[int], seed: jr.PRNGKey,
                 num_steps: int, batch_size: int, normalize: bool = True, replacement: bool = False):
        """
        :param dataset:
        :param DATA_ROOT:
        :param num_samples:
        :param split:
        :param seed:
        :param num_steps:
        :param batch_size:
        :param normalize:
        :param replacement:
        """


        path = Path(DATA_ROOT)
        path = path.joinpath(dataset)

        if num_samples not in [1000, 10000, 100000, 1000000, 10000000]:
            raise ValueError(f"num_samples ({num_samples}) must be in [1000, 10000, 100000, 1000000]")

        self.replacement = replacement

        self.normalize = normalize
        self.DATA_ROOT = DATA_ROOT
        self.dataset = dataset
        self.batch_size = batch_size
        self.rng = jr.PRNGKey(seed)

        self.current_step = 1
        self.dim_theta = 10

        split_start = split[0]
        split_end = split[1]

        self.y = self.load_file(path.joinpath(f"x_{num_samples}.csv"))
        self.X = self.load_file(path.joinpath(f"theta_{num_samples}.csv"))

        # samples seem to be sorted when loading from file; make sure to shuffle them!
        permutation_seed = jr.PRNGKey(0)
        perm = jr.permutation(permutation_seed, num_samples)
        self.X = self.X[perm]
        self.y = self.y[perm]

        self.mean_X = self.X.mean(axis=0)
        self.std_X = self.X.std(axis=0)

        self.mean_y = self.y.mean(axis=0)
        self.std_y = self.y.std(axis=0)

        if self.normalize:
            self.X = (self.X - self.mean_X) / self.std_X
            self.y = (self.y - self.mean_y) / self.std_y

        self.X = self.X[split_start:split_end]
        self.y = self.y[split_start:split_end]

        self.num_samples = self.X.shape[0]
        if not self.replacement:
            self.num_steps = int(ceil(self.num_samples / self.batch_size))
            print(f"Dataloader sampling without replacements (num_steps: {self.num_steps})")
        else:
            self.num_steps = num_steps

    def __next__(self):
        if self.current_step <= self.num_steps:

            if not self.replacement:
                X = self.X[self.batch_size * (self.current_step-1): self.batch_size * self.current_step]
                y = self.y[self.batch_size * (self.current_step-1): self.batch_size * self.current_step]
            else:
                X = self.X[jr.choice(self.rng, self.num_samples, (self.batch_size,))]
                y = self.y[jr.choice(self.rng, self.num_samples, (self.batch_size,))]
                self.rng = jr.split(self.rng)[0]

            self.current_step += 1
            return {'parameters': X, 'weighting': jnp.ones((self.batch_size, 1)), 'conditioning': y}

        else:

            raise StopIteration

    def __len__(self):
        return self.num_steps


class GaussianLinear(GeneratorDataloader):

    def __init__(self, *args, **kwargs):
        super().__init__("gaussian_linear", *args, **kwargs)


class BernoulliGLM(GeneratorDataloader):

    def __init__(self, *args, **kwargs):
        super().__init__("bernoulli_glm", *args, **kwargs)


class BernoulliGLMRaw(GeneratorDataloader):

    def __init__(self, *args, **kwargs):
        super().__init__("bernoulli_glm_raw", *args, **kwargs)

    def get_observation(self, idx):

        base_dir = Path(self.DATA_ROOT).joinpath(self.dataset).joinpath(f'num_observation_{idx}')

        if not base_dir.exists():
            raise FileNotFoundError(f"Directory {base_dir} does not exist")

        observation = self.load_file(base_dir.joinpath(f"observation_raw.csv"))

        true_theta = self.load_file(base_dir.joinpath(f"true_parameters.csv"))

        reference_posterior = self.load_file(base_dir.joinpath(f"reference_posterior_samples.csv"))

        if self.normalize:
            true_theta = (true_theta - self.mean) / self.std
            reference_posterior = (reference_posterior - self.mean) / self.std

        return observation, true_theta, reference_posterior


class GaussianLinearUniform(GeneratorDataloader):

    def __init__(self, *args, **kwargs):
        super().__init__("gaussian_linear_uniform", *args, **kwargs)


class GaussianMixture(GeneratorDataloader):

    def __init__(self, *args, **kwargs):
        super().__init__("gaussian_mixture", *args, **kwargs)


class LotkaVolterra(GeneratorDataloader):

    def __init__(self, *args, **kwargs):
        super().__init__("lotka_volterra", *args, **kwargs)

class LotkaVolterraJax(GeneratorDataloader):

    def __init__(self, *args, **kwargs):
        super().__init__("lotka_volterra_jax", *args, **kwargs)


class SIR(GeneratorDataloader):

    def __init__(self, *args, **kwargs):
        super().__init__("sir", *args, **kwargs)


class SLCP(GeneratorDataloader):

    def __init__(self, *args, **kwargs):
        super().__init__("slcp", *args, **kwargs)


class SLCPDistractors(GeneratorDataloader):

    def __init__(self, *args, **kwargs):
        super().__init__("slcp_distractors", *args, **kwargs)

    def get_observation(self, idx):

        base_dir = Path(self.DATA_ROOT).joinpath(self.dataset).joinpath(f'num_observation_{idx}')

        if not base_dir.exists():
            raise FileNotFoundError(f"Directory {base_dir} does not exist")

        observation = self.load_file(base_dir.joinpath(f"observation_distractors.csv"))

        true_theta = self.load_file(base_dir.joinpath(f"true_parameters.csv"))

        reference_posterior = self.load_file(base_dir.joinpath(f"reference_posterior_samples.csv"))

        if self.normalize:
            true_theta = (true_theta - self.mean) / self.std
            reference_posterior = (reference_posterior - self.mean) / self.std

        return observation, true_theta, reference_posterior


class TwoMoons(GeneratorDataloader):

    def __init__(self, *args, **kwargs):
        super().__init__("two_moons", *args, **kwargs)

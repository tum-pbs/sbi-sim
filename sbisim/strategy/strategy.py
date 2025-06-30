from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional

import jax.random as jr
import jax.numpy as jnp
from math import ceil

from tqdm import tqdm

from jaxtyping import PyTree

from ..optimization import Optimization


class Strategy(ABC):

    params: PyTree = None
    bound: bool = False
    batch_size: int = 128

    def bind(self, params: PyTree):
        self.params = params
        self.bound = True

    @abstractmethod
    def setup(self, opt: Optimization, example_data: PyTree, key: jr.PRNGKey, batch_size: int) \
            -> Tuple[PyTree, jr.PRNGKey]:
        pass

    @abstractmethod
    def train_step(self, i: int, opt_state: PyTree, rng: jr.PRNGKey, logs: Dict[str, Any],
                   batch: PyTree) -> Tuple[PyTree, jr.PRNGKey, jnp.ndarray]:
        pass

    @abstractmethod
    def eval_step(self, params: PyTree, rng: jr.PRNGKey, logs: Dict[str, Any],
                  batch: PyTree, testing: bool):
        pass

    @abstractmethod
    def loss_fn(self, params: PyTree, rng: jr.PRNGKey, batch: PyTree) \
            -> Tuple[jnp.ndarray, jr.PRNGKey]:
        pass

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def forward(self, x: jnp.ndarray, rng: jr.PRNGKey, ldj: Optional[jnp.ndarray]=None,
                 conditioning: Optional[jnp.ndarray]=None, batch_size: Optional[int] = None, **kwargs) \
            -> Tuple[PyTree, jr.PRNGKey]:
        if not self.bound:
            raise ValueError("Strategy not bound to parameters! Call .bind() first.")

        if batch_size is None:
            batch_size = self.batch_size

        _list = []

        size = x.shape[0]

        for n in range(ceil(size / batch_size)):

            x_ = x[n*batch_size:(n+1)*batch_size].astype(jnp.float32)

            if x_.shape[0] < batch_size:
                x_ = x_.repeat(ceil(batch_size / x_.shape[0]), axis = 0)[:batch_size]

            if ldj is not None:
                ldj_ = ldj[n*batch_size:(n+1)*batch_size].astype(jnp.float32)

                if ldj_.shape[0] < batch_size:
                    ldj_ = ldj_.repeat(ceil(batch_size / ldj_.shape[0]), axis = 0)[:batch_size]

            else:
                ldj_ = None

            if conditioning is not None:
                conditioning_ = conditioning[n*batch_size:(n+1)*batch_size].astype(jnp.float32)

                if conditioning_.shape[0] < batch_size:
                    conditioning_ = conditioning_.repeat(ceil(batch_size / conditioning_.shape[0]),
                                                         axis = 0)[:batch_size]

            else:
                conditioning_ = None

            data, rng = (self._forward(self.params, x_, rng, ldj_, conditioning_, **kwargs))

            _list.append(data)

        return self.collate(_list, max_elements=x.shape[0]), rng

    @abstractmethod
    def _forward(self, params: PyTree, x: jnp.ndarray, rng: jr.PRNGKey, *args, **kwargs) \
            -> Tuple[PyTree, jr.PRNGKey]:
        pass

    def collate(self, sample_list: list, max_elements: int) -> PyTree:

        if len(sample_list) == 0:
            return {}

        if len(sample_list) == 1:
            return {key: sample_list[0][key][:max_elements] for key in sample_list[0].keys()}

        keys = sample_list[0].keys()

        result = {}

        for key in keys:
            result[key] = jnp.concatenate([sample[key] for sample in sample_list], axis = 0)[:max_elements]

        return result

    def sample(self, num_samples: int, rng: jr.PRNGKey, *args, z_init: Optional[jnp.ndarray]=None,
               conditioning: Optional[jnp.ndarray] = None, batch_size: Optional[int] = None, **kwargs) -> Tuple[PyTree, jr.PRNGKey]:
        if not self.bound:
            raise ValueError("Strategy not bound to parameters! Call .bind() first.")

        if batch_size is None:
            batch_size = self.batch_size

        if z_init is not None and z_init.shape[0] != num_samples:
            raise ValueError("Initial samples must have the same number of samples as requested samples. "
                             "Found z_init.shape[0]: {} and num_samples {}".format(z_init.shape[0], num_samples))

        if isinstance(conditioning, tuple) or isinstance(conditioning, list):
            for _cond in conditioning:
                if _cond.shape[0] != num_samples:
                    raise ValueError("Conditioning must have the same number of samples as requested samples. "
                                     "Found conditioning.shape[0]: {} and num_samples {}".format(_cond.shape[0], num_samples))
        elif conditioning is not None and conditioning.shape[0] != num_samples:
            raise ValueError("Conditioning must have the same number of samples as requested samples. "
                             "Found conditioning.shape[0]: {} and num_samples {}".format(conditioning.shape[0], num_samples))

        _list = []

        pbar = tqdm(range(ceil(num_samples / batch_size)))
        for n in pbar:

            if z_init is not None:

                z_init_ = z_init[n*batch_size:(n+1)*batch_size]

                if z_init_.shape[0] < batch_size:
                    z_init_ = z_init_.repeat(ceil(batch_size / z_init_.shape[0]), axis = 0)[:batch_size]

            else:

                z_init_ = None

            if conditioning is not None:

                if isinstance(conditioning, tuple) or isinstance(conditioning, list):
                    conditioning_ = tuple([c[n*batch_size:(n+1)*batch_size] for c in conditioning])

                    def _expand(c):
                        if c.shape[0] < batch_size:
                            return c.repeat(ceil(batch_size / c.shape[0]), axis = 0)[:batch_size]
                        return c

                    conditioning_ = [_expand(c) for c in conditioning_]

                else:

                    conditioning_ = conditioning[n*batch_size:(n+1)*batch_size]

                    if conditioning_.shape[0] < batch_size:
                        conditioning_ = conditioning_.repeat(ceil(batch_size / conditioning_.shape[0]), axis = 0)[:batch_size]

            else:

                conditioning_ = None


            data, rng = (self._sample(self.params, batch_size, rng, *args,
                                      z_init_, conditioning_, **kwargs))

            _list.append(data)

        return self.collate(_list, max_elements=num_samples), rng

    @abstractmethod
    def _sample(self, params: PyTree, num_samples: int, rng: jr.PRNGKey, *args, **kwargs) \
            -> Tuple[PyTree, jr.PRNGKey]:
        pass

    def compute_likelihood(self, x: jnp.ndarray, rng: jr.PRNGKey, weighting: Optional[jnp.ndarray] = None,
                           conditioning: Optional[jnp.ndarray] = None, **kwargs) \
            -> Tuple[PyTree, jr.PRNGKey]:

        _list = []

        size = x.shape[0]

        pbar = tqdm(range(ceil(size / self.batch_size)))
        for n in pbar:

            x_ = x[n * self.batch_size:(n + 1) * self.batch_size]

            if x_.shape[0] < self.batch_size:
                x_ = x_.repeat(ceil(self.batch_size / x_.shape[0]), axis=0)[:self.batch_size]

            if weighting is not None:
                weighting_ = weighting[n * self.batch_size:(n + 1) * self.batch_size]

                if weighting_.shape[0] < self.batch_size:
                    weighting_ = weighting_.repeat(ceil(self.batch_size / weighting_.shape[0]),
                                                   axis=0)[:self.batch_size]

            else:
                weighting_ = None

            if conditioning is not None:

                if isinstance(conditioning, tuple):

                    conditioning_ = tuple([c[n * self.batch_size:(n + 1) * self.batch_size] for c in conditioning])

                    def _expand(c):
                        if c.shape[0] < self.batch_size:
                            return c.repeat(ceil(self.batch_size / c.shape[0]), axis = 0)[:self.batch_size]
                        return c

                    conditioning_ = [_expand(c) for c in conditioning_]

                else:

                    conditioning_ = conditioning[n * self.batch_size:(n + 1) * self.batch_size]

                    if conditioning_.shape[0] < self.batch_size:
                        conditioning_ = conditioning_.repeat(ceil(self.batch_size / conditioning_.shape[0]),
                                                             axis=0)[:self.batch_size]

            else:
                conditioning_ = None

            data, rng = (self._compute_likelihood(self.params, x_, rng, weighting_, conditioning_, **kwargs))

            _list.append(data)

        return self.collate(_list, max_elements=x.shape[0]), rng

    @abstractmethod
    def _compute_likelihood(self, params: PyTree, x: jnp.ndarray, rng: jr.PRNGKey, *args, **kwargs) \
            -> Tuple[PyTree, jr.PRNGKey]:
        pass

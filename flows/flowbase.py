from abc import ABC, abstractmethod
from functools import partial

from typing import Sequence, Tuple, Optional, Union

import flax.linen as nn
import jax.nn
import jax.numpy as jnp

import jax.random as jr
from jax import jit

from jax.scipy.stats.norm import logpdf

class FlowBlock(ABC, nn.Module):

    dim_flow: int
    dim_conditioning: Union[int, Tuple[int, ...]]

    def _zero_conditioning(self, batch_size: int) -> jnp.ndarray:
        if isinstance(self.dim_conditioning, int):
            return jnp.zeros((batch_size, self.dim_conditioning))
        else:
            return jnp.zeros(((batch_size,) + tuple(self.dim_conditioning)))
    def forward(self, x: jnp.ndarray, rng: jr.PRNGKey,
                 ldj: Optional[jnp.ndarray]=None,
                 conditioning: Optional[jnp.ndarray]=None) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:
        if ldj is None:
            ldj = jnp.zeros(x.shape[0])

        if conditioning is None:
            conditioning = self._zero_conditioning(x.shape[0])

        return self.forward_(x, rng, ldj, conditioning)

    def inverse(self, z: jnp.ndarray, rng: jr.PRNGKey,
                ldj: Optional[jnp.ndarray]=None,
                conditioning: Optional[jnp.ndarray]=None) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:
        if ldj is None:
            ldj = jnp.zeros(z.shape[0])
        if conditioning is None:
            conditioning = self._zero_conditioning(z.shape[0])
        return self.inverse_(z, rng, ldj, conditioning)

    @abstractmethod
    def forward_(self, x: jnp.ndarray, rng: jr.PRNGKey,
               ldj: jnp.ndarray,
               conditioning: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:
        pass

    @abstractmethod
    def inverse_(self, z: jnp.ndarray, rng: jr.PRNGKey,
               ldj: jnp.ndarray,
               conditioning: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:
        pass


class FlowBase(FlowBlock, ABC):

    dim_flow: int
    dim_conditioning: int
    import_samples: int

    def compute_likelihood(self, x: jnp.ndarray, rng: jr.PRNGKey,
                           weighting: Optional[jnp.ndarray] = None,
                           conditioning: Optional[jnp.ndarray] = None) \
            -> Tuple[jnp.ndarray, jr.PRNGKey]:
        if weighting is None:
            weighting = jnp.ones(x.shape[0])
        if conditioning is None:
            conditioning = self._zero_conditioning(x.shape[0])
        return self.compute_likelihood_(x, rng, weighting, conditioning)

    def compute_likelihood_(self, x: jnp.ndarray, rng: jr.PRNGKey,
                           weighting: jnp.ndarray,
                           conditioning: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jr.PRNGKey]:
        """
        Compute the negative log likelihood of a batch of data samples
        :param x: data samples
        :param rng:
        :param weighting: weighting of the samples
        :param conditioning: conditioning of the samples
        """
        z, ldj, rng = self.encode_(x, rng, conditioning=conditioning)
        log_pz = logpdf(z).sum(axis=1)
        log_px = ldj + log_pz

        if not weighting is None:
            log_px = weighting * log_px

        nll = -log_px

        return nll, rng

    def __call__(self, x: jnp.ndarray, rng: jr.PRNGKey,
                 weighting: jnp.ndarray,
                 conditioning: jnp.ndarray,
                 testing=False, bpd=False) \
            -> Tuple[jnp.ndarray, jr.PRNGKey]:
        """
        Compute the negative log likelihood of a batch of data samples
        :param x: Contains the data, weighting and conditioning
        :param rng:
        :param testing: for testing, we use importance sampling to estimate the likelihood
        :param bpd: if True, return bits per dimension instead of negative log likelihood
        :param weighting: weighting of the samples
        :param conditioning: conditioning of the samples
        :return:
        """
        if not testing:
            nll, rng = self.compute_likelihood_(x, rng, weighting=weighting,
                                               conditioning=conditioning)

        else:

            # importance sampling during testing => estimate likelihood M times for each image
            x = x.repeat(self.import_samples, 0)

            weighting = weighting.repeat(self.import_samples, 0)

            conditioning = conditioning.repeat(self.import_samples, 0)

            nll, rng = self.compute_likelihood_(x, rng, weighting=weighting,
                                               conditioning=conditioning)

            nll = nll.reshape(-1, self.import_samples)

            # go from log-space to exp, and back to log.
            nll = jax.nn.logsumexp(nll, axis=-1) - jnp.log(self.import_samples)

        if not bpd:
            return nll, rng
        else:
            dims = x[0].shape[1:]

            bpd = -nll * jnp.log2(jnp.exp(1)) / jnp.prod(dims)

            return bpd, rng


    def sample(self, num_samples: int, rng: jr.PRNGKey,
               z_init: Optional[jnp.ndarray] = None,
               conditioning: Optional[jnp.ndarray] = None) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:
        """
        Sample a batch of samples from the flow.
        :param num_samples: Number of samples (if z_init is None)
        :param rng: 
        :param z_init: Initial latent samples (if None, sample from standard normal)
        :param conditioning: 
        :return: 
        """

        if z_init is None:
            rng, normal_rng = jr.split(rng)
            z_init = jr.normal(normal_rng, shape=(num_samples, self.dim_flow))

        if conditioning is None:
            conditioning = self._zero_conditioning(num_samples)

        return self.sample_(rng, z_init, conditioning)

    def sample_(self, rng: jr.PRNGKey,
               z_init: jnp.ndarray,
               conditioning: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:

        log_probabilities = logpdf(z_init).sum(axis=1)

        x, ldf, rng = self.inverse_(z_init, rng,
                                    log_probabilities,
                                    conditioning)


        return x, ldf, rng

    def encode(self, x: jnp.ndarray, rng: jr.PRNGKey,
               conditioning: Optional[jnp.ndarray] = None) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:
        """
        Encode a batch of data samples  (i.e. compute the forward flow)
        :param x: data samples
        :param rng:
        :param conditioning:

        :return: transformed data samples, transformed log probabilities and rng
        """
        if conditioning is None:
            conditioning = self._zero_conditioning(x.shape[0])

        return self.encode_(x, rng, conditioning)

    def encode_(self, x: jnp.ndarray, rng: jr.PRNGKey,
                conditioning: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:
        """
        Encode a batch of data samples  (i.e. compute the forward flow)
        :param x: data samples
        :param rng:
        :param conditioning:

        :return: transformed data samples, transformed log probabilities and rng
        """
        log_probabilities = jnp.zeros(x.shape[0])
        return self.forward_(x, rng, log_probabilities,
                             conditioning)

    def decode(self, z: jnp.ndarray, rng: jr.PRNGKey,
               conditioning: Optional[jnp.ndarray]=None) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:
        """
        Decode a batch of data samples  (i.e. compute the inverse flow)
        :param z: latent data samples
        :param rng:
        :param conditioning:

        :return: transformed data samples, transformed log probabilities and rng
        """
        if conditioning is None:
            conditioning = self._zero_conditioning(z.shape[0])

        return self.decode_(z, rng, conditioning)

    def decode_(self, z: jnp.ndarray, rng: jr.PRNGKey,
               conditioning: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:
        """
        Decode a batch of data samples  (i.e. compute the inverse flow)
        :param z: latent data samples
        :param rng:
        :param conditioning:

        :return: transformed data samples, transformed log probabilities and rng
        """
        log_probabilities = logpdf(z).sum(axis=1)
        return self.inverse_(z, rng, log_probabilities,
                             conditioning)

class NormalizingFlow(FlowBase, ABC):

    def forward_(self, x: jnp.ndarray, rng: jr.PRNGKey,
                ldj: jnp.ndarray,
                conditioning: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:

        z = x
        for flow in self.flows:
            z, ldj, rng = flow.forward_(z, rng, ldj,
                                        conditioning)
        return z, ldj, rng

    def inverse_(self, z: jnp.ndarray, rng: jr.PRNGKey,
                ldj: jnp.ndarray,
                conditioning: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:

        x = z
        for flow in reversed(self.flows):
            x, ldj, rng = flow.inverse_(x, rng, ldj,
                                        conditioning)
        return x, ldj, rng







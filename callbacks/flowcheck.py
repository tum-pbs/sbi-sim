from typing import Tuple

import jax.random as jr
from jax.numpy import allclose

from callbacks.callback import Callback
from flows.flowbase import FlowBase

from jax.scipy.stats.norm import logpdf

from strategy import NeuralPosteriorEstimation


class FlowCheck(Callback):

    name: str = 'flow_check'

    def __init__(self, interrupt: bool = True, num_samples: int = 4):
        self.interrupt = interrupt
        self.num_samples = num_samples

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    def _call(self, logs: dict, rng: jr.PRNGKey, strategy: NeuralPosteriorEstimation,
                 *args, **kwargs) -> Tuple[dict, jr.PRNGKey]:

        print("checking flow...")

        z = jr.normal(rng, (self.num_samples, strategy.get_flow_dimension()))
        rng = jr.split(rng)[0]

        z_prob = logpdf(z).sum(axis=-1)

        samples, rng = strategy.sample(self.num_samples, rng, z_init=z)
        x = samples['samples']
        x_prob = samples['likelihood']

        z_samples, rng = strategy.forward(x, rng, ldj=x_prob)
        z_rec = z_samples['samples']
        z_rec_prob = z_samples['likelihood']

        if self.interrupt:
            assert allclose(z_rec, z, atol=1e-4), "Error: Flow is not bijective!"
            assert allclose(z_rec_prob, z_prob, atol=1e-2), "Error: Probability transform is not correct!"

            print("flow check passed!")

        else:
            passed = True
            if not allclose(z_rec, z, atol=1e-4):
                print("Error: Flow is not bijective!")
                passed = False
            if not allclose(z_rec_prob, z_prob, atol=1e-2):
                print("Error: Probability transform is not correct!")
                passed = False

            if passed:
                print("flow check passed!")
            else:
                print("flow check failed!")

        return logs, rng

    def on_train_begin(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)
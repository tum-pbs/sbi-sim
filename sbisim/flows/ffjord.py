from functools import partial
from typing import Sequence

from flax import linen as nn
from jax import jit

from .cnf import ContinuousNormalizingFlow
from .flowbase import NormalizingFlow, FlowBase

kernel_init_fn = nn.initializers.glorot_normal
bias_init_fn = nn.initializers.normal

import jax.numpy as jnp

class _BaseFFJORD(nn.Module):

    data_size: int
    width_size: int
    depth: int

    def setup(self):

        layers = []

        if self.depth==0:
            layers.append(
                ConcatSquash(out_size=self.data_size)
            )
        else:
            layers.append(
                ConcatSquash(out_size=self.width_size)
            )
            for i in range(self.depth - 1):
                layers.append(
                    ConcatSquash(out_size=self.width_size)
                )
            layers.append(
                    ConcatSquash(out_size=self.data_size)
                )

        self.layers = layers

    def __call__(self, t, y):

        for layer in self.layers[:-1]:
            y = layer(t, y)
            y = nn.tanh(y)
        y = self.layers[-1](t, y)
        return y

class ConcatSquash(nn.Module):

    out_size: int

    def setup(self):
        self.lin1 = nn.Dense(self.out_size,
                        kernel_init=kernel_init_fn(),
                        bias_init=bias_init_fn())

        self.lin2 = nn.Dense(self.out_size,
                        kernel_init=kernel_init_fn(),
                        bias_init=bias_init_fn())

        self.lin3 = nn.Dense(self.out_size,
                        kernel_init=kernel_init_fn(),
                        use_bias=False)

    def __call__(self, t, y):

        return self.lin1(y) * nn.sigmoid(self.lin2(t)) + self.lin3(t)

class BaseFFJORD(nn.Module):

    dim_flow: int
    dim_conditioning: int

    width_size: int
    depth: int

    def setup(self):
        self.ffjord = _BaseFFJORD(
                                data_size=self.dim_flow,
                                width_size=self.width_size, depth=self.depth
        )

    def __call__(self, t, y, conditioning):

        conditioning = jnp.reshape(conditioning,
                                   (conditioning.shape[0], -1))
        t = jnp.ones((y.shape[0], 1)) * t
        y = jnp.concatenate([y, conditioning], axis=-1)
        return self.ffjord(t, y)

class FFJORD(ContinuousNormalizingFlow):

    width_size: int = 64
    depth: int = 3

    def setup(self):

        super().setup()

        self.model = BaseFFJORD(dim_flow=self.dim_flow, dim_conditioning=self.dim_conditioning,
                                width_size=self.width_size, depth=self.depth)

class StackedFFJORD(NormalizingFlow):

    mode: str = 'exact'
    init_stepsize: float = 0.1
    stepsize_controller_name: str = 'constant'
    solver_name: str = 'tsit5'
    steps: int = 1
    width_size: int = 64
    depth: int = 3

    def setup(self):

        super().setup()

        self.flows: Sequence[FlowBase] = []

        for i in range(self.steps):

            self.flows += (FFJORD(import_samples=self.import_samples, dim_flow=self.dim_flow,
                                  dim_conditioning=self.dim_conditioning, init_stepsize=self.init_stepsize,
                                  mode=self.mode, stepsize_controller_name=self.stepsize_controller_name,
                                  width_size=self.width_size, depth=self.depth),)
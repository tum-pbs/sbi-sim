from dataclasses import field
from typing import List, Tuple, Union, Sequence

from jax import Array, dtypes, random
from jax._src.nn.initializers import RealNumeric, DTypeLikeInexact, Initializer, _compute_fans, lecun_uniform
from jax._src import core

from diffusers.models.embeddings_flax import FlaxTimesteps
from ..cnf import ContinuousNormalizingFlow
import jax.numpy as jnp

import flax.linen as nn

kernel_init_fn = nn.initializers.glorot_normal
bias_init_fn = nn.initializers.normal

def uniform_shifted(scale: RealNumeric = 1e-3,
            dtype: DTypeLikeInexact = jnp.float_) -> Initializer:

  def init(key,
           shape: core.Shape,
           dtype: DTypeLikeInexact = dtype) -> Array:
    dtype = dtypes.canonicalize_dtype(dtype)
    return random.uniform(key, shape, dtype) * 2 * jnp.array(scale, dtype) - jnp.array(scale, dtype)
  return init

def pytorch_kernel_init(in_axis: Sequence[int] = -2,
        out_axis: Sequence[int] = -1,
        batch_axis: Sequence[int] = (),
        dtype: DTypeLikeInexact = jnp.float_) -> Initializer:

      def init(key,
              shape: core.Shape,
              dtype: DTypeLikeInexact = dtype) -> Array:
             dtype = dtypes.canonicalize_dtype(dtype)
             shape = core.canonicalize_shape(shape)
             dtype = dtypes.canonicalize_dtype(dtype)
             fan_in, fan_out = _compute_fans(shape, in_axis, out_axis, batch_axis)
             return (random.uniform(key, shape, dtype) * 2 * jnp.sqrt(jnp.array(1 / fan_in, dtype))
                     - jnp.sqrt(jnp.array(1 / fan_in, dtype)))
      return init

def pytorch_bias_init(
        in_features: int,
        dtype: DTypeLikeInexact = jnp.float_) -> Initializer:

    def init(key,
            shape: core.Shape,
            dtype: DTypeLikeInexact = dtype) -> Array:
        dtype = dtypes.canonicalize_dtype(dtype)
        return (random.uniform(key, shape, dtype) * 2 * jnp.sqrt(jnp.array(1 / in_features, dtype))
                - jnp.sqrt(jnp.array(1 / in_features, dtype)))

    return init



class GLU(nn.Module):

    dim: int

    def setup(self):

        self.dense1 = nn.Dense(self.dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn())
        self.dense2 = nn.Dense(self.dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn())

    def __call__(self, x, t, theta):

        x = self.dense1(x)
        y = self.dense2(jnp.concatenate([t, theta], axis=-1))

        return x * nn.sigmoid(y)

class BaseFMPE(nn.Module):

    residual_blocks: List[Tuple[int, int]]
    gelu: List[bool]
    time_embed_dim: int
    dim_flow: int

    def setup(self):

        self.time_proj = FlaxTimesteps(
            self.time_embed_dim, flip_sin_to_cos=True, freq_shift=0
        )

        layers = []

        self.upsample = nn.Dense(self.residual_blocks[0][0],
                                 kernel_init=kernel_init_fn(), use_bias=False)

        for i, (out_dim, repetitions) in enumerate(self.residual_blocks):

            for j in range(repetitions):

                # Residual block
                layers.append(nn.Dense(out_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn()))
                layers.append(nn.Dense(out_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn()))

                if self.gelu[i]:
                    layers.append(GLU(out_dim))

            if i < len(self.residual_blocks) - 1:

                layers.append(nn.Dense(self.residual_blocks[i+1][0], kernel_init=kernel_init_fn(),
                                       bias_init=bias_init_fn()))


        layers.append(nn.Dense(self.dim_flow, kernel_init=kernel_init_fn(), use_bias=False))

        self.layers = layers

    def __call__(self,
            timesteps: Union[jnp.ndarray, float, int],
            theta: jnp.ndarray,
            y: jnp.ndarray,
            train: bool = False,):

        if not isinstance(timesteps, jnp.ndarray):
            timesteps = jnp.array([timesteps], dtype=jnp.int32)
        elif isinstance(timesteps, jnp.ndarray) and len(timesteps.shape) == 1:
            timesteps = timesteps.astype(dtype=jnp.float32)
            timesteps = jnp.expand_dims(timesteps, 1)

        if len(timesteps.shape) == 1:
            timesteps = jnp.repeat(timesteps, y.shape[0], axis=0)

        t_emb = self.time_proj(timesteps[:, 0])
        t_emb = jnp.concatenate([t_emb, timesteps[:, 1:]], axis=-1)

        y = jnp.reshape(y, (y.shape[0], -1))

        if y.shape[1] == 0:
            y = jnp.zeros((y.shape[0], 1))

        y = self.upsample(y)

        c = 0

        for i, (out_dim, repetitions) in enumerate(self.residual_blocks):

                for j in range(repetitions):

                    y_ = y
                    y = self.layers[c](y)
                    y = nn.activation.elu(y)
                    y = self.layers[c+1](y)
                    y = nn.activation.elu(y)
                    y += y_

                    c += 2

                    if self.gelu[i]:
                        y = self.layers[c](y, t_emb, theta)
                        c += 1

                if i < len(self.residual_blocks) - 1:
                    y = self.layers[c](y)
                    c += 1

        y = self.layers[-1](y)
        return y

class FMPE(ContinuousNormalizingFlow):

    residual_blocks: List[Tuple[int, int]] = field(default_factory=list)
    gelu: List[bool] = field(default_factory=list)
    time_embed_dim: int = 16

    def setup(self):
        super().setup()

        self.model = BaseFMPE(
            residual_blocks=self.residual_blocks,
            gelu=self.gelu,
            time_embed_dim=self.time_embed_dim,
            dim_flow=self.dim_flow
        )

class BaseBenchmarkFMPE(nn.Module):

    hidden_dim: int
    num_blocks: int
    out_dim: int
    activation_fn: str = 'elu'

    def setup(self):

        blocks = []

        if self.activation_fn == 'elu':
            self.activation = nn.elu
        elif self.activation_fn == 'gelu':
            self.activation = nn.gelu
        else:
            raise ValueError(f"Activation function {self.activation_fn} not supported")

        self.dense_in = nn.Dense(self.hidden_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn())
        self.dense_out = nn.Dense(self.out_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn())

        for i in range(self.num_blocks):

            block = [
                nn.Dense(self.hidden_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn()),
                nn.Dense(self.hidden_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn())
            ]

            blocks.append(block)

        self.blocks = blocks

    def __call__(self, t, theta, x=None, train=True):

        if not isinstance(t, jnp.ndarray):
            t = jnp.array([t], dtype=theta.dtype)
        elif isinstance(t, jnp.ndarray) and len(t.shape) == 0:
            t = t.astype(dtype=theta.dtype)
            t = jnp.expand_dims(t, 0)

        if len(t.shape) == 1:
            t = jnp.expand_dims(t, 1)

        if x is None:
            x = jnp.zeros((theta.shape[0], 0))

        x = jnp.concatenate([x, t, theta], axis=-1)

        x = self.dense_in(x)

        for block in self.blocks:

            x_ = x

            for layer in block:
                x = layer(x)
                x = self.activation(x)

            x += x_

        x = self.dense_out(x)

        return x

class DenseResidualBlocks(nn.Module):

    hidden_dim: int
    activation_fn: str = 'elu'
    context_dim: int = 0

    def setup(self) -> None:

        if self.activation_fn == 'elu':
            self.activation = nn.elu
        elif self.activation_fn == 'gelu':
            self.activation = nn.gelu
        else:
            raise ValueError(f"Activation function {self.activation_fn} not supported")

        self.layer1 = nn.Dense(self.hidden_dim, kernel_init=lecun_uniform(),
                               bias_init=pytorch_bias_init(self.hidden_dim))
        self.layer2 = nn.Dense(self.hidden_dim, kernel_init=uniform_shifted(), bias_init=uniform_shifted())

        if self.context_dim > 0:
            self.context_layer = nn.Dense(self.hidden_dim, kernel_init=lecun_uniform(),
                                          bias_init=pytorch_bias_init(self.context_dim))

    def __call__(self, x, context = None):

        x_skip = x
        x = self.activation(x)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)

        if context is not None:
            x = nn.glu(jnp.concatenate(
                [x, self.context_layer(context)], axis=1), axis=1)

        return x + x_skip

class Identity(nn.Module):

        def __call__(self, x):
            return x

class SBIResidualNet(nn.Module):

    hidden_dims: List[int]
    out_dim: int
    in_dim: int
    activation_fn: str = 'elu'

    def setup(self):

        self.model = DenseResidualNet(
            hidden_dims=self.hidden_dims,
            out_dim=self.out_dim,
            in_dim=self.in_dim,
            context_dim=0,
            activation_fn=self.activation_fn
        )

    def __call__(self, t, theta, x, train=True):

        if not isinstance(t, jnp.ndarray):
            t = jnp.array([t], dtype=x.dtype)
        elif isinstance(t, jnp.ndarray) and len(t.shape) == 0:
            t = t.astype(dtype=x.dtype)
            t = jnp.expand_dims(t, 0)

        if len(t.shape) == 1:
            t = jnp.expand_dims(t, 1)

        x = jnp.concatenate([x, t, theta], axis=-1)

        x = self.model(x, context=None, train=train)

        return x

class DenseResidualNet(nn.Module):

    hidden_dims: List[int]
    out_dim: int
    in_dim: int
    context_dim: int = 0
    activation_fn: str = 'elu'

    def setup(self):

        blocks = []
        projections = []

        self.dense_in = nn.Dense(self.hidden_dims[0], kernel_init=lecun_uniform(),
                                 bias_init=pytorch_bias_init(self.in_dim))

        for hidden_dim, hidden_dim_next in zip(self.hidden_dims, list(self.hidden_dims[1:]) + [self.out_dim]):
            blocks.append(DenseResidualBlocks(hidden_dim, context_dim = self.context_dim,
                                              activation_fn=self.activation_fn))
            if hidden_dim != hidden_dim_next:
                projections.append(nn.Dense(hidden_dim_next, use_bias=True,
                                           kernel_init=lecun_uniform(),
                                           bias_init=pytorch_bias_init(hidden_dim)))
            else:
                projections.append(Identity())

        self.blocks = blocks
        self.projections = projections

    def __call__(self, theta, context = None, train=True):

        x = self.dense_in(theta)

        for block, projection in zip(self.blocks, self.projections):

            x = block(x, context=context)
            x = projection(x)

        return x

class ShortcutMLP(nn.Module):

    hidden_dims: List[int]
    out_dim: int
    in_dim: int
    activation_fn: str = 'elu'

    def setup(self):

        self.model = DenseResidualNet(
            hidden_dims=self.hidden_dims,
            out_dim=self.out_dim,
            in_dim=self.in_dim,
            context_dim=0,
            activation_fn=self.activation_fn
        )

    def __call__(self, x, y, t, d, train=True):

        if not isinstance(t, jnp.ndarray):
            t = jnp.array([t], dtype=x.dtype)
        elif isinstance(t, jnp.ndarray) and len(t.shape) == 0:
            t = t.astype(dtype=x.dtype)
            t = jnp.expand_dims(t, 0)
        if len(t.shape) == 1:
            t = jnp.expand_dims(t, 1)

        if not isinstance(d, jnp.ndarray):
            d = jnp.array([d], dtype=x.dtype)
        elif isinstance(d, jnp.ndarray) and len(d.shape) == 0:
            d = d.astype(dtype=x.dtype)
            d = jnp.expand_dims(d, 0)
        if len(d.shape) == 1:
            d = jnp.expand_dims(d, 1)

        x = jnp.concatenate([x, t, d, y], axis=-1)

        x = self.model(x, context=None, train=train)

        return x

class BenchmarkFMPE(ContinuousNormalizingFlow):

    hidden_dim: int = 64
    num_blocks: int = 10
    out_dim: int = 2

    def setup(self):
        super().setup()

        self.model = BaseBenchmarkFMPE(
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            out_dim=self.out_dim
        )
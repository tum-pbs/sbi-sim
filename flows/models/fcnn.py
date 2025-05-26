from flax import linen as nn
import jax.nn

import jax.numpy as jnp

kernel_init_fn = nn.initializers.glorot_normal
bias_init_fn = nn.initializers.normal
# bias_init_fn = nn.initializers.zeros_init

class FCNN(nn.Module):

    out_dim: int
    hidden_dim: int
    num_layers: int = 5
    in_dim: int = 1

    def setup(self):

        layers = [nn.Dense(self.hidden_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn())
                  for _ in range(self.num_layers - 1)]

        layers.append(nn.Dense(self.out_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn()))

        self.layers = [nn.Dense(self.hidden_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn()),
                       nn.Dense(self.hidden_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn()),
                       nn.Dense(self.hidden_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn()),
                       nn.Dense(self.hidden_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn()),
                       nn.Dense(self.out_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn())]

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = jax.nn.relu(x)
        return x

class CNFFCNN(nn.Module):
    in_dim: int
    out_dim: int
    hidden_dim: int

    def setup(self):

        self.layers = [nn.Dense(self.hidden_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn()),
                       nn.Dense(self.hidden_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn()),
                       nn.Dense(self.hidden_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn()),
                       nn.Dense(self.hidden_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn()),
                       nn.Dense(self.out_dim, kernel_init=kernel_init_fn(), bias_init=bias_init_fn())]

    def __call__(self, t, x, conditioning):

        t = jnp.ones((x.shape[0], 1)) * t
        x = jnp.concatenate([t, x, conditioning], axis=-1)

        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = jax.nn.relu(x)
        return x
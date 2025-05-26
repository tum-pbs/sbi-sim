
import jax.numpy as jnp

import jax.random as jr

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

class SimpleMoonsDataLoader:

    def __init__(self, seed, num_samples, num_steps, batch_size, shuffle, conditioning=False):

        self.num_steps = num_steps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.conditioning = conditioning
        import matplotlib.pyplot as plt
        self.num_samples = num_samples
        noisy_moons = datasets.make_moons(n_samples=num_samples, noise=.05)
        X, y = noisy_moons

        self.X = jnp.array(StandardScaler().fit_transform(X))
        self.y = jnp.array(y)

        self.rng = jr.PRNGKey(seed)

        self.current_step = 1

    def __iter__(self):
        self.current_step = 1
        return self

    def __next__(self):
        if self.current_step <= self.num_steps:
            self.current_step += 1
            if self.shuffle:
                batch = self.X[jr.choice(self.rng, self.num_samples, (self.batch_size,))]
                batch_y = self.y[jr.choice(self.rng, self.num_samples, (self.batch_size,))][:, None]
                self.rng = jr.split(self.rng)[0]
            else:
                idx = (self.current_step * self.batch_size) % self.num_samples
                batch = self.X[idx:idx+self.batch_size]
                batch_y = self.y[idx:idx+self.batch_size][:, None]

            if not self.conditioning:
                batch_y = jnp.zeros((batch.shape[0], 0))

            return {'parameters': batch, 'weighting': jnp.ones((batch.shape[0], 1)), 'conditioning': batch_y}
        else:
            raise StopIteration

    def __len__(self):
        return self.num_steps
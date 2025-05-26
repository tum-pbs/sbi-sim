from torch.utils.data import DataLoader
import numpy as np
import jax.random as jr
import jax.numpy as jnp
class InverseParabolaDataLoader:

    def __init__(self, seed, num_samples, num_steps, batch_size, shuffle):

        self.num_steps = num_steps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = num_samples

        self.Y = np.random.random(self.num_samples)
        sign = (- np.ones((self.num_samples,))) ** np.random.randint(2, size=self.num_samples)
        self.X = np.sqrt(self.Y) * sign



        self.rng = jr.PRNGKey(seed)

        self.current_step = 1

    def __iter__(self):
        self.current_step = 1
        return self

    def __next__(self):
        if self.current_step <= self.num_steps:
            self.current_step += 1
            if self.shuffle:
                batch = self.X[jr.choice(self.rng, self.num_samples, (self.batch_size,))][:, None]
                batch_y = self.Y[jr.choice(self.rng, self.num_samples, (self.batch_size,))][:, None]
                self.rng = jr.split(self.rng)[0]
            else:
                idx = (self.current_step * self.batch_size) % self.num_samples
                batch = self.X[idx:idx+self.batch_size][:, None]
                batch_y = self.Y[idx:idx+self.batch_size][:, None]

            return {'parameters': batch, 'weighting': jnp.ones((batch.shape[0], 1)),
                    'conditioning': batch_y}
        else:
            raise StopIteration

    def __len__(self):
        return self.num_steps
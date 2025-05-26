
import jax.numpy as jnp
from jax import vmap

from tqdm import tqdm
import numpy as np

import time

from utils import instantiate_from_config
from torch.utils.data import Dataset, DataLoader
import jax.random as jr

class MoonsDataset(Dataset):

    def __init__(self, seed, num_samples):

        self.num_samples = num_samples

        rng = jr.PRNGKey(seed)

        theta = jr.choice(rng, jnp.array([-1.0,1.0]), shape=(num_samples,2), p=jnp.array([0.5, 0.5]), replace=True)

        # theta = jr.uniform(rng, (num_samples,2), minval=-1, maxval=1)
        rng, _ = jr.split(rng)

        a = jr.uniform(rng, (num_samples,), minval=-jnp.pi/2, maxval=jnp.pi/2)
        rng, _ = jr.split(rng)
        r = 0.1 + 0.01 * jr.normal(rng, (num_samples,))
        p = jnp.stack([r * jnp.cos(a), r * jnp.sin(a)], axis=1)
        x = p + jnp.stack([-jnp.abs(theta[:,0]+theta[:,1])/jnp.sqrt(2),
                           jnp.abs((-theta[:,0]+theta[:,1]))/jnp.sqrt(2)], axis=1)

        self.data = x
        self.weights = jnp.ones(num_samples)
        self.conditional_data = (x[:,0] * x[:,1] > 0).astype(jnp.float32)

    def load(self, key):
        return self.data[key], self.weights[key], self.conditional_data[key]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.load(index)

def get_dataloader(**config):

    dataset = MoonsDataset(config["seed"], config["num_samples"])

    def collate_fn(batch):

        data, weights, conditional = zip(*batch)
        return jnp.stack(data), jnp.stack(weights), jnp.stack(conditional)

    dataloader = DataLoader(dataset, collate_fn=collate_fn,
                                  batch_size=config["batch_size"], shuffle=config["shuffle"])

    return dataloader
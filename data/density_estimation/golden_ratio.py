from typing import Iterable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

class IsotropicGaussianMixtureModel:

    def __init__(self, means, stds, probabilities=None):

        self.means = np.array(means)
        self.stds = np.array(stds)
        self.probabilities = probabilities

        if len(means) != len(stds):
            raise ValueError("The number of means and standard deviations must be the same.")

        if len(means) < 1:
            raise ValueError("The number of components must be at least 1.")

        if probabilities is not None:
            if len(means) != len(probabilities):
                raise ValueError("The number of means and probabilities must be the same.")

            if not np.isclose(np.sum(probabilities), 1):
                raise ValueError("The sum of probabilities must be 1.")
        else:
            self.probabilities = np.ones(len(means)) / len(means)

    def sample(self, n_samples):

        z = np.random.normal(0, 1, size=(n_samples, 2))
        z_x, z_y = z[:, 0], z[:, 1]

        idx = np.random.choice(len(self.means), n_samples, p=self.probabilities).astype(np.int32)

        samples_x = z_x * self.stds[idx] + self.means[idx, 0]
        samples_y = z_y * self.stds[idx] + self.means[idx, 1]

        samples = np.column_stack((samples_x, samples_y))

        return samples

    def likelihood(self, x: jnp.ndarray) -> jnp.ndarray:

        means_ = jnp.array(jnp.expand_dims(self.means, axis=0)).repeat(x.shape[0], axis=0)
        stds_ = jnp.array(jnp.expand_dims(self.stds, axis=0)).repeat(x.shape[0], axis=0)
        probabilities_ = jnp.array(jnp.expand_dims(self.probabilities, axis=0)).repeat(x.shape[0], axis=0)

        stds_ = jnp.expand_dims(stds_, axis=2).repeat(x.shape[1], axis=2)
        probabilities_ = jnp.expand_dims(probabilities_, axis=2).repeat(x.shape[1], axis=2)
        x = jnp.expand_dims(x, axis=1).repeat(len(self.means), axis=1)

        inner = jnp.einsum('ijk,ijk->ij', (x - means_) / stds_, (x - means_) / stds_)
        inner = jnp.exp(- 0.5 * inner)

        likelihood = probabilities_[:, :, 0] * inner

        return jnp.sum(likelihood, axis=1)

    def grad_log_likelihood(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute the gradient of the likelihood of the Gaussian Mixture Model with respect to the input x.
        :param x:
        :return:
        """

        grad_fn = jax.grad(lambda y: jnp.log(self.likelihood(jnp.expand_dims(y, axis=0)))[0])
        vmap_fn = jax.vmap(grad_fn, in_axes=(0,))
        out = vmap_fn(x)

        return out, self.likelihood(x)


    def grad_likelihood(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute the gradient of the likelihood of the Gaussian Mixture Model with respect to the input x.
        :param x:
        :return:
        """

        means_ = jnp.array(jnp.expand_dims(self.means, axis=0)).repeat(x.shape[0], axis=0)
        stds_ = jnp.array(jnp.expand_dims(self.stds, axis=0)).repeat(x.shape[0], axis=0)
        probabilities_ = jnp.array(jnp.expand_dims(self.probabilities, axis=0)).repeat(x.shape[0], axis=0)

        stds_ = jnp.expand_dims(stds_, axis=2).repeat(x.shape[1], axis=2)
        probabilities_ = jnp.expand_dims(probabilities_, axis=2).repeat(x.shape[1], axis=2)
        x = jnp.expand_dims(x, axis=1).repeat(len(self.means), axis=1)

        inner = jnp.einsum('ijk,ijk->ij', (x - means_) / stds_, (x - means_) / stds_)

        inner = jnp.exp(- 0.5 * inner)

        inner_ = jnp.expand_dims(inner, axis=2).repeat(x.shape[2], axis=2)

        grad = probabilities_ * ((x-means_) / stds_) * inner_
        likelihood = probabilities_[:,:,0] * inner

        return jnp.sum(grad, axis=1), jnp.sum(likelihood, axis=1)

    def pdf(self, x):

        pdf = 0

        for i in range(len(self.means)):
            pdf += self.probabilities[i] * np.exp(-0.5 * np.power((x - self.means[i]) / self.stds[i], 2)) / (
                        self.stds[i] * np.sqrt(2 * np.pi))

        return pdf

def get_centers(fib_sequence):
    width = fib_sequence[0]

    top_right_corner = (width, width)

    corners = [top_right_corner]

    count = 0

    for i in range(1, len(fib_sequence)):

        new_width = fib_sequence[i]

        corner_x, corner_y = top_right_corner

        case = count % 4

        if case == 0:
            top_right_corner = (corner_x + new_width, corner_y)
        elif case == 1:
            top_right_corner = (corner_x, corner_y - width)
        elif case == 2:
            top_right_corner = (corner_x - width, corner_y - (width - new_width))
        elif case == 3:
            top_right_corner = (corner_x - (width - new_width), corner_y + new_width)

        corners.append(top_right_corner)

        width = new_width

        count += 1

    centers = []

    for i, corner in enumerate(corners):
        x, y = corner

        width = fib_sequence[i]

        centers.append((x - width / 2, y - width / 2))

    return centers


def fibonacci_ratio(n: int):
    phi = (1 + np.sqrt(5)) / 2
    return np.array([np.power(phi, -i) for i in range(n)])

class GoldenRatioDataset(Iterable):

    def __init__(self, num_steps: int, batch_size: int,
                 n: int = 7, std_factor: float = 0.1):

        self.num_steps = num_steps
        self.batch_size = batch_size
        self.n = n
        self.std_factor = std_factor

        self.fibonacci = fibonacci_ratio(n)
        self.centers = get_centers(self.fibonacci[:n])

        self.gmm = IsotropicGaussianMixtureModel(self.centers, self.std_factor * (self.fibonacci / 2))

    def __iter__(self):
        self.current_step = 1
        return self

    def __next__(self):
        if self.current_step <= self.num_steps:
            X = jnp.array(self.gmm.sample(self.batch_size))
            weighting = jnp.ones((self.batch_size, 1))
            y = jnp.zeros((self.batch_size, 0))
            self.current_step += 1
            return {'parameters': X, 'weighting': weighting, 'conditioning': y}
        else:
            raise StopIteration

    def __len__(self):
        return self.num_steps
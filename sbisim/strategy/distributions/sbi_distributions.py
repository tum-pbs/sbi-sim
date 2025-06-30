from .base_distribution import UniformBase, LogNormalBase, GaussianBase

import jax.numpy as jnp

def get_two_moons_prior():
    return UniformBase(
        low=jnp.array([-1.0, -1.0]),
        high=jnp.array([1.0, 1.0]),
        shape=(2,)
    )

def get_slcp_prior():
    return UniformBase(
        low=jnp.array([-3.0, -3.0, -3.0, -3.0, -3.0]),
        high=jnp.array([3.0, 3.0, 3.0, 3.0, 3.0]),
        shape=(5,)
    )

def get_sir_prior():
    return LogNormalBase(
        mean=jnp.array([jnp.log(0.4), jnp.log(1/8)]),
        std=jnp.array([0.5, 0.2]),
        shape=(2,)
    )

def get_normal_gaussian_prior(shape):
    return GaussianBase(
        shape=shape,
        mean=jnp.zeros(shape),
        std=jnp.ones(shape),
    )

def get_lotka_volterra_prior():
    return LogNormalBase(
        mean=jnp.array([-0.125, -3, -0.125, -3]),
        std=jnp.array([0.5, 0.5, 0.5, 0.5]),
        shape=(4,)
    )
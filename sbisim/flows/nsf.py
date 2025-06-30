from typing import Sequence, Tuple

from flows.layers.actnorm import ActNorm
from flows.models.fcnn import FCNN
from flows.flowbase import FlowBase, NormalizingFlow

import jax.numpy as jnp
import jax.random as jr
import jax.nn

import flax.linen as nn

from flows.onebyoneconv import OneByOneConv
from flows.utils import uniform_min_max_init

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations = bin_locations.at[..., -1].add(eps)
    return jnp.sum(
        inputs[..., None] >= bin_locations,
        axis=-1
    ) - 1


def RQS(inputs, unnormalized_widths, unnormalized_heights,
        unnormalized_derivatives, inverse=False, left=0., right=1.,
        bottom=0., top=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE):
    # we need to guarantee that the inputs are within the domain
    # if jnp.min(inputs) < left or jnp.max(inputs) > right:
    #     raise ValueError("Input outside domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    # constrained softmax

    def constrained_softmax(x, mask):
        x = jnp.exp(x) * mask
        x = x / jnp.sum(x, axis=-1, keepdims=True)
        return x

    # widths = constrained_softmax(unnormalized_widths, inside_intvl_mask)

    widths = jax.nn.softmax(unnormalized_widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = jnp.cumsum(widths, axis=-1)

    pad_width = tuple((0, 0) for _ in range(cumwidths.ndim - 1)) + ((1, 0),)

    cumwidths = jnp.pad(cumwidths, pad_width=pad_width, mode='constant')
    cumwidths = (right - left) * cumwidths + left

    cumwidths = cumwidths.at[..., 0].set(left)
    cumwidths = cumwidths.at[..., -1].set(right)

    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + jax.nn.softplus(unnormalized_derivatives)

    heights = jax.nn.softmax(unnormalized_heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = jnp.cumsum(heights, axis=-1)

    pad_width = tuple((0, 0) for _ in range(cumheights.ndim - 1)) + ((1, 0),)

    cumheights = jnp.pad(cumheights, pad_width=pad_width, mode='constant')
    cumheights = (top - bottom) * cumheights + bottom

    cumheights = cumheights.at[..., 0].set(bottom)
    cumheights = cumheights.at[..., -1].set(top)

    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = jnp.take_along_axis(cumwidths, bin_idx, -1)[..., 0]
    input_bin_widths = jnp.take_along_axis(widths, bin_idx, -1)[..., 0]

    input_cumheights = jnp.take_along_axis(cumheights, bin_idx, -1)[..., 0]
    delta = heights / widths
    input_delta = jnp.take_along_axis(delta, bin_idx, -1)[..., 0]

    input_derivatives = jnp.take_along_axis(derivatives, bin_idx, -1)[..., 0]
    input_derivatives_plus_one = jnp.take_along_axis(derivatives[..., 1:], bin_idx, -1)
    input_derivatives_plus_one = input_derivatives_plus_one[..., 0]

    input_heights = jnp.take_along_axis(heights, bin_idx, -1)[..., 0]

    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives
                                             + input_derivatives_plus_one - 2 * input_delta)
              + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives - (inputs - input_cumheights)
             * (input_derivatives + input_derivatives_plus_one
                - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = jnp.power(b, 2) - 4 * a * c

        # uncomment for jit compile
        # assert (discriminant >= 0).all()

        root = (2 * c) / (-b - jnp.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta \
                      + ((input_derivatives + input_derivatives_plus_one
                          - 2 * input_delta) * theta_one_minus_theta)
        derivative_numerator = jnp.power(input_delta, 2) \
                               * (input_derivatives_plus_one * jnp.power(root, 2)
                                  + 2 * input_delta * theta_one_minus_theta
                                  + input_derivatives * jnp.power(1 - root, 2))
        logabsdet = jnp.log(derivative_numerator) - 2 * jnp.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * jnp.power(theta, 2)
                                     + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives
                                      + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = jnp.power(input_delta, 2) \
                               * (input_derivatives_plus_one * jnp.power(theta, 2)
                                  + 2 * input_delta * theta_one_minus_theta
                                  + input_derivatives * jnp.power((1 - theta), 2))
        logabsdet = jnp.log(derivative_numerator) - 2 * jnp.log(denominator)
        return outputs, logabsdet


def unconstrained_RQS(inputs, unnormalized_widths, unnormalized_heights,
                      unnormalized_derivatives, inverse=False,
                      tail_bound=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                      min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                      min_derivative=DEFAULT_MIN_DERIVATIVE):
    inside_intvl_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_intvl_mask

    outputs = jnp.zeros_like(inputs)
    logabsdet = jnp.zeros_like(inputs)

    pad_width = tuple((0, 0) for _ in range(unnormalized_derivatives.ndim - 1)) + ((1, 1),)

    unnormalized_derivatives = jnp.pad(unnormalized_derivatives, pad_width=pad_width)
    constant = jnp.log(jnp.exp(1 - min_derivative) - 1)

    unnormalized_derivatives = unnormalized_derivatives.at[..., 0].set(constant)
    unnormalized_derivatives = unnormalized_derivatives.at[..., -1].set(constant)

    logabsdet = logabsdet * inside_intvl_mask.astype(jnp.float32)

    o_, l_ = RQS(
        inputs=inputs * inside_intvl_mask,  # [inside_intvl_mask],
        unnormalized_widths=unnormalized_widths,  # [inside_intvl_mask, :],
        unnormalized_heights=unnormalized_heights,  # [inside_intvl_mask, :],
        unnormalized_derivatives=unnormalized_derivatives,  # [inside_intvl_mask, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )

    outputs = (o_ * inside_intvl_mask.astype(jnp.float32) +
               inputs * outside_interval_mask.astype(jnp.float32))

    logabsdet = (l_ * inside_intvl_mask.astype(jnp.float32) +
                 logabsdet * outside_interval_mask.astype(jnp.float32))

    return outputs, logabsdet


class NSF_AR(FlowBase):
    """
    Neural spline flow, auto-regressive.

    [Durkan et al. 2019]
    """

    K: int = 5
    B: int = 3
    hidden_dim: int = 8
    base_network = FCNN

    def setup(self):
        self.layers: Sequence[nn.Module] = []

        if self.dim_conditioning > 0:
            self.init_param = self.base_network(self.dim_conditioning, 3 * self.K - 1, self.hidden_dim)
        else:
            self.init_param = self.param('init_param', uniform_min_max_init(-1 / 2, 1 / 2), (3 * self.K - 1,))

        for i in range(1, self.dim_flow):
            self.layers += (self.base_network(i + self.dim_conditioning, 3 * self.K - 1, self.hidden_dim),)

    def forward_(self, x: jnp.ndarray, rng: jr.PRNGKey,
                 ldj: jnp.ndarray,
                 conditioning: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:

        z = jnp.zeros_like(x)
        log_det = jnp.zeros(z.shape[0])
        for i in range(self.dim_flow):
            if i == 0:

                if self.dim_conditioning > 0:
                    init_param = self.init_param(conditioning)
                else:
                    init_param = jnp.broadcast_to(self.init_param, (x.shape[0], 3 * self.K - 1))

                W, H, D = jnp.array_split(init_param, [self.K, 2 * self.K], axis=1)
            else:
                layer_input = jnp.concatenate([x[:, :i], conditioning], axis=-1)
                out = self.layers[i - 1](layer_input)
                W, H, D = jnp.array_split(out, [self.K, 2 * self.K], axis=1)
            W, H = jax.nn.softmax(W, axis=1), jax.nn.softmax(H, axis=1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = jax.nn.softplus(D)
            z_, ld = unconstrained_RQS(
                x[:, i], W, H, D, inverse=False, tail_bound=self.B)
            z = z.at[:, i].set(z_)
            log_det += ld
        return z, ldj + log_det.reshape(-1), rng

    def inverse_(self, z: jnp.ndarray, rng: jr.PRNGKey,
                 ldj: jnp.ndarray,
                 conditioning: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:
        x = jnp.zeros_like(z)
        log_det = jnp.zeros(x.shape[0])
        for i in range(self.dim_flow):
            if i == 0:
                if self.dim_conditioning > 0:
                    init_param = self.init_param(conditioning)
                else:
                    init_param = jnp.broadcast_to(self.init_param, (x.shape[0], 3 * self.K - 1))

                W, H, D = jnp.array_split(init_param, [self.K, 2 * self.K], axis=1)
            else:
                layer_input = jnp.concatenate([x[:, :i], conditioning], axis=-1)
                out = self.layers[i - 1](layer_input)
                W, H, D = jnp.array_split(out, [self.K, 2 * self.K], axis=1)
            W, H = jax.nn.softmax(W, axis=1), jax.nn.softmax(H, axis=1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = jax.nn.softplus(D)
            x_, ld = unconstrained_RQS(
                z[:, i], W, H, D, inverse=True, tail_bound=self.B)
            x = x.at[:, i].set(x_)
            log_det += ld
        return x, ldj + log_det.reshape(-1), rng

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class NSF_CL(FlowBase):
    """
    Neural spline flow, coupling layer.

    [Durkan et al. 2019]
    """

    K: int = 5
    B: int = 3
    hidden_dim: int = 8
    base_network = FCNN

    def setup(self):
        self.f1 = self.base_network(self.dim_flow // 2 + self.dim_conditioning,
                                    (3 * self.K - 1) * self.dim_flow // 2, self.hidden_dim)
        self.f2 = self.base_network(self.dim_flow // 2 + self.dim_conditioning,
                                    (3 * self.K - 1) * self.dim_flow // 2, self.hidden_dim)

    def forward_(self, x: jnp.ndarray, rng: jr.PRNGKey,
                 ldj: jnp.ndarray,
                 conditioning: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:
        log_det = jnp.zeros(x.shape[0])
        lower, upper = x[:, :self.dim_flow // 2], x[:, self.dim_flow // 2:]

        f1_input = jnp.concatenate([lower, conditioning], axis=-1)

        out = self.f1(f1_input).reshape(-1, self.dim_flow // 2, 3 * self.K - 1)
        W, H, D = jnp.array_split(out, [self.K, 2 * self.K], axis=2)
        W, H = jax.nn.softmax(W, axis=2), jax.nn.softmax(H, axis=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = jax.nn.softplus(D)
        upper, ld = unconstrained_RQS(
            upper, W, H, D, inverse=False, tail_bound=self.B)
        log_det += jnp.sum(ld, axis=1)

        f2_input = jnp.concatenate([upper, conditioning], axis=-1)

        out = self.f2(f2_input).reshape(-1, self.dim_flow // 2, 3 * self.K - 1)
        W, H, D = jnp.array_split(out, [self.K, 2 * self.K], axis=2)
        W, H = jax.nn.softmax(W, axis=2), jax.nn.softmax(H, axis=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = jax.nn.softplus(D)
        lower, ld = unconstrained_RQS(
            lower, W, H, D, inverse=False, tail_bound=self.B)
        log_det += jnp.sum(ld, axis=1)
        return jnp.concatenate([lower, upper], axis=1), ldj + log_det.reshape(-1), rng

    def inverse_(self, z: jnp.ndarray, rng: jr.PRNGKey,
                 ldj: jnp.ndarray,
                 conditioning: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:
        log_det = jnp.zeros(z.shape[0])
        lower, upper = z[:, :self.dim_flow // 2], z[:, self.dim_flow // 2:]

        f2_input = jnp.concatenate([upper, conditioning], axis=-1)

        out = self.f2(f2_input).reshape(-1, self.dim_flow // 2, 3 * self.K - 1)
        W, H, D = jnp.array_split(out, [self.K, 2 * self.K], axis=2)
        W, H = jax.nn.softmax(W, axis=2), jax.nn.softmax(H, axis=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = jax.nn.softplus(D)
        lower, ld = unconstrained_RQS(
            lower, W, H, D, inverse=True, tail_bound=self.B)
        log_det += jnp.sum(ld, axis=1)

        f1_input = jnp.concatenate([lower, conditioning], axis=-1)

        out = self.f1(f1_input).reshape(-1, self.dim_flow // 2, 3 * self.K - 1)
        W, H, D = jnp.array_split(out, [self.K, 2 * self.K], axis=2)
        W, H = jax.nn.softmax(W, axis=2), jax.nn.softmax(H, axis=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = jax.nn.softplus(D)
        upper, ld = unconstrained_RQS(
            upper, W, H, D, inverse=True, tail_bound=self.B)
        log_det += jnp.sum(ld, axis=1)
        return jnp.concatenate([lower, upper], axis=1), ldj + log_det.reshape(-1), rng


class StackedNSF_CL(NormalizingFlow):
    """
    Stacked neural spline flow, coupling layer.

    [Durkan et al. 2019]
    """

    steps: int
    K: int = 5
    B: int = 3
    hidden_dim: int = 8
    seed: int = 0
    use_onebyoneconv: bool = False
    use_actnorm: bool = False
    base_network = FCNN

    def setup(self):
        self.flows: Sequence[FlowBase] = []
        seed = self.seed

        for i in range(self.steps):
            self.flows += (NSF_CL(self.dim_flow, self.dim_conditioning,
                                  self.import_samples, hidden_dim=self.hidden_dim,
                                  K=self.K, B=self.B),)

            if self.use_onebyoneconv:
                self.flows += (OneByOneConv(self.dim_flow, self.dim_conditioning,
                                            seed),)
                seed += 1

            if self.use_actnorm:
                self.flows += (ActNorm(self.dim_flow, self.dim_conditioning),)


class StackedNSF_AR(NormalizingFlow):
    """
    Stacked neural spline flow, auto-regressive.

    [Durkan et al. 2019]
    """

    steps: int
    K: int = 5
    B: int = 3
    hidden_dim: int = 8
    seed: int = 0
    use_onebyoneconv: bool = False
    use_actnorm: bool = False
    base_network = FCNN

    def setup(self):
        self.flows: Sequence[FlowBase] = []
        seed = self.seed
        for i in range(self.steps):
            self.flows += (NSF_AR(self.dim_flow, self.dim_conditioning,
                                  self.import_samples, hidden_dim=self.hidden_dim,
                                  K=self.K, B=self.B),)

            if self.use_onebyoneconv:
                self.flows += (OneByOneConv(self.dim_flow, self.dim_conditioning,
                                            seed),)
                seed += 1

            if self.use_actnorm:
                self.flows += (ActNorm(self.dim_flow, self.dim_conditioning),)

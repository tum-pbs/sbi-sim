# Source: https://github.com/DarshanDeshpande/jax-models/blob/main/jax_models/models/convnext.py

import jax.numpy as jnp
import flax.linen as nn

from ..layers import DepthwiseConv2D, DropPath

from typing import Optional, Iterable

__all__ = [
    "ConvNeXt",
    "convnext",
]

initializer = nn.initializers.variance_scaling(
    0.2, "fan_in", distribution="truncated_normal"
)


class ConvNeXtBlock(nn.Module):
    dim: int = 256
    layer_scale_init_value: float = 1e-6
    drop_path: float = 0.1
    deterministic: Optional[bool] = None

    def init_fn(self, key, shape, fill_value):
        return jnp.full(shape, fill_value)

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        x = DepthwiseConv2D((7, 7), weights_init=initializer, name="dwconv")(inputs)
        x = nn.LayerNorm(name="norm")(x)
        x = nn.Dense(4 * self.dim, kernel_init=initializer, name="pwconv1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim, kernel_init=initializer, name="pwconv2")(x)
        if self.layer_scale_init_value > 0:
            gamma = self.param(
                "gamma", self.init_fn, (self.dim,), self.layer_scale_init_value
            )
            x = gamma * x

        x = inputs + DropPath(self.drop_path)(x, deterministic)
        return x


class ConvNeXt(nn.Module):
    """
    ConvNeXt Module

    Attributes:

        depths (list or tuple): Depths for every block
        dims (list or tuple): Embedding dimension for every stage.
        drop_path (float): Dropout value for DropPath. Default is 0.1
        layer_scale_init_value (float): Initialization value for scale. Default is 1e-6.
        head_init_scale (float): Initialization value for head. Default is 1.0.
        attach_head (bool): Whether to attach classification head. Default is False.
        out_dim (int): Number of classification classes. Only works if attach_head is True. Default is 1000.
        deterministic (bool): Optional argument, if True, network becomes deterministic and dropout is not applied.

    """

    depths: Iterable = (3, 3, 9, 3)
    dims: Iterable = (96, 192, 384, 768)
    drop_path: float = 0.0
    layer_scale_init_value: float = 1e-6
    head_init_scale: float = 1.0
    attach_classification_head: bool = True
    attach_regression_head: bool = False
    out_dim: int = 1000
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        dp_rates = jnp.linspace(0, self.drop_path, sum(self.depths))
        curr = 0

        x = jnp.transpose(inputs, (0, 2, 3, 1))

        # Stem
        x = nn.Conv(
            self.dims[0], (4, 4), 4, kernel_init=initializer, name="downsample_layers00"
        )(x)
        x = nn.LayerNorm(name="downsample_layers01")(x)

        for j in range(self.depths[0]):
            x = ConvNeXtBlock(
                self.dims[0],
                drop_path=dp_rates[curr + j],
                layer_scale_init_value=self.layer_scale_init_value,
                name=f"stages0{j}",
            )(x, deterministic)
        curr += self.depths[0]

        # Downsample layers
        for i in range(3):
            x = nn.LayerNorm(name=f"downsample_layers{i + 1}0")(x)
            x = nn.Conv(
                self.dims[i + 1],
                (2, 2),
                2,
                kernel_init=initializer,
                name=f"downsample_layers{i + 1}1",
            )(x)

            for j in range(self.depths[i + 1]):
                x = ConvNeXtBlock(
                    self.dims[i + 1],
                    drop_path=dp_rates[curr + j],
                    layer_scale_init_value=self.layer_scale_init_value,
                    name=f"stages{i + 1}{j}",
                )(x, deterministic)

            curr += self.depths[i + 1]

        if self.attach_classification_head:
            x = nn.LayerNorm(name="norm")(jnp.mean(x, [1, 2]))
            x = nn.Dense(self.out_dim, kernel_init=initializer, name="head")(x)

        if self.attach_regression_head:
            x = jnp.mean(x, [1, 2])
            x = nn.Dense(self.out_dim, kernel_init=initializer, name="head")(x)

        return x


DEPTHS = {
    'tinier': (3, 3, 9, 3, 3),
    "tiny": (3, 3, 9, 3),
    "small": (3, 3, 27, 3),
    "base": (3, 3, 27, 3),
    "large": (3, 3, 27, 3),
    "xlarge": (3, 3, 27, 3),
}

DIMS = {
    'tinier': (96, 96, 96, 96, 96),
    'regression-tiny': (96, 96, 96, 96, 96),
    "tiny": (96, 192, 384, 768),
    "small": (96, 192, 384, 768),
    "base": (128, 256, 512, 1024),
    "large": (192, 384, 768, 1536),
    "xlarge": (256, 512, 1024, 2048),
}


def convnext(
        size: str = 'tiny',
        deterministic: bool = None,
        *args, **kwargs
):
    if size not in DEPTHS.keys() or size not in DIMS.keys():
        raise ValueError(f"ConvNeXt size should be one of {list(DEPTHS.keys())}")

    model = ConvNeXt(
        depths=DEPTHS[size],
        dims=DIMS[size],
        deterministic=deterministic,
        *args,
        **kwargs,
    )

    return model
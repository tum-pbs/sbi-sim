from typing import Dict, Optional, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from diffusers.configuration_utils import ConfigMixin, flax_register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.embeddings_flax import FlaxTimestepEmbedding, FlaxTimesteps
from diffusers.models.modeling_flax_utils import FlaxModelMixin
from diffusers.models.unets.unet_2d_blocks_flax import (
    FlaxCrossAttnDownBlock2D,
    FlaxCrossAttnUpBlock2D,
    FlaxDownBlock2D,
    FlaxUNetMidBlock2DCrossAttn,
    FlaxUpBlock2D,
)
from flows.cnf import ContinuousNormalizingFlow


@flax_register_to_config
class Conv2DConditionModel(nn.Module, FlaxModelMixin, ConfigMixin):
    r"""
    A conditional 2D convolutional model that takes a noisy sample, conditional state, and a timestep and returns a sample
    shaped output.

    Implementation based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_condition_flax.py

    This model inherits from [`FlaxModelMixin`]. Check the superclass documentation for its generic methods
    implemented for all models (such as downloading or saving).

    This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax Linen module and refer to the Flax documentation for all matters related to its
    general usage and behavior.

    Parameters:
        sample_size (`int`, *optional*):
            The size of the input sample.
        dim_flow (`int`, *optional*):
            The dimension of the flow.
        in_channels (`int`, *optional*, defaults to 4):
            The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4):
            The number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxDownBlock2D")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("FlaxUpBlock2D", "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D")`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        attention_head_dim (`int` or `Tuple[int]`, *optional*, defaults to 8):
            The dimension of the attention heads.
        num_attention_heads (`int` or `Tuple[int]`, *optional*):
            The number of attention heads.
        cross_attention_dim (`int`, *optional*, defaults to 768):
            The dimension of the cross attention features.
        dropout (`float`, *optional*, defaults to 0):
            Dropout probability for down, up and bottleneck blocks.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            Enable memory efficient attention as described [here](https://arxiv.org/abs/2112.05682).
        split_head_dim (`bool`, *optional*, defaults to `False`):
            Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
            enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
    """

    sample_size: int = 32
    dim_flow: int = 3
    in_channels: int = 4
    out_channels: int = 4
    down_block_types: Tuple[str, ...] = (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types: Tuple[str, ...] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")
    only_cross_attention: Union[bool, Tuple[bool]] = False
    block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280)
    layers_per_block: int = 2
    attention_head_dim: Union[int, Tuple[int, ...]] = 8
    num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None
    cross_attention_dim: int = 1280
    dropout: float = 0.0
    use_linear_projection: bool = False
    dtype: jnp.dtype = jnp.float32
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    use_memory_efficient_attention: bool = False
    split_head_dim: bool = False
    transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1
    addition_embed_type: Optional[str] = None
    addition_time_embed_dim: Optional[int] = None
    addition_embed_type_num_heads: int = 64
    projection_class_embeddings_input_dim: Optional[int] = None

    def init_weights(self, rng: jax.Array) -> FrozenDict:
        # init input tensors
        sample_shape = (1, self.in_channels, self.sample_size, self.sample_size)
        sample = jnp.zeros(sample_shape, dtype=jnp.float32)
        timesteps = jnp.ones((1,), dtype=jnp.int32)
        encoder_hidden_states = jnp.zeros((1, 1, self.cross_attention_dim), dtype=jnp.float32)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        added_cond_kwargs = None

        return self.init(rngs, sample, timesteps, encoder_hidden_states, added_cond_kwargs)["params"]

    def setup(self) -> None:
        block_out_channels = self.block_out_channels
        time_embed_dim = block_out_channels[0] * 4

        if self.num_attention_heads is not None:
            raise ValueError(
                "At the moment it is not possible to define the number of attention heads via `num_attention_heads` because of a naming issue as described in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131. Passing `num_attention_heads` will only be supported in diffusers v0.19."
            )

        # If `num_attention_heads` is not defined (which is the case for most models)
        # it will default to `attention_head_dim`. This looks weird upon first reading it and it is.
        # The reason for this behavior is to correct for incorrectly named variables that were introduced
        # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        # which is why we correct for the naming here.
        num_attention_heads = self.num_attention_heads or self.attention_head_dim

        # input
        self.conv_in = nn.Conv(
            block_out_channels[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # time
        self.time_proj = FlaxTimesteps(
            block_out_channels[0], flip_sin_to_cos=self.flip_sin_to_cos, freq_shift=self.config.freq_shift
        )
        self.time_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)

        # TODO add variable like time_embed_dim for params?
        self.param_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)

        only_cross_attention = self.only_cross_attention
        if isinstance(only_cross_attention, bool):
            only_cross_attention = (only_cross_attention,) * len(self.down_block_types)

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(self.down_block_types)

        # transformer layers per block
        transformer_layers_per_block = self.transformer_layers_per_block
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(self.down_block_types)

        # addition embed types
        if self.addition_embed_type is None:
            self.add_embedding = None
        elif self.addition_embed_type == "text_time":
            if self.addition_time_embed_dim is None:
                raise ValueError(
                    f"addition_embed_type {self.addition_embed_type} requires `addition_time_embed_dim` to not be None"
                )
            self.add_time_proj = FlaxTimesteps(self.addition_time_embed_dim, self.flip_sin_to_cos, self.freq_shift)
            self.add_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)
        else:
            raise ValueError(f"addition_embed_type: {self.addition_embed_type} must be None or `text_time`.")

        # down
        down_blocks = []
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(self.down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if down_block_type == "CrossAttnDownBlock2D":
                down_block = FlaxCrossAttnDownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    dropout=self.dropout,
                    num_layers=self.layers_per_block,
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    num_attention_heads=num_attention_heads[i],
                    add_downsample=not is_final_block,
                    use_linear_projection=self.use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    use_memory_efficient_attention=self.use_memory_efficient_attention,
                    split_head_dim=self.split_head_dim,
                    dtype=self.dtype,
                )
            else:
                down_block = FlaxDownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    dropout=self.dropout,
                    num_layers=self.layers_per_block,
                    add_downsample=not is_final_block,
                    dtype=self.dtype,
                )

            down_blocks.append(down_block)
        self.down_blocks = down_blocks

        # mid
        self.mid_block = FlaxUNetMidBlock2DCrossAttn(
            in_channels=block_out_channels[-1],
            dropout=self.dropout,
            num_attention_heads=num_attention_heads[-1],
            transformer_layers_per_block=transformer_layers_per_block[-1],
            use_linear_projection=self.use_linear_projection,
            use_memory_efficient_attention=self.use_memory_efficient_attention,
            split_head_dim=self.split_head_dim,
            dtype=self.dtype,
        )

        self.dense_out = nn.Dense(self.dim_flow, dtype=self.dtype)

        # out
        self.conv_norm_out = nn.GroupNorm(num_groups=32, epsilon=1e-5)
        self.conv_out = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(
            self,
            timesteps: Union[jnp.ndarray, float, int],
            sample: jnp.ndarray,
            encoder_hidden_states: jnp.ndarray,
            train: bool = False,
    ) -> jnp.ndarray:
        r"""
        Args:
            timesteps (`jnp.ndarray` or `float` or `int`): timesteps
            sample (`jnp.ndarray`): (batch, features) noisy inputs tensor
            encoder_hidden_states (`jnp.ndarray`): (batch_size, height, width) encoder hidden states
            train (`bool`, *optional*, defaults to `False`):
                Use deterministic functions and disable dropout when not training.

        Returns:
            jnp.ndarray
        """
        # 1. time
        if not isinstance(timesteps, jnp.ndarray):
            timesteps = jnp.array([timesteps], dtype=jnp.int32)
        elif isinstance(timesteps, jnp.ndarray) and len(timesteps.shape) == 0:
            timesteps = timesteps.astype(dtype=jnp.float32)
            timesteps = jnp.expand_dims(timesteps, 0)

        print("compiling.....")

        timesteps = jnp.reshape(timesteps, -1)

        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)

        # 1. swap sample (features) and encoder_hidden_states (conditioning for images)
        temp_ = sample
        sample = encoder_hidden_states
        encoder_hidden_states = temp_
        encoder_hidden_states = self.param_embedding(encoder_hidden_states)
        encoder_hidden_states = jnp.expand_dims(encoder_hidden_states, axis=1)


        # 2. pre-process
        # sample = jnp.transpose(sample, (0, 2, 3, 1))
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            if isinstance(down_block, FlaxCrossAttnDownBlock2D):
                sample, res_samples = down_block(sample, t_emb, encoder_hidden_states, deterministic=not train)
            else:
                sample, res_samples = down_block(sample, t_emb, deterministic=not train)
            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, t_emb, encoder_hidden_states, deterministic=not train)


        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = nn.silu(sample)
        sample = self.conv_out(sample)
        sample = jnp.transpose(sample, (0, 3, 1, 2))

        sample = jnp.reshape(sample, (sample.shape[0], -1))

        out = self.dense_out(sample)

        return out

def get_simple_cross_attention_conv2d(
        sample_size: int = 160,
        dim_flow: int = 3,
):
    return Conv2DConditionModel(
        sample_size=sample_size,
        dim_flow=dim_flow,
        in_channels=1,
        out_channels=1,
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        block_out_channels=(64, 64, 64, 64, 64, 64),
        layers_per_block=1,
        attention_head_dim=4,
        num_attention_heads=None,
        cross_attention_dim=64,
        dropout=0.0,
        flip_sin_to_cos=True,
        freq_shift=0,
        use_memory_efficient_attention=False,
        split_head_dim=False,
        transformer_layers_per_block=1,
        addition_embed_type=None,
        addition_time_embed_dim=None,
        addition_embed_type_num_heads=4,
        projection_class_embeddings_input_dim=None
    )

class CrossAttentionCNF(ContinuousNormalizingFlow):

    sample_size: int = 160

    def setup(self):

        super().setup()

        self.model = get_simple_cross_attention_conv2d(self.sample_size, self.dim_flow)
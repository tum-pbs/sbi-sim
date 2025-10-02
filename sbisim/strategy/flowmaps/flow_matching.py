import jax
import jax.numpy as jnp
import numpy as np

def get_targets(FLAGS, key, model, params_dict, batch,
                force_t=-1, force_dt=-1, **kwargs):
    label_key, time_key, noise_key = jax.random.split(key, 3)
    info = {}

    x_samples, y_samples = batch['parameters'], batch['conditioning']
    batch_size = x_samples.shape[0]

    labels = jnp.zeros((batch_size,), dtype=jnp.int32)

    labels_dropout = jax.random.bernoulli(label_key, FLAGS['model']['class_dropout_prob'],
                                          (labels.shape[0],))
    labels_dropped = jnp.where(labels_dropout, FLAGS['model']['num_classes'], labels)
    info['dropped_ratio'] = jnp.mean(labels_dropped == FLAGS['model']['num_classes'])

    # Sample t.
    t = jax.random.randint(time_key, (batch_size,), minval=0,
                           maxval=FLAGS['model']['denoise_timesteps']).astype(jnp.float32)
    t /= FLAGS['model']['denoise_timesteps']
    force_t_vec = jnp.ones(batch_size, dtype=jnp.float32) * force_t
    t = jnp.where(force_t_vec != -1, force_t_vec, t) # If force_t is not -1, then use force_t.
    t_full = jnp.expand_dims(t, axis=(1,))

    x_1 = x_samples
    x_0 = jax.random.normal(noise_key, x_samples.shape)
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    v_t = x_1 - (1 - 1e-5) * x_0

    dt_flow = np.log2(FLAGS['model']['denoise_timesteps']).astype(jnp.int32)
    dt_base = jnp.ones(batch_size, dtype=jnp.int32) * dt_flow

    return x_t, v_t, t, y_samples, dt_base, labels_dropped, info
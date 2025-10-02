import jax
import jax.numpy as jnp
import numpy as np

def get_targets(FLAGS, key, model, params_dict, batch,
                force_t=-1, force_dt=-1, epoch: int=0):

    time_key, noise_key = jax.random.split(key, 2)
    info = {}

    x_samples, y_samples = batch['parameters'], batch['conditioning']
    batch_size = x_samples.shape[0]

    labels = jnp.zeros((batch_size, ), dtype=jnp.int32)

    # 1) =========== Sample dt (based on current train step). ============
    dt_flow = np.log2(FLAGS['model']['denoise_timesteps']).astype(jnp.int32)
    dt_base = jnp.ceil(epoch / (FLAGS['max_epochs'] / dt_flow))
    dt_base = jnp.ones((batch_size,),  dtype=jnp.int32) * dt_base
    dt = 1 / (2 ** (dt_base))
    info['dt_base'] = jnp.mean(dt_base)

    # 2) =========== Sample t. ============
    dt_sections = jnp.power(2, dt_base) # [1, 2, 4, 8, 16, 32]
    t = jax.random.randint(time_key, (batch_size,), minval=0, maxval=dt_sections).astype(jnp.float32)
    t = t / dt_sections # Between 0 and 1.
    t_full = jnp.expand_dims(t, axis=(1,))
    t2 = t + dt
    t2_full = jnp.expand_dims(t2, axis=(1,))

    # 2) =========== Generate Bootstrap Targets ============
    x_1 = x_samples
    x_0 = jax.random.normal(noise_key, x_1.shape)
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    x_t2 = (1 - (1 - 1e-5) * t2_full) * x_0 + t2_full * x_1

    call_model_fn = lambda *args, **kwargs: model.apply({'params': params_dict['model_ema_params']}, *args, **kwargs)

    v_b2 = call_model_fn(x_t2, y_samples, t2, dt_base, train=False)
    pred_x1 = x_t2 + (1 - t2_full) * v_b2
    v_target = (pred_x1 - x_t) / (1 - jnp.expand_dims(t, axis=(1,)))

    info['v_magnitude_bootstrap'] = jnp.sqrt(jnp.mean(jnp.square(v_target)))
    info['v_magnitude_b1'] = jnp.sqrt(jnp.mean(jnp.square(x_t2 - x_t)))
    info['v_magnitude_b2'] = jnp.sqrt(jnp.mean(jnp.square(v_b2)))

    return x_t, v_target, t, y_samples, dt_base, labels, info
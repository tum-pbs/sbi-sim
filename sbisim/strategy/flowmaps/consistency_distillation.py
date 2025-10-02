import jax
import jax.numpy as jnp
import numpy as np

def get_targets(FLAGS, key, model, params_dict, batch,
                force_t=-1, force_dt=-1, **kwargs):

    time_key, noise_key = jax.random.split(key, 2)
    info = {}

    x_samples, y_samples = batch['parameters'], batch['conditioning']
    batch_size = x_samples.shape[0]

    labels = jnp.zeros((batch_size, ), dtype=jnp.int32)

    dt_flow = np.log2(FLAGS['model']['denoise_timesteps']).astype(jnp.int32)
    dt_base = jnp.ones(batch_size, dtype=jnp.int32) * dt_flow
    dt_bootstrap = 1 / FLAGS['model']['denoise_timesteps']

    # 1) =========== Sample t. ============
    t = jax.random.randint(time_key, (batch_size,), minval=0, maxval=FLAGS['model']['denoise_timesteps']).astype(jnp.float32)
    t /= FLAGS['model']['denoise_timesteps']
    t_full = jnp.expand_dims(t, axis=(1,))

    # 2) =========== Generate Bootstrap Targets ============
    x_1 = x_samples
    x_0 = jax.random.normal(noise_key, x_1.shape)
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1

    call_teacher_fn = lambda *args, **kwargs: model.apply({'params': params_dict['teacher_params']}, *args, **kwargs)
    call_model_ema_fn = lambda *args, **kwargs: model.apply({'params': params_dict['model_ema_params']}, *args, **kwargs)

    v_b1 = call_teacher_fn(x_t, y_samples, t, dt_base, train=False)

    t2 = t + dt_bootstrap
    x_t2 = x_t + dt_bootstrap * v_b1
    x_t2 = jnp.clip(x_t2, -10, 10)


    v_b2 = call_model_ema_fn(x_t2, y_samples, t2, dt_base, train=False)
    # v_b2 = train_state.call_model_ema(x_t2, t2, dt_base, labels, train=False)

    pred_x1 = x_t2 + (1 - jnp.expand_dims(t2, axis=(1,))) * v_b2
    v_target = (pred_x1 - x_t) / (1 - jnp.expand_dims(t, axis=(1,)))

    info['v_magnitude_bootstrap'] = jnp.sqrt(jnp.mean(jnp.square(v_target)))
    info['v_magnitude_b1'] = jnp.sqrt(jnp.mean(jnp.square(v_b1)))
    info['v_magnitude_b2'] = jnp.sqrt(jnp.mean(jnp.square(v_b2)))

    return x_t, v_target, t, y_samples, dt_base, labels, info
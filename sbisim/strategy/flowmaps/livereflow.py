import jax
import jax.numpy as jnp
import numpy as np

def get_targets(FLAGS, key, model, params, batch,
                force_t=-1, force_dt=-1):
    label_key, time_key, noise_key = jax.random.split(key, 3)
    info = {}

    x_samples, y_samples = batch['parameters'], batch['conditioning']
    batch_size = x_samples.shape[0]

    labels = jnp.zeros((batch_size, ), dtype=jnp.int32)

    # Batch of flow targets.
    labels_dropout = jax.random.bernoulli(label_key, FLAGS.model['class_dropout_prob'], (labels.shape[0],))
    labels_dropped = jnp.where(labels_dropout, FLAGS.model['num_classes'], labels)
    info['dropped_ratio'] = jnp.mean(labels_dropped == FLAGS.model['num_classes'])
    t = jax.random.randint(time_key, (batch_size,), minval=0, maxval=FLAGS.model['denoise_timesteps']).astype(jnp.float32)
    t /= FLAGS.model['denoise_timesteps']
    force_t_vec = jnp.ones(batch_size, dtype=jnp.float32) * force_t
    t = jnp.where(force_t_vec != -1, force_t_vec, t)         # If force_t is not -1, then use force_t.
    t_full = t[:, None, None, None] # [batch, 1, 1, 1]
    x_0 = jax.random.normal(noise_key, x_samples.shape)
    x_1 = x_samples
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    v_t = x_1 - (1 - 1e-5) * x_0
    dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
    dt_base = jnp.ones(batch_size, dtype=jnp.int32) * dt_flow

    # Batch of reflow targets.
    bootstrap_size = FLAGS.batch_size // FLAGS.model['bootstrap_every']
    x_0_reflow = jax.random.normal(noise_key, x_1.shape)[:bootstrap_size]
    x = x_0_reflow
    t_iter = jnp.zeros(bootstrap_size, dtype=jnp.float32)
    labels_uncond = jnp.ones(bootstrap_size, dtype=jnp.int32) * FLAGS.model['num_classes']

    call_model_fn = lambda *args, **kwargs: model.apply({'params': params}, *args, **kwargs)

    for _ in range(8):
        if FLAGS.model['cfg_scale'] == 0:

            v = call_model_fn(x, y_samples[:bootstrap_size], t_iter,
                              dt_base[:bootstrap_size], train=False)

            # v = train_state.call_model(x, t_iter, dt_base[:bootstrap_size], labels_uncond, train=False)

        else:
            x_ext = jnp.concatenate([x, x], axis=0)
            t_iter_ext = jnp.concatenate([t_iter, t_iter], axis=0)
            dt_base_ext = jnp.concatenate([dt_base[:bootstrap_size], dt_base[:bootstrap_size]], axis=0)
            labels_ext = jnp.concatenate([labels[:bootstrap_size], labels_uncond], axis=0)
            y_samples_ext = jnp.concatenate([y_samples[:bootstrap_size], y_samples[:bootstrap_size]], axis=0)

            v_all = call_model_fn(x_ext, y_samples_ext, t_iter_ext,
                              dt_base_ext, train=False)

            # v_all = train_state.call_model(x_ext, t_iter_ext, dt_base_ext, labels_ext, train=False)

            v_cond = v_all[:bootstrap_size]
            v_uncond = v_all[bootstrap_size:]
            v = v_uncond + FLAGS.model['cfg_scale'] * (v_cond - v_uncond)
        t_iter += 1 / 8
        x = x + (1 / 8) * v
    v_reflow = (x - x_0_reflow)
    dt_base_reflow = jnp.zeros(bootstrap_size, dtype=jnp.int32)

    t_reflow = jax.random.randint(time_key, (bootstrap_size,), minval=0, maxval=FLAGS.model['denoise_timesteps']).astype(jnp.float32)
    t_reflow /= FLAGS.model['denoise_timesteps']
    t_reflow_full = t_reflow[:, None, None, None] # [batch, 1, 1, 1]
    x_t_reflow = (1 - (1 - 1e-5) * t_reflow_full) * x_0_reflow + t_reflow_full * x

    x_t = jnp.concatenate([x_t_reflow, x_t[:-bootstrap_size]], axis=0)
    v_t = jnp.concatenate([v_reflow, v_t[:-bootstrap_size]], axis=0)
    t = jnp.concatenate([t_reflow, t[:-bootstrap_size]], axis=0)
    dt_base = jnp.concatenate([dt_base_reflow, dt_base[:-bootstrap_size]], axis=0)
    labels_dropped = jnp.concatenate([labels[:bootstrap_size], labels_dropped[:-bootstrap_size]], axis=0)

    return x_t, v_t, t, y_samples, dt_base, labels_dropped, info
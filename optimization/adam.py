from jax.example_libraries import optimizers

from optimization.optimization import OptimizerWrapper, OptState
import optax


class OptaxWrapper(OptimizerWrapper):
    def __init__(self, optimizer, lr_schedule, **config):
        super(OptaxWrapper, self).__init__(lr_schedule)

        self.opt = optax.inject_hyperparams(optimizer)(learning_rate=lr_schedule(0), **config)

        self.config = config.update({'learning_rate': lr_schedule(0)})

    def init(self, params, **kwargs):
        return OptState(params=params, state=self.opt.init(params, **kwargs))

    def update(self, i, opt_state, grads):

        updates, state = self.opt.update(grads, opt_state.state, opt_state.params)
        opt_state.state = state
        opt_state.params = optax.apply_updates(opt_state.params, updates)

        return opt_state

    def update_learning_rate(self, opt_state, learning_rate):
        opt_state.state.hyperparams['learning_rate'] = learning_rate
        return opt_state

    def get_params(self, state):
        return state.params

class AdamOptax(OptaxWrapper):
    def __init__(self, *args, **config):
        optimizer = optax.adam
        super(AdamOptax, self).__init__(optimizer, *args, **config)

class AdamWOptax(OptaxWrapper):
    def __init__(self, *args, **config):
        optimizer = optax.adamw
        super(AdamWOptax, self).__init__(optimizer, *args, **config)




class AdamStax(OptimizerWrapper):
    def __init__(self, lr_schedule, **config):
        super(AdamStax, self).__init__(lr_schedule)

        self.opt_fn = optimizers.adam(step_size=self.lr_schedule)

        self.opt_init = self.opt_fn[0]
        self.opt_update_ = self.opt_fn[1]
        self.get_params_ = self.opt_fn[2]

    def init(self, *args, **kwargs):
        return self.opt_init(*args, **kwargs)

    def update(self, i, opt_state, grads):
        opt_state = self.opt_update_(i, grads, opt_state)
        return opt_state

    def get_params(self, state):
        return self.get_params_(state)

    def update_learning_rate(self, opt_state, learning_rate):
        self.opt_fn = optimizers.adam(step_size=learning_rate)
        self.opt_init = self.opt_fn[0]
        self.opt_update_ = self.opt_fn[1]
        self.get_params_ = self.opt_fn[2]
        return opt_state

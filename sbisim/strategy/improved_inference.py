from abc import abstractmethod, ABC
from functools import partial
from typing import Union, Tuple, Optional, Dict

from jax.scipy.stats.norm import logpdf
from diffrax import diffeqsolve, ODETerm, ConstantStepSize, PIDController, SaveAt
from flax.core import FrozenDict
import flax.linen as nn
import jax.random as jr
import jax.numpy as jnp
from jax import jit

import jax

from jaxtyping import PyTree

from ..flows.cnf import no_logp_wrapper, get_solver, get_prob_wrapper, get_stepsize_controller
from ..utils import instantiate_from_config, generate_apply_rngs


class BaseSampler(ABC):

    @abstractmethod
    def sample(self, num_samples: int, dim: int, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
               z_init: Optional[jnp.ndarray], conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> Tuple[
        PyTree, jr.PRNGKey]:
        pass

    @abstractmethod
    def compute_likelihood(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                           conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> Tuple[PyTree, jr.PRNGKey]:
        pass

    @abstractmethod
    def forward(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> Tuple[PyTree, jr.PRNGKey]:
        pass


class SigmaSchedule(ABC):

    @abstractmethod
    def log_sigma(self, t: float) -> float:
        pass

    @abstractmethod
    def grad_log_sigma(self, t: float) -> float:
        pass


class LinearSchedule(SigmaSchedule):

    def __init__(self, init_log_sigma: float, final_log_sigma: float, t_0: float, t_1: float):
        self.init_log_sigma = init_log_sigma
        self.final_log_sigma = final_log_sigma
        self.t_0 = t_0
        self.t_1 = t_1

    def log_sigma(self, t: float) -> float:
        return (self.init_log_sigma + (self.final_log_sigma - self.init_log_sigma)
                * (t - self.t_0) / (self.t_1 - self.t_0))

    def grad_log_sigma(self, t: float) -> float:
        return (self.final_log_sigma - self.init_log_sigma) / (self.t_1 - self.t_0)

    def init_sample(self, rng: jr.PRNGKey, shape: Tuple[int]) \
            -> Tuple[jnp.ndarray, jr.PRNGKey]:
        sample = jr.normal(rng, shape) * jnp.power(10, self.init_log_sigma)
        rng = jr.split(rng)[0]
        return sample, rng


class SGLDSampler(BaseSampler):

    def __init__(self, path: Dict, init_stepsize: float = 0.1, t_0: float = 0.0,
                 t_1: float = 1.0, mode: str = 'none', sigma_init: float = 1.0,
                 sigma_rescale: float = 0.0, correction_steps: int = 2, r: float = 0.01):
        """

        :param path:
        :param init_stepsize:
        :param t_0:
        :param t_1:
        :param mode:
        :param sigma_init:
        :param correction_steps:
        :param correction_steps:
        :param epsilon:
        :param r: signal-to-noise ratio as defined in Song et al. 2021 appendix G
        """

        self.path = instantiate_from_config(path)
        self.init_stepsize = init_stepsize
        self.t_0 = t_0
        self.t_1 = t_1
        self.r = r
        self.correction_steps = correction_steps
        self.num_steps = int((t_1 - t_0) / init_stepsize)
        self.sigma_init = sigma_init
        self.sigma_rescale = 0.0
        self.prob_wrapper = get_prob_wrapper(mode)

    def rescale(self, x: jnp.ndarray, rng: jr.PRNGKey):
        x = x + jr.normal(rng, x.shape) * self.sigma_rescale
        rng = jr.split(rng)[0]
        return x, rng

    def flow_to_score(self, x: jnp.ndarray, t: jnp.ndarray):

        scaling = 1 / (self.path.grad_sigma(t))
        return jnp.einsum("ab,a->ab", x, scaling)

    def compute_likelihood(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                           conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> PyTree:

        data_dict = {
            'likelihood': jnp.zeros(x.shape[0]),
        }

        return data_dict, rng

    def _inference(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                   conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> Tuple[PyTree, jr.PRNGKey]:

        data_dict = {
            'trajectory': jnp.zeros((self.num_steps,) + x.shape),
            'features': jnp.zeros((self.num_steps,) + x.shape),
            'gradient': jnp.zeros((self.num_steps,) + x.shape),
            'nll': jnp.zeros((self.num_steps, x.shape[0]))
        }

        def loop_body(i, vals, final=False):

            data_dict, x, conditioning, t, params, rng = vals

            t_in = jnp.repeat(t, x.shape[0])

            if hasattr(model.__class__, 'forward'):

                out = model.apply(params, t_in, x, conditioning,
                                  method=model.__class__.forward)

                x = x + self.init_stepsize * out[0]

                data_dict['trajectory'] = data_dict['trajectory'].at[i].set(x)
                data_dict['features'] = data_dict['features'].at[i].set(out[3])
                data_dict['gradient'] = data_dict['gradient'].at[i].set(out[2])
                data_dict['nll'] = data_dict['nll'].at[i].set(out[1])

            else:

                out = model.apply(params, t_in, x, conditioning)

                print('out: ', out)

                x = x + self.init_stepsize * out

                data_dict['trajectory'] = data_dict['trajectory'].at[i].set(x)

            for _ in range(self.correction_steps - 1):
                epsilon = jr.normal(rng, x.shape)
                rng = jr.split(rng)[0]

                out = model.apply(params, t_in, x, conditioning)

                score = self.flow_to_score(out, t_in)

                norm_score = jnp.mean(jnp.linalg.norm(score, axis=1))
                norm_noise = jnp.mean(jnp.linalg.norm(epsilon, axis=1))

                eps = 2 * ((self.r * norm_noise / norm_score) ** 2)

                x = x + eps * score + epsilon * jnp.sqrt(2 * eps)

            if self.correction_steps > 0:

                epsilon = jr.normal(rng, x.shape)
                rng = jr.split(rng)[0]

                out = model.apply(params, t_in, x, conditioning)

                score = self.flow_to_score(out, t_in)

                norm_score = jnp.mean(jnp.linalg.norm(score, axis=1))
                norm_noise = jnp.mean(jnp.linalg.norm(epsilon, axis=1))

                eps = 2 * ((self.r * norm_noise / norm_score) ** 2)

                print('eps:', eps)

                x = x + score * eps

                if not final:
                    x = x + epsilon * jnp.sqrt(2 * eps)

            return data_dict, x, conditioning, t + self.init_stepsize, params, rng

        # LOOP 0 to num_steps-1

        data_dict, x, conditioning, t, params, rng = jax.lax.fori_loop(0, self.num_steps - 1, loop_body,
                                                                       (data_dict, x, conditioning,
                                                                        self.t_0, params, rng))

        # FINAL ITERATION

        data_dict, x, conditioning, t, params, rng = loop_body(self.num_steps - 1,
                                                               (data_dict, x, conditioning, t, params, rng),
                                                               final=True)

        data_dict['samples'] = x

        return data_dict, rng

    def sample(self, num_samples: int, dim: int, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
               z_init: Optional[jnp.ndarray], conditioning: Union[Tuple[jnp.ndarray], PyTree], ) -> Tuple[
        PyTree, jr.PRNGKey]:

        if z_init is not None:
            x_init = z_init
        else:
            x_init = jr.normal(rng, (num_samples, dim)) * self.sigma_init
        rng = jr.split(rng)[0]

        return self._inference(x_init, model, params, rng, conditioning)

    def forward(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> Tuple[PyTree, jr.PRNGKey]:

        return self._inference(x, model, params, rng, conditioning)


class FlowMapSampler(BaseSampler):

    def __init__(self, num_steps: int = 1, base_steps: int = 128,
                 t_0: float = 0.0, t_1: float = 1.0, rtol: float = 1e-5, mode: str = 'euler',
                 atol: float = 1e-5, sigma_init: float = 1.0, sigma_rescale: float = 0.0, dt_base_steps: Optional[int] = None):

        self.base_steps = base_steps
        self.num_steps = num_steps
        self.dt_base_steps = dt_base_steps
        self.mode = mode

        assert base_steps == num_steps or (base_steps // num_steps) % 2 == 0, \
            f"num_steps {num_steps} needs to be a power of two and divide base_steps {base_steps}"

        self.t_0 = t_0
        self.t_1 = t_1
        self.sigma_rescale = sigma_rescale
        self.sigma_init = sigma_init
        self.prob_wrapper = get_prob_wrapper('none')

    def rescale(self, x: jnp.ndarray, rng: jr.PRNGKey):
        x = x + jr.normal(rng, x.shape) * self.sigma_rescale
        rng = jr.split(rng)[0]
        return x, rng

    def compute_likelihood(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                           conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> PyTree:

        return self._inference(x, model, params, rng, conditioning, backward=False)

    def _inference(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                   conditioning: Union[Tuple[jnp.ndarray], PyTree], num_steps: int = 100, backward=False) -> Tuple[PyTree, jr.PRNGKey]:

        data_dict = {
            'trajectory': [],
            'features': [],
            'gradient': [],
            'nll': []
        }

        dt_base = jnp.log2(num_steps)
        dt = 1 / (2 ** dt_base)
        dt = jnp.repeat(dt, x.shape[0])

        if self.dt_base_steps is None:

            dt_base = jnp.repeat(dt_base, x.shape[0])

        else:

            dt_base = jnp.log2(self.dt_base_steps)
            dt_base = jnp.repeat(dt_base, x.shape[0])


        rng_apply, rng = generate_apply_rngs(rng)
        steps = jnp.linspace(self.t_0, self.t_1, num_steps+1)[:-1]

        def loop_body(iteration, values):

            x_, steps_, conditioning_, dt_base_, dt_, params_, rng_ = values

            t_ = jnp.repeat(jnp.array([steps_[iteration]]), x_.shape[0])

            v_ = model.apply(params_, x_, conditioning_, t_, dt_base_, rngs=rng_)
            rng_ = jr.split(rng_)[0]

            if self.mode=='euler':

                x_ = x_ + jnp.einsum('i...,i->i...', v_, dt_)

            elif self.mode=='consistency':

                eps = jr.normal(rng_, x_.shape)
                rng_ = jr.split(rng_)[0]
                x1_ = x_ + jnp.einsum('i...,i->i...', v_, (1-t_))
                x_ = (jnp.einsum('i...,i->i...', x1_, (t_ + dt_))
                      + jnp.einsum('i...,i->i...', eps, (1 - t_ - dt_)))

            else:
                raise NotImplementedError

            values = (
                x_, steps_, conditioning_, dt_base_, dt_, params_, rng_
            )

            return values

        x, _, _, _, _, _, rng = jax.lax.fori_loop(0, num_steps, loop_body,
                                                                         (x, steps, conditioning, dt_base, dt,
                                                                                 params, rng))

        data_dict['samples'] = x
        data_dict = {k: jnp.array(v) for k, v in data_dict.items()}

        return data_dict, rng

    def sample(self, num_samples: int, dim: int, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
               z_init: Optional[jnp.ndarray], conditioning: Union[Tuple[jnp.ndarray], PyTree], num_steps: Optional[int] = None) -> Tuple[
        PyTree, jr.PRNGKey]:

        if num_steps is None:
            num_steps = self.num_steps

        if z_init is not None:
            x_init = z_init
        else:
            x_init = self.sigma_init * jr.normal(rng, (num_samples, dim))
            rng = jr.split(rng)[0]

        return self._inference(x_init, model, params, rng, conditioning, num_steps=num_steps)

    def forward(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> Tuple[PyTree, jr.PRNGKey]:

        return self._inference(x, model, params, rng, conditioning)


class ODESolver(BaseSampler):

    def __init__(self, solver_name: str = 'euler', init_stepsize: float = 0.1, num_steps: int = 10,
                 stepsize_controller_name: str = 'constant', t_0: float = 0.0, t_1: float = 1.0, rtol: float = 1e-5,
                 atol: float = 1e-5, mode: str = 'none', sigma_init: float = 1.0, sigma_rescale: float = 0.0, ):

        self.solver_name = solver_name
        self.init_stepsize = init_stepsize
        self.num_steps = num_steps
        self.t_0 = t_0
        self.t_1 = t_1
        self.sigma_rescale = sigma_rescale
        self.sigma_init = sigma_init
        self.prob_wrapper = get_prob_wrapper(mode)
        self.solver = get_solver(self.solver_name)
        self.term = ODETerm(self.prob_wrapper)
        self.stepsize_controller = get_stepsize_controller(stepsize_controller_name, rtol, atol)

    def rescale(self, x: jnp.ndarray, rng: jr.PRNGKey):
        x = x + jr.normal(rng, x.shape) * self.sigma_rescale
        rng = jr.split(rng)[0]
        return x, rng

    def compute_likelihood(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                           conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> PyTree:

        return self._inference(x, model, params, rng, conditioning, backward=False)

    def get_ode_fn(self, model):

        @jit
        def ode_fn(t, y, args, params, rngs=None):
            out = model.apply(params, t, y, args, rngs=rngs, train=False)

            return out

        return ode_fn

    def _inference(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                   conditioning: Union[Tuple[jnp.ndarray], PyTree], backward=False, num_steps: int = 100) -> Tuple[PyTree, jr.PRNGKey]:

        data_dict = {
            'trajectory': [],
            'features': [],
            'gradient': [],
            'nll': []
        }

        if isinstance(x, tuple):
            eps = None
        else:
            eps = jr.normal(rng, x.shape)
            rng = jr.split(rng)[0]

        ode_fn = self.get_ode_fn(model)

        rng_apply, rng = generate_apply_rngs(rng)

        stepsize = 1 / num_steps

        if backward:

            y = (x, jnp.zeros(x.shape[0]))

            saveat = SaveAt(ts=jnp.linspace(self.t_1, self.t_0, num_steps))

            sol = diffeqsolve(self.term, self.solver, t0=self.t_1, t1=self.t_0,
                              dt0=-stepsize, saveat=saveat,
                              y0=y, args=(conditioning, eps, ode_fn, params, rng_apply),
                              stepsize_controller=self.stepsize_controller)

        else:

            if isinstance(x, tuple):
                prob_x = 0
            else:
                prob_x = logpdf(x).sum(axis=1)

            y = (x, prob_x)

            saveat = SaveAt(ts=jnp.linspace(self.t_0, self.t_1, num_steps))

            sol = diffeqsolve(self.term, self.solver, t0=self.t_0, t1=self.t_1, dt0=stepsize,
                              saveat=saveat, y0=y, args=(conditioning, eps, ode_fn, params, rng_apply),
                              stepsize_controller=self.stepsize_controller)

        x, ldj_cnf = sol.ys

        data_dict['samples'] = x[-1]

        data_dict['trajectory'] = x

        data_dict = {k: jnp.array(v) for k, v in data_dict.items()}

        return data_dict, rng

    def sample(self, num_samples: int, dim: int, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
               z_init: Optional[jnp.ndarray], conditioning: Union[Tuple[jnp.ndarray], PyTree], ) -> Tuple[
        PyTree, jr.PRNGKey]:

        if z_init is not None:
            x_init = z_init
        else:
            x_init = self.sigma_init * jr.normal(rng, (num_samples, dim))
            rng = jr.split(rng)[0]

        return self._inference(x_init, model, params, rng, conditioning)

    def forward(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> Tuple[PyTree, jr.PRNGKey]:

        return self._inference(x, model, params, rng, conditioning)


class PriorODESolver(ODESolver):

    def __init__(self, prior: Dict, solver_name: str = 'euler', init_stepsize: float = 0.1, num_steps: int = 10,
                 stepsize_controller_name: str = 'constant', t_0: float = 0.0, t_1: float = 1.0, rtol: float = 1e-5,
                 atol: float = 1e-5, mode: str = 'none', sigma_init: float = 1.0, sigma_rescale: float = 0.0, ):

        super().__init__(solver_name, init_stepsize, num_steps, stepsize_controller_name, t_0, t_1, rtol, atol,
                         mode, sigma_init, sigma_rescale)

        self.prior = instantiate_from_config(prior)

    def sample(self, num_samples: int, dim: int, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
               z_init: Optional[jnp.ndarray], conditioning: Union[Tuple[jnp.ndarray], PyTree],
               num_steps: Optional[int] = 100) -> Tuple[PyTree, jr.PRNGKey]:

        if num_steps is None:
            num_steps = self.num_steps

        if z_init is not None:
            x_init = z_init
        else:
            x_init, rng = self.prior.sample(rng, num_samples)

        return self._inference(x_init, model, params, rng, conditioning, num_steps = num_steps)


class FastCorrectorSampler(BaseSampler):
    """
    Fast Corrector Sampler
    """

    def __init__(self, num_steps: int, t_0: float = 0.0,
                 t_1: float = 1.0, start_time: float = 0.8):
        self.num_steps = num_steps
        self.t_0 = t_0
        self.t_1 = t_1
        self.start_time = start_time
        self.dt = (t_1 - t_0) / num_steps

    def forward(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> Tuple[PyTree, jr.PRNGKey]:

        return self._inference(x, model, params, rng, conditioning)

    def compute_likelihood(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                           conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> PyTree:

        return self._inference(x, model, params, rng, conditioning, backward=False)

    def _inference(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                   conditioning: Union[Tuple[jnp.ndarray], PyTree], backward=False) -> Tuple[PyTree, jr.PRNGKey]:

        data_dict = {
            'trajectory': [],
            'features': [],
            'gradient': [],
            'nll': []
        }

        @jit
        def ode_fn_flow(t, y, args, params, rngs=None):
            out = model.forward_flow(params, t, y, args, train=False, rngs=rngs)
            return out

        @jit
        def ode_fn_corrected(t, y, args, params, rngs=None):
            out = model.forward(params, t, y, args, train=False, rngs=rngs)
            return out

        t = jnp.repeat(jnp.array([self.t_0]), x.shape[0])

        def loop_body(i, vals, corrected=False):

            x, t, rng = vals

            rngs, rng = generate_apply_rngs(rng)

            if corrected:

                flow = ode_fn_corrected(t, x, conditioning, params, rngs=rngs)

            else:

                flow = ode_fn_flow(t, x, conditioning, params, rngs=rngs)

            x = x + self.dt * flow

            t += self.dt

            return x, t, rng

        iteration_phase_1 = int(self.num_steps * self.start_time)
        iteration_phase_2 = self.num_steps

        loop_body_corrected = partial(loop_body, corrected=True)
        loop_body_no_correction = partial(loop_body, corrected=False)

        x, t, rng = jax.lax.fori_loop(0, iteration_phase_1, loop_body_no_correction,
                                      (x, t, rng))

        x, t, rng = jax.lax.fori_loop(iteration_phase_1, iteration_phase_2, loop_body_corrected,
                                      (x, t, rng))

        data_dict['samples'] = x
        data_dict = {k: jnp.array(v) for k, v in data_dict.items()}

        return data_dict, rng

    def sample(self, num_samples: int, dim: int, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
               z_init: Optional[jnp.ndarray], conditioning: Union[Tuple[jnp.ndarray], PyTree], ) -> Tuple[
        PyTree, jr.PRNGKey]:

        if z_init is not None:
            x_init = z_init
        else:
            x_init = jr.normal(rng, (num_samples, dim))
            rng = jr.split(rng)[0]

        return self._inference(x_init, model, params, rng, conditioning)


class SelfConditionedSampler(BaseSampler):
    """
    Sampler solver with prior as initial condition and self-conditioning for the model
    """

    def __init__(self, prior: Dict, num_steps: int, t_0: float = 0.0, t_1: float = 1.0):
        self.num_steps = num_steps
        self.t_0 = t_0
        self.t_1 = t_1
        self.prior = instantiate_from_config(prior)
        self.dt = (t_1 - t_0) / num_steps

    def forward(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> Tuple[PyTree, jr.PRNGKey]:

        return self._inference(x, model, params, rng, conditioning)

    def compute_likelihood(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                           conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> PyTree:

        return self._inference(x, model, params, rng, conditioning, backward=False)

    def _inference(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                   conditioning: Union[Tuple[jnp.ndarray], PyTree], backward=False) -> Tuple[PyTree, jr.PRNGKey]:

        data_dict = {
            'trajectory': [],
            'features': [],
            'gradient': [],
            'nll': []
        }

        @jit
        def ode_fn(t, y, args, params):
            out = model.forward_inference(params, t, y, args, train=False)
            return out

        x, x_pred = x
        t = jnp.repeat(jnp.array([self.t_0]), x.shape[0])

        def loop_body(i, vals):
            x, x_pred, t, rng = vals

            flow, x_pred = ode_fn(t, [x, x_pred], conditioning, params)

            flow = jnp.nan_to_num(flow)
            x_pred = jnp.nan_to_num(x_pred)

            x = x + self.dt * flow

            t += self.dt

            return x, x_pred, t, rng

        x, x_pred, t, rng = jax.lax.fori_loop(0, self.num_steps, loop_body,
                                              (x, x_pred, t, rng))

        data_dict['samples'] = x
        data_dict = {k: jnp.array(v) for k, v in data_dict.items()}

        return data_dict, rng

    def sample(self, num_samples: int, dim: int, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
               z_init: Optional[jnp.ndarray], conditioning: Union[Tuple[jnp.ndarray], PyTree], ) -> Tuple[
        PyTree, jr.PRNGKey]:

        if z_init is not None:
            x_init = z_init
        else:
            x_init, rng = self.prior.sample(rng, num_samples)
            x_init = (x_init, jnp.zeros_like(x_init))

        return self._inference(x_init, model, params, rng, conditioning)


class DiffEqSolver(BaseSampler):
    rtol: float = 1e-5
    atol: float = 1e-5

    def __init__(self, sigma_schedule: Dict, solver_name: str = 'euler', num_steps: int = 10,
                 stepsize_controller_name: str = 'constant', t_0: float = 0.0, t_1: float = 10.0):

        self.sigma_schedule = instantiate_from_config(sigma_schedule)
        self.solver_name = solver_name
        self.init_stepsize = (t_1 - t_0) / num_steps
        self.stepsize_controller_name = stepsize_controller_name
        self.t_0 = t_0
        self.t_1 = t_1
        self.num_steps = num_steps

        self.solver = get_solver(self.solver_name)
        self.term = ODETerm(no_logp_wrapper)

        if self.stepsize_controller_name == 'constant':
            self.stepsize_controller = ConstantStepSize()
        else:
            self.stepsize_controller = PIDController(rtol=self.rtol, atol=self.atol)

    def compute_likelihood(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                           conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> PyTree:

        data_dict = {
            'likelihood': jnp.zeros(x.shape[0]),
        }

        return data_dict, rng

    def _inference(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                   conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> Tuple[PyTree, jr.PRNGKey]:

        data_dict = {
            'trajectory': [],
            'features': [],
            'gradient': [],
            'nll': []
        }

        eps = jr.normal(rng, x.shape)
        rng = jr.split(rng)[0]

        @jit
        def ode_fn(t, y, args, params):
            log_std = self.sigma_schedule.log_sigma(t)

            out = model.apply(params, log_std, y, args)

            update = jnp.einsum("ab,a->ab", out, jnp.power(10, log_std))

            return update

        y = (x, jnp.zeros(x.shape[0]))

        saveat = SaveAt(ts=jnp.linspace(self.t_0, self.t_1, self.num_steps))

        sol = diffeqsolve(self.term, self.solver, t0=self.t_0, t1=self.t_1, dt0=self.init_stepsize,
                          saveat=saveat, y0=y, args=(conditioning, eps, ode_fn, params),
                          stepsize_controller=self.stepsize_controller)

        x, ldj_cnf = sol.ys

        data_dict['samples'] = x[-1]

        data_dict['trajectory'] = x[:-1]

        data_dict = {k: jnp.array(v) for k, v in data_dict.items()}

        return data_dict, rng

    def sample(self, num_samples: int, dim: int, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
               z_init: Optional[jnp.ndarray], conditioning: Union[Tuple[jnp.ndarray], PyTree], ) -> Tuple[
        PyTree, jr.PRNGKey]:

        if z_init is not None:
            x_init = z_init
        else:
            x_init, rng = self.sigma_schedule.init_sample(rng, (num_samples, dim))

        return self._inference(x_init, model, params, rng, conditioning)

    def forward(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> Tuple[PyTree, jr.PRNGKey]:

        return self._inference(x, model, params, rng, conditioning)


class DDIMSampler(BaseSampler):

    def __init__(self, init_log_sigma, final_log_sigma, num_steps):

        self.init_log_sigma = init_log_sigma
        self.final_log_sigma = final_log_sigma
        self.reduction_factor = (init_log_sigma - final_log_sigma) / num_steps
        self.num_steps = num_steps

    def compute_likelihood(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                           conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> PyTree:

        data_dict = {
            'likelihood': jnp.zeros(x.shape[0]),
        }

        return data_dict, rng

    def _inference(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                   conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> Tuple[PyTree, jr.PRNGKey]:

        data_dict = {
            'trajectory': jnp.zeros((self.num_steps,) + x.shape),
            'features': jnp.zeros((self.num_steps,) + x.shape),
            'gradient': jnp.zeros((self.num_steps,) + x.shape),
            'nll': jnp.zeros((self.num_steps, x.shape[0]))
        }

        def loop_body(i, vals, final=False):

            data_dict, x, conditioning, std, params, rng = vals

            std_in = jnp.repeat(std, x.shape[0])

            if hasattr(model.__class__, 'forward'):

                out = model.apply(params, std_in, x, conditioning, method=model.__class__.forward)

                x = x + jnp.power(10, std) * out[0]

                data_dict['trajectory'] = data_dict['trajectory'].at[i].set(x)
                data_dict['features'] = data_dict['features'].at[i].set(out[3])
                data_dict['gradient'] = data_dict['gradient'].at[i].set(out[2])
                data_dict['nll'] = data_dict['nll'].at[i].set(out[1])

            else:

                out = model.apply(params, std_in, x, conditioning)

                x = x + jnp.power(10, std) * out

                data_dict['trajectory'] = data_dict['trajectory'].at[i].set(x)

            if not final:
                std -= self.reduction_factor
                x = x + jr.normal(rng, x.shape) * jnp.power(10, std)
                rng = jr.split(rng)[0]

            return data_dict, x, conditioning, std, params, rng

        # LOOP 0 to num_steps-1

        data_dict, x, conditioning, std, params, rng = jax.lax.fori_loop(0, self.num_steps - 1, loop_body,
                                                                         (data_dict, x, conditioning,
                                                                          self.init_log_sigma, params, rng))

        # FINAL ITERATION

        data_dict, x, conditioning, std, params, rng = loop_body(self.num_steps - 1,
                                                                 (data_dict, x, conditioning, std, params, rng),
                                                                 final=True)

        data_dict['samples'] = x

        return data_dict, rng

    def sample(self, num_samples: int, dim: int, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
               z_init: Optional[jnp.ndarray], conditioning: Union[Tuple[jnp.ndarray], PyTree], ) -> Tuple[
        PyTree, jr.PRNGKey]:

        if z_init is not None:
            x_init = z_init
        else:
            x_init = jr.normal(rng, (num_samples, dim)) * jnp.power(10, self.init_log_sigma)
        rng = jr.split(rng)[0]

        return self._inference(x_init, model, params, rng, conditioning)

    def forward(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> Tuple[PyTree, jr.PRNGKey]:

        return self._inference(x, model, params, rng, conditioning)


class DDIM(BaseSampler):

    def __init__(self, path: Dict, init_stepsize: float = 0.1, t_0: float = 0.0,
                 t_1: float = 1.0, mode: str = 'none', sigma_init: float = 1.0,
                 sigma_rescale: float = 0.0, correction_steps: int = 2, r: float = 0.01):
        """

        :param path:
        :param init_stepsize:
        :param t_0:
        :param t_1:
        :param mode:
        :param sigma_init:
        :param correction_steps:
        :param correction_steps:
        :param epsilon:
        :param r: signal-to-noise ratio as defined in Song et al. 2021 appendix G
        """

        self.path = instantiate_from_config(path)
        self.init_stepsize = init_stepsize
        self.t_0 = t_0
        self.t_1 = t_1
        self.r = r
        self.correction_steps = correction_steps
        self.num_steps = int((t_1 - t_0) / init_stepsize)
        self.sigma_init = sigma_init
        self.sigma_rescale = 0.0
        self.prob_wrapper = get_prob_wrapper(mode)

    def rescale(self, x: jnp.ndarray, rng: jr.PRNGKey):
        x = x + jr.normal(rng, x.shape) * self.sigma_rescale
        rng = jr.split(rng)[0]
        return x, rng

    def flow_to_score(self, x: jnp.ndarray, t: jnp.ndarray):

        scaling = 1 / (self.path.grad_sigma(t))
        return jnp.einsum("ab,a->ab", x, scaling)

    def compute_likelihood(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                           conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> PyTree:

        data_dict = {
            'likelihood': jnp.zeros(x.shape[0]),
        }

        return data_dict, rng

    def _inference(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                   conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> Tuple[PyTree, jr.PRNGKey]:

        data_dict = {
            'trajectory': jnp.zeros((self.num_steps,) + x.shape),
            'features': jnp.zeros((self.num_steps,) + x.shape),
            'gradient': jnp.zeros((self.num_steps,) + x.shape),
            'nll': jnp.zeros((self.num_steps, x.shape[0]))
        }

        def loop_body(i, vals, final=False):

            data_dict, x, conditioning, t, params, rng = vals

            t_in = jnp.repeat(t, x.shape[0])

            if hasattr(model.__class__, 'forward'):

                out = model.apply(params, t_in, x, conditioning,
                                  method=model.__class__.forward)

                x = x + self.init_stepsize * out[0]

                data_dict['trajectory'] = data_dict['trajectory'].at[i].set(x)
                data_dict['features'] = data_dict['features'].at[i].set(out[3])
                data_dict['gradient'] = data_dict['gradient'].at[i].set(out[2])
                data_dict['nll'] = data_dict['nll'].at[i].set(out[1])

            else:

                out = model.apply(params, t_in, x, conditioning)

                print('out: ', out)

                x = x + self.init_stepsize * out

                data_dict['trajectory'] = data_dict['trajectory'].at[i].set(x)

            for _ in range(self.correction_steps - 1):
                epsilon = jr.normal(rng, x.shape)
                rng = jr.split(rng)[0]

                out = model.apply(params, t_in, x, conditioning)

                score = self.flow_to_score(out, t_in)

                norm_score = jnp.mean(jnp.linalg.norm(score, axis=1))
                norm_noise = jnp.mean(jnp.linalg.norm(epsilon, axis=1))

                eps = 2 * ((self.r * norm_noise / norm_score) ** 2)

                x = x + eps * score + epsilon * jnp.sqrt(2 * eps)

            if self.correction_steps > 0:

                epsilon = jr.normal(rng, x.shape)
                rng = jr.split(rng)[0]

                out = model.apply(params, t_in, x, conditioning)

                score = self.flow_to_score(out, t_in)

                norm_score = jnp.mean(jnp.linalg.norm(score, axis=1))
                norm_noise = jnp.mean(jnp.linalg.norm(epsilon, axis=1))

                eps = 2 * ((self.r * norm_noise / norm_score) ** 2)

                print('eps:', eps)

                x = x + score * eps

                if not final:
                    x = x + epsilon * jnp.sqrt(2 * eps)

            return data_dict, x, conditioning, t + self.init_stepsize, params, rng

        # LOOP 0 to num_steps-1

        data_dict, x, conditioning, t, params, rng = jax.lax.fori_loop(0, self.num_steps - 1, loop_body,
                                                                       (data_dict, x, conditioning,
                                                                        self.t_0, params, rng))

        # FINAL ITERATION

        data_dict, x, conditioning, t, params, rng = loop_body(self.num_steps - 1,
                                                               (data_dict, x, conditioning, t, params, rng),
                                                               final=True)

        data_dict['samples'] = x

        return data_dict, rng

    def sample(self, num_samples: int, dim: int, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
               z_init: Optional[jnp.ndarray], conditioning: Union[Tuple[jnp.ndarray], PyTree], ) -> Tuple[
        PyTree, jr.PRNGKey]:

        if z_init is not None:
            x_init = z_init
        else:
            x_init = jr.normal(rng, (num_samples, dim)) * self.sigma_init
        rng = jr.split(rng)[0]

        return self._inference(x_init, model, params, rng, conditioning)

    def forward(self, x: jnp.ndarray, model: nn.Module, params: PyTree, rng: jr.PRNGKey,
                conditioning: Union[Tuple[jnp.ndarray], PyTree]) -> Tuple[PyTree, jr.PRNGKey]:

        return self._inference(x, model, params, rng, conditioning)

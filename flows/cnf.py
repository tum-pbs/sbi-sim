from abc import ABC
from functools import partial
from typing import Tuple

import diffrax
from diffrax import PIDController, ODETerm, diffeqsolve, ConstantStepSize

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit

import flax.linen as nn

from flows.models.fcnn import CNFFCNN
from flows.flowbase import FlowBase


def get_solver(solver, scan_stages=False):

    # use scan https://github.com/patrick-kidger/diffrax/issues/94 for lower compile time

    if solver == 'tsit5':
        return diffrax.Tsit5(scan_stages=scan_stages)
    elif solver == 'dopri5':
        return diffrax.Dopri5(scan_stages=scan_stages)
    elif solver == 'euler':
        return diffrax.Euler()
    else:
        raise ValueError(f"diffrax solver \"{solver}\" not recognized")

def approx_logp_wrapper(t, y, args):
    y, _ = y
    *args, eps, func, params, rngs = args
    fn = lambda y: func(t, y, args[0], params, rngs=rngs)
    f, vjp_fn = jax.vjp(fn, y)
    (eps_dfdy,) = vjp_fn(eps)
    logp = jnp.sum(eps_dfdy * eps, axis=-1)
    return f, logp

def wrap_expansion(y):
    if isinstance(y, tuple):
        return tuple([y_[None] for y_ in y])
    else:
        return y[None]

def get_stepsize_controller(stepsize_controller_name, rtol, atol):

        if stepsize_controller_name == 'constant':
            return ConstantStepSize()
        elif stepsize_controller_name == 'pid':
            return PIDController(rtol=rtol, atol=atol)
        else:
            raise ValueError(f"stepsize controller \"{stepsize_controller_name}\" not recognized")

def get_prob_wrapper(mode):

    if mode == 'approx':
        return approx_logp_wrapper
    elif mode == 'exact':
        return exact_logp_wrapper
    elif mode == 'none':
        return no_logp_wrapper
    else:
        raise ValueError(f"log p computation mode \"{mode}\" not recognized")

def no_logp_wrapper(t, y, args):

    y, _ = y
    *args, _, func, params, rngs = args

    if isinstance(y, tuple):
        t = jnp.repeat(t, y[0].shape[0])

    else:
        t = jnp.repeat(t, y.shape[0])

    out = func(t, y, args[0], params, rngs=rngs)
    return out, 0


def exact_logp_wrapper(t, y, args):
    y, _ = y
    *args, _, func, params, rngs = args

    def exact_logp_sample(y_sample, conditioning_sample):
        fn = lambda y_: func(t[None], wrap_expansion(y_), wrap_expansion(conditioning_sample), params, rngs=rngs)[0]
        f, vjp_fn = jax.vjp(fn, y_sample)
        (size,) = y_sample.shape  # only 1D input
        eye = jnp.eye(size)
        (dfdy,) = jax.vmap(vjp_fn)(eye)
        logp = jnp.trace(dfdy)
        return f, logp

    f, logp = jax.vmap(exact_logp_sample, in_axes=0)(y, args[0])

    return f, logp

# CNF with diffrax thanks to https://docs.kidger.site/diffrax/examples/continuous_normalising_flow/
# For issues when combining diffrax and Flax, see
#   1. https://github.com/patrick-kidger/diffrax/issues/115
#   2. https://github.com/google/flax/discussions/2891

class ContinuousNormalizingFlow(FlowBase, ABC):

    time_0: float = 0.0
    time_T: float = 1.0
    init_stepsize: float = 0.1

    rtol: float = 1e-5
    atol: float = 1e-5

    stepsize_controller_name: str = 'constant'

    solver_name: str = 'dopri5'
    mode: str = 'approx'

    def get_variables(self, *args):
        if self.is_initializing():
            self.model(*args)
        return self.model.variables["params"]

    def setup(self):

        if self.mode == 'approx':
            self.logp_wrapper = approx_logp_wrapper
        elif self.mode == 'exact':
            self.logp_wrapper = exact_logp_wrapper
        elif self.mode == 'none':
            self.logp_wrapper = no_logp_wrapper
        else:
            raise ValueError(f"log p computation mode \"{self.mode}\" not recognized")

        if self.stepsize_controller_name == 'constant':
            self.stepsize_controller = ConstantStepSize()
        else:
            self.stepsize_controller = PIDController(rtol=self.rtol, atol=self.atol)

        self.term = ODETerm(self.logp_wrapper)
        self.solver = get_solver(self.solver_name)

    def __call__(self, x: jnp.ndarray, rng: jr.PRNGKey,
                 weighting: jnp.ndarray,
                 conditioning: jnp.ndarray,
                 testing=False, bpd=False):

        return super().__call__(x, rng, weighting, conditioning, testing, bpd)

    def forward_(self, x: jnp.ndarray, rng: jr.PRNGKey,
                ldj: jnp.ndarray,
                conditioning: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:

        eps = jr.normal(rng, x.shape)
        rng = jr.split(rng)[0]

        # just initialize the model with batch size 1 here to avoid retracing if possible

        if isinstance(conditioning, tuple):
            ode_params = self.get_variables(jnp.zeros(shape=(1,)).astype(jnp.float32), x[:1],
                                            tuple([c[:1] for c in conditioning]))
        else:
            ode_params = self.get_variables(jnp.array(0.0).astype(jnp.float32), x[:1], conditioning[:1])

        def ode_fn(t, y, args, params):
            return self.model.apply({'params': params}, t, y, args)

        y = (x, jnp.zeros(x.shape[0]))

        sol = diffeqsolve(self.term, self.solver, t0=self.time_0, t1=self.time_T, dt0=self.init_stepsize,
                        y0=y, args=(conditioning, eps, ode_fn, ode_params), stepsize_controller=self.stepsize_controller)

        (z,), (ldj_cnf,) = sol.ys

        return z, ldj+ldj_cnf, rng

    def inverse_(self, z: jnp.ndarray, rng: jr.PRNGKey,
                ldj: jnp.ndarray,
                conditioning: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jr.PRNGKey]:

        eps = jr.normal(rng, z.shape)
        rng = jr.split(rng)[0]

        if isinstance(conditioning, tuple):
            ode_params = self.get_variables(jnp.zeros(shape=(1,)).astype(jnp.float32), z[:1],
                                            tuple([c[:1] for c in conditioning]))
        else:
            ode_params = self.get_variables(jnp.array(0.0).astype(jnp.float32), z[:1], conditioning[:1])

        def ode_fn(t, y, args, params):
            return self.model.apply({'params': params}, t, y, args)

        y = (z, jnp.zeros(z.shape[0]))

        sol = diffeqsolve(self.term, self.solver, t0=self.time_T, t1=self.time_0, dt0=-self.init_stepsize,
                          y0=y, args=(conditioning, eps, ode_fn, ode_params),
                          stepsize_controller=self.stepsize_controller)

        (x,), (ldj_cnf,) = sol.ys

        return x, ldj+ldj_cnf, rng

class MLPCNF(ContinuousNormalizingFlow):

    solver_name: str = 'dopri5'
    mode: str = 'approx'
    hidden_dim: int = 32

    def setup(self):

        super().setup()

        self.model = CNFFCNN(in_dim=self.dim_flow+self.dim_conditioning+1,
                          out_dim=self.dim_flow, hidden_dim=self.hidden_dim)

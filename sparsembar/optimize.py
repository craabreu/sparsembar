"""
.. module:: optimize
   :platform: Linux, MacOS
   :synopsis: A module for handling optimization algorithms.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import typing as t

from jax import numpy as jnp
from scipy import optimize

_METHODS_THAT_REQUIRE_HESSIAN = [
    "Newton-CG",
    "dogleg",
    "trust-ncg",
    "trust-krylov",
    "trust-exact",
    "trust-constr",
]


def argmin(
    objective_function: t.Callable,
    initial_guess: jnp.ndarray,
    *args,
    method: str = "BFGS",
    tolerance: float = 1e-12,
    allow_unconverged: bool = True,
    jac: t.Optional[t.Callable] = None,
    hess: t.Optional[t.Callable] = None,
    **kwargs,
) -> jnp.ndarray:
    """
    Minimize a function using a specified method and return the argument that
    minimizes the function.

    Parameters
    ----------
    fun
        The function to be minimized.
    x0
        The initial guess.
    args
        Additional arguments to be passed to the function.
    method
        The minimization method to use. The options are the same as for
        :func:`scipy.optimize.minimize`.
    tolerance
        The tolerance for termination. When specified, the selected minimization
        algorithm sets some relevant solver-specific tolerance(s) equal to this
        value.
    allow_unconverged
        Whether to allow unconverged minimization results due to precision loss.
    jac
        The Jacobian of `fun`.
    hess
        The Hessian of `fun`.

    Returns
    -------
    jnp.ndarray
        The optimization result.
    """
    kwargs = {**kwargs, "method": method, "tol": tolerance, "jac": jac}
    if method in _METHODS_THAT_REQUIRE_HESSIAN:
        kwargs["hess"] = hess
    result = optimize.minimize(objective_function, initial_guess, args, **kwargs)
    if not (result.success or (result.status == 2 and allow_unconverged)):
        raise ValueError(result.message)
    return jnp.array(result.x)

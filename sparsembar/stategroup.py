"""
.. module:: stategroup
   :platform: Linux, MacOS
   :synopsis: A class for representing a group of states.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import typing as t
from functools import partial

import jax
from jax import numpy as jnp
from jax.scipy import special

from .optimize import argmin


@partial(jax.jit, static_argnames=("prepend_zero",))
def mbar_negative_log_likelihood(
    free_energies: jnp.ndarray,
    potentials: jnp.ndarray,
    sample_sizes: jnp.ndarray,
    prepend_zero: bool = True,
) -> float:
    """The negative of the log-likelihood function that is maximized by the MBAR"""
    if prepend_zero:
        free_energies = jnp.insert(free_energies, 0, 0.0)
    return special.logsumexp(
        free_energies - potentials.T,
        b=sample_sizes,
        axis=1,
    ).sum() - jnp.dot(sample_sizes, free_energies)


mbar_gradient = jax.grad(mbar_negative_log_likelihood)
mbar_hessian = jax.hessian(mbar_negative_log_likelihood)


class StateGroup:
    """
    A class for representing a group of states with sampled configurations.

    .. _scipy.optimize.minimize:
        https://tinyurl.com/yx3by4p3

    Parameters
    ----------
    states
        A sequence of hashable identifiers for the states in the group. The number
        :math:`K` of states is the length of this sequence. If an identical state
        appears in other groups, it should have the same hashable identifier in all
        groups.
    potentials
        A matrix of reduced potentials having one of the following shapes:
        1. :math:`(K, N_{\\rm sum})`, with :math:`N_{\\rm sum} = \\sum_{i=0}^{K-1} N_i`,
        where :math:`N_i` is the number of samples drawn from state :math:`i`. Thus,
        :math:`u_{k,n}` is the reduced potential of the :math:`n`-th sample evaluated
        in the :math:`k`-th state, independently of which state the sample was actually
        drawn from.
        2. :math:`(K, K, N_{\\rm max})`, where :math:`N_{\\rm max} = \\max(N_0,
        \\ldots, N_{K-1})`. In this case, :math:`u_{k,l,n}` is the reduced potential
        evaluated at the :math:`k`-th state for the :math:`n`-th sample drawn from the
        :math:`l`-th state.
    sample_sizes
        The number of samples drawn from each state. If not provided, it is assumed
        that the samples are evenly split among the states.
    method
        The minimization method to use for free energy calculation. The options are the
        same as for `scipy.optimize.minimize`_.
    tolerance
        The tolerance for termination of the minimization. Each method sets some
        relevant solver-specific tolerance(s) equal to this value.
    allow_unconverged
        Whether to allow unconverged minimization results due to lack of numerical
        precision.
    **kwargs
        Additional keyword arguments that will be passed to `scipy.optimize.minimize`_,
        except for ``method``, ``tol``, ``jac`` and ``hess``.

    Examples
    --------
    >>> import sparsembar as smbar
    >>> means = [0, 1, 2]
    >>> model = smbar.MultiGaussian(means, 1)
    >>> samples = model.draw_samples(100)
    >>> matrix = model.compute_reduced_potentials(samples)
    >>> state_group = smbar.StateGroup(means, matrix)
    >>> matrix = model.compute_reduced_potentials(
    ...     samples, kn_format=False,
    ... )
    >>> state_group = smbar.StateGroup(
    ...     means, matrix, sample_sizes=[80, 90, 100],
    ... )
    >>> state_group
    StateGroup(states=[0, 1, 2], sample_sizes=[ 80  90 100])
    >>> state_group.get_free_energies()
    Array([...], dtype=float64)
    """

    def __init__(
        self,
        states: t.Sequence[t.Hashable],
        potentials: jnp.ndarray,
        sample_sizes: t.Optional[t.Sequence[int]] = None,
        *,
        method: str = "BFGS",
        tolerance: float = 1e-12,
        allow_unconverged: bool = True,
        **kwargs,
    ) -> None:
        num_states = len(states)
        shape = potentials.shape
        ndim = len(shape)
        args = (num_states, shape, ndim, sample_sizes)
        self._validate_input(*args)
        self._states = states
        self._sample_sizes = self._get_sample_sizes_array(*args)
        self._potentials = self._get_potentials_array(*args, potentials)
        self._free_energies = self._compute_free_energies(
            method, tolerance, allow_unconverged, **kwargs
        )

    def __len__(self) -> int:
        return len(self._states)

    def __repr__(self) -> str:
        return f"StateGroup(states={self._states}, sample_sizes={self._sample_sizes})"

    @staticmethod
    def _validate_input(
        num_states: int,
        shape: t.Tuple[int, ...],
        ndim: int,
        sample_sizes: t.Optional[t.Sequence[int]],
    ) -> None:
        if shape[0] != num_states:
            raise ValueError("Wrong number of states in potentials")
        if sample_sizes is not None and len(sample_sizes) != num_states:
            raise ValueError("Wrong number of states in sample_sizes")
        if ndim not in [2, 3] or (ndim == 3 and shape[1] != num_states):
            raise ValueError("Wrong shape of potentials")
        if ndim == 2 and sample_sizes is None and shape[1] % num_states != 0:
            raise ValueError("Cannot split potentials evenly")

    @staticmethod
    def _get_sample_sizes_array(
        num_states: int,
        shape: t.Tuple[int, ...],
        ndim: int,
        sample_sizes: t.Optional[t.Sequence[int]],
    ) -> jnp.ndarray:
        if sample_sizes is None:
            return jnp.array(
                [shape[1] // num_states if ndim == 2 else shape[2]] * num_states
            )
        if (ndim == 2 and sum(sample_sizes) == shape[1]) or (
            ndim == 3 and max(sample_sizes) <= shape[2]
        ):
            return jnp.array(sample_sizes)
        raise ValueError("Wrong numbers of samples in sample_sizes")

    @staticmethod
    def _get_potentials_array(
        num_states: int,
        shape: t.Sequence[int],
        ndim: int,
        sample_sizes: t.Optional[t.Sequence[int]],
        potentials: jnp.ndarray,
    ) -> jnp.ndarray:
        if ndim == 3:
            if any(x < shape[2] for x in sample_sizes):
                return jnp.concatenate(
                    [potentials[i, :, :num] for i, num in enumerate(sample_sizes)],
                    axis=-1,
                )
            return jnp.reshape(potentials, (num_states, -1))
        return jnp.array(potentials)

    def _compute_free_energies(
        self,
        method: str,
        tolerance: float,
        allow_unconverged: bool,
        **kwargs,
    ) -> jnp.ndarray:
        xmin = argmin(
            mbar_negative_log_likelihood,
            jnp.zeros(len(self._states) - 1),
            (self._potentials, self._sample_sizes),
            method,
            tolerance,
            allow_unconverged,
            mbar_gradient,
            mbar_hessian,
            **kwargs,
        )
        return jnp.insert(xmin, 0, 0.0)

    @property
    def states(self) -> t.Sequence[t.Hashable]:
        """The states in the group."""
        return self._states

    @property
    def potentials(self) -> jnp.ndarray:
        """The matrix of reduced potentials."""
        return self._potentials

    @property
    def sample_sizes(self) -> jnp.ndarray:
        """The number of samples drawn from each state in the group."""
        return self._sample_sizes

    def get_free_energies(
        self, return_dict: bool = False
    ) -> t.Union[t.Dict[t.Hashable, float], jnp.ndarray]:
        """
        Return the free energies of the states.

        Parameters
        ----------
        return_dict
            Whether to return a dictionary of free energies with the states as keys.

        Returns
        -------
        t.Union[t.Dict[t.Hashable, float], jnp.ndarray]
            The reduced free energies of the states. If ``return_dict`` is ``True``,
            the return value is a dictionary of free energies with the states as keys.
            Otherwise, it is an array of free energies.

        Examples
        --------
        >>> import sparsembar as smbar
        >>> from pymbar import MBAR
        >>> from numpy import allclose
        >>> model = smbar.MultiGaussian([0, 1, 2], [1, 2, 3], seed=1)
        >>> samples = model.draw_samples(100)
        >>> potentials = model.compute_reduced_potentials(samples)
        >>> state_group = smbar.StateGroup([0, 1, 2], potentials, method="Newton-CG")
        >>> free_energies = state_group.get_free_energies()
        >>> free_energies
        Array([...], dtype=float64)
        >>> mbar = MBAR(state_group.potentials, state_group.sample_sizes)
        >>> result = mbar.compute_free_energy_differences()
        >>> assert allclose(free_energies, result["Delta_f"][0, :])
        """
        return (
            dict(zip(self._states, self._free_energies))
            if return_dict
            else self._free_energies
        )

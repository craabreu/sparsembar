"""
.. module:: sparsembar
   :platform: Linux, MacOS
   :synopsis: A module for performing MBAR estimation on groups of states with
              sparse connections.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import itertools as it
import typing as t

import jax
from jax import numpy as jnp

from .optimize import argmin
from .stategroup import StateGroup, mbar_negative_log_likelihood


@jax.jit
def sparse_mbar_negative_log_likelihood(
    free_energies: jnp.ndarray,
    potentials: t.Sequence[jnp.ndarray],
    sample_sizes: t.Sequence[jnp.ndarray],
    state_indices: t.Sequence[jnp.ndarray],
) -> float:
    """The negative of the log-likelihood function that is maximized by the
    Sparse MBAR estimator."""
    free_energies = jnp.insert(free_energies, 0, 0.0)
    return sum(
        map(
            mbar_negative_log_likelihood,
            [jnp.take(free_energies, indices) for indices in state_indices],
            potentials,
            sample_sizes,
            it.repeat(False),
        )
    )


sparse_mbar_gradient = jax.grad(sparse_mbar_negative_log_likelihood)
sparse_mbar_hessian = jax.hessian(sparse_mbar_negative_log_likelihood)


class SparseMBAR:  # pylint: disable=too-few-public-methods
    """
    A class for performing MBAR estimation on groups of states with sparse
    connections.

    Parameters
    ----------
    state_groups
        A sequence of state groups.
    method
        The minimization method to use for free energy calculation. The options are the
        same as for :func:`scipy.optimize.minimize`.
    tolerance
        The tolerance for termination of the minimization. Each method sets some
        relevant solver-specific tolerance(s) equal to this value.
    allow_unconverged
        Whether to allow unconverged minimization results due to lack of numerical
        precision.
    **kwargs
        Additional keyword arguments that will be passed to the
        :func:`scipy.optimize.minimize` function, except for ``method``, ``tol``,
        ``jac`` and ``hess``.

    Examples
    --------
    >>> import sparsembar as smbar
    >>> state_id_lists = [0, 1, 2], [2, 3, 4]
    >>> models = [
    ...     smbar.MultiGaussian(state_ids, 1)
    ...     for state_ids in state_id_lists
    ... ]
    >>> potentials = [
    ...     model.compute_reduced_potentials(model.draw_samples(100))
    ...     for model in models
    ... ]
    >>> estimator = smbar.SparseMBAR(
    ...     smbar.StateGroup(ids, matrix)
    ...     for ids, matrix in zip(state_id_lists, potentials)
    ... )
    >>> states = estimator.all_states
    >>> states
    (0, 1, 2, 3, 4)
    >>> [estimator.groups_with_state(state) for state in states]
    [(0,), (0,), (0, 1), (1,), (1,)]
    """

    def __init__(
        self,
        state_groups: t.Iterable[StateGroup],
        *,
        method: str = "BFGS",
        tolerance: float = 1e-12,
        allow_unconverged: bool = True,
        **kwargs,
    ) -> None:
        self._groups = list(state_groups)
        self._num_groups = len(self._groups)

        self._groups_with_state = {}
        for group_index, group in enumerate(self._groups):
            for state_index, state in enumerate(group.states):
                if state not in self._groups_with_state:
                    self._groups_with_state[state] = []
                self._groups_with_state[state].append((group_index, state_index))

        self._all_states = tuple(self._groups_with_state)
        self._num_states = len(self._all_states)

        self._state_indices = [
            jnp.array([self._all_states.index(state) for state in group.states])
            for group in self._groups
        ]

        self._rough_free_energies = self._compute_rough_free_energies()
        self._free_energies = self._compute_free_energies(
            method, tolerance, allow_unconverged, **kwargs
        )

    def _compute_rough_free_energies(self) -> jnp.ndarray:
        zeros = jnp.zeros(self._num_states)
        sample_sizes = jnp.vstack(
            jnp.put(zeros, state_indices, group.sample_sizes, inplace=False)
            for group, state_indices in zip(self._groups, self._state_indices)
        )
        free_energies = jnp.vstack(
            jnp.put(zeros, state_indices, group.get_free_energies(), inplace=False)
            for group, state_indices in zip(self._groups, self._state_indices)
        )
        w_fractions = sample_sizes / sample_sizes.sum(axis=1, keepdims=True)
        z_fractions = sample_sizes / sample_sizes.sum(axis=0, keepdims=True)
        unshifted_means = (z_fractions * free_energies).sum(axis=0)
        alpha_matrix = w_fractions.dot(z_fractions.T) - jnp.identity(self._num_groups)
        b_vector = (w_fractions * (free_energies - unshifted_means)).sum(axis=1)
        shifts = jnp.linalg.solve(
            alpha_matrix.at[0, :].set(z_fractions.at[:, 0].get()),
            b_vector.at[0].set(-unshifted_means.at[0].get()),
        )
        return unshifted_means + z_fractions.T.dot(shifts)

    def _compute_free_energies(
        self, method: str, tolerance: float, allow_unconverged: bool, **kwargs
    ) -> None:
        free_energies = argmin(
            sparse_mbar_negative_log_likelihood,
            self._rough_free_energies.at[1:].get(),
            (
                [group.potentials for group in self._groups],
                [group.sample_sizes for group in self._groups],
                self._state_indices,
            ),
            method,
            tolerance,
            allow_unconverged,
            sparse_mbar_gradient,
            sparse_mbar_hessian,
            **kwargs,
        )
        print(self._rough_free_energies)
        print(jnp.insert(free_energies, 0, 0.0))
        return jnp.insert(free_energies, 0, 0.0)

    @property
    def all_states(self) -> t.Tuple[t.Hashable, ...]:
        """
        Return a tuple of all distinct states in the groups.

        Returns
        -------
        t.Tuple[t.Hashable, ...]
            A tuple of all distinct states in the groups.
        """
        return self._all_states

    @property
    def groups(self) -> t.List[StateGroup]:
        """The list of state groups."""
        return self._groups

    def groups_with_state(self, state: t.Hashable) -> t.Tuple[int]:
        """
        The index of the state groups that contain the given state.

        Parameters
        ----------
        state
            The state whose groups will be returned.

        Returns
        -------
        t.Tuple[int]
            The indices of the state groups that contain the given state.
        """
        if state not in self._groups_with_state:
            raise ValueError(f"Unknown state {state}")
        return tuple(index for index, _ in self._groups_with_state[state])

    def get_free_energies(
        self, return_dict: bool = False
    ) -> t.Union[t.Dict[t.Hashable, float], t.Tuple[jnp.ndarray, ...]]:
        """
        Return the free energies of all states in the groups.

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
        >>> mean_lists = [0, 1, 2], [2, 2, 1, 0]
        >>> sigma_lists = [1, 2, 3], [3, 2, 1, 1]

        Using tuples of means and standard deviations as state IDs:

        >>> state_id_lists = [
        ...     [(mean, sigma) for mean, sigma in zip(means, sigmas)]
        ...     for means, sigmas in zip(mean_lists, sigma_lists)
        ... ]
        >>> state_id_lists
        [[(0, 1), (1, 2), (2, 3)], [(2, 3), (2, 2), (1, 1), (0, 1)]]

        Sampling from multi-dimensional Gaussian distributions:

        >>> models = [
        ...     smbar.MultiGaussian(means, sigmas, seed=123)
        ...     for means, sigmas in zip(mean_lists, sigma_lists)
        ... ]
        >>> potentials = [
        ...     model.compute_reduced_potentials(model.draw_samples(100))
        ...     for model in models
        ... ]

        Estimating the free energies of the states:

        >>> estimator = smbar.SparseMBAR(
        ...     smbar.StateGroup(ids, matrix)
        ...     for ids, matrix in zip(state_id_lists, potentials)
        ... )
        >>> estimator.get_free_energies()
        (Array([...], ...dtype=float64), Array([...], ...dtype=float64))
        >>> estimator.get_free_energies(return_dict=True)
        {(0, 1): 0.0, (1, 2): ..., (2, 3): ..., (2, 2): ..., (1, 1): ...}
        """
        if return_dict:
            return dict(zip(self._all_states, self._free_energies.tolist()))
        return tuple(
            jnp.take(self._free_energies, indices) for indices in self._state_indices
        )

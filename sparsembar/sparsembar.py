"""
.. module:: sparsembar
   :platform: Linux, MacOS
   :synopsis: A module for performing MBAR estimation on groups of states with
              sparse connections.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import typing as t

import numpy as np
from jax import numpy as jnp

from .optimize import argmin
from .stategroup import StateGroup


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

        self._overlapping_states = {
            state: group_state_pairs
            for state, group_state_pairs in self._groups_with_state.items()
            if len(group_state_pairs) > 1
        }

        self._all_states = tuple(self._groups_with_state)
        self._num_states = len(self._all_states)
        self._free_energies = self._compute_free_energy_initial_guess(
            method, tolerance, allow_unconverged, **kwargs
        )

    def _compute_free_energy_initial_guess(
        self,
        method: str,
        tolerance: float,
        allow_unconverged: bool,
        **kwargs,
    ) -> None:
        free_energies = [np.array(group.get_free_energies()) for group in self._groups]

        def _sum_square_deviations(values: t.Iterable[float]) -> float:
            values = np.fromiter(values, dtype=np.float64)
            return np.square(values - values.mean()).sum()

        def _misfit(shifts: np.ndarray) -> float:
            shifts = np.insert(shifts, 0, 0.0)
            return sum(
                _sum_square_deviations(
                    free_energies[igroup][istate] + shifts[igroup]
                    for igroup, istate in pairs
                )
                for pairs in self._overlapping_states.values()
            )

        shifts = argmin(
            _misfit,
            np.zeros(self._num_groups - 1),
            method=method,
            tol=tolerance,
            allow_unconverged=allow_unconverged,
            jac=None,
            hess=None,
            **kwargs,
        ).tolist()
        for index, shift in enumerate(shifts, 1):
            free_energies[index] += shift
        for _, pairs in self._overlapping_states.items():
            mean_free_energy = np.mean(
                [free_energies[igroup][istate].item() for igroup, istate in pairs]
            )
            for igroup, istate in pairs:
                free_energies[igroup][istate] = mean_free_energy
        return tuple(map(jnp.array, free_energies))

    @property
    def all_states(self) -> set:
        """
        Return the set of states in the groups.

        Returns
        -------
        set
            The set of states in the groups.

        """
        return self._all_states

    @property
    def groups(self) -> t.List[StateGroup]:
        """The list of state groups."""
        return self._groups

    def groups_with_state(self, state: t.Hashable) -> int:
        """The index of the state groups with the given state."""
        if state not in self._groups_with_state:
            raise ValueError(f"Unknown state {state}")
        return tuple(index for index, _ in self._groups_with_state[state])

    def get_free_energies(self) -> jnp.ndarray:
        """
        Examples
        --------
        >>> import sparsembar as smbar
        >>> mean_lists = [0, 1, 2], [2, 1, 0]
        >>> sigma_lists = [1, 2, 3], [3, 1, 2]
        >>> state_id_lists = [
        ...     [(mean, sigma) for mean, sigma in zip(means, sigmas)]
        ...     for means, sigmas in zip(mean_lists, sigma_lists)
        ... ]
        >>> state_id_lists
        [[(0, 1), (1, 2), (2, 3)], [(2, 3), (1, 1), (0, 2)]]
        >>> models = [
        ...     smbar.MultiGaussian(means, sigmas, seed=123)
        ...     for means, sigmas in zip(mean_lists, sigma_lists)
        ... ]
        >>> potentials = [
        ...     model.compute_reduced_potentials(model.draw_samples(100))
        ...     for model in models
        ... ]
        >>> estimator = smbar.SparseMBAR(
        ...     smbar.StateGroup(ids, matrix)
        ...     for ids, matrix in zip(state_id_lists, potentials)
        ... )
        >>> estimator.get_free_energies()
        (Array([...], dtype=float64), Array([...], dtype=float64))
        """
        return self._free_energies

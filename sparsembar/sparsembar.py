"""
.. module:: sparsembar
   :platform: Linux, MacOS
   :synopsis: A module for performing MBAR estimation on groups of states with
              sparse connections.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import typing as t

from jax import numpy as jnp

from .stategroup import StateGroup


class SparseMBAR:  # pylint: disable=too-few-public-methods
    """
    A class for performing MBAR estimation on groups of states with sparse
    connections.

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
    >>> states = sorted(estimator.all_states)
    >>> states
    [0, 1, 2, 3, 4]
    >>> [estimator.groups_with_state(state) for state in states]
    [[0], [0], [0, 1], [1], [1]]
    """

    def __init__(self, state_groups: t.Iterable[StateGroup]) -> None:
        self._groups = list(state_groups)
        self._all_states = {state for group in self._groups for state in group.states}
        self._groups_with_state = {
            state: jnp.array(
                [
                    index
                    for index, group in enumerate(self._groups)
                    if state in group.states
                ]
            )
            for state in self._all_states
        }

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
        if state not in self._all_states:
            raise ValueError(f"Unknown state {state}")
        return self._groups_with_state[state].tolist()

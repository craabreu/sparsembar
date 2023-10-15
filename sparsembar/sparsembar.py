"""
.. module:: sparsembar
   :platform: Linux, MacOS
   :synopsis: A module for performing MBAR estimation on groups of states with
              sparse connections.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import typing as t

from .stategroup import StateGroup


class SparseMBAR:  # pylint: disable=too-few-public-methods
    """
    A class for performing MBAR estimation on groups of states with sparse
    connections.

    Examples
    --------
    >>> import sparsembar as smbar
    >>> state_id_lists = [0, 1, 2], [2, 3, 4]
    >>> models = [smbar.MultiGaussian(ids, 1, 123) for ids in state_id_lists]
    >>> samples = [model.draw_samples(100) for model in models]
    >>> potentials = [
    ...     model.compute_reduced_potentials(model.draw_samples(100))
    ...     for model in models
    ... ]
    >>> estimator = smbar.SparseMBAR(
    ...     smbar.StateGroup(ids, matrix)
    ...     for ids, matrix in zip(state_id_lists, potentials)
    ... )
    >>> sorted(estimator.states)
    [0, 1, 2, 3, 4]
    """

    def __init__(self, state_groups: t.Iterable[StateGroup]) -> None:
        self._state_groups = list(state_groups)
        self._states = {state for group in self._state_groups for state in group.states}

    @property
    def states(self) -> set:
        """
        Return the set of states in the groups.

        Returns
        -------
        set
            The set of states in the groups.

        """
        return self._states

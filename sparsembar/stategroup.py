"""
.. module:: stategroup
   :platform: Linux, MacOS
   :synopsis: A class for representing a group of states.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import typing as t

from jax import numpy as jnp


class StateGroup:
    """
    A class for representing a group of states.

    Parameters
    ----------
    states
        A sequence of hashable identifiers for the states in the group. The number
        :math:`K` of states is inferred from the length of this sequence. If a given
        state appears in more than one group, it must be represented by the same
        identifier in all groups.
    potentials
        A matrix of reduced potentials. This matrix can have one of the following
        shapes:

        1. :math:`(K, N_{\\rm sum})`, where :math:`N_{\\rm sum} = \\sum_{i=0}^{K-1}
        N_i` and :math:`N_i` is the number of samples drawn from state :math:`i`.
        In this case, :math:`u_{k,n}` is the reduced potential of the :math:`n`-th
        sample evaluated in the :math:`k`-th state, independently of which state
        the sample was drawn from.

        2. :math:`(K, K, N_{\\rm max})`, where :math:`N_{\\rm max} = \\max(N_0,
        \\ldots, N_{K-1})`. In this case, :math:`u_{k,l,n}` is the reduced potential
        of the :math:`n`-th sample drawn from the :math:`l`-th state, but evaluated
        in the :math:`k`-th state.
    sample_sizes
        The number of samples drawn from each state in the group. If not provided,
        it will be assumed that all states have the same number of samples and
        this number will be inferred from the shape of the reduced potential
        matrix.

    Examples
    --------
    >>> import sparsembar as smbar
    >>> means = [0, 1, 2]
    >>> model = smbar.MultiGaussian(means, 1, 123)
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
    """

    def __init__(
        self,
        states: t.Sequence[t.Hashable],
        potentials: jnp.ndarray,
        sample_sizes: t.Optional[t.Sequence[int]] = None,
    ) -> None:
        num_states = len(states)
        shape = potentials.shape
        ndim = len(shape)
        self._validate_input(num_states, shape, ndim, sample_sizes)
        self._states = states
        self._sample_sizes = self._get_sample_sizes_array(
            num_states,
            shape,
            ndim,
            sample_sizes,
        )
        self._potentials = self._get_potentials_array(
            num_states,
            shape,
            ndim,
            sample_sizes,
            potentials,
        )

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

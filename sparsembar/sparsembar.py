"""
.. module:: sparsembar
   :platform: Linux, MacOS
   :synopsis: A module for performing MBAR estimation on groups of states with
              sparse connections.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import typing as t
from functools import reduce

from jax import numpy as jnp


class SparseMBAR:  # pylint: disable=too-few-public-methods
    """
    A class for performing MBAR estimation on groups of states with sparse
    connections.

    Examples
    --------
    >>> import sparsembar as smbar
    >>> smbar.__version__
    '0.0.0'
    """

    def __init__(self) -> None:
        self._groups = []
        self._states = set()
        self._reduced_potentials = []
        self._sample_sizes = []

    def add_state_group(
        self,
        states: t.Sequence[t.Hashable],
        potentials: jnp.ndarray,
        sample_sizes: t.Optional[t.Sequence[int]] = None,
    ) -> None:
        """
        Add a group of states to the MBAR estimator.

        Parameters
        ----------
        states
            A sequence of hashable objects representing the states in the group.
            The number :math:`K` of states in the group if inferred from the length
            of this sequence.
        potentials
            A matrix of reduced potentials. This matrix can have one of the following
            shapes:

            1. :math:`(K, N_{\\rm sum})`, where :math:`N_{\\rm sum} = \\sum_{i=0}^{K-1}
            N_i` and :math:`N_i` is the number of samples drawn from state :math:`i`.

            2. :math:`(K, K, N_{\\rm max})`, where :math:`N_{\\rm max} = \\max(N_0,
            \\ldots, N_{K-1})`.
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
        >>> estimator = smbar.SparseMBAR()
        >>> estimator.add_state_group(means, matrix)
        >>> matrix = model.compute_reduced_potentials(samples, kn_format=False)
        >>> estimator.add_state_group(means, matrix, sample_sizes=[80, 90, 100])
        """
        num_states = len(states)
        shape = potentials.shape
        rank = len(shape)

        if shape[0] != num_states:
            raise ValueError("Wrong number of states in potentials")
        if sample_sizes is not None and len(sample_sizes) != num_states:
            raise ValueError("Wrong number of states in sample_sizes")
        if rank not in [2, 3] or (rank == 3 and shape[1] != num_states):
            raise ValueError("Wrong shape of potentials")
        if rank == 2 and sample_sizes is None and shape[1] % num_states != 0:
            raise ValueError("Cannot split potentials evenly")

        if sample_sizes is None:
            samples_per_state = shape[1] // num_states if rank == 2 else shape[2]
            sample_sizes = jnp.array([samples_per_state] * num_states)
        elif (rank == 2 and sum(sample_sizes) == shape[1]) or (
            rank == 3 and reduce(max, sample_sizes) <= shape[2]
        ):
            sample_sizes = jnp.array(sample_sizes)
        else:
            raise ValueError("Wrong numbers of samples in sample_sizes")

        if rank == 3:
            if any(map(lambda x: x < shape[2], sample_sizes)):
                potentials = jnp.concatenate(
                    [potentials[i, :, :num] for i, num in enumerate(sample_sizes)],
                    axis=-1,
                )
            else:
                potentials = jnp.reshape(potentials, (num_states, -1))

        self._groups.append(states)
        self._states.update(states)
        self._reduced_potentials.append(potentials)
        self._sample_sizes.append(sample_sizes)

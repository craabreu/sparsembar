"""
.. module:: multigaussian
   :platform: Linux, MacOS
   :synopsis: A set of multidimensional, independent Gaussian distributions.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>
"""

import typing as t
import jax
from jax import numpy as jnp


@jax.jit
def _draw_samples(
    means: jnp.ndarray,
    sigmas: jnp.ndarray,
    draws: int,
    prng_key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """
    Draw samples from a collection of multidimensional Gaussian distributions.

    Parameters
    ----------
    means
        The location of the Gaussian distributions. The shape must be (num, dim), where
        num is the number of distributions and dim is the number of dimensions.
    sigmas
        The standard deviations of the Gaussian distributions. The shape must be
        (num,).
    draws
        The number of samples to be drawn from each distribution.
    prng_key
        The key of the pseudo-random number generator.

    Returns
    -------
    jnp.ndarray
        The samples drawn from the distributions. The shape is (draws, num, dim).
    """
    return sigmas[:, None] * jax.random.normal(prng_key, (draws, *means.shape)) + means


@jax.jit
def _compute_reduced_energy_matrix(
    samples: jnp.ndarray,
    means: jnp.ndarray,
    sigmas: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the reduced energy matrix of a sample taken from a collection of
    multidimensional Gaussian distributions.

    Parameters
    ----------
    samples
        The samples drawn from the distributions. The shape must be (draws, num, dim),
        where draws is the number of samples per distribution, num is the number of
        distributions, and dim is the number of dimensions.
    means
        The means of the Gaussian distributions. The shape must be (num, dim).
    sigmas
        The standard deviations of the Gaussian distributions. The shape must be
        (num,).

    Returns
    -------
    jnp.ndarray
        The energy matrix of the sample, whose shape is (num, num, draws).
    """
    devs = samples[:, :, None, :] - means
    return 0.5 * jnp.square(devs / sigmas[:, None]).sum(axis=3).transpose()


class MultiGaussian:
    """
    A set of multidimensional, independent Gaussian distributions. The means of the
    distributions are located at nodes of an integer lattice. All the distributions
    are isotropic, i.e., the covariance matrix of every distribution :math:`k` is
    :math:`\\sigma_k^2 \\mathbf{I}`, where :math:`\\sigma_k` is the standard
    deviation and :math:`\\mathbf{I}` is the identity matrix.

    The number of distributions is :math:`K` and the number of dimensions is :math:`D`.

    Parameters
    ----------
    means
        The locations of the means of the distributions. It can be a sequence of
        integers for one-dimensional distributions or a sequence of tuples of integers
        for multi-dimensional distributions. All tuples must have the same length. Both
        :math:`K` and :math:`D` are inferred from the passed sequence.
    sigmas
        The standard deviation of the Gaussian distributions. If a single value is
        passed, all distributions have the same standard deviation.
    seed
        A seed for the pseudo-random number generator.
    """

    def __init__(
        self,
        means: t.Sequence[t.Union[int, t.Tuple[int, ...]]],
        sigmas: t.Union[float, t.Sequence[float]],
        seed: int,
    ) -> None:
        def are_all(item_type):
            return lambda seq: all(isinstance(x, item_type) for x in seq)

        self._num = len(means)
        if are_all(int)(means):
            self._dim = 1
            self._mean_tuples = [(mean,) for mean in means]
            self._means = jnp.array(means, dtype=jnp.float64).reshape((-1, 1))
        elif are_all(tuple)(means) and all(map(are_all(int), means)):
            items = iter(means)
            self._dim = len(next(items))
            if any(len(item) != self._dim for item in items):
                raise ValueError("All tuples must have the same length.")
            self._mean_tuples = means
            self._means = jnp.array(means, dtype=jnp.float64)
        else:
            raise ValueError("Each mean must be an integer or a tuple of integers.")
        if isinstance(sigmas, (int, float)):
            self._sigmas = jnp.ones(self._num, dtype=jnp.float64) * sigmas
        elif len(sigmas) == self._num and are_all((int, float))(sigmas):
            self._sigmas = jnp.array(sigmas, dtype=jnp.float64)
        else:
            raise ValueError("Each sigma must be a float or a sequence of floats.")
        if any(self._sigmas <= 0.0):
            raise ValueError("All sigmas must be positive.")
        self._prng_key = jax.random.PRNGKey(seed)

    def draw_samples(self, num_draws: int) -> jnp.ndarray:
        """
        Draw samples from this collection of multidimensional Gaussian distributions.

        The probability density function of a configuration :math:`\\mathbf{X} =
        \\{\\mathbf{x}_0, \\ldots, \\mathbf{x}_{K-1}\\}`, where each
        :math:`\\mathbf{x}_i` is a :math:`D`-dimensional random vector, is given by

        .. math::

            p(\\mathbf{X}) =  \\frac{1}{(2 \\pi \\sigma^2)^{KD/2}}
                \\exp \\left( - \\frac{1}{2 \\sigma^2} \\sum_{i=0}^{K-1}
                    \\left\\| \\mathbf{x}_i - \\boldsymbol{\\mu}_i \\right\\|^2
                \\right),

        where :math:`\\boldsymbol{\\mu}_i` is the mean of the :math:`i`-th Gaussian
        distribution and :math:`\\|\\cdot\\|` is the Euclidean norm.

        This method returns a sample :math:`\\{\\mathbf{X}_0, \\ldots,
        \\mathbf{X}_{N-1}\\}` in the form of an array of shape (N, K, D).

        Parameters
        ----------
        num_draws
            The number of samples :math:`N` to be drawn from the multigaussian
            distribution.

        Returns
        -------
        jnp.ndarray
            The samples drawn from the distributions, whose shape is (N, K, D).
        """
        return _draw_samples(self._means, self._sigmas, num_draws, self._prng_key)

    def compute_reduced_energy_matrix(
        self, samples: jnp.ndarray, kn_format: bool = True
    ) -> jnp.ndarray:
        """
        Compute the reduced energy matrix of a sample.

        The sample is assumed to have been drawn using :func:`draw_samples`, so that
        its shape is (N, K, D), where N is the number of samples per distribution,
        K is the number of distributions, and D is the number of dimensions.

        The reduced energy matrix can be returned one of two ways:

        1. An array of shape :math:`(K, K, N)` whose elements are given by

        .. math::

            u_{k,l,n} = \\frac{1}{2 \\sigma^2} \\left\\|
                \\mathbf{x}_k(\\mathbf{X}_n) - \\boldsymbol{\\mu}_k
            \\right\\|^2

        2. (default) An array of shape :math:`(K, KN)` whose elements are given by

        .. math::

            u_{k,n} = \\frac{1}{2 \\sigma^2} \\left\\|
                \\mathbf{x}_{n \\bmod K}\\left(
                    \\mathbf{X}_{\\lfloor n/K \\rfloor}
                \\right) - \\boldsymbol{\\mu}_k
            \\right\\|^2

        Parameters
        ----------
        samples
            The samples drawn from the distributions. The shape must be (N, K, D),
            where N is the number of samples per distribution, K is the number of
            distributions, and D is the number of dimensions.
        kn_format
            Whether to return the matrix in the :math:`u_{k,n}` format with shape
            :math:`(K, KN)`. If False, the matrix is returned in the :math:`u_{k,l,n}`
            format with shape :math:`(K, K, N)`.

        Returns
        -------
        jnp.ndarray
            The reduced energy matrix of the sample. The shape is :math:`(K, KN)` or
            :math:`(K, K, N)`, depending on the value of `kn_format`.
        """
        matrix = _compute_reduced_energy_matrix(samples, self._means, self._sigmas)
        return matrix.reshape(self._num, -1) if kn_format else matrix

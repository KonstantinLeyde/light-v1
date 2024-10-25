import jax
import numpy as np
import numpyro.distributions as dist
import scipy
from jax import lax
from jax import numpy as jnp
from jax.scipy.special import gammaln, xlog1py, xlogy
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import (
    binomial,
    clamp_probs,
    lazy_property,
    promote_shapes,
    validate_sample,
)
from numpyro.util import not_jax_tracer


def stirling_approximation(n):
    """
    Compute the Stirling approximation for n!.

    Parameters:
    - n: Non-negative integer.

    Returns:
    - Approximation of n! using the Stirling formula.
    """
    #     if jnp.any(n < 0):
    #         raise ValueError("n must be a non-negative integer.")

    #     if n == 0:
    #         return 1.0  # 0! is defined to be 1

    sqrt_term = jnp.sqrt(2 * jnp.pi * n)
    power_term = (n / jnp.exp(1)) ** n

    return sqrt_term * power_term


def log_stirling_approximation(n):
    """
    Write the above approximation in log space, for better
    accuracy.

    """

    log_sqrt_term = 1 / 2 * jnp.log(2 * jnp.pi * n)
    log_power_term = n * (jnp.log(n) - 1)

    return log_sqrt_term + log_power_term


# inherite from the true bionomial distribution
class BinomialProbsStirling(dist.BinomialProbs):
    arg_constraints = {
        "probs": constraints.unit_interval,
        # "total_count": constraints.nonnegative_integer,
    }
    has_enumerate_support = True

    def __init__(self, probs, total_count=1, *, validate_args=None):
        self.probs, self.total_count = promote_shapes(probs, total_count)
        batch_shape = lax.broadcast_shapes(jnp.shape(probs), jnp.shape(total_count))
        super(dist.BinomialProbs, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    @validate_sample
    def log_prob(self, value):
        log_factorial_n = log_stirling_approximation(self.total_count)
        log_factorial_k = log_stirling_approximation(value)
        log_factorial_nmk = log_stirling_approximation(self.total_count - value)
        return (
            log_factorial_n
            - log_factorial_k
            - log_factorial_nmk
            + xlogy(value, self.probs)
            + xlog1py(self.total_count - value, -self.probs)
        )


class BinomialProbs(Distribution):
    """
    I copied the source code from numpyro and simply implemented an
    cutoff for values larger than the total_count. This did not make
    a difference for the inference.

    """

    arg_constraints = {
        "probs": constraints.unit_interval,
        "total_count": constraints.nonnegative_integer,
    }
    has_enumerate_support = True

    def __init__(self, probs, total_count=1, *, validate_args=None):
        self.probs, self.total_count = promote_shapes(probs, total_count)
        batch_shape = lax.broadcast_shapes(jnp.shape(probs), jnp.shape(total_count))
        super(BinomialProbs, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return binomial(
            key, self.probs, n=self.total_count, shape=sample_shape + self.batch_shape
        )

    @validate_sample
    def log_prob(self, value):
        log_factorial_n = gammaln(self.total_count + 1)
        log_factorial_k = gammaln(value + 1)
        log_factorial_nmk = gammaln(self.total_count - value + 1)
        probs = clamp_probs(self.probs)
        heaviside = jnp.where(value < self.total_count, 1, 100 * self.total_count)
        return (
            log_factorial_n
            - log_factorial_k
            - log_factorial_nmk
            + xlogy(value, probs)
            + xlog1py(self.total_count - value, -probs)
        )

    @lazy_property
    def logits(self):
        return _to_logits_bernoulli(self.probs)

    @property
    def mean(self):
        return jnp.broadcast_to(self.total_count * self.probs, self.batch_shape)

    @property
    def variance(self):
        return jnp.broadcast_to(
            self.total_count * self.probs * (1 - self.probs), self.batch_shape
        )

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        return constraints.integer_interval(0, self.total_count)

    def enumerate_support(self, expand=True):
        if not_jax_tracer(self.total_count):
            total_count = np.amax(self.total_count)
            # NB: the error can't be raised if inhomogeneous issue happens when tracing
            if np.amin(self.total_count) != total_count:
                raise NotImplementedError(
                    "Inhomogeneous total count not supported" " by `enumerate_support`."
                )
        else:
            total_count = jnp.amax(self.total_count)
        values = jnp.arange(total_count + 1).reshape(
            (-1,) + (1,) * len(self.batch_shape)
        )
        if expand:
            values = jnp.broadcast_to(values, values.shape[:1] + self.batch_shape)
        return values


# adapted from https://num.pyro.ai/en/stable/tutorials/truncated_distributions.html#3
def scipy_truncated_poisson_icdf(args):  # Note: all arguments are passed inside a tuple
    rate, high, u = args
    rate = np.asarray(rate)
    high = np.asarray(high)
    u = np.asarray(u)
    density = scipy.stats.poisson(rate)
    normalizer = density.cdf(high)
    x = normalizer * u
    return density.ppf(x)


class RightTruncatedPoisson(dist.Distribution):
    """
    A truncated Poisson distribution.
    :param numpy.ndarray high: high bound at which truncation happens
    :param numpy.ndarray rate: rate of the Poisson distribution.
    """

    arg_constraints = {
        "high": dist.constraints.nonnegative_integer,
        "rate": dist.constraints.positive,
    }
    has_enumerate_support = True

    def __init__(self, rate=1.0, high=0, validate_args=None):
        batch_shape = jax.lax.broadcast_shapes(jnp.shape(high), jnp.shape(rate))
        self.high, self.rate = dist.util.promote_shapes(high, rate)
        super().__init__(batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        m = jax.scipy.stats.poisson.cdf(self.high, self.rate)
        log_p = jax.scipy.stats.poisson.logpmf(value, self.rate)
        return jnp.where(value <= self.high, log_p - jnp.log(m), -jnp.inf)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        float_type = jnp.result_type(float)
        minval = jnp.finfo(float_type).tiny
        u = jax.random.uniform(key, shape, minval=minval)
        # return self.icdf(u)        # Brute force
        # return self.icdf_faster(u) # For faster sampling.
        return self.icdf(u)  # Using `host_callback`

    # def icdf(self, u):
    #     def cond_fn(val):
    #         n, cdf = val
    #         return jnp.any(cdf < u)

    #     def body_fn(val):
    #         n, cdf = val
    #         n_new = jnp.where(cdf < u, n + 1, n)
    #         return n_new, self.cdf(n_new)

    #     high = self.high * jnp.ones_like(u)
    #     cdf = self.cdf(high)
    #     n, _ = jax.lax.while_loop(cond_fn, body_fn, (high, cdf))
    #     return n.astype(jnp.result_type(int))

    # def icdf_faster(self, u):
    #     num_bins = 500 # Choose a reasonably large value
    #     bins = jnp.arange(num_bins)
    #     cdf = self.cdf(bins)
    #     indices = jnp.searchsorted(cdf, u)
    #     return bins[indices]

    def icdf(self, u):
        result_shape = jax.ShapeDtypeStruct(u.shape, jnp.result_type(float))
        result = jax.experimental.host_callback.call(
            scipy_truncated_poisson_icdf,
            (self.rate, self.high, u),
            result_shape=result_shape,
        )
        return result.astype(jnp.result_type(int))

    def cdf(self, value):
        m = jax.scipy.stats.poisson.cdf(self.high, self.rate)
        f = jax.scipy.stats.poisson.cdf(value, self.rate)
        return jnp.where(value <= self.high, f / m, 0)

    @dist.constraints.dependent_property(is_discrete=True)
    def support(self):
        return dist.constraints.integer_greater_than(self.high)

    # in order to do sampling, we have to first write a function for
    # enumerate_support
    def enumerate_support(self, expand=True):
        if not_jax_tracer(self.high):
            high = np.amax(self.high)
            # NB: the error can't be raised if inhomogeneous issue happens when tracing
            if np.amin(self.high) != high:
                raise NotImplementedError(
                    "Inhomogeneous total count not supported" " by `enumerate_support`."
                )
        else:
            high = jnp.amax(self.high)
        values = jnp.arange(high + 1).reshape((-1,) + (1,) * len(self.batch_shape))
        if expand:
            values = jnp.broadcast_to(values, values.shape[:1] + self.batch_shape)
        return values


class RightTruncatedPoissonCategorical(dist.Distribution):
    def __init__(self, rate, high, validate_args=None):
        self.rate = rate
        self.high = high
        super(RightTruncatedPoissonCategorical, self).__init__(
            probs=self._calculate_probs(), validate_args=validate_args
        )

    def _calculate_probs(self):
        # Calculate the probabilities for each integer up to the truncation point (self.high)
        probs = jax.scipy.stats.poisson.pmf(jnp.arange(self.high + 1), self.rate)
        return probs

import copy

import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax
from numpyro.distributions import constraints
from numpyro.distributions.transforms import (
    IdentityTransform,
    ParameterFreeTransform,
    Transform,
)


def arctan_0_1_interpolation(x):
    return (jnp.arctan(x) + jnp.pi / 2) / jnp.pi


def logistic_0_1_interpolation(x):
    return 1 / (1 + jnp.exp(-x))


def a_b_interpolate(x, mu, sigma, a, delta):
    argument = (x - mu) / sigma

    term_1 = a
    term_2 = delta * logistic_0_1_interpolation(argument)

    return term_1 + term_2


class AsinhTransform(ParameterFreeTransform):
    domain = dist.constraints.real
    codomain = dist.constraints.positive

    def __call__(self, x):
        return jnp.arcsinh(x)

    def _inverse(self, y):
        return jnp.sinh(y)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return -1 / 2 * jnp.log(1 + x**2)


class Arcsinh(ParameterFreeTransform):
    domain = dist.constraints.real
    codomain = dist.constraints.positive

    def __call__(self, x):
        return jnp.arcsinh(x)

    def _inverse(self, y):
        return jnp.sinh(y)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return -1 / 2 * jnp.log(1 + x**2)


class CustomTransformLinear(ParameterFreeTransform):
    domain = dist.constraints.real
    codomain = dist.constraints.positive

    def __call__(self, x):
        return None

    def _inverse(self, y):
        return jnp.sinh(y)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return -1 / 2 * jnp.log(1 + x**2)


def positive_polynomial_7(x, a1, a2, a3, b1, b2, b3):
    return (
        x**7 / 7
        - a1 * x**6 / 3
        - a2 * x**6 / 3
        - a3 * x**6 / 3
        + a1**2 * x**5 / 5
        + a2**2 * x**5 / 5
        + a3**2 * x**5 / 5
        + b1**2 * x**5 / 5
        + b2**2 * x**5 / 5
        + b3**2 * x**5 / 5
        + 4 / 5 * a1 * a2 * x**5
        + 4 / 5 * a1 * a3 * x**5
        + 4 / 5 * a2 * a3 * x**5
        - 1 / 2 * a1 * a2**2 * x**4
        - 1 / 2 * a1 * a3**2 * x**4
        - 1 / 2 * a2 * a3**2 * x**4
        - 1 / 2 * a2 * b1**2 * x**4
        - 1 / 2 * a3 * b1**2 * x**4
        - 1 / 2 * a1 * b2**2 * x**4
        - 1 / 2 * a3 * b2**2 * x**4
        - 1 / 2 * a1 * b3**2 * x**4
        - 1 / 2 * a2 * b3**2 * x**4
        - 1 / 2 * a1**2 * a2 * x**4
        - 1 / 2 * a1**2 * a3 * x**4
        - 1 / 2 * a2**2 * a3 * x**4
        - 2 * a1 * a2 * a3 * x**4
        + 1 / 3 * a1**2 * a2**2 * x**3
        + 1 / 3 * a1**2 * a3**2 * x**3
        + 1 / 3 * a2**2 * a3**2 * x**3
        + 4 / 3 * a1 * a2 * a3**2 * x**3
        + 1 / 3 * a2**2 * b1**2 * x**3
        + 1 / 3 * a3**2 * b1**2 * x**3
        + 4 / 3 * a2 * a3 * b1**2 * x**3
        + 1 / 3 * a1**2 * b2**2 * x**3
        + 1 / 3 * a3**2 * b2**2 * x**3
        + 1 / 3 * b1**2 * b2**2 * x**3
        + 4 / 3 * a1 * a3 * b2**2 * x**3
        + 1 / 3 * a1**2 * b3**2 * x**3
        + 1 / 3 * a2**2 * b3**2 * x**3
        + 1 / 3 * b1**2 * b3**2 * x**3
        + 1 / 3 * b2**2 * b3**2 * x**3
        + 4 / 3 * a1 * a2 * b3**2 * x**3
        + 4 / 3 * a1 * a2**2 * a3 * x**3
        + 4 / 3 * a1**2 * a2 * a3 * x**3
        - a1 * a2**2 * a3**2 * x**2
        - a1**2 * a2 * a3**2 * x**2
        - a2 * a3**2 * b1**2 * x**2
        - a2**2 * a3 * b1**2 * x**2
        - a1 * a3**2 * b2**2 * x**2
        - a3 * b1**2 * b2**2 * x**2
        - a1**2 * a3 * b2**2 * x**2
        - a1 * a2**2 * b3**2 * x**2
        - a2 * b1**2 * b3**2 * x**2
        - a1 * b2**2 * b3**2 * x**2
        - a1**2 * a2 * b3**2 * x**2
        - a1**2 * a2**2 * a3 * x**2
        + a1**2 * a2**2 * a3**2 * x
        + a2**2 * a3**2 * b1**2 * x
        + a1**2 * a3**2 * b2**2 * x
        + a3**2 * b1**2 * b2**2 * x
        + a1**2 * a2**2 * b3**2 * x
        + a2**2 * b1**2 * b3**2 * x
        + a1**2 * b2**2 * b3**2 * x
        + b1**2 * b2**2 * b3**2 * x
    )


def d_positive_polynomial_7_dx(x, a1, a2, a3, b1, b2, b3):
    return (
        (a1**2 + b1**2 - 2 * a1 * x + x**2)
        * (a2**2 + b2**2 - 2 * a2 * x + x**2)
        * (a3**2 + b3**2 - 2 * a3 * x + x**2)
    )


def positive_polynomial_3(x, a1, b1):
    numerator = x * (3 * (a1**2 + b1**2) - 3 * a1 * x + x**2)
    denominator = 1 - 3 * a1 + 3 * (a1**2 + b1**2)
    return numerator / denominator


def d_positive_polynomial_3_dx(x, a1, b1):
    numerator = 3 * (a1**2 + b1**2 - 2 * a1 * x + x**2)
    denominator = 1 - 3 * a1 + 3 * (a1**2 + b1**2)
    return numerator / denominator


def positive_polynomial_3_inverse(x, a1, b1):
    f_x = (
        -27 * a1**3
        - 81 * a1 * b1**2
        + 27 * x
        - 81 * a1 * x
        + 81 * a1**2 * x
        + 81 * b1**2 * x
    )
    root = jnp.sqrt(2916 * b1**6 + f_x**2)
    return (
        a1
        - (3 * 2 ** (1 / 3) * b1**2) / (f_x + root) ** (1 / 3)
        + (f_x + root) ** (1 / 3) / (3 * 2 ** (1 / 3))
    )


class PositivePolynomial3(Transform):
    def __init__(self, a1, b1, domain=constraints.unit_interval):
        self.a1 = a1
        self.b1 = b1
        self.domain = domain

    @property
    def codomain(self):
        if self.domain is constraints.real:
            return constraints.real
        elif self.domain is constraints.unit_interval:
            return constraints.unit_interval
        else:
            raise NotImplementedError

    def __call__(self, x):
        polynomial = positive_polynomial_3(x, self.a1, self.b1)
        return polynomial

    def _inverse(self, y):
        return positive_polynomial_3_inverse(y, self.a1, self.b1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        log_jacobian = jnp.log(d_positive_polynomial_3_dx(x, self.a1, self.b1))
        return jnp.broadcast_to(log_jacobian, jnp.shape(x))

    def tree_flatten(self):
        return (
            self.a1,
            self.b1,
        ), (("a1", "b1"), dict())

    def forward_shape(self, shape):
        return lax.broadcast_shapes(
            shape, getattr(self.a1, "a1", ()), getattr(self.b1, "b1", ())
        )

    def inverse_shape(self, shape):
        return lax.broadcast_shapes(
            shape, getattr(self.a1, "a1", ()), getattr(self.b1, "b1", ())
        )


def construct_inverse_transform(transform_list):
    return [t.inv for t in transform_list[::-1]]


def apply_transform_list(x, transformations):
    y = copy.copy(x)
    for t in transformations:
        y = t(y)
    return y


class PiecewisePositivePolynomial3(Transform):
    domain = constraints.real
    codomain = constraints.real

    def __init__(self, a1, b1, a2, b2):
        self.a1 = a1
        self.b1 = b1
        self.a2 = a2
        self.b2 = b2

        self.threshold = 0
        self.f1 = PositivePolynomial3(self.a1, self.b1, domain=constraints.real)
        self.f2 = PositivePolynomial3(self.a2, self.b2, domain=constraints.real)

    def __call__(self, x):
        comp_polynomial = jnp.where(x < self.threshold, self.f1(x), self.f2(x))
        return comp_polynomial

    def _inverse(self, y):
        return jnp.where(y < self.threshold, self.f1.inv(y), self.f2.inv(y))

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        log_jacobian = jnp.where(
            x < self.threshold,
            self.f1.log_abs_det_jacobian(x, y),
            self.f2.log_abs_det_jacobian(x, y),
        )
        return jnp.broadcast_to(log_jacobian, jnp.shape(x))

    def tree_flatten(self):
        return (self.a1, self.b1, self.a2, self.b2), (("a1", "b1", "a2", "b2"), dict())

    def forward_shape(self, shape):
        return lax.broadcast_shapes(
            shape,
            getattr(self.a1, "a1", ()),
            getattr(self.b1, "b1", ()),
            getattr(self.a2, "a2", ()),
            getattr(self.b2, "b2", ()),
        )

    def inverse_shape(self, shape):
        return lax.broadcast_shapes(
            shape,
            getattr(self.a1, "a1", ()),
            getattr(self.b1, "b1", ()),
            getattr(self.a2, "a2", ()),
            getattr(self.b2, "b2", ()),
        )


class RightPositivePolynomial3(Transform):
    domain = constraints.real
    codomain = constraints.real

    def __init__(self, b):
        self.a = 1 / 3.0  # fix to value for smooth behavior at 0
        self.b = b

        self.threshold = 0
        self.f1 = IdentityTransform()
        self.f2 = PositivePolynomial3(self.a, self.b, domain=constraints.real)

    def __call__(self, x):
        comp_polynomial = jnp.where(x < self.threshold, self.f1(x), self.f2(x))
        return comp_polynomial

    def _inverse(self, y):
        return jnp.where(y < self.threshold, self.f1.inv(y), self.f2.inv(y))

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        log_jacobian = jnp.where(
            x < self.threshold,
            self.f1.log_abs_det_jacobian(x, y),
            self.f2.log_abs_det_jacobian(x, y),
        )
        return jnp.broadcast_to(log_jacobian, jnp.shape(x))

    def tree_flatten(self):
        return (self.b), (("b", dict()))

    def forward_shape(self, shape):
        return lax.broadcast_shapes(
            shape,
            getattr(self.b, "b", ()),
        )

    def inverse_shape(self, shape):
        return lax.broadcast_shapes(
            shape,
            getattr(self.b, "b", ()),
        )

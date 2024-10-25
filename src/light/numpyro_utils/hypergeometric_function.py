import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.special import gammaln

# ====================================
# ====================================

# https://www.grad.hr/nastava/gs/prg/NumericalRecipesinC.pdf

MAXITSICIINT = 200
TMIN = 2.0
FPMIN = 1e-30
EPS = 1e-6


def t_greater_TMIN_cond(val):
    b, c, d, h, i = val
    stop = c * d
    cond = jnp.fabs(stop.real - 1.0) + jnp.fabs(stop.imag) < EPS
    return jnp.where(i > MAXITSICIINT, False, jnp.where(cond, False, True))


def t_greater_TMIN_loop(val):
    b, c, d, h, i = val
    a = -(i - 1.0) * (i - 1.0)
    b = b + jax.lax.complex(2.0, 0.0)
    d = 1.0 / (a * d + b)
    c = b + jax.lax.complex(a, 0.0) / c
    h = h * c * d
    val = (b, c, d, h, i)
    i += 1
    return val


def t_greater_TMIN(t):
    b = jax.lax.complex(1.0, t)
    c = jax.lax.complex(1.0 / FPMIN, 0.0)
    d = h = 1.0 / b
    i = 2
    b, c, d, h, i = jax.lax.while_loop(
        t_greater_TMIN_cond, t_greater_TMIN_loop, (b, c, d, h, i)
    )
    h = jax.lax.complex(jnp.cos(t), -jnp.sin(t)) * h
    ci = -h.real
    si = jnp.pi / 2.0 + h.imag
    return jnp.array([si, ci])


def t_less_TMIN_cond(val):
    t, fact, sum, sums, sumc, sign, odd, k = val
    cond = fact / k / jnp.fabs(sum) < EPS
    return jnp.where(k > MAXITSICIINT, False, jnp.where(cond, False, True))


def t_less_TMIN_loop(val):
    t, fact, sum, sums, sumc, sign, odd, k = val
    fact *= t / k
    term = fact / k
    sum += sign * term
    sign = jnp.where(odd == 1, -sign, sign)
    sums = jnp.where(odd == 1, sum, sums)
    sumc = jnp.where(odd == -1, sum, sumc)
    sum = jnp.where(odd == 1, sumc, sums)
    odd = jnp.where(odd == 1, -1, 1)
    k += 1
    val = (t, fact, sum, sums, sumc, sign, odd, k)
    return val


def t_less_TMIN(t):
    sum = sums = sumc = 0.0
    sign = fact = 1.0
    odd = 1
    k = 1
    t, fact, sum, sums, sumc, sign, odd, k = jax.lax.while_loop(
        t_less_TMIN_cond, t_less_TMIN_loop, (t, fact, sum, sums, sumc, sign, odd, k)
    )
    sumc = jnp.where(t < jnp.sqrt(FPMIN), 0, sumc)
    sums = jnp.where(t < jnp.sqrt(FPMIN), t, sums)
    si = sums
    ci = sumc + jnp.log(t) + jnp.euler_gamma
    return jnp.array([si, ci])


def sici_(x):
    t = jnp.fabs(x)
    cond = t > TMIN
    si, ci = jax.lax.cond(cond, t_greater_TMIN, t_less_TMIN, t)
    si = jnp.where(x < 0.0, -si, si)
    return si, ci


sici = jit(jnp.vectorize(sici_))

# ====================================
# ====================================

from jax.experimental.ode import odeint

MAXITHYP2F1 = 3000


def z_less_1_loop(i, val):
    a, b, c, term, sum, z = val
    term *= (a + i - 1) * (b + i - 1) / (c + i - 1) / i * z
    sum += term
    val = (a, b, c, term, sum, z)
    return val


def z_less_1(a, b, c, z):
    sum = term = 1.0
    a, b, c, term, sum, z = jax.lax.fori_loop(
        1, MAXITHYP2F1, z_less_1_loop, (a, b, c, term, sum, z)
    )
    return sum


value_and_grad_z_less_1 = jax.value_and_grad(z_less_1, 3, holomorphic=True)


def hypdrv(F, s, dz, z0, a, b, c):
    z = z0 + s * dz
    dF2 = (a * b * F[0] - (c - (a + b + 1) * z) * F[1]) / (z * (1 - z))
    return F[1] * dz, dF2 * dz


def z_greater_1(a, b, c, z):
    z0 = jnp.where(z.real <= 1.0, 0.5 * jnp.sign(z.real), 1j * jnp.sign(z.imag) * 0.5)
    dz = z - z0
    y0 = value_and_grad_z_less_1(a, b, c, z0)
    yfin = odeint(
        hypdrv, y0, jnp.array([0.0, 1.0]), *(dz, z0, a, b, c), rtol=1e-6, atol=1e-5
    )[0]
    return yfin[-1].real


@jit
def hyp2f1_(a, b, c, z):
    args = (a, b, c, z)
    # sum = jax.lax.cond(jnp.abs(z) <= 0.5, z_less_1, z_greater_1, *args)
    # sum = jax.numpy.piecewise(
    #     jnp.abs(z),
    #     [jnp.abs(z) <= 0.5, jnp.abs(z) > 0.5],
    #     [lambda x: z_less_1(*args), lambda x: z_greater_1(*args)]
    # )
    sum = jnp.where(jnp.abs(z) <= 0.5, z_less_1(*args), z_greater_1(*args))
    return sum


hyp2f1 = jnp.vectorize(hyp2f1_)


# my implementation
def log_pochhammer(a, n):
    return gammaln(a + n) - gammaln(a)


def hypergeometric2_1_x_smaller_1(a, b, c, x, n_terms=200):
    # Determine new shape
    if isinstance(x, float):
        new_shape = (n_terms,)
    else:
        new_shape = (n_terms,) + (1,) * x.ndim

    # Create reshaped array of term indices
    nnn = jnp.reshape(jnp.arange(n_terms), new_shape)

    # Compute log numerator and denominator
    log_num = (
        log_pochhammer(a, nnn) + log_pochhammer(b, nnn) + jnp.log(jnp.abs(x)) * nnn
    )
    log_denom = log_pochhammer(c, nnn) + gammaln(nnn + 1)

    # Compute sign term
    sign_term = jnp.power(jnp.sign(x), nnn % 2)

    return jnp.sum(sign_term * jnp.exp(log_num - log_denom), axis=0)

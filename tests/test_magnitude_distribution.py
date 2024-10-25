import random

import jax
import jax.numpy as jnp
from light.astrophysics import luminosity

magnitude_model_name = "NF_sigmoid_regularized"
n_pre = 6
n_post = 0

magn_dist = luminosity.MagnitudeDistribution(magnitude_model_name, n_pre, n_post)

M_MAX = 200
M_MIN = -50


def test_magnitude_distribution_integration():
    params = {
        "eps": -1.0552065808399074,
        "eps_2": 0.1,
        "f_faint": 0.003121041689228923,
        "f_mu": -3,
        "mu": 85.00607436769789,
        "pre_a0": 0.08225775569452146,
        "pre_a1": -0.4095653816434679,
        "pre_a2": -0.1520475612540095,
        "pre_a3": -0.03571316994064743,
        "pre_a4": 0.5565621456219333,
        "pre_a5": -0.00036206529872759774,
        "pre_b0": 0.495659655504072,
        "pre_b1": 0.016470299632362566,
        "pre_b2": 0.4861196835981026,
        "pre_b3": 0.22749708703533075,
        "pre_b4": 0.5508711191655272,
        "pre_b5": 0.0028766271808116787,
        "sigma": 72.42886669532163,
        "H0": 67,
    }

    for key, value in params.items():
        if key not in ["eps", "eps_2"]:
            random_change = random.uniform(-0.01, 0.01)
        else:
            random_change = random.uniform(-0.1, 0.1)
        params[key] *= 1 + random_change

    numpyro_dist = magn_dist.construct_numpyro_magnitude_distribution(
        params, H0_REF=params["H0"]
    )

    M_vals = jnp.linspace(M_MIN, M_MAX, 100_000)

    log_prob = numpyro_dist.log_prob(M_vals)
    prob = jnp.exp(log_prob)
    prob = jnp.where(jnp.isnan(prob), 0, prob)

    test_integral = jax.scipy.integrate.trapezoid(prob, x=M_vals)
    assert jnp.allclose(
        test_integral, 1, atol=1e-5
    ), "Integral of probability density function must be close to 1."

    M_vals = jnp.linspace(params["f_mu"], M_MAX, 100_000)

    log_prob = numpyro_dist.log_prob(M_vals)
    prob = jnp.exp(log_prob)
    prob = jnp.where(jnp.isnan(prob), 0, prob)

    test_integral = jax.scipy.integrate.trapezoid(prob, x=M_vals)
    assert jnp.allclose(
        test_integral, params["f_faint"], atol=1e-5
    ), f"Integral of probability density function must be approximately equal to the parameter 'f_faint'. Expected: {params['f_faint']}, Found: {test_integral}"

import jax.numpy as jnp
import pytest

from light.field import field


@pytest.fixture
def real_field():
    power_spectrum_of_k = (
        lambda x: 2e-1
        * (x + 1e-5) ** (-0.8)
        * jnp.heaviside(x - 0.7, 0)
        * (1 + 0.9 * jnp.sin(x))
        * jnp.heaviside(45 - x, 0)
    )
    real_field = field.RealLogNormalField(
        box_size_d=[10, 10],
        box_shape_d=(150, 150),
        power_spectrum_of_k=power_spectrum_of_k,
    )
    return real_field
import jax.numpy as jnp
import numpy as np

from light.utils import jax_utils


def test_integrator():
    nZ = 4

    # compute a random x array
    x = 1 + 0.5 * np.random.random((2, 3, nZ))
    original_coords = jnp.linspace(0, 1, nZ + 1)

    integrator = jax_utils.Integrator(x, original_coords)

    weight = jnp.ones((nZ + 1,)) * 0.1
    weight = weight.at[0].set(0)
    weight = weight.at[-1].set(0)

    # pick some new coordinates that are random (and do not differ at the boundary)
    new_coordinates = original_coords + weight * np.random.random((nZ + 1,))
    new_coordinates_centers, _ = jax_utils.compute_centers_and_delta_from_array(
        new_coordinates
    )

    # now compute by hand the expected integral
    values_comp = np.zeros(x.shape)
    values_comp[:, :, 0] = (new_coordinates[1] - new_coordinates[0]) * x[:, :, 0]

    for i in range(0, len(new_coordinates_centers)):
        values_comp[:, :, i] = (
            integrator.original_coord_boundaries[i + 1]
            - integrator.original_coord_boundaries[i]
        ) * x[:, :, i]

        delta = new_coordinates[i + 1] - integrator.original_coord_boundaries[i + 1]
        if delta > 0:
            value = x[:, :, i + 1] if (i != (nZ - 1)) else 0
        else:
            value = x[:, :, i]

        values_comp[:, :, i] += delta * value

        delta = new_coordinates[i] - integrator.original_coord_boundaries[i]
        if delta > 0:
            value = x[:, :, i]
        else:
            value = x[:, :, i - 1] if (i != 0) else 0

        values_comp[:, :, i] -= delta * value

    values_comp[:, :, -1] = (
        integrator.original_coord_boundaries[-1] - new_coordinates[-1 - 1]
    ) * x[:, :, -1]

    values_comp = jnp.array(values_comp)

    assert jnp.allclose(
        values_comp, integrator.get_x_from_integral_approximation(new_coordinates)
    )


def test_centers_and_deltas():
    x_boundaries = jnp.array([0, 2, 3, 4, 10])
    x_centers_should = jnp.array([1, 2.5, 3.5, 7])
    deltas_should = jnp.array([2, 1, 1, 6])

    x_centers, deltas = jax_utils.compute_centers_and_delta_from_array(x_boundaries)

    assert jnp.allclose(x_centers_should, x_centers)
    assert jnp.allclose(deltas_should, deltas)

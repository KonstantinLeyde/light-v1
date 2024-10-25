import jax.numpy as jnp


def test_box_volume_change(real_field):
    print(real_field.box_volume)
    assert jnp.allclose(
        real_field.box_volume, 100
    ), "Box volume must be the product of the box lengths"

    real_field.box_size_d = [20, 20]
    print(real_field.box_volume)
    assert jnp.allclose(
        real_field.box_volume, 400
    ), "Box volume must be the product of the box lengths, after change. "

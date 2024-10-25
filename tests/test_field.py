import jax.numpy as jnp

def test_unit_fourier_vectors(real_field):
    uvec_FT = real_field.get_unit_fourier_vector_with_FT()
    uvec = real_field.get_unit_fourier_vector()
    assert jnp.allclose(uvec_FT, uvec)

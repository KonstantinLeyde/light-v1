import unittest

import jax
import jax.numpy as jnp

from light.field import field
from light.numpyro_utils import pack_fftn_values


class TestCorrelationCoefficients(unittest.TestCase):
    def setUp(self, n=6, batch=5_000, box_size_d=[3, 3, 3]):
        self.n = n
        self.batch = batch
        self.box_size_d = box_size_d

        self.m_array = jnp.zeros((self.batch, self.n, self.n, self.n))
        self.m_array_pack = jnp.zeros((self.batch, self.n, self.n, self.n))

        self.k_array = jnp.zeros(
            (self.batch, self.n, self.n, self.n // 2 + 1), dtype=complex
        )
        self.k_array_pack = jnp.zeros(
            (self.batch, self.n, self.n, self.n // 2 + 1), dtype=complex
        )

        # For measuring power spectrum need k grid, hence also field
        self.field_instance = field.RealField(
            box_size_d=self.box_size_d,
            box_shape_d=(self.n, self.n, self.n),
            power_spectrum_of_k=lambda x: x,
        )
        self.FT_kwargs = self.field_instance.FT_kwargs

        key = jax.random.PRNGKey(0)

        # Sample a (n, n, n) array from a standard Gaussian distribution
        sampled_array = jax.random.normal(key, (batch, self.n, self.n, self.n))

        self.k_array = jnp.fft.rfftn(sampled_array, **self.FT_kwargs)
        self.k_array_pack = pack_fftn_values.vpack_fft_values_3d(sampled_array)

        self.m_array = sampled_array
        self.m_array_pack = jnp.fft.irfftn(self.k_array_pack, **self.FT_kwargs)

        self.cor = jnp.corrcoef(
            self.k_array.reshape((self.batch, -1)),
            self.k_array.reshape((self.batch, -1)).conjugate(),
            rowvar=0,
        )
        self.cor_pack = jnp.corrcoef(
            self.k_array_pack.reshape((self.batch, -1)),
            self.k_array_pack.reshape((self.batch, -1)).conjugate(),
            rowvar=0,
        )

    def test_correlation_differences(self):
        real_diff = jnp.abs(self.cor.real - self.cor_pack.real)
        imag_diff = jnp.abs(self.cor.imag - self.cor_pack.imag)

        real_std = jnp.std(self.cor.real, axis=0)
        imag_std = jnp.std(self.cor.imag, axis=0)

        threshold_real = 6 * real_std
        threshold_imag = 6 * imag_std

        print(imag_std)

        # they have to agree only statistically
        self.assertTrue(
            jnp.all(real_diff < threshold_real), "Real part differences exceed 4 * std"
        )
        self.assertTrue(
            jnp.all(imag_diff < threshold_imag),
            "Imaginary part differences exceed 4 * std",
        )


if __name__ == "__main__":
    # Example of running the test with custom parameters
    ns = [3, 6, 8, 20]
    batch = 5_000
    box_size_ds = [[3, 3, 3], [100, 100, 100], [1, 1, 1], [0.1, 0.1, 0.1]]

    test_suite = unittest.TestSuite()
    test_case = TestCorrelationCoefficients("test_correlation_differences")

    for n, box_size_d in zip(ns, box_size_ds):
        test_case.setUp(n=n, batch=batch, box_size_d=box_size_d)
        test_suite.addTest(test_case)

    runner = unittest.TextTestRunner()
    runner.run(test_suite)

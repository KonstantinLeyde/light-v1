import jax
import jax.numpy as jnp

from ..numpyro_utils.hypergeometric_function import hyp2f1 as hyp2f1
from ..utils import jax_utils

C_KM_PER_SEC = 2.99792e5
Z_MAX = 5


def compute_abs_magnitude_from_apparent_magnitude(m, luminosity_distance):
    """
    Computes the apparent magnitude from an absolute magnitude for a
    luminosity_distance in Mpc.

    Parameters
    ----------
    absolute_magnitude : float
        The absolute magnitude of the astronomical object.

    luminosity_distance : float
        The luminosity distance to the object in megaparsecs (Mpc).

    Returns
    -------
    float
        The calculated apparent magnitude.

    https://en.wikipedia.org/wiki/Luminosity_distance

    """

    return m - 5 * jnp.log10(luminosity_distance) - 25


def apparent_magnitude_from_abs_magnitude(M, luminosity_distance):
    """
    Computes the apparent magnitude from an absolute magnitude for a
    luminosity_distance in Mpc.

    Parameters
    ----------
    absolute_magnitude : float
        The absolute magnitude of the astronomical object.

    luminosity_distance : float
        The luminosity distance to the object in megaparsecs (Mpc).

    Returns
    -------
    float
        The calculated apparent magnitude.

    https://en.wikipedia.org/wiki/Luminosity_distance

    """

    return M + 5 * jnp.log10(luminosity_distance) + 25


class Cosmology:
    def __init__(self, params, numerics=dict(z_min=1e-7, z_max=Z_MAX, z_steps=200_000)):
        self._H0 = params["H0"]
        self._Omega_m = params["Omega_m"]

        self.numerics = numerics
        self.initialize_comoving_distance()

    def set_cosmological_parameters(self, params):
        self._H0 = params["H0"]
        self._Omega_m = params["Omega_m"]

        self.initialize_comoving_distance()

    def get_luminosity_distance_from_z(self, z):
        luminosity_distance = jnp.interp(
            z, self.z_interp, self.luminosity_distance_interp, left=0, right=None
        )

        return luminosity_distance

    def get_angular_diameter_distance_from_z(self, z):
        angular_diameter_distance = jnp.interp(
            z, self.z_interp, self.angular_diameter_distance_interp, left=0, right=None
        )

        return angular_diameter_distance

    def get_comoving_distance_from_z(self, z):
        """
        Compute the comoving distance in Mpc.

        """

        comoving_distance = jnp.interp(
            z, self.z_interp, self.comoving_distance_interp, left=0, right=None
        )

        return comoving_distance

    def get_z_from_comoving_distance(self, comoving_distance):
        """
        Compute the redshift from a comoving distance in Mpc.

        """

        z = jnp.interp(comoving_distance, self.comoving_distance_interp, self.z_interp)

        return z

    def get_z_from_luminosity_distance(self, luminosity_distance):
        """
        Compute the redshift from a luminosity distance in Mpc.

        """

        z = jnp.interp(
            luminosity_distance, self.luminosity_distance_interp, self.z_interp
        )

        return z

    def get_absolute_magnitude_from_redshift(self, apparent_magnitude, redshift):
        luminosity_distance = self.get_luminosity_distance_from_z(z=redshift)

        absolute_magnitudes = compute_abs_magnitude_from_apparent_magnitude(
            apparent_magnitude, luminosity_distance
        )

        return absolute_magnitudes

    def get_comoving_volume_differential(self, z):
        comoving_slice = 4 * jnp.pi * self.get_comoving_distance_from_z(z=z) ** 2

        return comoving_slice * self.get_dcomoving_distance_over_dz_from_z(z=z)

    def get_dcomoving_distance_over_dz_from_z(self, z):
        return self.E_function(z=z) / self._H0

    def get_dcomoving_distance_over_dz_from_comoving_distance(self, comoving_distance):
        z = self.get_z_from_comoving_distance(comoving_distance)

        return self.get_dcomoving_distance_over_dz_from_z(z=z)

    def get_dluminosity_distance_over_dz_from_z(self, z):
        luminosity_distance = self.get_luminosity_distance_from_z(z=z)

        term_1 = 1 / (1 + z) * luminosity_distance
        term_2 = (1 + z) * self.get_dcomoving_distance_over_dz_from_z(z=z)

        return term_1 + term_2

    # TODO: test this function
    def get_dangular_diameter_distance_over_dz_from_z(self, z):
        angular_diameter_distance = self.get_angular_diameter_distance_from_z(z=z)

        term_1 = -(1 + z) * angular_diameter_distance
        term_2 = 1 / (1 + z) * self.get_dcomoving_distance_over_dz_from_z(z=z)

        raise NotImplementedError

        return term_1 + term_2

    def get_log_prob_uniform_in_comoving_distance_1d(self, z, normalized=True):
        # not tested yet

        prob = self.get_dcomoving_distance_over_dz_from_z(z=z)

        if normalized:
            integrand = self.get_dcomoving_distance_over_dz_from_z(
                z=self.z_interp_boundaries
            )
            normalization = jax.scipy.integrate.trapezoid(
                integrand, x=self.z_interp_boundaries
            )
        else:
            normalization = 1

        return jnp.log(prob) - jnp.log(normalization)

    def initialize_comoving_distance(self):
        self.z_interp_boundaries = jnp.logspace(
            jnp.log10(self.numerics["z_min"]),
            jnp.log10(self.numerics["z_max"]),
            self.numerics["z_steps"],
        )
        self.z_interp, self.delta_z = jax_utils.compute_centers_and_delta_from_array(
            self.z_interp_boundaries
        )

        # Integral for comoving distance
        integrand = self.E_function(z=self.z_interp)
        integral_cum = jnp.cumsum(integrand * self.delta_z)

        # precompute the distance measures that are then used for interpolation
        self.comoving_distance_interp = integral_cum / self._H0
        self.luminosity_distance_interp = integral_cum * (1 + self.z_interp) / self._H0
        self.angular_diameter_distance_interp = (
            integral_cum / (1 + self.z_interp) / self._H0
        )

    def E_function(self, z):
        Omega_Lambda = 1.0 - self._Omega_m
        return C_KM_PER_SEC / jnp.sqrt(self._Omega_m * (1 + z) ** 3 + Omega_Lambda)


class CosmologyAnalytic:
    def __init__(self, params, comoving_distance_function=None):
        self._H0 = params["H0"]
        self._Omega_m = params["Omega_m"]
        self.set_Omega_L()

        self.comoving_distance_function = comoving_distance_function

    def set_cosmological_parameters(self, params):
        self._H0 = params["H0"]
        self._Omega_m = params["Omega_m"]
        self.set_Omega_L()

    def set_Omega_L(self):
        self._Omega_Lambda = 1.0 - self._Omega_m

    def get_luminosity_distance_from_z(self, z):
        luminosity_distance = (1 + z) * self.get_comoving_distance_from_z(z=z)

        return luminosity_distance

    def get_angular_diameter_distance_from_z(self, z):
        angular_diameter_distance = 1 / (1 + z) * self.get_comoving_distance_from_z(z=z)

        return angular_diameter_distance

    def get_comoving_distance_from_z(self, z):
        return self.comoving_distance_function(z, self._H0, self._Omega_m)

    def get_absolute_magnitude_from_redshift(self, apparent_magnitude, redshift):
        luminosity_distance = self.get_luminosity_distance_from_z(z=redshift)

        absolute_magnitudes = compute_abs_magnitude_from_apparent_magnitude(
            apparent_magnitude, luminosity_distance
        )

        return absolute_magnitudes

    def get_comoving_volume_differential(self, z):
        comoving_slice = 4 * jnp.pi * self.get_comoving_distance_from_z(z=z) ** 2

        return comoving_slice * self.get_dcomoving_distance_over_dz_from_z(z=z)

    def get_dcomoving_distance_over_dz_from_z(self, z):
        return self.E_function(z=z) / self._H0

    def get_dluminosity_distance_over_dz_from_z(self, z):
        luminosity_distance = self.get_luminosity_distance_from_z(z=z)

        term_1 = 1 / (1 + z) * luminosity_distance
        term_2 = (1 + z) * self.get_dcomoving_distance_over_dz_from_z(z=z)

        return term_1 + term_2

    # TODO: test this function
    def get_dangular_diameter_distance_over_dz_from_z(self, z):
        angular_diameter_distance = self.get_angular_diameter_distance_from_z(z=z)

        term_1 = -(1 + z) * angular_diameter_distance
        term_2 = 1 / (1 + z) * self.get_dcomoving_distance_over_dz_from_z(z=z)

        raise NotImplementedError

        return term_1 + term_2

    def E_function(self, z):
        return C_KM_PER_SEC / jnp.sqrt(
            self._Omega_m * (1 + z) ** 3 + self._Omega_Lambda
        )


class CosmologyHyper(CosmologyAnalytic):
    def __init__(self, params):
        super().__init__(
            params,
            comoving_distance_function=comoving_distance_from_hypergeometric_function,
        )

def comoving_distance_from_hypergeometric_function(z, H0, Omega_m):

    Omega_Lambda = 1 - Omega_m
    
    term1 = - hyp2f1(1/3, 1/2, 4/3, -(Omega_m/Omega_Lambda))
    term2 = (1 + z) * hyp2f1(1/3, 1/2, 4/3, -((Omega_m * (1 + z)**3)/Omega_Lambda))

    term_final = term1 + term2

    return C_KM_PER_SEC * term_final / H0 / jnp.sqrt(Omega_Lambda)
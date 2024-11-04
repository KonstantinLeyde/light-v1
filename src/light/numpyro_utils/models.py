import copy

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import AffineTransform

from ..cosmology.cosmology import Cosmology
from ..field.field import (
    RealLogNormalField,
    bias_function,
    bias_function_exp,
)
from ..numpyro_utils.sampling_utils import get_prior_dict_samples
from ..utils.conventions import complete_cosmopower_dict, get_Omega_m
from ..utils.jax_utils import (
    compute_centers_and_delta_from_array,

)

from . import custom_distributions as custom_distributions
from . import custom_transformations as custom_transformations
from . import priors as priors
from . import sampling_utils as sampling_utils

hist_kwargs = {"bins": "auto", "histtype": "step", "density": True}
params_power_spectrum_list = [
    "omega_b",
    "omega_cdm",
    "h",
    "n_s",
    "ln10^{10}A_s",
    "cmin",
    "eta_0",
    "z",
]
FIELD_LEVEL_NAME = "gaussian_F_whitened_spatial"


# Define the model
def model(
    counts_galaxies_observed,
    analysis,
    prior_dict={},
    sample_number_counts=True,
    infer_density_contrast_DM=False,
    infer_density_g=False,
    z_boundaries=None,
    apparent_magnitude_threshold_box2d=None,
    m_list=None,
    z_list=None,
    prior_settings={},
):
    angle_perpendicular_0 = analysis.kwargs_catalog["angle_box_0"]
    angle_perpendicular_1 = analysis.kwargs_catalog["angle_box_1"]
    RightTruncatedPoissonHigh = prior_settings["RightTruncatedPoissonHigh"]

    if "replace_FT_with_packing" in analysis.kwargs_field.keys():
        replace_FT_with_packing = analysis.kwargs_field["replace_FT_with_packing"]
    else:
        replace_FT_with_packing = False

    if analysis.kwargs_field["galaxy_model"].startswith("simple-bias-gamma"):
        bias_func = bias_function
    elif analysis.kwargs_field["galaxy_model"].startswith("simple-bias-exp-gamma"):
        bias_func = bias_function_exp
    else:
        raise NotImplementedError

    params = {}
    for key in prior_dict.keys():
        params[key] = {}
        for param in prior_dict[key].keys():
            params[key][param] = get_prior_dict_samples(
                param, prior_dict[key][param]
            )

    if analysis.kwargs_field["power_spectrum_model"] in [
        "analytical-phenom",
        "flat",
        "analytical-phenom-all",
    ]:
        params["power_spectrum"] = dict(**params["power_spectrum"], cut_off=1e-6)
    elif analysis.kwargs_field["power_spectrum_model"] in ["smooth-turnover"]:
        pass
    elif analysis.kwargs_field["power_spectrum_model"] == "cosmopower":
        params_power_spectrum = complete_cosmopower_dict(
            params_power_spectrum=params["power_spectrum"],
            params_cosmology=params["cosmology"],
            z=0,
        )
        params_power_spectrum_array = jnp.array(
            [params_power_spectrum[k] for k in params_power_spectrum_list]
        ).T
    else:
        raise "Power spectrum model not known. "

    # cosmological inference
    Omega_m = get_Omega_m(params_cosmology=params["cosmology"])
    Omega_m = numpyro.deterministic("Omega_m", Omega_m)
    params_cosmo = dict(H0=params["cosmology"]["H0"], Omega_m=Omega_m)

    if "cosmo_numerics" in analysis.kwargs_cosmology.keys():
        # print('Using cosmology with numerical interpolation. ')
        if analysis.kwargs_cosmology["cosmo_numerics"]["z_max"] == None:
            analysis.kwargs_cosmology["cosmo_numerics"]["z_max"] = (
                z_boundaries[-1] + 0.1
            )

        cosmological_model = Cosmology(
            params=params_cosmo, numerics=analysis.kwargs_cosmology["cosmo_numerics"]
        )
    else:
        raise NotImplementedError

    if analysis.kwargs_catalog["pixelation"] == "redshift":
        comoving_distance_boundaries = cosmological_model.get_comoving_distance_from_z(
            z_boundaries
        )
        box_size_comoving_Z_direction = (
            comoving_distance_boundaries[-1] - comoving_distance_boundaries[0]
        )

        # compute the physical size of the box
        angular_diameter_distance = (
            cosmological_model.get_angular_diameter_distance_from_z(z_boundaries[0])
        )
        box_size_perpendicular_0 = angular_diameter_distance * angle_perpendicular_0
        box_size_perpendicular_1 = angular_diameter_distance * angle_perpendicular_1

        new_box_size_d = jnp.array(
            [
                box_size_perpendicular_0,
                box_size_perpendicular_1,
                box_size_comoving_Z_direction,
            ]
        )

        box_sub_shape_d = analysis.box_shape_d[:2] + [
            analysis.box_shape_d[2] * prior_settings["number_fine_redshift"]
        ]

    elif analysis.kwargs_catalog["pixelation"] == "comoving_distance":
        new_box_size_d = analysis.box_size_d
        box_sub_shape_d = analysis.box_shape_d[:2] + [
            analysis.box_shape_d[2] * prior_settings["number_fine_redshift"]
        ]
    else:
        raise NotImplementedError

    # initiate new field class
    field_instance = RealLogNormalField(
        box_size_d=new_box_size_d,
        box_shape_d=analysis.box_shape_d,
        power_spectrum_of_k=analysis.power_spectrum_of_k,
        set_zero_mode_to_zero=analysis.kwargs_field["set_zero_mode_to_zero"],
        replace_FT_with_packing=replace_FT_with_packing,
    )

    if analysis.kwargs_catalog["pixelation"] == "redshift":
        z_line_sub = jnp.linspace(
            z_boundaries[0], z_boundaries[-1], box_sub_shape_d[2] + 1
        )
        z_centers_sub, _ = compute_centers_and_delta_from_array(z_line_sub)

        comoving_distance_boundaries_sub = (
            cosmological_model.get_comoving_distance_from_z(z_line_sub)
        )
        _, deltas_comoving_distance_boundaries_long_sub = (
            compute_centers_and_delta_from_array(
                comoving_distance_boundaries_sub
            )
        )

        new_shape = (
            (
                1,
                1,
            )
            + (analysis.box_shape_d[2],)
            + (prior_settings["number_fine_redshift"],)
        )

        z_box_d_sub = z_centers_sub.reshape(new_shape)
        deltas_comoving_distance_boundaries_sub = (
            deltas_comoving_distance_boundaries_long_sub.reshape(new_shape)
        )

    M_thresholdi_box_sub = cosmological_model.get_absolute_magnitude_from_redshift(
        apparent_magnitude=apparent_magnitude_threshold_box2d[..., None, None],
        redshift=z_box_d_sub,
    )

    norm_magnitudes = -5 * jnp.log10(
        params["cosmology"]["H0"]
        / analysis.kwargs_cosmology["cosmo_ref_parameters"]["H0"]
    )
    transformation_M_hat_to_magnitude = AffineTransform(-norm_magnitudes, 1)

    # could be also moved outside of the model
    magnitude_model = copy.deepcopy(analysis.magnitude_model)

    magnitude_model.set_transformation_M_hat_to_magnitude(
        transformation_M_hat_to_magnitude
    )
    magnitude_model.compute_transformation_M_hat_to_latent_M(
        params["magnitude_distribution"]
    )

    # define the overall magnitude distribution
    dist_magnitudes = dist.TransformedDistribution(
        dist.Uniform(0, 1, validate_args=True),
        magnitude_model.transformation_latent_M_to_M_hat,
    )

    plate_x3 = numpyro.plate(
        FIELD_LEVEL_NAME + "2a", size=analysis.box_shape_d[2], dim=-1
    )
    plate_x2 = numpyro.plate(
        FIELD_LEVEL_NAME + "1a", size=analysis.box_shape_d[1], dim=-2
    )
    plate_x1 = numpyro.plate(
        FIELD_LEVEL_NAME + "0a", size=analysis.box_shape_d[0], dim=-3
    )

    with plate_x3:
        with plate_x2:
            with plate_x1:
                gaussian_F_whitened_spatial = numpyro.sample(
                    FIELD_LEVEL_NAME,
                    dist.Normal(0, 1),
                )

                field_instance.set_gaussian_F_whitened_from_gaussian_F_whitened_spatial(
                    gaussian_F_whitened_spatial
                )

                if analysis.kwargs_field["power_spectrum_model"] in [
                    "analytical-phenom",
                    "flat",
                    "smooth-turnover",
                    "analytical-phenom-all",
                ]:
                    field_instance.compute_gaussian_F_fourier(params["power_spectrum"])
                elif analysis.kwargs_field["power_spectrum_model"] == "cosmopower":
                    field_instance.compute_gaussian_F_fourier(
                        cosmo_params=params_power_spectrum_array
                    )

                field_instance.compute_gaussian_F_spatial()
                field_instance.compute_density_contrast_DM()
                field_instance.compute_density_g(
                    bias_kwargs=params["galaxy_bias"], bias_function=bias_func
                )

    # get the latent variable for the magnitude distribution
    latent_M_box_sub = magnitude_model.get_latent_M_from_magnitude(M_thresholdi_box_sub)

    # compute the prior of a galaxy being at a redshift
    prior_redshift_box_sub = cosmological_model.get_dcomoving_distance_over_dz_from_z(
        z_box_d_sub
    )
    # this is nothing than the delta comoving elements, but for numerical reasons, we compute it
    prior_redshift_norm_box_sub = jnp.mean(prior_redshift_box_sub, axis=-1)[..., None]

    prior_redshift_box_sub /= prior_redshift_norm_box_sub

    mu_galaxies_box_sub = (
        prior_redshift_box_sub
        * field_instance.density_g[..., None]
        * deltas_comoving_distance_boundaries_sub
    )

    # since the cdf of the uniform the detection probability is simply x itself
    mudet_box_sub = latent_M_box_sub * mu_galaxies_box_sub

    mu_galaxies_box = jnp.sum(mu_galaxies_box_sub, axis=-1)
    pdet_box = jnp.sum(mudet_box_sub, axis=-1) / mu_galaxies_box

    # add these for post-computing, but these are not saved during inference
    if infer_density_contrast_DM:
        numpyro.deterministic("density_contrast_DM", field_instance.density_contrast_DM)
    if infer_density_g:
        numpyro.deterministic("density_g", field_instance.density_g)

    if analysis.kwargs_catalog["pixelation"] == "redshift":
        if analysis.kwargs_field["integration_method"] == "stretch":
            pixel_volume_2D = (
                field_instance.box_size_d[0]
                / field_instance.box_shape_d[0]
                * field_instance.box_size_d[1]
                / field_instance.box_shape_d[1]
            )

            number_g_per_pixel = mu_galaxies_box * pixel_volume_2D
        elif analysis.kwargs_field["integration_method"] == "constant":
            number_g_per_pixel = mu_galaxies_box * field_instance.pixel_volume
        else:
            raise NotImplementedError

    elif analysis.kwargs_catalog["pixelation"] == "comoving_distance":
        number_g_per_pixel = field_instance.density_g * field_instance.pixel_volume

    if sample_number_counts:
        with plate_x3:
            with plate_x2:
                with plate_x1:
                    counts_galaxies_true = numpyro.sample(
                        "counts_galaxies_true",
                        custom_distributions.RightTruncatedPoisson(
                            rate=number_g_per_pixel, high=RightTruncatedPoissonHigh
                        ),
                        infer={"enumerate": "parallel"},
                    )

                    numpyro.sample(
                        "counts_galaxies_observed",
                        dist.Binomial(total_count=counts_galaxies_true, probs=pdet_box),
                        obs=counts_galaxies_observed,
                    )

                    # correct for selection effect (compensate binomial likelihood)
                    log_factor_normalization_binomial_term = (
                        -jnp.log(pdet_box) * counts_galaxies_observed
                    )
                    numpyro.factor(
                        "log_factor_hierarchical_likelihood",
                        log_factor_normalization_binomial_term,
                    )

    else:
        with plate_x3:
            with plate_x2:
                with plate_x1:
                    rate_obs = number_g_per_pixel * pdet_box

                    log_factor_hierarchical_likelihood = (
                        -rate_obs
                        + jnp.log(field_instance.density_g * pixel_volume_2D)
                        * counts_galaxies_observed
                    )
                    numpyro.factor(
                        "log_factor_hierarchical_likelihood",
                        log_factor_hierarchical_likelihood,
                    )

    with numpyro.plate("obs_plate", len(z_list)):
        M_obs_list = cosmological_model.get_absolute_magnitude_from_redshift(
            apparent_magnitude=m_list, redshift=z_list
        )

        M_hat_obs_list = transformation_M_hat_to_magnitude.inv(M_obs_list)
        numpyro.sample("magnitudes_obs", dist_magnitudes, obs=M_hat_obs_list)

        # compute Jacobian
        log_factor_hierarchical_likelihood_redshift = jnp.log(
            cosmological_model.get_dcomoving_distance_over_dz_from_z(z_list)
        )

        numpyro.factor(
            "log_factor_hierarchical_likelihood_redshift",
            log_factor_hierarchical_likelihood_redshift,
        )

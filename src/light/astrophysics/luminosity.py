import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import (
    AffineTransform,
    SigmoidTransform,
    SoftplusTransform,
)

from ..numpyro_utils import custom_transformations as custom_transformations
from ..utils import conventions as conventions


def get_mu_faintest(analysis):
    if "mu_faintest_value" in analysis.kwargs_sampler["priors"].keys():
        mu_faintest = analysis.kwargs_sampler["priors"]["mu_faintest_value"]
    else:
        mu_faintest = analysis.kwargs_catalog["additional_information"][
            "faintest_absolute_magnitude"
        ]

    return mu_faintest


class MagnitudeDistribution:
    def __init__(self, magnitude_model_name, n_pre, n_post):
        self.magnitude_model_name = magnitude_model_name
        self.n_pre = n_pre
        self.n_post = n_post

        self.n_pre_params_names = [
            ("pre_a" + str(i), ("pre_b" + str(i))) for i in range(self.n_pre)
        ]
        self.n_post_params_names = [
            ("post_a" + str(i), ("post_b" + str(i))) for i in range(self.n_post)
        ]

        self.list_all_params = ["mu", "sigma", "eps", "eps_2"]
        self.list_all_params += [
            item for sublist in self.n_pre_params_names for item in sublist
        ]
        self.list_all_params += [
            item for sublist in self.n_post_params_names for item in sublist
        ]

        if self.magnitude_model_name.endswith("_faint"):
            self.list_all_params += ["faint_mu", "faint_b1", "faint_b2"]

        if self.magnitude_model_name.endswith("_regularized"):
            self.list_all_params += ["f_faint", "f_mu"]

    def set_transformation_M_hat_to_magnitude(self, transformation_M_hat_to_magnitude):
        self.transformation_M_hat_to_magnitude = transformation_M_hat_to_magnitude
        self.transformation_magnitude_to_M_hat = transformation_M_hat_to_magnitude.inv

    def compute_transformation_M_hat_to_latent_M(self, params_M):
        self.transformation_latent_M_to_M_hat = construct_latent_M_to_M_hat_transform(
            self,
            params_M,
        )

        self.transformation_M_hat_to_latent_M = (
            custom_transformations.construct_inverse_transform(
                self.transformation_latent_M_to_M_hat
            )
        )

        self.transformation_magnitude_to_latent_M = [
            self.transformation_magnitude_to_M_hat
        ] + self.transformation_M_hat_to_latent_M

    def get_latent_M_from_magnitude(self, magnitude):
        return custom_transformations.apply_transform_list(
            magnitude, self.transformation_magnitude_to_latent_M
        )

    def __str__(self):
        attributes = {
            "n_pre": self.n_pre,
            "n_post": self.n_post,
        }
        summary_params = "_".join(
            [f"{key}_{value}" for key, value in attributes.items()]
        )
        return self.magnitude_model_name + "_" + summary_params

    def construct_numpyro_magnitude_distribution(self, params, H0_REF):
        norm_magnitudes = -5 * jnp.log10(params["H0"] / H0_REF)
        transformation_M_hat_to_magnitude = AffineTransform(-norm_magnitudes, 1)

        self.set_transformation_M_hat_to_magnitude(transformation_M_hat_to_magnitude)
        self.compute_transformation_M_hat_to_latent_M(params)

        dist_magnitudes = dist.TransformedDistribution(
            dist.Uniform(0, 1), self.transformation_latent_M_to_M_hat
        )

        return dist_magnitudes


def construct_latent_M_to_M_hat_transform(magnitude_model, kwargs):
    transformations = []

    n_iteration = 2
    for i in range(magnitude_model.n_pre):
        args = kwargs["pre_a" + str(i)], kwargs["pre_b" + str(i)]
        if i % n_iteration == 0:
            trans = custom_transformations.PositivePolynomial3(*args).inv
        elif i % n_iteration == 1:
            trans = custom_transformations.PositivePolynomial3(*args)
        else:
            pass

        transformations += [trans]

    transformations += [SigmoidTransform().inv]
    transformations += [AffineTransform(kwargs["eps"], kwargs["eps_2"])]
    transformations += [SoftplusTransform()]

    for i in range(magnitude_model.n_post):
        args = kwargs["post_a" + str(i)], kwargs["post_b" + str(i)]
        if i % n_iteration == 0:
            trans = custom_transformations.PositivePolynomial3(
                *args, domain=constraints.real
            )
        elif i % n_iteration == 1:
            trans = custom_transformations.PositivePolynomial3(
                *args, domain=constraints.real
            )
        else:
            pass

        transformations += [trans]

    mu, sigma = kwargs["mu"], kwargs["sigma"]
    transformations += [AffineTransform(mu, sigma)]

    latent_temp_fraction = custom_transformations.apply_transform_list(
        1 - kwargs["f_faint"], transformations
    )
    transformations += [AffineTransform(kwargs["f_mu"] - latent_temp_fraction, 1)]

    if magnitude_model.magnitude_model_name.endswith("_faint"):
        transformations += [AffineTransform(kwargs["f_mu"], 1)]
        transformations += [
            custom_transformations.RightPositivePolynomial3(kwargs["faint_b1"]).inv
        ]
        transformations += [
            custom_transformations.RightPositivePolynomial3(kwargs["faint_b2"]).inv
        ]

    return transformations

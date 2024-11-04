import copy

import jax.numpy as jnp

from ..astrophysics import luminosity
from ..utils import conventions


def get_prior_dict(analysis):
    magnitude_model = analysis.magnitude_model
    cosmology_model = analysis.kwargs_cosmology["cosmology_model"]
    power_spectrum_model = analysis.kwargs_field["power_spectrum_model"]
    galaxy_model = analysis.kwargs_field["galaxy_model"]

    prior_params_power_spectrum = {}
    prior_params_galaxy = {}
    prior_params_magnitudes = {}
    prior_params_cosmology = {}

    prior_params_galaxy["ng_bar"] = dict(
        min=analysis.kwargs_sampler["priors"]["ng_bar_min"],
        max=analysis.kwargs_sampler["priors"]["ng_bar_max"],
        dist_type="Uniform",
    )

    prior_settings = {}
    prior_settings["RightTruncatedPoissonHigh"] = (
        conventions.get_RightTruncatedPoissonHigh_from_n(analysis)
    )
    prior_settings["number_fine_redshift"] = 10

    if galaxy_model in ["simple-bias-gamma", "simple-bias-exp-gamma"]:
        for k in ["gamma", "epsilon_g", "field_cut_exp"]:
            prior_params_galaxy[k] = get_prior_from_analysis_and_param(
                priors=analysis.kwargs_sampler["priors"], param=k
            )

    else:
        raise NotImplementedError

    if magnitude_model.magnitude_model_name != None:
        if magnitude_model.magnitude_model_name.startswith("NF_sigmoid"):
            params_init = analysis.kwargs_sampler["magnitude_model_params_init"]
            error_param = 0.14
            params_init_err = jnp.array(
                [
                    4,
                    30,
                    0.7,
                    0.04,
                    error_param,
                    error_param,
                    error_param,
                    error_param,
                    error_param,
                    error_param,
                    error_param,
                    error_param,
                    error_param,
                    error_param,
                    error_param,
                    error_param,
                ]
            )
            
            for i, k in enumerate(["mu", "sigma", "eps", "eps_2"]):
                prior_params_magnitudes[k] = dict(
                    min=params_init[k] - params_init_err[i],
                    max=params_init[k] + params_init_err[i],
                )

            n_previous = len(prior_params_magnitudes)
            for i in range(2 * magnitude_model.n_pre):
                if i % 2 == 0:
                    k = "pre_a" + str(i // 2)
                else:
                    k = "pre_b" + str(i // 2)

                idx = i + n_previous
                try:
                    dict_prior = dict(
                        min=params_init[k] - params_init_err[idx],
                        max=params_init[k] + params_init_err[idx],
                    )
                except:
                    dict_prior = dict(min=1e-5, max=0.2)

                if dict_prior["min"] <= 0 and (i % 2 == 1):
                    dict_prior["min"] = 1e-6

                prior_params_magnitudes[k] = dict_prior

            n_previous = len(prior_params_magnitudes)
            for i in range(2 * magnitude_model.n_post):
                idx = i + n_previous
                dict_prior = dict(
                    min=params_init[k] - params_init_err[idx],
                    max=params_init[k] + params_init_err[idx],
                )
                if i % 2 == 0:
                    prior_params_magnitudes["post_a" + str(i // 2)] = dict_prior
                else:
                    prior_params_magnitudes["post_b" + str(i // 2)] = dict_prior

            if magnitude_model.magnitude_model_name.endswith("_faint"):
                if analysis.catalog_name in [
                    "millenium_bertone2007a_0.01.csv",
                    "millenium_bertone2007a_0.05.csv",
                ]:
                    prior_params_magnitudes["faint_mu"] = dict_prior = dict(
                        min=0, max=15
                    )
                    prior_params_magnitudes["faint_b1"] = dict_prior = dict(
                        min=1, max=10, dist_type="LogUniform"
                    )
                    prior_params_magnitudes["faint_b2"] = dict_prior = dict(
                        min=1, max=10, dist_type="LogUniform"
                    )

                elif analysis.catalog_name == "millenium_bertone2007a_0.001.csv":
                    prior_params_magnitudes["faint_mu"] = dict_prior = dict(
                        min=0, max=15
                    )
                    prior_params_magnitudes["faint_b1"] = dict_prior = dict(
                        min=1, max=10, dist_type="LogUniform"
                    )
                    prior_params_magnitudes["faint_b2"] = dict_prior = dict(
                        min=1, max=10, dist_type="LogUniform"
                    )
                else:
                    raise NotImplementedError

            if magnitude_model.magnitude_model_name.endswith("_regularized"):
                f = analysis.kwargs_catalog["additional_information"][
                    "fraction_never_observed"
                ]

                if "f_faint_max" in analysis.kwargs_sampler["priors"].keys():
                    f_max = analysis.kwargs_sampler["priors"]["f_faint_max"]
                else:
                    f_max = min(0.9, max(2 * f, 0.1))

                if "f_faint_min" in analysis.kwargs_sampler["priors"].keys():
                    f_min = analysis.kwargs_sampler["priors"]["f_faint_min"]
                else:
                    f_min = f / 2

                mu_faintest = luminosity.get_mu_faintest(analysis)

                prior_params_magnitudes["f_mu"] = dict(
                    value=mu_faintest, dist_type="Delta"
                )
                prior_params_magnitudes["f_faint"] = dict(
                    min=f_min, max=f_max
                )  # , dist_type='LogUniform')

    else:
        prior_params_magnitudes = {}

    if cosmology_model == "flatLCDM":
        prior_params_cosmology["H0"] = dict(min=64, max=84)
        prior_params_cosmology["omega_b"] = dict(min=0.0185, max=0.026)
        prior_params_cosmology["omega_cdm"] = dict(min=0.05, max=0.25)
    elif cosmology_model == None:
        H0, _ = conventions.get_H0_OMEGA_M_REF()
        omega_b, omega_cdm = conventions.get_omega_b_REF_omega_cdm_REF()

        prior_params_cosmology["H0"] = dict(value=H0, dist_type="Delta")
        prior_params_cosmology["omega_b"] = dict(value=omega_b, dist_type="Delta")
        prior_params_cosmology["omega_cdm"] = dict(value=omega_cdm, dist_type="Delta")
    else:
        raise NotImplementedError

    if power_spectrum_model == "cosmopower":
        prior_params_power_spectrum["cmin"] = dict(min=2 + 0.1, max=4 - 0.1)
        prior_params_power_spectrum["n_s"] = dict(min=0.82 + 0.01, max=1.1 - 0.01)
        prior_params_power_spectrum["ln10^{10}A_s"] = dict(
            min=1.61, max=3.9, dist_type="Uniform"
        )
        prior_params_power_spectrum["eta_0"] = dict(min=0.5 + 0.05, max=1 - 0.05)

    elif power_spectrum_model in ["analytical-phenom", "flat"]:
        # TODO, adapt here to different gamma values
        if galaxy_model.startswith("simple-bias"):
            if power_spectrum_model == "analytical-phenom":
                for k in ["A_s", "n_s"]:
                    min_val = analysis.kwargs_sampler["priors"][f"{k}_min"]
                    max_val = analysis.kwargs_sampler["priors"][f"{k}_max"]

                    prior_params_power_spectrum[k] = dict(min=min_val, max=max_val)

            elif power_spectrum_model == "flat":
                prior_params_power_spectrum["n_s"] = dict(value=1.0, dist_type="Delta")
                prior_params_power_spectrum["alpha_s"] = dict(
                    value=0, dist_type="Delta"
                )

        else:
            raise "Prior not defined"
    elif power_spectrum_model in ["analytical-phenom-all"]:
        min_val = analysis.kwargs_sampler["priors"]["A_s_min"]
        max_val = analysis.kwargs_sampler["priors"]["A_s_max"]

        prior_params_power_spectrum["A_s"] = dict(min=min_val, max=max_val)
        prior_params_power_spectrum["n_s"] = dict(min=-0.8, max=0.3)
        prior_params_power_spectrum["alpha_s"] = dict(min=-2, max=1)
        prior_params_power_spectrum["k0"] = dict(min=0.005, max=0.14)

    elif power_spectrum_model in ["smooth-turnover"]:
        prior_params_power_spectrum["A_s"] = dict(min=1e4, max=5e7)
        prior_params_power_spectrum["alpha_1_s"] = dict(min=1, max=3)
        prior_params_power_spectrum["alpha_2_s"] = dict(min=-3, max=-1)
        prior_params_power_spectrum["k_turn"] = dict(min=0.001, max=0.05)

    elif power_spectrum_model == "analytical-phenom-fix-ns--0.3":
        prior_params_power_spectrum["A_s"] = dict(min=1e2, max=9e4)
        prior_params_power_spectrum["n_s"] = dict(value=-0.3, dist_type="Delta")
        prior_params_power_spectrum["alpha_s"] = dict(value=0, dist_type="Delta")

    else:
        print(power_spectrum_model)
        raise "Power spectrum model not known. "

    prior_dict = {}

    prior_dict["power_spectrum"] = prior_params_power_spectrum
    prior_dict["galaxy_bias"] = prior_params_galaxy
    prior_dict["magnitude_distribution"] = prior_params_magnitudes
    prior_dict["cosmology"] = prior_params_cosmology

    return prior_dict, prior_settings


def extract_1d_samples(list_in):
    exclude_params = [
        "gaussian_F_whitened_spatial",
        "counts_galaxies_observed",
        "magnitudes_obs",
    ]
    return [
        l for l in list_in if l not in exclude_params and not l.startswith("log_factor")
    ]


def seperate_gibbs_params(analysis, params_list_init, mode=0):
    params_list = copy.copy(params_list_init)
    params_list = [
        l
        for l in params_list
        if not l.startswith("log_factor")
        and not l.endswith("_obs")
        and not l.endswith("_observed")
        and not l == "z_list_measured"
        and not l == "z_list_true"
        and not l == "z_list_true_base"
    ]

    print("List of all parameters: ", params_list)

    params_cosmo_list = [
        "H0",
        "omega_b",
        "omega_cdm",
        "ng_bar",
        "ng_bar_tilde",
        "A_s_tilde",
        "n_s",
        "k0",
        "cmin",
        "ln10^{10}A_s",
        "eta_0",
        "alpha_s",
        "A_s",
        "gamma",
        "epsilon_g",
        "field_cut_exp",
        "alpha_1_s",
        "alpha_2_s",
        "k_turn",
    ]
    params_cosmo_list = [l + "_base" for l in params_cosmo_list]
    list_cosmo = [l for l in params_list if l in params_cosmo_list]

    params_magnitude_list = analysis.magnitude_model.list_all_params
    params_magnitude_list = [l + "_base" for l in params_magnitude_list]
    list_magnitude = [l for l in params_list if l in params_magnitude_list]

    if (len(list_cosmo) + len(list_magnitude) + 1) != len(params_list):
        print(params_list)
        raise "Division into parameters does not add up. "

    if mode == 0:
        list_kernels = [list_cosmo + list_magnitude + ["gaussian_F_whitened_spatial"]]
        list_dense_params = [[tuple(list_cosmo + list_magnitude)]]
    elif mode == 1:
        list_kernels = [list_cosmo, ["gaussian_F_whitened_spatial"], list_magnitude]
        list_dense_params = [True, False, True]
    elif mode == 2:
        list_kernels = [list_cosmo + list_magnitude, ["gaussian_F_whitened_spatial"]]
        list_dense_params = [True, False]
    elif mode == 3:
        list_kernels = [list_cosmo, list_magnitude, ["gaussian_F_whitened_spatial"]]
        list_dense_params = [True, True, False]
    elif mode == 4:
        list_kernels = [list_magnitude, list_cosmo, ["gaussian_F_whitened_spatial"]]
        list_dense_params = [True, True, False]
    else:
        raise NotImplementedError

    return list_dense_params, list_kernels


def get_prior_from_analysis_and_param(priors, param):
    if f"{param}_value" in priors.keys():
        val = priors[f"{param}_value"]
        out_dict = dict(value=val, dist_type="Delta")
    elif f"{param}_min" in priors.keys():
        min_val = priors[f"{param}_min"]
        max_val = priors[f"{param}_max"]
        out_dict = dict(min=min_val, max=max_val)
    else:
        if param == "gamma":
            out_dict = dict(value=0.7, dist_type="Delta")
        elif param == "epsilon_g":
            out_dict = dict(value=0.25, dist_type="Delta")
        elif param == "field_cut_exp":
            out_dict = dict(value=0.5, dist_type="Delta")

    return out_dict

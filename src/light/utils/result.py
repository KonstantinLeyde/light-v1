import argparse
import os

import arviz as az
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml

from ..astrophysics.luminosity import MagnitudeDistribution
from ..field import power_spectrum
from ..field.field import RealLogNormalField
from ..numpyro_utils import models, priors
from ..utils import helper_functions


def bool_arg(val):
    return val.lower() in ("y", "yes", "t", "true", "on", "1")


def convert_string_to_tuple(input_string):
    if input_string in [None, ""]:
        return tuple()
    else:
        return tuple(map(int, input_string.strip("()").split(",")))


class Analysis(RealLogNormalField):
    def __init__(
        self,
        kwargs_catalog,
        kwargs_magnitude_model,
        kwargs_cosmology,
        kwargs_field,
        kwargs_sampler=dict(
            num_posterior_samples=200,
        ),
        data_location=None,
        results_location=None,
        id_job=0,
    ):
        self.kwargs_catalog = kwargs_catalog
        self.kwargs_magnitude_model = kwargs_magnitude_model
        self.magnitude_model_name = self.kwargs_magnitude_model["magnitude_model_name"]
        self.kwargs_cosmology = kwargs_cosmology
        self.kwargs_field = kwargs_field
        self.data_location = data_location
        self.results_location = results_location

        if self.kwargs_field["power_spectrum_model"] in [
            "analytical-phenom",
            "flat",
            "analytical-phenom-all",
        ]:
            power_spectrum_of_k = power_spectrum.power_spectrum_analytical_form
        elif self.kwargs_field["power_spectrum_model"] in ["smooth-turnover"]:
            power_spectrum_of_k = power_spectrum.smooth_power_law_turnover
        elif self.kwargs_field["power_spectrum_model"] == "cosmopower":
            power_spectrum_of_k = None
        else:
            raise "Power spectrum model not known. "

        super().__init__(
            box_shape_d=self.kwargs_catalog["box_shape_d"],
            box_size_d=self.kwargs_catalog["box_size_d"],
            power_spectrum_of_k=power_spectrum_of_k,
        )

        self.magnitude_model = MagnitudeDistribution(
            **self.kwargs_magnitude_model,
        )

        self.kwargs_sampler = kwargs_sampler
        self.id_job = id_job
        self.nb_subresults = (
            self.kwargs_sampler["num_posterior_samples"]
            // self.kwargs_sampler["num_posterior_samples_per_batch"]
        )

    def __str__(self):
        attributes = {
            "magnitude_model": self.magnitude_model,
            "kwargs_cosmology": self.kwargs_cosmology,
            "data_location": self.data_location,
            "results_file_name": self.get_results_file_name(),
            "id_job": self.id_job,
        }

        for k in self.kwargs_field.keys():
            attributes[k] = self.kwargs_field[k]

        for k in self.kwargs_catalog.keys():
            attributes[k] = self.kwargs_catalog[k]

        for k in self.kwargs_sampler.keys():
            attributes[k] = self.kwargs_sampler[k]

        summary = "\n".join([f"{key}: {value}" for key, value in attributes.items()])
        return summary

    def get_data(self):
        return helper_functions.load_data(
            data_folder=self.data_location,
            catalog_file_name=self.kwargs_catalog["catalog_file_name"],
            dimensions=self.dimensions,
            pixelation=self.kwargs_catalog["pixelation"],
        )

    def get_data_as_array(self):
        (
            truth_df,
            data_df,
            abs_magnitudes_df,
            abs_magnitudes_obs_df,
            apparent_magnitudes_obs,
            redshift_obs,
            z_boundaries,
        ) = self.get_data()

        truth_array = truth_df.to_numpy().reshape(self.box_shape_d)
        data_array = data_df.to_numpy().reshape(self.box_shape_d)

        M_list = abs_magnitudes_df.to_numpy().flatten()
        M_obs_list = abs_magnitudes_obs_df.to_numpy().flatten()

        m_list = apparent_magnitudes_obs.to_numpy().flatten()
        z_list = redshift_obs.to_numpy().flatten()

        if z_boundaries is not None:
            z_boundaries = jnp.array(z_boundaries.to_numpy()[:, 0])

        truth_array = jnp.array(truth_array)
        data_array = jnp.array(data_array)

        M_list = jnp.array(M_list)
        M_obs_list = jnp.array(M_obs_list)

        return truth_array, data_array, M_list, M_obs_list, m_list, z_list, z_boundaries

    def get_data_kwargs(self):
        truth_array, data_box, M_list, M_obs_list, m_list, z_list, z_boundaries = (
            self.get_data_as_array()
        )

        apparent_magnitude_threshold_box2d = self.get_apparent_magnitude_threshold()

        data_kwargs = dict(
            counts_galaxies_observed=data_box,
            apparent_magnitude_threshold_box2d=apparent_magnitude_threshold_box2d,
            m_list=m_list,
            z_list=z_list,
            z_boundaries=z_boundaries,
        )

        if self.kwargs_catalog["pixelation"] == "redshift":
            pass
        elif self.kwargs_catalog["pixelation"] == "comoving_distance":
            del data_kwargs["z_boundaries"]
        else:
            raise NotImplementedError

        return data_kwargs

    def get_apparent_magnitude_threshold(self):
        apparent_magnitude_threshold_box2d = self.kwargs_catalog[
            "m_threshold"
        ] * jnp.ones(self.box_shape_d[:2])

        for pix in self.kwargs_catalog["special_deep_pixels"]:
            print(
                "Preparing the apperent magnitude threshold map for selected pixels. "
            )

            ix = pix["X_bin"]
            iy = pix["Y_bin"]
            val = pix["m_threshold"]

            apparent_magnitude_threshold_box2d = apparent_magnitude_threshold_box2d.at[
                ix, iy
            ].set(val)

        return apparent_magnitude_threshold_box2d

    def get_truth_kwargs(self):
        truth_array, _, _, _, _, _, _ = self.get_data_as_array()

        truth_kwargs = dict(
            truth_box=truth_array,
        )

        return truth_kwargs

    def get_deviations(self):
        counts_galaxies_true = self.hypersamples_d["counts_galaxies_true"]

        print("Loaded {} posterior samples. ".format(counts_galaxies_true.shape[0]))

        # load the truth counts
        truth_array = self.get_truth_kwargs()["truth_box"]

        statistics_number_counts = {}
        statistics_number_counts["mean"] = jnp.mean(counts_galaxies_true, 0)
        statistics_number_counts["std"] = jnp.std(counts_galaxies_true, 0)
        statistics_number_counts["median"] = jnp.median(counts_galaxies_true, 0)

        statistics_number_counts["deviations_absolute"] = (
            truth_array - statistics_number_counts["mean"]
        )
        statistics_number_counts["deviations_relative"] = (
            statistics_number_counts["deviations_absolute"]
            / statistics_number_counts["std"]
        )

        return statistics_number_counts

    def get_results_folder(self):
        return (
            self.results_location
        )

    def get_results_file_name(self):
        return self.get_results_folder() + "/result.av"

    def get_sub_results_file_name(self, i):
        return self.get_results_folder() + f"/result_{i}.av"

    def get_data_folder(self):
        return helper_functions.get_data_folder(self.kwargs_catalog["catalog_origin"])

    def create_result_folder(self):
        folder_name = self.get_results_folder() + "/preliminary"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' created successfully.")
        else:
            print(f"Folder '{folder_name}' already exists.")

    def save_settings_as_yaml(self):
        # Define the path to save the YAML file
        results_folder = self.get_results_folder()
        self.create_result_folder()
        yaml_file_path = os.path.join(results_folder, "settings.yaml")

        # Create a dictionary with settings data
        settings_data = {
            "kwargs_catalog": self.kwargs_catalog,
            "kwargs_magnitude_model": self.kwargs_magnitude_model,
            "kwargs_cosmology": self.kwargs_cosmology,
            "kwargs_field": self.kwargs_field,
            "data_location": self.data_location,
            "results_location": self.results_location,
            "kwargs_sampler": self.kwargs_sampler,
            "id_job": self.id_job,
        }

        settings_data = convert_all_tuples_to_lists(settings_data)
        settings_data = convert_all_jnp_arrays_to_lists(settings_data)

        # Save the settings data to a YAML file
        with open(yaml_file_path, "w") as yaml_file:
            yaml.dump(settings_data, yaml_file)

    def load_inf_data(
        self, paths=None, idxs=None, skip_tensors=False, skip_irrelevant_data=True
    ):
        if idxs == None:
            idxs = list(range(self.nb_subresults))

        if paths == None:
            paths = [self.get_sub_results_file_name(idx) for idx in idxs]

        self.inf_datas = []
        for i, path in enumerate(paths):
            print(
                "Loading {} / {} result (#total {}). ".format(
                    i + 1, len(paths), self.nb_subresults
                )
            )

            res = az.from_netcdf(path)

            if skip_tensors:
                print("Omitting tensors in posterior. ")
                res.posterior = remove_tensors_from_arviz(res.posterior)

                try:
                    del res.posterior_predictive
                except:
                    print("Posterior predictive not available to omit. ")

            if skip_irrelevant_data:
                try:
                    del res.observed_data
                    del res.log_likelihood
                except:
                    pass

            self.inf_datas.append(res)

        self.inf_data = az.concat(self.inf_datas, dim="draw")

    def load_hypersamples(self, skip_diverging_samples):
        self.hypersamples = helper_functions.convert_arviz_to_numpy_dict(self.inf_data)

        # get rid of the chain dimension
        if skip_diverging_samples:
            diverging = self.inf_data.sample_stats.diverging.values

            # if using Gibbs sampler
            if len(diverging.shape) == 3:
                # get the max diverging per Gibbs sampling
                diverging = np.max(diverging, axis=(-1,))
            elif len(diverging.shape) == 2:
                pass
            else:
                raise NotImplementedError

            self.hypersamples = {k: v[~diverging] for k, v in self.hypersamples.items()}

        else:
            self.hypersamples = helper_functions.flatten_dict_along_chain_dim(
                self.hypersamples
            )
        self.hypersamples, self.hypersamples_d = (
            helper_functions.divide_scalars_and_arrays_from_dict(self.hypersamples)
        )
        self.hypersamples_df = pd.DataFrame(self.hypersamples)

    def load_numpyro_result(
        self,
        paths=None,
        idxs=None,
        skip_tensors=False,
        skip_diverging_chains=False,
        skip_diverging_samples=False,
    ):
        self.load_inf_data(paths=paths, idxs=idxs, skip_tensors=skip_tensors)
        self.post_process_inf_data(
            skip_diverging_chains=skip_diverging_chains,
        )
        self.load_hypersamples(skip_diverging_samples=skip_diverging_samples)

    def post_process_inf_data(self, skip_diverging_chains=False):
        if isinstance(skip_diverging_chains, bool):
            if skip_diverging_chains:
                _ = drop_diverging_chains(self.inf_data)
        else:
            _ = drop_chains(self.inf_data, chains_to_drop=skip_diverging_chains)

    def setup_default_priors(self):
        try:
            self.kwargs_sampler["prior"], self.kwargs_sampler["prior_settings"] = (
                self.kwargs_sampler["prior"],
                self.kwargs_sampler["prior_settings"],
            )
        except:
            print("Could not find prior settings. ")
            self.kwargs_sampler["prior"], self.kwargs_sampler["prior_settings"] = (
                priors.get_prior_dict(self)
            )

    @classmethod
    def from_yaml(cls, id_job=None, yaml_file_path=None):
        """
        Load the analysis from a yaml file, we have to also specify the id_job,
        since this is not always a priori known if the job is defined by the file
        before it is slurm-submitted.

        """

        if yaml_file_path == None:
            raise "No yaml file path specified. "

        # Load settings from the YAML file
        with open(yaml_file_path, "r") as yaml_file:
            settings_data = yaml.safe_load(yaml_file)

        if "id_job" not in settings_data.keys():
            settings_data["id_job"] = id_job

        # Initialize a new instance of the Analysis class with the loaded settings
        return cls(**settings_data)

    def get_model(self):
        if "redshift_error_model" in self.kwargs_field.keys():
            if self.kwargs_field["redshift_error_model"] == "convolution-1d":
                print("Using redshift error model. ")
                raise NotImplementedError('Redshift error model not implemented in public version. ')
            else:
                model = models.model
        else:
            model = models.model

        return model


def get_settings_from_parse_command_line(mode):
    parser = argparse.ArgumentParser(
        description="Process command line arguments for the analysis."
    )

    parser.add_argument("--id_job", type=int, help="Job ID")
    parser.add_argument(
        "--init_file",
        default=None,
        help="Whether the settings are defined from the launch file. This overrides all other CL args. ",
    )
    parser.add_argument(
        "--settings_file",
        default=None,
        help="Whether the settings are defined from the settings file. This overrides all other CL args. ",
    )

    if mode in ["default", "settings"]:
        pass
    else:
        print(mode)
        raise "Mode not known. "

    args = vars(parser.parse_args())

    if args["init_file"] != None:
        print("Loading init_file from yaml. ")
        with open(args["init_file"], "r") as yaml_file:
            new_args = yaml.safe_load(yaml_file)

        for k in ["init_file", "id_job", "settings_file"]:
            new_args[k] = args[k]
    else:
        new_args = args

    if mode == "default":
        new_args = post_processing_default_command_line_args(new_args)

    return new_args


def post_processing_default_command_line_args(args):
    print(args["catalog_settings_path"])
    with open(args["catalog_settings_path"], "r") as yaml_file:
        kwargs_catalog = yaml.safe_load(yaml_file)

    new_args = {}
    new_args["kwargs_catalog"] = kwargs_catalog

    for k in [
        "kwargs_magnitude_model",
        "kwargs_sampler",
        "kwargs_cosmology",
        "kwargs_field",
        "data_location",
        "results_location",
    ]:
        new_args[k] = args[k]

    # add the rest of the arguments
    for k in args.keys():
        if k not in new_args.keys() and k not in ["catalog_settings_path"]:
            print(f'Keyword not used: {k}')
            new_args[k] = args[k]

    return new_args


def get_settings_class_from_parse_command_line():
    args = get_settings_from_parse_command_line(mode="default")

    if args["settings_file"] != None:
        print("Initialize analysis from yaml file. ")
        print("All other command line arguments are overwritten. ")
        return Analysis.from_yaml(
            id_job=args["id_job"], yaml_file_path=args["settings_file"]
        )
    else:
        for k in ["settings_file", "init_file"]:
            if k in args.keys():
                del args[k]
        return Analysis(**args)


def convert_all_lists_to_tuples(input_dict):
    """
    Convert all lists to tuples within a dictionary.

    Args:
        input_dict (dict): The input dictionary.

    Returns:
        dict: The modified dictionary with lists converted to tuples.
    """
    converted_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, list):
            converted_dict[key] = tuple(value)
        elif isinstance(value, dict):
            converted_dict[key] = convert_all_lists_to_tuples(value)
        else:
            converted_dict[key] = value
    return converted_dict


def convert_all_tuples_to_lists(input_dict):
    """
    Convert all tuples to lists within a dictionary.

    Args:
        input_dict (dict): The input dictionary.

    Returns:
        dict: The modified dictionary with tuples converted to lists.
    """
    converted_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, tuple):
            converted_dict[key] = list(value)
        elif isinstance(value, dict):
            converted_dict[key] = convert_all_tuples_to_lists(value)
        else:
            converted_dict[key] = value
    return converted_dict


def convert_all_jnp_arrays_to_lists(input_dict):
    converted_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, jnp.ndarray):
            converted_dict[key] = value.tolist()
        elif isinstance(value, dict):
            converted_dict[key] = convert_all_jnp_arrays_to_lists(value)
        else:
            converted_dict[key] = value
    return converted_dict


def remove_tensors_from_arviz(ds):
    return ds[[var for var in ds.data_vars if len(ds[var].dims) == 2]]


def drop_diverging_chains(inf_data, threshold=0.02):
    """
    Drops chains from inf_data where the majority of samples are diverging.

    Parameters
    ----------

    inf_data (arviz.InferenceData):
        The InferenceData object containing the sample statistics.
    threshold (float):
        The proportion of True values in a chain required to drop that chain.

    Returns
    -------
    arviz.InferenceData: The modified InferenceData object with diverging chains removed.
    """
    diverging = inf_data.sample_stats.diverging.values

    # if using Gibbs sampler
    if len(diverging.shape) == 3:
        # Calculate the proportion of diverging samples in each chain
        diverging_proportion = np.mean(diverging, axis=(1,))

        # get the max diverging per Gibbs sampling
        diverging_proportion = np.max(diverging_proportion, axis=(-1,))

    elif len(diverging.shape) == 2:
        # Calculate the proportion of diverging samples in each chain
        diverging_proportion = np.mean(diverging, axis=(1,))
    else:
        raise NotImplementedError

    # Identify chains to keep (those with less than the threshold of diverging samples)
    chains_to_drop = np.where(diverging_proportion >= threshold)[0]

    print(f"Chains have the following property: {diverging_proportion}. ")

    return drop_chains(inf_data, chains_to_drop)


def drop_chains(inf_data, chains_to_drop):
    print(f"Dropping chains {chains_to_drop}.")

    # Check existing chain indices
    existing_chains = inf_data.sample_stats.chain.values
    valid_chains_to_keep = [
        chain for chain in existing_chains if chain not in chains_to_drop
    ]

    if not valid_chains_to_keep:
        raise ValueError("No valid chains to keep based on the given threshold.")

    # Apply the selection to each group in the InferenceData object
    for group in inf_data.groups():
        if hasattr(inf_data, group):
            dataset = getattr(inf_data, group)
            if "chain" in dataset.dims:
                # Check if 'chain' is in the dataset dimensions
                available_chains = dataset.chain.values
                valid_chains = [
                    chain for chain in valid_chains_to_keep if chain in available_chains
                ]
                if valid_chains:
                    setattr(inf_data, group, dataset.sel(chain=valid_chains))
                else:
                    raise KeyError(
                        f"Not all values found in index 'chain' for group '{group}'"
                    )

    return None

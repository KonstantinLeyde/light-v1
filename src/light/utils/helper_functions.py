import pickle

import numpy as np
import pandas as pd


def get_coordinates(dimensions):
    if dimensions == 2:
        coordinates = ["Y", "Z"]
    else:
        coordinates = ["X", "Y", "Z"]

    return coordinates


def get_data_folder(catalog_origin, data_location=""):
    if catalog_origin.startswith("millenium"):
        data_path = "./millenium/"
    else:
        raise ValueError(f"Unknown catalog origin '{catalog_origin}'")

    return data_location + data_path


def get_data_name(
    box_shape_d,
    m_threshold,
    sample_from_schechter,
    Z_offset,
    catalog_name=None,
    pixelation="comoving_distance",
    cut_magnitudes_above=False,
    special_deep_pixels=[],
    redshift_error_settings={},
    postfix_name="",
):
    """
    Generate a data path based on input parameters.

    Parameters
    ----------
    box_shape_d : tuple
        Dimensions of the box.
    m_threshold : float
        Magnitude threshold.
    sample_from_schechter : bool
        Whether the magnitude of the catalog is sampled from Schechter distribution.
    Z_offset : float
        Offset for the computation of distance.

    Returns
    -------
    str
        Data name.
    """
    dimensions = len(box_shape_d)
    coordinates = get_coordinates(dimensions)

    file_name_1 = "dim_{}_mtresh_{}_matrix_schechter_{}_z_off_{}".format(
        dimensions, m_threshold, sample_from_schechter, Z_offset
    )

    if catalog_name != None:
        file_name_1 += "_{}".format(catalog_name)

    file_name_2 = "_pixels_" + "_".join(
        [f"{key}_{box_shape}" for key, box_shape in zip(coordinates, box_shape_d)]
    )

    file_name = file_name_1 + file_name_2

    if pixelation == "redshift":
        file_name += "_redshift_pixelation"

    if cut_magnitudes_above:
        file_name += "_cut_magnitudes_above"

    if special_deep_pixels != []:
        file_name += f"_special_pixels_{len(special_deep_pixels)}"

    if redshift_error_settings != {}:
        file_name += "_redshift_with_errors"

    if postfix_name != "":
        file_name += "_" + postfix_name

    return file_name


def get_results_name(
    box_shape_d,
    m_threshold,
    sample_from_schechter,
    Z_offset,
    magnitude_model=None,
    cosmology_model=None,
    catalog_name=None,
    galaxy_model=None,
    power_spectrum_model=None,
):
    file_path_1 = get_data_name(
        box_shape_d, m_threshold, sample_from_schechter, Z_offset, catalog_name
    )
    file_path_2 = ""

    if magnitude_model != None:
        file_path_2 += "_magnitude_model_{}".format(magnitude_model)

    if cosmology_model != None:
        file_path_2 += "_cosmology_model_{}".format(cosmology_model)

    if power_spectrum_model != None:
        file_path_2 += "_power_spectrum_model_{}".format(power_spectrum_model)

    if galaxy_model != None:
        file_path_2 += "_galaxy_model_{}".format(galaxy_model)

    return file_path_1 + file_path_2


def load_data(data_folder, catalog_file_name, dimensions=3, pixelation="comoving"):
    data_name = catalog_file_name

    name_path = data_folder + data_name
    print("Loading from {}".format(name_path))

    if dimensions == 2:
        # Load truth dataframe
        truth_d = pd.read_csv(name_path).drop(["Y_bin"], axis=1)

        # Load data dataframe
        data = pd.read_csv(name_path + "_obs").drop(["Y_bin"], axis=1)

    elif dimensions == 3:
        truth_d = pd.read_csv(name_path, header=[0, 1], index_col=[0])
        # Load data dataframe
        data = pd.read_csv(name_path + "_obs", header=[0, 1], index_col=[0])

    abs_magnitudes = pd.read_csv(name_path + "_abs_magnitudes", index_col=[0])
    abs_magnitudes_obs = pd.read_csv(name_path + "_abs_magnitudes_obs", index_col=[0])

    apparent_magnitudes_obs = pd.read_csv(
        name_path + "_apparent_magnitudes_obs", index_col=[0]
    )

    redshifts_obs = pd.read_csv(name_path + "_redshifts_obs", index_col=[0])

    if pixelation == "redshift":
        z_boundaries = pd.read_csv(name_path + "_z_boundaries")
    else:
        z_boundaries = None

    return (
        truth_d,
        data,
        abs_magnitudes,
        abs_magnitudes_obs,
        apparent_magnitudes_obs,
        redshifts_obs,
        z_boundaries,
    )


def convert_arviz_to_numpy_dict(inf_data, skip_vars_end_with="_base"):
    """
    Convert ArviZ data variables to a dictionary of NumPy arrays.

    Parameters:
    - inf_data (arviz.InferenceData): ArviZ InferenceData object containing posterior samples.

    Returns:
    - numpy_dict (dict): Dictionary containing selected data variables as NumPy arrays.
    """

    # Extract data variables from inf_data
    data_vars = inf_data.posterior.data_vars

    # Initialize an empty dictionary to store NumPy arrays
    numpy_dict = {}

    # Iterate over data variables and convert them to NumPy arrays
    for var_name, var_data in data_vars.items():
        # Skip variables ending with 'base'
        if var_name.endswith(skip_vars_end_with):
            continue

        # Check if the data type of the variable is integer or float
        if np.issubdtype(var_data.dtype, np.integer) or np.issubdtype(
            var_data.dtype, np.floating
        ):
            # Extract the data as a NumPy array
            var_array = np.asarray(var_data)
            # Add the NumPy array to the dictionary with the variable name as key
            numpy_dict[var_name] = var_array

    # add also predictions

    # Extract data variables from inf_data
    try:
        data_vars = inf_data.posterior_predictive.data_vars
    except:
        data_vars = {}

    # Iterate over data variables and convert them to NumPy arrays
    for var_name, var_data in data_vars.items():
        # Check if the data type of the variable is integer or float
        if np.issubdtype(var_data.dtype, np.integer) or np.issubdtype(
            var_data.dtype, np.floating
        ):
            # Extract the data as a NumPy array
            var_array = np.asarray(var_data)
            # Add the NumPy array to the dictionary with the variable name as key
            numpy_dict[var_name] = var_array

    return numpy_dict


def squeeze_dict(input_dict):
    return {k: v.squeeze() for k, v in input_dict.items()}


def flatten_dict_along_chain_dim(input_dict):
    return {k: v.reshape((-1,) + v.shape[2:]) for k, v in input_dict.items()}


def divide_scalars_and_arrays_from_dict(input_dict):
    return {k: v for k, v in input_dict.items() if len(v.shape) == 1}, {
        k: v for k, v in input_dict.items() if len(v.shape) != 1
    }


def get_ith_entry_of_array(dictionary, i):
    result_dict = {}
    for key, value in dictionary.items():
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise ValueError(f"The value for key '{key}' is not an array")
        if i < 0 or i >= len(value):
            raise IndexError(f"Index {i} out of range for key '{key}'")
        result_dict[key] = value[i]
    return result_dict


def filter_columns_with_dynamic_range(df):
    """
    Filter columns of a DataFrame that have a dynamic range.

    Args:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: DataFrame containing only columns with a dynamic range.
    """
    dynamic_columns = []
    for col in df.columns:
        if df[col].max() != df[col].min():
            dynamic_columns.append(col)
    return df[dynamic_columns]


def flatten_dict(d, parent_key="", sep="_"):
    """
    Flatten a nested dictionary.

    Parameters:
    d (dict): The dictionary to flatten.
    parent_key (str): The base key string for recursion (used internally).
    sep (str): The separator between keys.

    Returns:
    dict: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def last_state_write(last_state, file_out):
    with open(file_out, "wb") as f:
        pickle.dump(last_state, f)


def last_state_read(file_in):
    with open(file_in, "rb") as f:
        last_state = pickle.load(f)
    return last_state

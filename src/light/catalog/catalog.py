import copy

import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import yaml
from matplotlib import pyplot as plt

from ..utils import conventions
from ..utils import helper_functions as _helper_functions

# these are only needed if we sample from the Schechter function and
# do not use the micecat absolute magnitudes
M_STAR = -20
ALPHA = -1.25
M_FAINT_MAX = -20.48399
REDSHIFT_BUFFER = 1e-4

SUN_ABSOLUTE_MAGNITUDE = 4.83
DATA_LOCATION = "/users/kleyde/galaxy_completion/gaussian_toy_model/data/"


def schechter(M_vals, M_star=M_STAR, alpha=ALPHA):
    prob = (
        0.4
        * np.log(10)
        * np.power(10, 0.4 * (M_star - M_vals) * (alpha + 1))
        * np.exp(-np.power(10, 0.4 * (M_star - M_vals)))
    )
    norm = np.trapz(prob, x=M_vals)

    return prob / norm


def schechter_cdf(M_vals, M_star=M_STAR, alpha=ALPHA):
    pdf = schechter(M_vals, M_star, alpha)
    cdf = np.cumsum(pdf) / np.sum(pdf)

    return cdf


def get_cdf_schechter_interpolation():
    M_values = np.linspace(-27, -5, 10000)

    cdf_values = schechter_cdf(M_values, M_STAR, ALPHA)

    interpolation = scipy.interpolate.interp1d(
        M_values, cdf_values, kind="linear", fill_value=(0, 1), bounds_error=False
    )

    return interpolation


def get_samples_from_schechter(
    num_samples, M_star=M_STAR, alpha=ALPHA, M_faint_max=M_FAINT_MAX
):
    num_proposal_samples = 10 * num_samples

    M_vals = np.linspace(M_star - 5, M_faint_max, num_proposal_samples)

    prob = schechter(M_vals, M_star, alpha)
    prob /= np.sum(prob)

    idx = np.random.choice(num_proposal_samples, p=prob, size=num_samples, replace=True)

    return M_vals[idx]


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

    return M + 5 * np.log10(luminosity_distance) + 25


def abs_magnitude_from_apparent_magnitude(m, luminosity_distance):
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

    return m - 5 * np.log10(luminosity_distance) - 25


def luminosity_to_absolute_magnitude(luminosity):
    """
    Convert luminosity to absolute magnitude.

    Parameters:
    - luminosity (float): Luminosity of the object in solar luminosity.

    Returns:
    - float: Absolute magnitude of the object.
    """

    absolute_magnitude = -2.5 * np.log10(luminosity) + SUN_ABSOLUTE_MAGNITUDE
    return absolute_magnitude


def absolute_magnitude_to_luminosity(absolute_magnitude):
    """
    Convert absolute magnitude to luminosity.

    Parameters:
    - absolute_magnitude (float): absolute_magnitude of the object.

    Returns:
    - float: Absolute magnitude of the object.
    """
    SUN_ABSOLUTE_MAGNITUDE = 4.83

    exponent = (absolute_magnitude - SUN_ABSOLUTE_MAGNITUDE) / (-2.5)
    luminosity = 10 ** (exponent)
    return luminosity


def load_catalog_data_v1(
    x_file, y_file, z_file, abs_M_path, sample_from_schechter=False
):
    # Load X, Y, Z data from text files
    x_data = pd.read_csv(x_file, header=None, names=["X"])
    y_data = pd.read_csv(y_file, header=None, names=["Y"])
    z_data = pd.read_csv(z_file, header=None, names=["Z"])

    if sample_from_schechter:
        number_galaxies = len(x_data)
        abs_M_data = pd.DataFrame(dict(M=get_samples_from_schechter(number_galaxies)))
    else:
        abs_M_data = pd.read_csv(abs_M_path, header=None, names=["M"])

    # Combine X, Y, Z into a single DataFrame
    catalog_data = pd.concat([x_data, y_data, z_data, abs_M_data], axis=1)

    return catalog_data


def load_micecat_catalog_v1(
    abs_path="./data/micecat_data_slice/", sample_from_schechter=False
):
    x_file_path = abs_path + "GalPosxMICECAT.txt"
    y_file_path = abs_path + "GalPosyMICECAT.txt"
    z_file_path = abs_path + "GalPoszMICECAT.txt"
    abs_M_path = abs_path + "MagAbsMICECAT.txt"

    return load_catalog_data_v1(
        x_file_path, y_file_path, z_file_path, abs_M_path, sample_from_schechter
    )


def load_catalog_from_csv(
    file_path, header, rename_dict, file_name, abs_path=DATA_LOCATION
):
    """
    Parameters
    ----------

    header (int):
        The number of lines skipped (because of descriptive text)

    """

    raw_catalog = pd.read_csv(
        abs_path + file_path + file_name,
        header=header,
    )

    catalog_keys = list(raw_catalog.keys())
    print("keys in catalog: ", catalog_keys)

    if any([key not in catalog_keys for key in ["X", "Y", "Z", "M"]]):
        # we have to rename columns for the micecat catalog

        raw_catalog.rename(columns=rename_dict, inplace=True)

    return raw_catalog


class SimulatedCatalog:
    """
    A class for handling galaxy catalogs and performing various computations.

    Parameters
    ----------
    catalog_origin (str): 
        Specifies the origin of the galaxy catalog. Options include:
        - "micecat_box": Loads MICECAT simulation catalog data.
        - "molino": Loads MOLINO simulation catalog data.
        - "millenium": Loads Millennium simulation catalog data.
    pixel_length_in_Mpc_comoving (float or list): 
        Length of a pixel side in comoving Mpc. If list, defines 
        each dimension.
    m_threshold (float): 
        Threshold for the absolute magnitude of galaxies included in the catalog.
    cosmology (dict): 
        Cosmological parameters (e.g., H0, Omega_m) required for distance calculations.
    galaxies_percentage (float): 
        Fraction of galaxies to sample from the catalog.
    geometry (str): 
        Shape of the simulation volume (e.g., "toy" for toy-model geometry).
    Z_offset (float): 
        Offset for the galaxy redshift distribution.
    debug (bool): 
        If True, loads a subset (e.g., 10,000 galaxies) for testing and faster debugging.
    sample_from_schechter (bool): 
        If True, samples galaxies according to a Schechter luminosity function.
    zmax (float): 
        Maximum redshift limit for galaxy selection.
    pixelation (str): 
        Mode for pixel division (e.g., "comoving_distance").
    box_size_d (float): 
        Dimension of the simulation box. Only applies if pixelation is "comoving_distance".
    redshift_num_bins (int): 
        Number of bins for redshift slicing.
    cut_magnitudes_above (bool): 
        If True, removes galaxies above a certain magnitude threshold.
    special_deep_pixels (list): 
        List of specific pixels to apply additional criteria, e.g., deeper cuts.
    redshift_error_settings (dict): 
        Settings for simulating redshift measurement errors.
    postfix_name (str): 
        Optional suffix for catalog file naming.

    """

    def __init__(
        self,
        catalog_origin,
        pixel_length_in_Mpc_comoving,
        m_threshold,
        cosmology,
        galaxies_percentage,
        geometry="toy",
        Z_offset=0,
        debug=True,
        sample_from_schechter=False,
        zmax=2,
        pixelation="comoving_distance",
        box_size_d=None,
        redshift_num_bins=None,
        cut_magnitudes_above=False,
        special_deep_pixels=[],
        redshift_error_settings={},
        postfix_name="",
    ):
        self.catalog_origin = catalog_origin
        self.data_folder = _helper_functions.get_data_folder(self.catalog_origin)
        self.pixel_length_in_Mpc_comoving = pixel_length_in_Mpc_comoving
        self.m_threshold = m_threshold
        self.cosmology = cosmology
        self.galaxies_percentage = galaxies_percentage
        self.geometry = geometry
        self.debug = debug
        self.zmax = zmax
        self.Z_offset = Z_offset  # for the computation of the distance
        self.sample_from_schechter = sample_from_schechter
        self.pixelation = pixelation  # whether the galaxies are binned in redshift or comoving coordinates
        self.redshift_num_bins = redshift_num_bins
        self.box_size_d = box_size_d
        self.cut_magnitudes_above = cut_magnitudes_above
        self.special_deep_pixels = special_deep_pixels
        self.redshift_error_settings = redshift_error_settings
        self.postfix_name = postfix_name

        self.catalog_file_name_original = conventions.get_original_catalog_file_name(
            self.catalog_origin, self.galaxies_percentage, self.box_size_d[2]
        )

        self.dimensions = len(pixel_length_in_Mpc_comoving)
        self.set_coordinates(self.dimensions)

        if catalog_origin == "micecat_box":
            if sample_from_schechter:
                raise "Cannot sample from Schechter in this configuration. "
            rename_dict = {"xgal": "X", "ygal": "Y", "zgal": "Z", "mr_gal": "M"}
            self.catalog = load_catalog_from_csv(
                self.data_folder,
                file_name=self.catalog_file_name_original,
                rename_dict=rename_dict,
                header=11,
            )
        elif catalog_origin == "molino":
            if not sample_from_schechter:
                raise "We have created samples from Schechter in this configuration. "
            rename_dict = {"xgal": "X", "ygal": "Y", "zgal": "Z", "mr_gal": "M"}
            self.catalog = load_catalog_from_csv(
                self.data_folder,
                file_name=self.catalog_file_name_original,
                rename_dict=rename_dict,
                header=0,
            )
        elif catalog_origin.startswith("millenium"):
            rename_dict = {"x": "X", "y": "Y", "z": "Z", "mag_k": "M"}
            self.catalog = load_catalog_from_csv(
                self.data_folder,
                file_name=self.catalog_file_name_original,
                rename_dict=rename_dict,
                header=9,
            )

        if self.redshift_error_settings != {}:
            if self.pixelation == "comoving_distance":
                raise NotImplementedError

        if self.debug:
            # only load 10000 galaxies if in debug mode
            self.catalog = self.catalog[:10000]

        self.set_subbins_per_bin()
        self.precompute_interpolation_for_redshift()
        self.compute_comoving_distance()
        self.compute_redshift_and_luminosity_distance()

        self.add_bin_columns_in_comoving_distance_and_angles()
        self.complete_apparent_magnitudes()
        self.compute_observed_catalog_and_additional_selection(
            self.cut_magnitudes_above
        )
        self.add_bin_columns()
        self.calculate_observed_catalog()
        self.add_bin_counts()

        self.add_truth()
        self.put_bin_counts_in_matrix()

        self.file_name = _helper_functions.get_data_name(
            box_shape_d=self.box_shape_d,
            m_threshold=self.m_threshold,
            sample_from_schechter=self.sample_from_schechter,
            Z_offset=self.Z_offset,
            catalog_name=self.catalog_file_name_original,
            pixelation=self.pixelation,
            cut_magnitudes_above=self.cut_magnitudes_above,
            special_deep_pixels=self.special_deep_pixels,
            redshift_error_settings=redshift_error_settings,
            postfix_name=self.postfix_name,
        )

        self.compute_fraction_galaxies_never_observed()
        self.sanity_check()

    def set_coordinates(self, dimensions):
        if dimensions == 2:
            self.coordinates = ["Y", "Z"]
            self.coordinate_bins = ["Y_bin", "Z_bin"]

            if self.pixelation == "redshift":
                raise NotImplementedError

        elif dimensions == 3:
            self.coordinates = ["X", "Y", "Z"]
            self.coordinate_bins = ["X_bin", "Y_bin", "Z_bin"]

        if self.pixelation == "redshift":
            self.bin_counts_vars = ["X", "Y", "z"]
            self.bin_counts_vars_labels = ["X_bin", "Y_bin", "z_bin"]

            if self.redshift_error_settings == {}:
                # no redshift errors
                self.bin_counts_vars_obs = self.bin_counts_vars
                self.bin_counts_vars_labels_obs = self.bin_counts_vars_labels
            else:
                # redshift errors
                self.bin_counts_vars_obs = ["X", "Y", "z_with_errors"]
                self.bin_counts_vars_labels_obs = [
                    "X_bin",
                    "Y_bin",
                    "z_with_errors_bin",
                ]

        elif self.pixelation == "comoving_distance":
            self.bin_counts_vars = self.coordinates
            self.bin_counts_vars_labels = self.coordinate_bins

            if self.redshift_error_settings == {}:
                # no redshift errors
                self.bin_counts_vars_obs = self.bin_counts_vars
                self.bin_counts_vars_labels_obs = self.bin_counts_vars_labels

        else:
            raise NotImplementedError

    def set_subbins_per_bin(self):
        self.NUMBER_SUBBINS_PER_BIN = conventions.get_NUMBER_SUBBINS_PER_BIN(
            self.pixel_length_in_Mpc_comoving
        )

    def precompute_interpolation_for_redshift(self):
        z_vals_log_space = np.logspace(-6, np.log10(self.zmax), 100_000)

        comoving_distance_vals = self.cosmology.comoving_distance(z_vals_log_space)

        self.redshift_from_comoving_distance_interp = scipy.interpolate.interp1d(
            comoving_distance_vals,
            z_vals_log_space,
            kind="linear",
            bounds_error=False,
            fill_value=(0, "nan"),
        )

    def compute_comoving_distance(self):
        if self.geometry == "toy":
            self.catalog["comoving_distance"] = abs(self.catalog["Z"]) + self.Z_offset

    def compute_redshift_and_luminosity_distance(self):
        print("Computing redshifts ...")
        print(
            "Minimum and maximum comoving distance: {:.2f}, {:.2f}".format(
                np.min(self.catalog["comoving_distance"]),
                np.max(self.catalog["comoving_distance"]),
            )
        )
        self.catalog["z"] = self.redshift_from_comoving_distance_interp(
            self.catalog["comoving_distance"]
        )
        print("Added redshifts.")

        if self.redshift_error_settings != {}:
            std_devs = self.redshift_error_settings["z_error"] * (1 + self.catalog["z"])

            # fix random seed for reproducibility
            seed = 6564743783635883188080801232342353452086803839626
            rng = np.random.default_rng(seed)

            self.catalog["z_with_errors"] = rng.normal(
                loc=self.catalog["z"], scale=std_devs, size=std_devs.shape
            )
        else:
            self.catalog["z_with_errors"] = self.catalog["z"]

        self.catalog["luminosity_distance"] = self.cosmology.luminosity_distance(
            self.catalog["z"]
        ).values

    def complete_apparent_magnitudes(self):
        self.catalog["m"] = apparent_magnitude_from_abs_magnitude(
            M=self.catalog["M"], luminosity_distance=self.catalog["luminosity_distance"]
        )

        self.catalog["m_threshold"] = self.m_threshold

        for pix in self.special_deep_pixels:
            print("Selecting a deeper survey for pixel {}".format(pix))

            new_idx_X = self.catalog["X_bin"] == pix["X_bin"]
            new_idx_Y = self.catalog["Y_bin"] == pix["Y_bin"]

            # collect the galaxies in the special pixel
            new_idx = new_idx_X & new_idx_Y

            self.catalog.loc[new_idx, "m_threshold"] = pix["m_threshold"]

    def compute_observed_catalog_and_additional_selection(self, cut_magnitudes_above):
        if cut_magnitudes_above:
            idx = self.catalog["m"] < self.m_threshold

            # compute the faintest available galaxy
            M_faintest = np.max(self.catalog["M"][idx])
            self.M_cut_high = M_faintest + 1

            print(f"Cutting the catalog above a M = {self.M_cut_high}. ")

            idx_bright = self.catalog["M"] < self.M_cut_high

            # select all galaxies brighter than this one
            self.catalog = self.catalog[idx_bright].reset_index(drop=True)
        else:
            self.M_cut_high = np.inf

        self.idx_selected = self.catalog["m"] < self.catalog["m_threshold"]

    def calculate_observed_catalog(self):
        self.idx_selected_and_z_range = copy.copy(self.idx_selected)
        self.idx_selected_and_z_range &= (
            self.catalog["z_with_errors"] < self.bins_z_boundaries[-1]
        )
        self.idx_selected_and_z_range &= (
            self.catalog["z_with_errors"] > self.bins_z_boundaries[0]
        )

        self.catalog_obs = self.catalog[self.idx_selected_and_z_range]

    def add_bin_columns_in_comoving_distance_and_angles(self):
        self.num_bins = {}
        self.length_var = {}
        # Define the number of bins
        for i, key in enumerate(self.coordinates):
            min_var = self.catalog[key].min()
            max_var = self.catalog[key].max()

            measured_box_length = max_var - min_var

            # check whether the box size is more or less compatible to the measure one.
            if abs(measured_box_length - self.box_size_d[i]) > 0.2:
                print(measured_box_length)
                raise "Error in the given box size. "

            self.length_var[key] = self.box_size_d[i]

            self.num_bins[key] = round(
                self.length_var[key] / self.pixel_length_in_Mpc_comoving[i]
            )

        for key in self.num_bins.keys():
            label = key + "_bin"

            self.catalog[label] = pd.cut(
                self.catalog[key], bins=self.num_bins[key], labels=False
            )

    def add_bin_columns(self):
        if self.geometry == "toy":
            redshift_buffer = REDSHIFT_BUFFER

            if self.redshift_error_settings != {}:
                redshift_buffer -= 4 * self.redshift_error_settings["z_error"]

            redshift_min = (
                np.min(self.catalog["z"][self.idx_selected]) - redshift_buffer
            )
            redshift_max = (
                np.max(self.catalog["z"][self.idx_selected]) + redshift_buffer
            )

            print(
                "z_min = {:.2f} and z_max = {:.2f} for the redshift grid".format(
                    redshift_min, redshift_max
                )
            )

            # compute the angular distance to the start of the box
            # this is a convention (using the smallest redshift), since the angle would vary going along redshift
            self.angular_diameter_distance = self.cosmology.angular_diameter_distance(
                z=redshift_min
            ).value
            self.angle_galaxy_0 = self.box_size_d[0] / self.angular_diameter_distance
            self.angle_galaxy_1 = self.box_size_d[1] / self.angular_diameter_distance

            if self.pixelation == "redshift":
                self.bins_z = np.linspace(
                    redshift_min,
                    redshift_max,
                    self.NUMBER_SUBBINS_PER_BIN * self.redshift_num_bins,
                )
                self.bins_z_boundaries = np.linspace(
                    redshift_min, redshift_max, self.redshift_num_bins + 1
                )

                # grid computation, both sub-grid and boundaries
                self.z_information = pd.DataFrame({"bins_z": self.bins_z})
                self.bins_z_boundaries_df = pd.DataFrame(
                    {"z_boundaries": self.bins_z_boundaries}
                )

            elif self.pixelation == "comoving_distance":
                self.bins_z = np.linspace(
                    redshift_min,
                    redshift_max,
                    self.NUMBER_SUBBINS_PER_BIN * self.num_bins["Z"],
                )

        if self.pixelation == "redshift":
            self.catalog["z_bin"] = pd.cut(
                self.catalog["z"], bins=self.bins_z_boundaries, labels=False
            )
            self.catalog["z_with_errors_bin"] = pd.cut(
                self.catalog["z_with_errors"], bins=self.bins_z_boundaries, labels=False
            )

            self.num_bins["z"] = self.redshift_num_bins

            if self.redshift_error_settings != {}:
                self.num_bins["z_with_errors"] = self.num_bins["z"]

    def add_bin_counts(self):
        print("Binning truth by {}".format(self.bin_counts_vars_labels))
        print("Binning observed survey by {}".format(self.bin_counts_vars_labels_obs))

        # bin the true catalog without errors in redshift
        self.bin_counts = (
            self.catalog.groupby(self.bin_counts_vars_labels)
            .size()
            .reset_index(name="GalaxyCount")
        )
        self.bin_counts_obs = (
            self.catalog_obs.groupby(self.bin_counts_vars_labels_obs)
            .size()
            .reset_index(name="GalaxyCountObserved")
        )

    def add_truth(self):
        mean = self.bin_counts.mean()["GalaxyCount"]
        std = self.bin_counts.std()["GalaxyCount"]

        self.truth = pd.DataFrame({"mean": [mean], "std": [std]})

    def put_bin_counts_in_matrix(self):
        if self.dimensions == 2:
            self.bin_counts = self.add_missing_cells_2d(self.bin_counts)
            self.bin_counts_obs = self.add_missing_cells_2d(self.bin_counts_obs)

            self.matrix_bin_counts = self.get_matrix_in_2d(
                self.bin_counts, name="GalaxyCount"
            )
            self.matrix_bin_counts_obs = self.get_matrix_in_2d(
                self.bin_counts_obs, name="GalaxyCountObserved"
            )

            self.box_shape_d = list(self.matrix_bin_counts.values.shape)

        elif self.dimensions == 3:
            self.matrix_bin_counts = self.add_missing_cells_and_get_matrix_in_3d(
                df=self.bin_counts,
                name="GalaxyCount",
                bin_counts_vars=self.bin_counts_vars,
                bin_counts_vars_labels=self.bin_counts_vars_labels,
            )
            self.matrix_bin_counts_obs = self.add_missing_cells_and_get_matrix_in_3d(
                df=self.bin_counts_obs,
                name="GalaxyCountObserved",
                bin_counts_vars=self.bin_counts_vars_obs,
                bin_counts_vars_labels=self.bin_counts_vars_labels_obs,
            )

            self.box_shape_d = [
                self.matrix_bin_counts.index.size,
                self.matrix_bin_counts.columns.levels[0].size,
                self.matrix_bin_counts.columns.levels[1].size,
            ]

    def add_missing_cells_2d(self, df):
        df = df.reset_index(drop=True)
        df.set_index(self.coordinate_bins, inplace=True)

        all_Y_bins = list(range(self.num_bins["Y"]))
        all_Z_bins = list(range(self.num_bins["Z"]))

        idx = pd.MultiIndex.from_product(
            [all_Y_bins, all_Z_bins], names=self.coordinate_bins
        )
        df = df.reindex(index=idx, fill_value=0)

        return df.reset_index()

    def get_matrix_in_2d(self, df, name):
        """
        Fill in missing columns and reshape the bin count such that it is in the shape of the number of bins in each
        dimension.

        Parameters
        ----------
        name (str):
            The name of the counts, either GalaxyCount or GalaxyCountObserved

        Returns
        -------
        The df but reshaped in (pixels Y, pixels Z)
        """

        if self.dimensions != 2:
            raise "Error, this only applies in two dimensions. "

        matrix = df.pivot(index="Y_bin", columns="Z_bin", values=name)

        return matrix

    def add_missing_cells_and_get_matrix_in_3d(
        self, df, name, bin_counts_vars, bin_counts_vars_labels
    ):
        """
        Fill in missing columns and reshape the bin count such that it is in the shape of the number of bins in each
        dimension.

        Parameters
        ----------
        name (str):
            The name of the counts, either GalaxyCount or GalaxyCountObserved

        Returns
        -------
        The df but reshaped in (pixels X, pixels Y, pixels Z)
        """

        if self.dimensions != 3:
            raise "Error, this only applies in three dimensions. "

        matrix = df.pivot_table(
            index=bin_counts_vars_labels[0],
            columns=bin_counts_vars_labels[1:],
            values=name,
            fill_value=0,
        )

        # Get the complete set of X_bin, Y_bin, and Z_bin (or z_bin)
        bins = {}
        for key in self.num_bins.keys():
            bins[key] = list(range(self.num_bins[key]))

        # Reindex to ensure all bins are present
        matrix = matrix.reindex(
            columns=pd.MultiIndex.from_product(
                [bins["Y"], bins[bin_counts_vars[2]]], names=bin_counts_vars_labels[1:]
            ),
            fill_value=0,
        )
        matrix = matrix.reindex(index=bins["X"], fill_value=0)

        return matrix

    def compute_luminosity_distance_of_bins(self):
        if self.geometry == "toy":
            self.bins_luminosity_distance = self.cosmology.luminosity_distance(
                self.bins_z
            ).value
            self.bins_z_sub_box = self.bins_z.reshape((-1, self.NUMBER_SUBBINS_PER_BIN))
            self.bins_luminosity_distance = self.bins_luminosity_distance.reshape(
                (-1, self.NUMBER_SUBBINS_PER_BIN)
            )

    def compute_bookkeeping_vars(self):
        if self.geometry == "toy":
            # for book keeping
            self.bins_luminosity_distance_median = (
                self.catalog.groupby([self.bin_counts_vars_labels[-1]])[
                    ["luminosity_distance"]
                ]
                .median()
                .values[:, 0]
            )
            self.bins_z_median = (
                self.catalog.groupby([self.bin_counts_vars_labels[-1]])[["z"]]
                .median()
                .values[:, 0]
            )
            self.bins_comoving_distance_median = (
                self.catalog.groupby([self.bin_counts_vars_labels[-1]])[
                    ["comoving_distance"]
                ]
                .median()
                .values[:, 0]
            )

    def compute_stats_luminosity_function(self):
        """
        This method computes some summary statistics, but is not used for data or comparisons.

        """

        self.stats_df = (
            self.catalog.groupby("M")["M"]
            .agg("count")
            .pipe(pd.DataFrame)
            .rename(columns={"M": "frequency"})
        )
        self.stats_df["pdf"] = self.stats_df["frequency"] / sum(
            self.stats_df["frequency"]
        )
        self.stats_df["cdf"] = self.stats_df["pdf"].cumsum()
        self.stats_df = self.stats_df.reset_index()

        self.cdf_from_absolute_magnitude = scipy.interpolate.interp1d(
            self.stats_df["M"],
            self.stats_df["cdf"],
            kind="linear",
            fill_value=(0, 1),
            bounds_error=False,
        )

        abs_magnitude_threshold = abs_magnitude_from_apparent_magnitude(
            self.m_threshold, self.bins_luminosity_distance
        )

        cdf_from_M_val_interp = get_cdf_schechter_interpolation()

        if self.sample_from_schechter:
            pdet = cdf_from_M_val_interp(abs_magnitude_threshold)
        else:
            pdet = self.cdf_from_absolute_magnitude(abs_magnitude_threshold)

        abs_magnitude_threshold = np.mean(abs_magnitude_threshold, axis=1)
        pdet = np.mean(pdet, axis=1)

        # only computed for comparison
        pdet_from_median = self.cdf_from_absolute_magnitude(
            abs_magnitude_from_apparent_magnitude(
                self.m_threshold, self.bins_luminosity_distance_median
            )
        )

        self.p_det_for_luminsity_bins = {
            "z": self.bins_z_median,
            "comoving_distance": self.bins_comoving_distance_median,
            "pdet": pdet,
            "luminosity_distance": self.bins_luminosity_distance_median,
            "pdet_from_median": pdet_from_median,
            "abs_magnitude_threshold": abs_magnitude_threshold,
        }

        self.p_det_for_luminsity_bins = pd.DataFrame(self.p_det_for_luminsity_bins)
        self.z_information = pd.DataFrame(self.z_information)

    def compute_fraction_galaxies_never_observed(self):
        M_faintest = np.max(self.catalog["M"][self.idx_selected])
        idx_bright = self.catalog["M"] > M_faintest

        self.fraction_never_observed = np.sum(idx_bright == 1) / len(self.catalog)
        self.faintest_absolute_magnitude = M_faintest

    def plot_matrix_bins(self, observed=True):
        if self.dimensions == 2:
            if observed:
                matrix = self.matrix_bin_counts_obs
            else:
                matrix = self.matrix_bin_counts
        else:
            print("Plotting the first X slice")

            if observed:
                matrix = self.matrix_bin_counts_obs.iloc[0].values.reshape(
                    self.box_shape_d[1:]
                )
            else:
                matrix = self.matrix_bin_counts.iloc[0].values.reshape(
                    self.box_shape_d[1:]
                )

        # Plotting the color matrix
        plt.figure(figsize=(14, 14))
        sns.heatmap(matrix, annot=True, fmt="g", cmap="viridis")
        plt.title("Galaxy Count Matrix")
        plt.show()

    def save_matrices(self):
        file_name = DATA_LOCATION + "/".join([self.data_folder, self.file_name])

        self.matrix_bin_counts.to_csv(file_name, index=True)

        file_name_obs = file_name + "_obs"
        self.matrix_bin_counts_obs.to_csv(file_name_obs, index=True)

        file_name_abs_magnitudes = file_name + "_abs_magnitudes"
        self.catalog[["M"]].to_csv(file_name_abs_magnitudes, index=True)

        file_name_abs_magnitudes = file_name + "_abs_magnitudes_obs"
        self.catalog_obs[["M"]].to_csv(file_name_abs_magnitudes, index=True)

        file_name_abs_magnitudes = file_name + "_apparent_magnitudes_obs"
        self.catalog_obs[["m"]].to_csv(file_name_abs_magnitudes, index=True)

        file_name_abs_magnitudes = file_name + "_redshifts_obs"
        self.catalog_obs[["z_with_errors"]].to_csv(file_name_abs_magnitudes, index=True)

        file_name_z_information = file_name + "_z_information"
        (self.z_information).to_csv(file_name_z_information, index=False)

        if self.pixelation == "redshift":
            file_name_z_boundaries = file_name + "_z_boundaries"
            (self.bins_z_boundaries_df).to_csv(file_name_z_boundaries, index=False)

        additional_information = dict(
            fraction_never_observed=float(self.fraction_never_observed),
            angular_diameter_distance=float(self.angular_diameter_distance),
            number_observed_galaxies=float(self.catalog_obs.shape[0]),
            faintest_absolute_magnitude=float(self.faintest_absolute_magnitude),
        )

        settings_data = {
            "catalog_origin": self.catalog_origin,
            "data_folder": self.data_folder,
            "pixel_length_in_Mpc_comoving": self.pixel_length_in_Mpc_comoving,
            "m_threshold": self.m_threshold,
            "geometry": self.geometry,
            "debug": self.debug,
            "zmax": self.zmax,
            "catalog_file_name_original": self.catalog_file_name_original,
            "Z_offset": self.Z_offset,
            "sample_from_schechter": self.sample_from_schechter,
            "pixelation": self.pixelation,
            "redshift_num_bins": self.redshift_num_bins,
            "box_size_d": self.box_size_d,
            "catalog_file_name": self.file_name,
            "box_shape_d": self.box_shape_d,
            "galaxies_percentage": self.galaxies_percentage,
            "box_shape_d": self.box_shape_d,
            "M_cut_high": self.M_cut_high,
            "angle_box_0": float(self.angle_galaxy_0),
            "angle_box_1": float(self.angle_galaxy_1),
            "additional_information": additional_information,
            "special_deep_pixels": self.special_deep_pixels,
        }

        file_name_settings = file_name + "_settings"
        with open(file_name_settings, "w") as yaml_file:
            yaml.dump(settings_data, yaml_file)

    def sanity_check(self):
        idx_non_zero = self.catalog["comoving_distance"] != 0
        should_be_NULL = self.catalog["luminosity_distance"][
            idx_non_zero
        ] / self.catalog["comoving_distance"][idx_non_zero] - (
            self.catalog["z"][idx_non_zero] + 1
        )

        if self.debug:
            print("Maximum deviation {:.2g}".format(np.max(abs(should_be_NULL))))

        assert np.all(abs(should_be_NULL) < 1e-6)
        assert np.all(self.catalog["luminosity_distance"][~idx_non_zero] == 0)
        print("Passed sanity check. ")

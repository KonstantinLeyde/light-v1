from matplotlib.colors import LinearSegmentedColormap

my_gradient = [
    # Edit this gradient at https://eltos.github.io/gradient/#0:000000-22:4A298F-44.4:0648AC-73.4:13C521-87:A7E60E-99.8:E7FF00
    [0.000, "#000000"],
    [0.220, "#4A298F"],
    [0.444, "#0648AC"],
    [0.734, "#13C521"],
    [0.870, "#A7E60E"],
    [0.998, "#E7FF00"],
    [1.000, "#E7FF00"],
]

# Extract positions and colors from the gradient
positions, colors = zip(*my_gradient)

# Create the custom colormap
custom_cmap = LinearSegmentedColormap.from_list(
    "my_custom_cmap", list(zip(positions, colors))
)


def get_NUMBER_SUBBINS_PER_BIN(pixel_length_in_Mpc_comoving):
    # dependent on how many Z pixels there are
    L_Z = pixel_length_in_Mpc_comoving[2]

    if L_Z >= 50:
        NUMBER_SUBBINS_PER_BIN = 50
    elif L_Z >= 25:
        NUMBER_SUBBINS_PER_BIN = 50
    elif L_Z >= 12.5:
        NUMBER_SUBBINS_PER_BIN = 50
    elif L_Z >= 10:
        NUMBER_SUBBINS_PER_BIN = 20
    elif L_Z >= 5:
        NUMBER_SUBBINS_PER_BIN = 10
    else:
        NUMBER_SUBBINS_PER_BIN = 10

    return NUMBER_SUBBINS_PER_BIN


def get_RightTruncatedPoissonHigh_from_n(analysis):
    pixel_volume = float(analysis.pixel_volume)
    galaxies_percentage = analysis.kwargs_catalog["galaxies_percentage"]

    # read from the 250^3 box with full completion
    calibration_peak = 5000
    pixel_volume_calibration = (250 / 50) ** 3

    RightTruncatedPoissonHigh = (
        calibration_peak * galaxies_percentage * pixel_volume / pixel_volume_calibration
    )

    print(
        "Fixing the upper bound of the Poisson distribution to {:.2f}".format(
            RightTruncatedPoissonHigh
        )
    )
    return int(RightTruncatedPoissonHigh)


def get_H0_OMEGA_M_REF():
    return 73, 0.25


def get_dict_cosmo_ref_parameters():
    H0, Omega_M = get_H0_OMEGA_M_REF()
    return dict(H0=H0, Omega_M=Omega_M)


def get_omega_b_REF_omega_cdm_REF():
    H0, Omega_M = get_H0_OMEGA_M_REF()
    h = H0 / 100

    Omega_b = 0.045
    Omega_cdm = Omega_M - Omega_b

    return Omega_b * h**2, Omega_cdm * h**2


def complete_cosmopower_dict(params_power_spectrum, params_cosmology, z):
    rel_keys = ["H0", "omega_cdm", "omega_b"]
    dict_cosmo = {k: v for k, v in params_cosmology.items() if k in rel_keys}
    dict_cosmo["h"] = dict_cosmo["H0"] / 100

    dict_out = dict(z=z, **params_power_spectrum, **dict_cosmo)

    return dict_out


def get_Omega_m(params_cosmology):
    if "get_Omega_m" in params_cosmology.keys():
        return params_cosmology["get_Omega_m"]
    else:
        omega_cdm = params_cosmology["omega_cdm"]
        omega_b = params_cosmology["omega_b"]
        h = params_cosmology["H0"] / 100

        return (omega_cdm + omega_b) / h**2


def add_catalog_name(kwargs_catalog):
    p = kwargs_catalog["galaxies_percentage"]
    L = kwargs_catalog["box_size_d"][0]

    if kwargs_catalog["catalog_origin"] == "millenium":
        if L == 500:
            return f"millenium_bertone2007a_{p}.csv"
        elif L == 150:
            return f"millenium_bertone2007a_{p:.1g}_{L}Mpc.csv"
        else:
            raise NotImplementedError


def get_original_catalog_file_name(catalog_origin, galaxies_percentage, box_size):
    if galaxies_percentage == 1:
        galaxies_percentage = int(galaxies_percentage)

    return f"{catalog_origin}_{galaxies_percentage}_{box_size}Mpc.csv"


def get_catalog_properties_from_case(case):
    """
    We follow the convention: Z_offset, fraction_galaxies, number_pizels, box_size, magnitude_limit
    """

    if case == 0:
        return 250, 0.01, 20, 150, 21
    elif case == 1:
        return 100, 0.01, 60, 500, 21
    elif case == 2:
        return 100, 0.001, 60, 500, 21
    elif case == 3:
        return 100, 0.01, 40, 500, 21
    elif case == 4:
        return 300, 1, 50, 250, 21
    elif case == 5:
        return 100, 1, 70, 250, 21
    elif case == 6:
        # same as 4 but deeper survey
        return 300, 1, 50, 250, 23
    elif case == 7:
        # same as 4 but less fine
        return 300, 1, 20, 250, 21
    elif case == 8:
        # same as 4 but less fine
        return 300, 1, 100, 250, 21
    elif case == 9:
        # same as 4 but very deep survey
        return 300, 1, 50, 250, 28
    elif case == 10:
        # same as 1 but almost no offset
        return 1, 0.01, 60, 500, 21
    elif case == 11:
        # same as 1 but almost no offset
        return 150, 1, 50, 150, 21
    elif case == 12:
        # same as 0 but 10% galaxies
        return 250, 0.1, 20, 150, 21
    elif case == 13:
        # same as 0 but 100% galaxies
        return 250, 1, 20, 150, 21
    elif case == 14:
        # same as 0 but 100% galaxies and finer pixelation
        return 250, 1, 40, 150, 21
    elif case == 15:
        # same as 0 but 100% galaxies, finer pixelation, larger box
        return 250, 1, 40, 250, 21
    elif case == 16:
        # same as 14 but further away and larger
        return 1000, 1, 40, 250, 21
    elif case == 17:
        return 300, 1, 50, 250, 17
    elif case == 18:
        # same as 14 but 100% galaxies and less fine pixelation
        return 250, 1, 20, 150, 21
    elif case == 19:
        # same as 17, but less fine
        return 300, 1, 20, 250, 17
    elif case == 20:
        # same as 14 but 100% galaxies and less fine pixelation
        return 250, 1, 10, 150, 21
    elif case == 21:
        # same as 17, but less fine
        return 300, 1, 10, 250, 17
    elif case == 22:
        # similar to 0, but far far away
        return 1500, 0.01, 20, 150, 23


def get_dense_parameters(power_spectrum_model):
    params_magnitude = [
        "eps",
        "eps_2",
        "f_faint",
        "f_mu",
        "mu",
        "pre_a0",
        "pre_a1",
        "pre_a2",
        "pre_a3",
        "pre_a4",
        "pre_a5",
        "pre_b0",
        "pre_b1",
        "pre_b2",
        "pre_b3",
        "pre_b4",
        "pre_b5",
        "sigma",
    ]
    params_cosmology = ["H0", "omega_b", "omega_cdm"]
    params_galaxies = ["ng_bar_tilde"]

    if power_spectrum_model == "analytical-phenom":
        params_power_spectrum = ["A_s", "n_s"]
    elif power_spectrum_model == "smooth-turnover":
        params_power_spectrum = ["A_s", "alpha_1_s", "alpha_2_s", "k_turn"]
    elif power_spectrum_model == "cosmopower":
        params_power_spectrum = [
            "ln10^{10}A_s",
            "cmin",
            "eta_0",
        ]
    else:
        raise NotImplementedError

    # From numpyro, see
    # https://num.pyro.ai/en/latest/tutorials/bad_posterior_geometry.html
    return params_magnitude + params_cosmology + params_galaxies + params_power_spectrum

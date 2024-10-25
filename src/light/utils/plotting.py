import corner
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..astrophysics import luminosity as luminosity
from . import conventions, helper_functions


class ShiftedLogNorm(LogNorm):
    def __init__(self, vmin=None, vmax=None, clip=True, epsilon=1e-5, value_epsilon=0):
        """
        A custom LogNorm class that shifts all data by a small epsilon to avoid log(0).

        Parameters:
        - vmin: Minimum data value for normalization.
        - vmax: Maximum data value for normalization.
        - clip: If True, data values outside [vmin, vmax] are clipped.
        - epsilon: The small value to shift all data by, defaults to 1e-5.
        """
        self.epsilon = epsilon
        self.value_epsilon = value_epsilon

        if vmin is not None:
            vmin = max(vmin, epsilon)  # Ensure vmin is at least epsilon

        super().__init__(vmin=vmin, vmax=vmax, clip=clip)

    def __call__(self, value, clip=None):
        """
        Shift the input data by epsilon before applying LogNorm.
        """
        shifted_value = value + self.value_epsilon
        return super().__call__(shifted_value, clip)


def determine_symmetric_vmin_vmax(x):
    vmin, vmax = jnp.percentile(x, 0), jnp.percentile(x, 100)
    vmax = max(vmax, -vmin)
    return dict(vmin=-vmax, vmax=vmax)


def plot_imshows_side_by_side(
    images,
    titles=None,
    cmap_list="viridis",
    figsize=(12, 7),
    c_range=dict(vmin=None, vmax=None),
    n_plots_per_row=3,
    scale_list=None,
):
    """
    Plot multiple images using imshow side by side.

    Parameters
    ----------
    images : list of numpy.ndarray
        List of images to be plotted.
    titles : list of str, optional
        List of titles for each subplot, default is None.
    cmap : str, optional
        Colormap to be used, default is 'viridis'.
    figsize : tuple, optional
        Figure size, default is (12, 4).

    Returns
    -------
    None
    """

    if type(c_range) == dict:
        c_range_list = [c_range] * len(images)
    else:
        c_range_list = c_range

    if type(cmap_list) == str:
        cmap_list = [cmap_list] * len(images)

    num_plots = len(images)

    if titles is None:
        titles = [""] * num_plots

    n_plots_x = (num_plots + n_plots_per_row - 1) // n_plots_per_row
    n_plots_fields = int(n_plots_x * n_plots_per_row)
    n_plots_remainder = n_plots_fields - num_plots

    fig, axes = plt.subplots(n_plots_x, n_plots_per_row, figsize=figsize)

    for ii, (scale, cmap) in enumerate(zip(scale_list, cmap_list)):
        ix, iy = ii // n_plots_per_row, ii % n_plots_per_row

        # treat extra case for 1d subplots
        if n_plots_x == 1:
            idx = iy
        else:
            idx = (ix, iy)

        if scale == "log":
            norm = ShiftedLogNorm(**c_range_list[ii])
            kwargs = dict(norm=norm)
        else:
            norm = None
            kwargs = dict(norm=norm, **c_range_list[ii])

        im = axes[idx].imshow(
            images[ii], cmap=cmap, extent=(-1 / 2, 1 / 2, -1 / 2, 1 / 2), **kwargs
        )
        axes[idx].set_title(titles[ii])
        axes[idx].axis("off")

        # Add colorbar
        divider = make_axes_locatable(axes[idx])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, orientation="vertical")

    # delete the remaining axes, if any
    if n_plots_remainder != 0:
        for ii in range(n_plots_fields - 1, num_plots - 1, -1):
            ix, iy = ii // n_plots_per_row, ii % n_plots_per_row

            if n_plots_x == 1:
                idx = iy
            else:
                idx = (ix, iy)

            fig.delaxes(axes[idx])

    return fig, axes


# not tested yet
def plot_plots_side_by_side(
    x_values,
    y_values,
    titles=None,
    colors="b",
    markers=None,
    linestyle="-",
    figsize=(12, 7),
    n_plots_per_row=3,
):
    """
    Plot multiple line plots side by side.

    Parameters
    ----------
    x_values : list of numpy.ndarray
        List of x-values for each plot.
    y_values : list of numpy.ndarray
        List of y-values for each plot.
    titles : list of str, optional
        List of titles for each subplot, default is None.
    colors : str or list of str, optional
        Color(s) for the line plots, default is 'b' (blue).
    markers : str or list of str, optional
        Marker(s) for the line plots, default is None.
    linestyle : str, optional
        Line style for the line plots, default is '-' (solid line).
    figsize : tuple, optional
        Figure size, default is (12, 7).
    n_plots_per_row : int, optional
        Number of plots to display in each row, default is 3.

    Returns
    -------
    None
    """

    # Ensure colors and markers are lists
    if not isinstance(colors, list):
        colors = [colors] * len(x_values)
    if not isinstance(markers, list):
        markers = [markers] * len(x_values)

    num_plots = len(x_values)

    if titles is None:
        titles = [""] * num_plots

    n_plots_x = (num_plots + n_plots_per_row - 1) // n_plots_per_row
    n_plots_fields = int(n_plots_x * n_plots_per_row)
    n_plots_remainder = n_plots_fields - num_plots

    fig, axes = plt.subplots(n_plots_x, n_plots_per_row, figsize=figsize)

    for ii in range(num_plots):
        ix, iy = ii // n_plots_per_row, ii % n_plots_per_row

        # treat extra case for 1d subplots
        if n_plots_x == 1:
            idx = iy
        else:
            idx = (ix, iy)

        axes[idx].plot(
            x_values[ii],
            y_values[ii],
            color=colors[ii],
            marker=markers[ii],
            linestyle=linestyle,
        )
        axes[idx].set_title(titles[ii])

    # delete the remaining axes, if any
    if n_plots_remainder != 0:
        for ii in range(n_plots_fields - 1, num_plots - 1, -1):
            ix, iy = ii // n_plots_per_row, ii % n_plots_per_row

            if n_plots_x == 1:
                idx = iy
            else:
                idx = (ix, iy)

            fig.delaxes(axes[idx])

    return fig, axes


latex_labels_dict = {
    "A_s": r"$\mathcal{A}_{\rm PS}$",
    "n_s": r"$n_{\rm PS}$",
    "alpha_s": r"$\alpha_{\rm PS}$",
    "ng_bar": r"$\bar n_{\rm g}$",
}


def get_latex_labels(list_params):
    return [latex_labels_dict[param] for param in list_params]


def make_corner_plot(
    analysis,
    path_name=None,
    corner_kwargs=dict(fill_contours=True, plot_datapoints=False, plot_density=False),
):
    # get all 1d samples
    samples_df = analysis.hypersamples_df

    samples_df = helper_functions.filter_columns_with_dynamic_range(samples_df)

    fig = corner.corner(samples_df, **corner_kwargs)
    if path_name != None:
        plt.savefig(path_name)


def make_summary_plot(
    field,
    idx_X,
    path_name=None,
    plot_kwargs={},
    cmap=conventions.custom_cmap,
    cmap_errors="PuOr_r",
):
    truth_box, data_box, _, _, _, _, _ = field.get_data_as_array()
    # magnitude_model = field.magnitude_model

    density_contrast_DM = field.hypersamples_d["density_contrast_DM"]
    density_g = field.hypersamples_d["density_g"]
    counts_galaxies_true = field.hypersamples_d["counts_galaxies_true"]

    density_contrast_DM_plus_1_mean = jnp.mean(density_contrast_DM, 0) + 1

    statistics_density_g = {}
    statistics_density_g["mean"] = jnp.mean(density_g, 0)
    statistics_density_g["std"] = jnp.std(density_g, 0)

    statistics_number_counts = {}
    statistics_number_counts["mean"] = jnp.mean(counts_galaxies_true, 0)
    statistics_number_counts["std"] = jnp.std(counts_galaxies_true, 0)

    # deviations_relative_mean_density_g = (truth_box - statistics_density_g['mean']) / statistics_density_g['std']
    ratio_std_over_mean_density_g = (
        statistics_density_g["std"] / statistics_density_g["mean"]
    )

    deviations_absolute_mean_number_counts = (
        truth_box - statistics_number_counts["mean"]
    )
    deviations_relative_mean_number_counts = (
        deviations_absolute_mean_number_counts / statistics_number_counts["std"]
    )
    ratio_std_over_mean_number_counts = (
        statistics_number_counts["std"] / statistics_number_counts["mean"]
    )

    images_list = [
        data_box[idx_X],
        truth_box[idx_X],
        statistics_number_counts["mean"][idx_X],
        #
        statistics_density_g["mean"][idx_X],
        statistics_number_counts["std"][idx_X],
        ratio_std_over_mean_density_g[idx_X],
        #
        density_contrast_DM_plus_1_mean[idx_X],
        deviations_absolute_mean_number_counts[idx_X],
        deviations_relative_mean_number_counts[idx_X],
    ]
    titles = [
        r"Observed galaxy count, $n_{{\rm c}, I}$",
        "Truth" + r", $n_{{\rm g}, I}^{\rm True}$",
        "Reconstructed number counts\n(Mean)" + r", $n_{{\rm g}, I}$",
        #
        "Reconstructed galaxy density\n(Mean)" + r", $\mu_{I}$",
        "Reconstructed number counts\n(Std)",
        "Std / Mean of  galaxy density",
        #
        "Reconstructed DM density constrast\n(Mean)" + r", $\delta_{{\rm DM}, I} + 1$",
        "Deviations number counts (mean)\nfrom truth" + r", $\Delta_{I}$",
        "Deviations number counts (mean) from\ntruth in units of stds"
        + r", $\Delta_{{\rm rel},I}$",
    ]

    # the epsilon is for plotting purposes, since we have a log color scale
    c_range_field = dict(
        vmin=0, vmax=jnp.percentile(truth_box[idx_X], 100), epsilon=1e-1
    )
    c_range_None = dict(vmin=None, vmax=None)

    c_range_Delta = determine_symmetric_vmin_vmax(
        deviations_absolute_mean_number_counts[idx_X]
    )
    c_range_Delta_rel = determine_symmetric_vmin_vmax(
        deviations_relative_mean_number_counts[idx_X]
    )

    c_range_list = [
        c_range_field,
        c_range_field,
        c_range_field,
        #
        c_range_None,
        c_range_None,
        c_range_None,
        #
        c_range_None,
        c_range_Delta,
        c_range_Delta_rel,
    ]

    scale_list = ["log", "log", "log", "log", "log", "log", "log", None, None]

    cmap_list = [
        cmap,
        cmap,
        cmap,
        #
        cmap,
        cmap,
        cmap,
        #
        cmap,
        cmap_errors,
        cmap_errors,
    ]

    fig, axes = plot_imshows_side_by_side(
        images_list,
        titles=titles,
        **plot_kwargs,
        c_range=c_range_list,
        cmap_list=cmap_list,
        scale_list=scale_list,
    )

    if path_name != None:
        plt.tight_layout()
        plt.savefig(path_name, dpi=240)


def make_magnitude_data(
    analysis, number_samples, use_normalized_magnitudes, number_bins=200
):
    if use_normalized_magnitudes:
        M_vals_normalized = np.linspace(-30, -3, number_bins)
    else:
        M_vals = np.linspace(-30, -3, number_bins)

    params = analysis.hypersamples
    probs = np.zeros((number_samples, number_bins))

    for i in range(number_samples):
        params = helper_functions.get_ith_entry_of_array(analysis.hypersamples, i)

        if "f_mu" not in params.keys():
            params["f_mu"] = luminosity.get_mu_faintest(analysis)

        magnitude_params_keys = (
            analysis.magnitude_model.list_all_params
            + ["mu_M_NF", "sigma_M_NF", "M_hat_cut", "eps", "eps_2", "mu", "sigma"]
            + ["H0"]
        )

        params_magnitude_distribution = {
            k: v for k, v in params.items() if k in magnitude_params_keys
        }

        dist_magnitudes = (
            analysis.magnitude_model.construct_numpyro_magnitude_distribution(
                params_magnitude_distribution,
                H0_REF=analysis.kwargs_cosmology["cosmo_ref_parameters"]["H0"],
            )
        )

        if use_normalized_magnitudes:
            M_vals = analysis.magnitude_model.transformation_M_hat_to_magnitude(
                M_vals_normalized
            )
        else:
            M_vals_normalized = (
                analysis.magnitude_model.transformation_M_hat_to_magnitude.inv(M_vals)
            )

        probs[i, :] = jnp.exp(dist_magnitudes.log_prob(M_vals_normalized))

    probs = np.where(np.isnan(probs), 0, probs)

    return M_vals, M_vals_normalized, probs


def make_magnitude_plot(
    analysis,
    path_name=None,
    fig_kwargs={"figsize": (14, 6)},
    plot_kwargs={"alpha": 0.01},
    label_kwargs={"fontsize": 18},
    plot_maximum=False,
    nb_plot_lines_drawn=None,
    include_log_plot=True,
    set_kwargs={"title": None},
    plot_bundles=False,
    add_rescaled_plot_to_M=None,
    use_normalized_magnitudes=False,
):
    """
    Produces the magnitude plot.

    Parameters
    ----------

    analysis:
        Results object that collects all properties of the analysis
    path_name: str
        Name where file is saved. If not given, it is not saved.
    fig_kwargs: dict
        Kwargs of the figure


    add_rescaled_plot_to_M: int
        If given, another median is added, s.t. it is rescaled to the given bin.

    """

    if nb_plot_lines_drawn == None:
        nb_plot_lines_drawn = analysis.hypersamples_df.shape[0]

    M_vals, M_vals_normalized, probs = make_magnitude_data(
        analysis,
        number_samples=nb_plot_lines_drawn,
        use_normalized_magnitudes=use_normalized_magnitudes,
    )

    if use_normalized_magnitudes:
        x_vals = M_vals_normalized
    else:
        x_vals = M_vals

    truth_array, data_array, M_list, M_obs_list, m_list, z_list, _ = (
        analysis.get_data_as_array()
    )

    numerical_array = analysis.hypersamples["H0"]
    normalized_array = (numerical_array - np.percentile(numerical_array, 1)) / (
        np.percentile(numerical_array, 99) - np.percentile(numerical_array, 1)
    )
    color_array = plt.cm.gnuplot2(normalized_array)

    if include_log_plot:
        fig, axes = plt.subplots(1, 2, **fig_kwargs)
    else:
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
        axes = [ax]

    # plot the non-normalized histograms
    hist_kwargs_nd = {"histtype": "step", "density": False, "lw": 2}

    idx = 0
    hist_vals, bins, _ = axes[idx].hist(
        np.array(M_list), bins="auto", **hist_kwargs_nd, alpha=0
    )  # this is only to get the bins
    max_magnitudes_plot = np.max(hist_vals) * 1.3
    norm = len(M_list) * (bins[1] - bins[0])

    if plot_bundles:
        for i in range(nb_plot_lines_drawn):
            axes[idx].plot(
                x_vals,
                np.array(probs[i, :] * norm),
                color=color_array[i],
                **plot_kwargs,
            )
    axes[idx].hist(np.array(M_list), bins=bins, **hist_kwargs_nd, label="All")
    axes[idx].hist(np.array(M_obs_list), **hist_kwargs_nd, bins=bins, label="Observed")

    # plot contours
    if not plot_bundles:
        alpha = 0.1
        if add_rescaled_plot_to_M == None:
            percentiles_not_norm = jnp.percentile(
                norm * probs, np.array([4.55, 31.37, 50, 68.27, 95.45]), axis=0
            ).T

            axes[idx].fill_between(
                x_vals,
                percentiles_not_norm[:, 0],
                percentiles_not_norm[:, 4],
                alpha=alpha,
                color="blue",
            )
            axes[idx].fill_between(
                x_vals,
                percentiles_not_norm[:, 1],
                percentiles_not_norm[:, 3],
                alpha=alpha,
                color="blue",
            )
            axes[idx].plot(
                x_vals,
                percentiles_not_norm[:, 2],
                color="blue",
                lw=1,
                alpha=0.8,
                label="Reconstructed\n(median)",
            )

        else:
            # find closted magnitude value to that bin
            idx_bin = np.argmin(np.abs(bins - add_rescaled_plot_to_M))
            idx_M_vals = np.argmin(np.abs(x_vals - add_rescaled_plot_to_M))

            rescale_factor = hist_vals[idx_bin] / np.median(probs[:, idx_M_vals])
            percentiles_rescaled = jnp.percentile(
                rescale_factor * probs,
                np.array([4.55, 31.37, 50, 68.27, 95.45]),
                axis=0,
            ).T

            axes[idx].fill_between(
                x_vals,
                percentiles_rescaled[:, 0],
                percentiles_rescaled[:, 4],
                alpha=alpha,
                color="blue",
            )
            axes[idx].fill_between(
                x_vals,
                percentiles_rescaled[:, 1],
                percentiles_rescaled[:, 3],
                alpha=alpha,
                color="blue",
            )
            axes[idx].plot(
                x_vals,
                percentiles_rescaled[:, 2],
                color="blue",
                lw=1,
                alpha=0.8,
                label="Reconstructed\n(median), rescaled",
            )

    if use_normalized_magnitudes:
        label_x = r"$\tilde M$"
        label_y = r"$\mathbb{P}(\tilde M)$"
    else:
        label_x = r"$M$"
        label_y = r"$\mathbb{P}(M)$"

    if set_kwargs["title"] == None:
        axes[idx].set_title("Observation vs fitting")
    axes[idx].set_xlabel(label_x, **label_kwargs)
    axes[idx].set_ylabel(label_y, **label_kwargs)
    axes[idx].set_ylim(0, max_magnitudes_plot)
    axes[idx].legend(**label_kwargs)
    axes[idx].text(
        0.05,
        -0.08,
        "Bright end",
        horizontalalignment="left",
        verticalalignment="top",
        transform=axes[idx].transAxes,
        alpha=0.7,
        **label_kwargs,
    )
    axes[idx].text(
        0.9,
        -0.08,
        "Faint end",
        horizontalalignment="center",
        verticalalignment="top",
        transform=axes[idx].transAxes,
        alpha=0.7,
        **label_kwargs,
    )

    if include_log_plot:
        idx = 1
        norm = len(M_list) * (bins[1] - bins[0])

        if plot_bundles:
            for i in range(nb_plot_lines_drawn):
                axes[idx].plot(
                    x_vals,
                    np.array(probs[i, :] * norm),
                    color=color_array[i],
                    **plot_kwargs,
                )
        axes[idx].hist(
            np.array(M_list),
            bins=bins,
            **hist_kwargs_nd,
            label="All magnitudes",
            color="C1",
        )
        axes[idx].hist(
            np.array(M_obs_list),
            bins=bins,
            **hist_kwargs_nd,
            label="Observed magnitudes",
            color="C2",
        )

        # make contour plot
        axes[idx].fill_between(
            x_vals,
            percentiles_not_norm[:, 0],
            percentiles_not_norm[:, 4],
            alpha=alpha,
            color="blue",
        )
        axes[idx].fill_between(
            x_vals,
            percentiles_not_norm[:, 1],
            percentiles_not_norm[:, 3],
            alpha=alpha,
            color="blue",
        )
        axes[idx].plot(
            x_vals, percentiles_not_norm[:, 2], color="blue", lw=1, alpha=0.8
        )

        if set_kwargs["title"] == None:
            axes[idx].set_title("Observation vs fitting (log)")
        axes[idx].set_yscale("log")
        axes[idx].set_xlabel(label_x, **label_kwargs)
        axes[idx].set_ylabel(label_y, **label_kwargs)
        axes[idx].set_ylim(1, max_magnitudes_plot)
        axes[idx].legend(**label_kwargs)

    if plot_maximum:
        for ax in axes:
            ax.axvspan(
                np.max(M_obs_list) - 1 / 2,
                np.max(M_obs_list) + 1 / 2,
                color="grey",
                alpha=0.3,
            )

    if path_name != None:
        fig.savefig(path_name)


def make_deviation_plot(
    analysis,
    plot_kwargs={},
    path_name=None,
    include_absolute_deviations=True,
    bins=None,
):
    statistics_number_counts = analysis.get_deviations()
    statistics_number_counts = clean_statistics_if_nan_inf_or_large(
        statistics_number_counts
    )

    all_deviations_abs = statistics_number_counts["deviations_absolute"].flatten()
    deviations_abs_perc_low = int(np.percentile(all_deviations_abs, 1))
    deviations_abs_perc_high = int(np.percentile(all_deviations_abs, 99))

    hist_kwargs = dict(histtype="step")

    if bins == None:
        hist_kwargs["bins"] = "auto"
    else:
        hist_kwargs["bins"] = bins

    if include_absolute_deviations:
        fig, axes = plt.subplots(1, 2, **plot_kwargs)
    else:
        fig, ax = plt.subplots(1, 1, **plot_kwargs)
        axes = [ax]

    sigma_vals = np.linspace(-5, 5, 100)
    gaussian_vals = np.exp(-(sigma_vals**2) / 2) / np.sqrt(2 * np.pi)

    idx = 0
    axes[idx].hist(
        np.array(statistics_number_counts["deviations_relative"].flatten()),
        density=True,
        **hist_kwargs,
        label="Relative deviations",
    )
    axes[idx].set_yscale("log")
    axes[idx].set_ylim(1e-4, 3)
    axes[idx].plot(sigma_vals, gaussian_vals, label="Gaussian deviations")
    axes[idx].legend()
    axes[idx].set_xlabel(r"$\Delta_{\rm rel}$", fontsize=20)
    axes[idx].set_ylabel(r"#$\Delta_{\rm rel}$", fontsize=20)

    if include_absolute_deviations:
        idx = 1
        axes[idx].hist(
            np.array(statistics_number_counts["deviations_absolute"].flatten()),
            density=True,
            histtype="step",
            bins="auto",
        )
        axes[idx].set_yscale("log")
        axes[idx].set_ylim(1e-4, 2)
        axes[idx].set_xlabel(r"$\Delta$", fontsize=20)
        axes[idx].set_ylabel(r"#$\Delta$", fontsize=20)

    if path_name != None:
        fig.savefig(path_name)


def clean_statistics_if_nan_inf_or_large(statistics_number_counts, threshold_val=100):
    replace_val = 0
    # correct if some of them are None
    if np.isnan(statistics_number_counts["deviations_relative"]).sum() != 0:
        idx = np.isnan(statistics_number_counts["deviations_relative"])
        statistics_number_counts["deviations_relative"] = (
            statistics_number_counts["deviations_relative"].at[idx].set(replace_val)
        )

        n_replaced = jnp.sum(idx == 1)
        print(
            "Replaced {} relative deviations since they were nan. ".format(n_replaced)
        )
        print(
            "The replace values had absolute deviations of: {}".format(
                statistics_number_counts["deviations_absolute"][idx]
            )
        )

    if np.isinf(statistics_number_counts["deviations_relative"]).sum() != 0:
        idx = np.isinf(statistics_number_counts["deviations_relative"])
        statistics_number_counts["deviations_relative"] = (
            statistics_number_counts["deviations_relative"].at[idx].set(replace_val)
        )

        n_replaced = jnp.sum(idx == 1)
        print(
            "Replaced {} relative deviations since they were inf. ".format(n_replaced)
        )
        print(statistics_number_counts["deviations_absolute"][idx])

    if np.any(abs(statistics_number_counts["deviations_relative"]) > threshold_val):
        idx = abs(statistics_number_counts["deviations_relative"]) > threshold_val
        statistics_number_counts["deviations_relative"] = (
            statistics_number_counts["deviations_relative"].at[idx].set(replace_val)
        )

        n_replaced = jnp.sum(idx == 1)
        print(
            "Replaced {} relative deviations since they were very large ({}). ".format(
                n_replaced, threshold_val
            )
        )
        print(statistics_number_counts["deviations_absolute"][idx])

    return statistics_number_counts

import light.utils.plotting as plotting
import light.utils.result as result

yaml_file_path = './PATH_TO_YOUR_RESULTS/'

analysis = result.Analysis.from_yaml(yaml_file_path=yaml_file_path)
analysis.load_numpyro_result(skip_diverging_chains=True)

plotting.make_corner_plot(
    analysis,
    path_name=analysis.get_results_folder() + 'plot_corner.png'
)

plotting.make_summary_plot(
    analysis, 
    idx_X=25, 
    path_name=analysis.get_results_folder() + f'plot_summary_{analysis.id_job}.png', 
    plot_kwargs={'figsize': (12,12)},
)

plotting.make_magnitude_plot(
    analysis, 
    path_name=analysis.get_results_folder() + 'plot_magnitudes_reconstructed.png', 
    fig_kwargs={'figsize': (9,7)},
    plot_maximum=True,
    set_kwargs={'title': ''},
    include_log_plot=False, 
)

plotting.make_deviation_plot(
    analysis, 
    path_name=analysis.get_results_folder() + 'plot_deviations.png', 
    plot_kwargs={'figsize': (15,7)},
)
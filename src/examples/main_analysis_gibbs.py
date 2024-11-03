import copy
import numpy as np

from matplotlib import pyplot as plt

import jax
jax.config.update('jax_enable_x64', True)

import numpyro
numpyro.enable_x64()

from numpyro.infer import MCMC, NUTS, autoguide
from numpyro.handlers import seed, trace
import optax

import arviz as az

import light.numpyro_utils.priors as priors
import light.astrophysics.luminosity as luminosity
import light.utils.result as result
import light.utils.plotting as plotting
import light.utils.helper_functions as helper_functions

start_from_subresult_i = 0

if(start_from_subresult_i!=0):
    first_reloaded_run = True
else:
    first_reloaded_run = False

if(first_reloaded_run):
    args = result.get_settings_from_parse_command_line(mode='default')
    analysis = result.Analysis.from_yaml(id_job=args['id_job'])
    analysis.setup_default_priors()
else:
    analysis = result.get_settings_class_from_parse_command_line()
    analysis.create_result_folder()
    analysis.setup_default_priors()
    analysis.save_settings_as_yaml()

print(analysis)

magnitude_model = luminosity.MagnitudeDistribution(**analysis.kwargs_magnitude_model)
data_kwargs = analysis.get_data_kwargs()
model = analysis.get_model()

use_svi_initialization = (analysis.kwargs_sampler['num_svi_samples'] != 0)

if(use_svi_initialization):

    init_fn = numpyro.infer.init_to_sample()
    guide = autoguide.AutoLowRankMultivariateNormal(model)

    max_iterations = analysis.kwargs_sampler['num_svi_samples']
    scheduler = optax.exponential_decay(
        init_value=0.01,
        decay_rate=0.99,
        transition_steps=100
    )

    optim = optax.adabelief(learning_rate=scheduler)
    loss = numpyro.infer.TraceMeanField_ELBO()
    svi_normal = numpyro.infer.SVI(model, guide, optim, loss)

    rng_key = jax.random.PRNGKey(6)
    rng_key, rng_key_ = jax.random.split(rng_key)

    svi_normal_result = svi_normal.run(
        rng_key_, 
        max_iterations,
        analysis=analysis,
        **data_kwargs, 
        prior_dict=analysis.kwargs_sampler['prior'],
        prior_settings=analysis.kwargs_sampler['prior_settings'],
        sample_number_counts=False,
        progress_bar=True, 
        stable_update=True
    )

    # loss plot with zoom inset for last 1/3
    fig, ax = plt.subplots(figsize=(15, 3.5))
    ax.plot(svi_normal_result.losses)
    ax.set_yscale('asinh')

    axins = ax.inset_axes([0.3, 0.5, 0.64, 0.45])
    N_end = max_iterations // 3
    x_plot = np.linspace(max_iterations - N_end, max_iterations, N_end)
    axins.plot(x_plot, svi_normal_result.losses[max_iterations - N_end:])
    ax.indicate_inset_zoom(axins, edgecolor='k')

    fig.savefig('./plots/svi_result.png')

    svi_median = guide.median(svi_normal_result.params)

    
num_samples = analysis.kwargs_sampler['num_posterior_samples']
num_warmup = analysis.kwargs_sampler['num_warmup']
num_posterior_samples_per_batch = analysis.kwargs_sampler['num_posterior_samples_per_batch']
num_batches = num_samples // num_posterior_samples_per_batch

if use_svi_initialization:
    init_strategy = numpyro.infer.initialization.init_to_value(values=svi_median)  
else:
    init_strategy = numpyro.infer.initialization.init_to_sample

# to extract the 1d sample parameters
exec_trace = trace(seed(model,jax.random.PRNGKey(0))).get_trace(
    **data_kwargs,
    analysis=analysis,
    prior_dict=analysis.kwargs_sampler['prior'],
    prior_settings=analysis.kwargs_sampler['prior_settings'],
    sample_number_counts=False,
)
sample_vars = [key for key, value in exec_trace.items() if exec_trace[key]['type'] == 'sample']

dense_mass_list, params_list = priors.seperate_gibbs_params(
    analysis, 
    sample_vars,
    mode=analysis.kwargs_sampler['gibbs_mode']
)
inner_kernels = [
    NUTS(
        model, 
        init_strategy=init_strategy, 
        max_tree_depth=analysis.kwargs_sampler['max_tree_depth'], 
        dense_mass=dm, 
        target_accept_prob=analysis.kwargs_sampler['target_accept_prob']
    ) for dm in dense_mass_list
    ]

mcmc = MCMC(
    inner_kernels[0], 
    num_warmup=num_warmup, 
    num_samples=num_posterior_samples_per_batch, 
    progress_bar=True, 
    num_chains=analysis.kwargs_sampler['num_chains'],
    chain_method='vectorized',
)

for i in range(start_from_subresult_i, analysis.nb_subresults):
    print(f"Batch {i+1} / {num_batches}")

    if(i==0):
        random_key = jax.random.PRNGKey(0)
    else:
        if(first_reloaded_run):
            print('Starting from a pre-loaded run. ')

            # if we start from a reloaded run
            last_state = helper_functions.last_state_read(analysis.get_results_folder() + f'./preliminary/mcmc_last_state.pkl')

            mcmc._last_state = jax.device_put(last_state)
            mcmc._warmup_state = jax.device_put(last_state)
            mcmc.post_warmup_state = mcmc.last_state

            random_key = mcmc.post_warmup_state.rng_key[0,0]

            first_reloaded_run = False
        else:
            random_key = mcmc.post_warmup_state.rng_key

    print('Starting running MCMC')
    # run MCMC for num_posterior_samples_per_batch samples
    mcmc.run(
        random_key,
        analysis=analysis,
        **data_kwargs,
        prior_dict=analysis.kwargs_sampler['prior'],
        prior_settings=analysis.kwargs_sampler['prior_settings'],
        sample_number_counts=False,
    )
    print('Ran chain. ')

    last_state = jax.device_get(mcmc.last_state)
    helper_functions.last_state_write(last_state, analysis.get_results_folder() + f'./preliminary/mcmc_last_state.pkl' )
    
    # to reduce memory usage, might be obsolete with the next numyro release
    mcmc._states = jax.device_get(mcmc._states)
    mcmc._states_flat = jax.device_get(mcmc._states_flat)

    mcmc.post_warmup_state = mcmc.last_state
    
    idata = az.from_numpyro(mcmc)
    print(f'#divergences: {idata.sample_stats.diverging.values.sum()}')

    # save result
    file_path = analysis.get_sub_results_file_name(i)
    az.to_netcdf(idata, file_path)

    # make preliminary plot
    preliminary_analysis = copy.copy(analysis)
    preliminary_analysis.load_numpyro_result(idxs=[i], skip_tensors=False)

    try:
        plotting.make_magnitude_plot(
            preliminary_analysis, 
            path_name=analysis.get_results_folder() + './preliminary/plot_magnitudes_reconstructed.png', 
            fig_kwargs={'figsize': (13,6.5)},
            plot_maximum=True,
            nb_plot_lines_drawn=analysis.kwargs_sampler['num_posterior_samples_per_batch']
        )
        plotting.make_corner_plot(
            preliminary_analysis,
            path_name=analysis.get_results_folder() + './preliminary/plot_corner.png'
        )
    except:
        pass

prior = numpyro.infer.Predictive(model, num_samples=250)(
    jax.random.PRNGKey(2), 
    analysis=analysis,
    **data_kwargs,
    prior_dict=analysis.kwargs_sampler['prior'],
    prior_settings=analysis.kwargs_sampler['prior_settings'],
    sample_number_counts=False,
)
idata_prior = az.from_numpyro(
    prior=prior
)
print(idata_prior.prior)
file_path = analysis.get_results_folder() + '/prior.av'
az.to_netcdf(idata_prior, file_path)



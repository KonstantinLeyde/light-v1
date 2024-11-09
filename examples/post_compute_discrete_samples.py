import sys
import numpy as np
import os

import jax
jax.config.update('jax_enable_x64', True)

import arviz as az
import numpyro

import light.utils.result as result
import light.utils.helper_functions as helper_functions

yaml_file_path = './PATH_TO_YOUR_RESULTS/'

analysis = result.Analysis.from_yaml(yaml_file_path=yaml_file_path)
analysis.setup_default_priors()
model = analysis.get_model()

data_kwargs = analysis.get_data_kwargs()

for idx in range(analysis.nb_subresults):

    analysis.load_inf_data(idxs=[idx])

    data_dict = {}
    data = analysis.inf_data.posterior
    
    try:
        del analysis.inf_data.posterior_predictive
    except:
        pass

    # Iterate over data variables and their values
    for var_name, var_data in data.data_vars.items():
        # Convert data variable to numpy array if it's numeric
        if np.issubdtype(var_data.dtype, np.number):
            data_dict[var_name] = np.asarray(var_data)

    return_sites = ['counts_galaxies_true']
    predictive = numpyro.infer.Predictive(model, data_dict, infer_discrete=True, return_sites=return_sites, batch_ndims=2)
    discrete_samples = predictive(
        jax.random.PRNGKey(0), 
        analysis=analysis, 
        **data_kwargs, 
        prior_dict=analysis.kwargs_sampler['prior'],
        prior_settings=analysis.kwargs_sampler['prior_settings'],
    )

    return_sites = ['density_g', 'density_contrast_DM']
    predictive = numpyro.infer.Predictive(model, data_dict, infer_discrete=False, return_sites=return_sites, batch_ndims=2)
    densities = predictive(
        jax.random.PRNGKey(0),
        analysis=analysis,
        **data_kwargs,
        infer_density_contrast_DM=True,
        infer_density_g=True,
        prior_dict=analysis.kwargs_sampler['prior'],
        prior_settings=analysis.kwargs_sampler['prior_settings'],
    )

    posterior_predictive = {**discrete_samples, **densities}
    posterior_predictive = helper_functions.flatten_dict_along_chain_dim(posterior_predictive)
    inf_data_predictive = az.from_numpyro(posterior_predictive=posterior_predictive, num_chains=analysis.kwargs_sampler['num_chains'])
    idata = az.concat(analysis.inf_data, inf_data_predictive, dim=None)

    # save result
    file_path = analysis.get_sub_results_file_name(idx) + '_temp'
    az.to_netcdf(idata, file_path)

    # Rename the file using the terminal command because of saving issues
    # with arviz of modified projects
    os.system("mv {0} {1}".format(file_path, analysis.get_sub_results_file_name(idx)))


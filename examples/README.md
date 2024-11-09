# Examples

## Producing the mock data

### `01_make_redshift_catalog.ipynb `

The first jupyter notebook produces the input data for the later analysis. You will either have to

- provide a pandas dataframe with the galaxy catalog you want to analyze, or
- load the galaxy catalog from the millennium simulation. For this, you will need to provide a username and a password, since this catalog is not publically available.

### `02_make_launch_file.ipynb `

The second script prepares a yaml file that includes all assumptions that are taken during the inferece. You will have to provide the path to which you want to save the results. This is the `results_location` variable. Please also include the data location where you have save the catalog files (computed in the jupyter notebook `01_make_redshift_catalog.ipynb`) - this variables is called `data_location`.

## Running the HMC

After you have run the two notebooks, please change directory in the `examples` folder.
You can start the analysis with

`python main_analysis_gibbs.py --id_job YOUR_JOB_ID_HERE --init_file PATH_TO_LAUNCH_SCRIPT/launch.yaml`

The `id_job` should be a unique identifier so that you can analyze the same data under different assumptions.

## Post-processing

### Produce discrete posterior samples

Change the path to the folder that contains your results (called `yaml_file_path`) in `post_compute_discrete_samples.py`. Then run

`python post_compute_discrete_samples.py`

### Plotting

Again, change the path to the folder that contains your results (called `yaml_file_path`) in `post_plotting.py`.

To make relevant plots, run

`python post_plotting.py`.

The plots can be found in the same directory as the result files.

## Running on a cluster with slurm

If you are running on a cluster with slurm, you can also modify the `job_slurm.sh`, including the partition you want to run on. The job can then be launched via

` sbatch job_slurm.sh`

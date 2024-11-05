# LIGHT

**L**arge-scale structure for the **I**nference of the **G**alaxy field with **H**amiltonian **M**onte-**C**arlo

**LIGHT** is a Bayesian framework for reconstructing the galaxy field and underlying dark matter field using Hamiltonian Monte Carlo (HMC), and more specifically [NumPyro](https://num.pyro.ai/en/stable/). This package is designed to improve the inference of cosmological parameters with gravitational waves and galaxy catalogs, a.k.a. the dark siren method.

---

## Installation Instructions

### Basic Installation

To create a new environment and install **LIGHT**, follow these steps:

```bash
conda create -n light-env python=3.11
conda activate light-env
pip install -e .
```

## Running

For examples, please first run the two notebooks in the examples folder.

### `01_make_redshift_catalog.ipynb `

The first jupyter notebook produces the input data for the later analysis. You will either have to

- provide a pandas dataframe with the galaxy catalog you want to analyze, or
- load the galaxy catalog from the millennium simulation. For this, you will need to provide a username and a password, since this catalog is not publically available.

### `02_make_launch_file.ipynb `

The second script prepares a yaml file that includes all assumptions that are taken during the inferece. You will have to provide the path to which you want to save the results. This is the `results_location` variable. Please also include the data location where you have save the catalog files (computed in the jupyter notebook `01_make_redshift_catalog.ipynb`) - this variables is called `data_location`.

### Running the HMC

After you have run the two notebooks, please change directory in the `examples` folder.
You can start the analysis with

`python main_analysis_gibbs.py --id_job YOUR_JOB_ID_HERE --init_file PATH_TO_LAUNCH_SCRIPT/launch.yaml`

The `id_job` should be a unique identifier so that you can analyze the same data under different assumptions.

### Running on a cluster with slurm

If you are running on a cluster with slurm, you can also modify the `job_slurm.sh`, including the partition you want to run on. The job can then be launched via

` sbatch job_slurm.sh`

## Citation

If you use this software, please cite:

[Cosmic Cartography: Bayesian reconstruction of the galaxy density informed by large-scale structure](https://arxiv.org/pdf/2409.20531)

```bibtex
@article{Leyde:2024tov,
    author = "Leyde, Konstantin and Baker, Tessa and Enzi, Wolfgang",
    title = "{Cosmic Cartography: Bayesian reconstruction of the galaxy density informed by large-scale structure}",
    eprint = "2409.20531",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "9",
    year = "2024"
}
```

## Contact

If you have any questions, feedback, or would like to discuss this project, please don't hesitate to reach out:

Email: Konstantin.Leyde@gmail.com

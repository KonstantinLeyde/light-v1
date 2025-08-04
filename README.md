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

For examples, please navigate to the examples folder, and follow the readme there. notebooks in the examples folder.

## Citation

If you use this software, please cite:

[Cosmic Cartography: Bayesian reconstruction of the galaxy density informed by large-scale structure](https://arxiv.org/pdf/2409.20531)

```bibtex
@article{Leyde:2024tov,
    author = "Leyde, Konstantin and Baker, Tessa and Enzi, Wolfgang",
    title = "{Cosmic cartography: Bayesian reconstruction of the galaxy density informed by large-scale structure}",
    eprint = "2409.20531",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1088/1475-7516/2024/12/013",
    journal = "JCAP",
    volume = "12",
    pages = "013",
    year = "2024"
}
```

## Contact

If you have any questions, feedback, or would like to discuss this project, please don't hesitate to reach out:

Email: Konstantin.Leyde@gmail.com

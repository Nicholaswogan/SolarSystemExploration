# Reproducing Wogan et al. (2025)

Version 1.0.0

This repository reproduces most of the main text calculations in Wogan et al. (2025). The article formally publishes the [Photochem](https://github.com/Nicholaswogan/photochem) software package, and benchmarks the model against the observed composition and climates of many of the Solar System atmospheres and one exoplanet (WASP-39b). To reproduce our calculations, follow these instructions:

## Step 1: Installation and setup

If you do not have Anaconda on your system, install it here or in any way you perfer: https://www.anaconda.com/download. Next, run the following code to cerate a conda environment `solarsystem` with `photochem` v0.6.7 installed.

```sh
conda create -n solarsystem -c conda-forge photochem=0.6.7 matplotlib
```

## Step 2: Run the code

To do all calculations, and reproduce many of the Figures in the paper, run the `main.py` script:

```sh
conda activate solarsystem
python main.py
```

Once completed, the resulting figures are in the `figures/` directory.



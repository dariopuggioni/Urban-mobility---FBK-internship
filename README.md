# Urban-mobility---FBK-internship

This repository contains an implementation of the model presented in the paper *"Modeling the Polycentric Transition of Cities"* by Rémi Louf and Marc Barthélemy (https://doi.org/10.1103/PhysRevLett.111.198702).

The model simulates the growth of a city by sequentially adding workers and assigning each of them to the activity center that maximizes the utility function

\[
Z_{ij} = \eta_j - \frac{d_{ij}}{l}\left[1+\left(\frac{T(j)}{c}\right)^\mu\right],
\]

where the congestion term depends on the current traffic attracted by each center. As the population increases, the model reproduces the transition from a monocentric to a polycentric urban structure.

## Repository structure

- **functions.py** contains the implementation of the simulation algorithm together with the visualization utilities.
- **Simulation_calling_module.ipynb** shows how to run the simulation, choose the model parameters, and reproduce the figures.

The computationally intensive parts of the algorithm have been implemented using **Numba** (`@njit` and parallelization where appropriate) to speed up the simulation.

## Sampling methods

The original model assumes that new workers are uniformly distributed over the city area before choosing their workplace.

In addition to this original implementation, I introduced an alternative sampling strategy that can be selected through the `sampling` parameter:

- `sampling="uniform"`: workers are sampled uniformly inside the city disk (original model).
- `sampling="clark"`: workers are sampled from a Kernel Density Estimate built as a sum of exponential Clark profiles centered on the urban centers. The kernel radius is chosen as the typical distance between neighboring centers.

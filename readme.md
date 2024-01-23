# Simulation-based inference for simple RL models

This repository contains code for parameter estimation using simulation-based inference. 

It primarily wraps functions from the [SBI package](https://www.mackelab.org/sbi/), implementing neural posterior estimation (NPE) and neural likelihood estimation (NLE) in a way that is straightforward to use for simple RL models.

## Installation

To install this package, clone the repository and run `pip install -e .` from the root directory. It can then be used as a regular Python package, e.g. `from simulation_based_inference.npe import NPE`.

## Examples

An example notebook is provided in the `notebooks` directory. 
# Uncertainty in Deep Learning Assessment, Hillary Term 2024

The following is the code used for the UDL assessment for Candidate #1080462

## Acknowledgements / What is Original vs Taken from Online Sources

The base repo used for this project was https://github.com/NixGD/variational-continual-learning.

They provided the overall repo structure, code for the PermutedMNIST dataset used, as well as reference implementations for the VCL training and models.

I expanded on the repo by:
1. Fully implementing their provided model code for VCL in "Pytorch style". As stated in their repo: "We have two implementations of a Discriminative VCL model: in models.vcl_nn, and in models.contrib. The former was our first implementation, has been more thoroughly tested, and was used to obtain our experimental data; the latter is our attempt at a cleaner implementation more similar to the PyTorch standard module style.". In order to get the latter working, multiple bugs (including loss calculations/normalizations, test vs train sampling, numerical stability issues etc.) had to be fixed. Cleaning up this implementation made it a lot easier to extend to the other novel contributions in my assessment such as the different variational layers.
2. Adding Bayesian Hypernetworks, kTied Normal Distributions, and the novel combination of Bayesian Hypernetworks and GMFVI. This required adding new variational layers, changing the model's code for the KL calculation to support Monte Carlo Estimation, and adding normalizing flow models.
3. Adding a Hydra-based configuration system. Previously, the config arguments were stored in the python experiment files, which made overriding experiments in the command line and tracking results with different hyperparameters difficult.
4. Adding the PermutedMNIST dataset.
5. Adding wandb logging, as well as logging more granular details such as the different KL term losses at every step. This helped a lot for the Bayesian hypernetwork experiments that were very sensitive to the KL loss weighting.
6. Adding plotting code.

## Replicating Results from the Paper

All the commands used to produce the paper results are in `sweeps/paper.py`. These results can then be used to produce the plots in the paper in `results.ipynb`.

## Using the code

The main entry point for experiments is `src/main.py`. Experiments can be started by running `python main.py CLI_ARGS`. Where `CLI_ARGS` are key value mapping of the form `key1=value1` following standard Hydra syntax. The `config/base.yaml` directory contains all of the options that can be configured. See `sweeps/paper.py` for all the exact command lines to reproduce the paper results.

The repository structure for notable files is as follows:
- src
    - main.py -> entry point for experiments
    - sweeps
        - paper.py -> contains all commands to reproduce experiments
    - notebooks
        - plot.ipynb -> code to produce plots from experiment logs
    - models
        - contrib.py -> main model code, fixed from old repo to work with Python-style module code
        - coreset.py -> code to train and predict with coresets
        - vcl_nn.py -> old main model code, in more rigid/less-extensible syntax
    - layers
        - variational.py -> contains code for all weight layers including GMF, BHMF, kTied posteriors, and KL calculation (closed-form and Monte Carlo)
    - experiments
        - discriminative.py -> instantiation code for experiments
    - util
        - datasets.py -> dataset code
        - experiment_utils.py -> prediction and training loops

## Installation

We assume a conda environment with pytorch 1.13.1 and python 3.7.16 installed. The rest of the dependencies can be installed by running:
```
pip install pandas matplotlib hydra-core wandb torchvision tqdm tensorboardX graphviz nflows einops
```
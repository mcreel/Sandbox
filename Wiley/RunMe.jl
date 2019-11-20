using Pkg
Pkg.activate(".")
# generate the training data and the Monte Carlo design replications
#include("MakeData.jl")

# train the net using the training data (not the design)
#include("Train.jl")

# do full statistic MSM Bayesian estimation
include("MSM_MCMC_full_stat.jl");
#
# do NN statistic MSM Bayesian estimation
#include("MSM_MCMC_NN_stat.jl");



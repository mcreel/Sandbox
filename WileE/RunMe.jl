function RunProject()

# generate the raw training data
#include("MakeData.jl")

# transform the raw statistics, and split out params and stats
#include("Transform.jl")
## when this is done, can delete raw_data.bson

# train the net using the transformed training/testing data
#include("Train.jl")
# when this is done, can delete cooked_data.bson

# do full statistic MSM Bayesian estimation
#@time include("MSM_MCMC_full_stat.jl");
#
# do NN statistic MSM Bayesian estimation
@time include("MSM_MCMC_NN_stat.jl");

end
RunProject()

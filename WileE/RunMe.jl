function RunProject()

# generate a draw at true params
m = WileE_model(θtrue)    
@show m

# generate the raw training data
#include("MakeData.jl")

# transform the raw statistics, and split out params and stats
#include("Transform.jl")
## when this is done, can delete raw_data.bson

# train the net using the transformed training/testing data
#include("Train.jl")
# when this is done, can delete cooked_data.bson

# do full statistic MSM Bayesian estimation
include("MSM_MCMC_raw.jl")
@time MSM_MCMC_raw(m)

# do NN statistic MSM Bayesian estimation
include("MSM_MCMC_NN.jl")
@time MSM_MCMC_NN(m)

end
RunProject()

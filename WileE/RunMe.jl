include("MakeData.jl")
include("Transform.jl")
include("Train.jl")
include("Analyze.jl")
include("MCMC.jl")

function RunProject()

# generate the raw training data
MakeData()

# transform the raw statistics, and split out params and stats
Transform()
## when this is done, can delete raw_data.bson

# train the net using the transformed training/testing data
Train()
# when this is done, can delete cooked_data.bson


mcreps = 100
results_raw = zeros(mcreps,12)
results_NN = zeros(mcreps,12)
for mcrep = 1:mcreps
    # generate a draw at true params
    m = WileE_model(θtrue)    
    # do full statistic MSM Bayesian estimation
    chain = MCMC(m, false)
    results_raw[mcrep,:] = Analyze(chain)
    # do NN statistic MSM Bayesian estimation
    chain = MCMC(m, true)
    results_NN[mcrep,:] = Analyze(chain)
    println("__________ replication: ", mcrep, "_______________")
    println("Results so far, raw stat")
    dstats(results_raw[1:mcrep,:])
    println("Results so far, NN stat")
    dstats(results_NN[1:mcrep,:])
    println("____________________________")
end
writedlm("results_raw", results_raw)
writedlm("results_NN", results_NN)
end
RunProject()

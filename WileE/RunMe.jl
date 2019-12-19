include("MakeData.jl")
include("Transform.jl")
include("Train.jl")
include("Analyze.jl")
include("MCMC.jl")

function RunProject()

# generate the raw training data
MakeData()

# transform the raw statistics, and split out params and stats
info = Transform()
# when this is done, can delete raw_data.bson

# train the net using the transformed training/testing data
Train()
# when this is done, can delete cooked_data.bson

results_raw = zeros(mcreps,15)
results_NN = zeros(mcreps,15)
for mcrep = 1:mcreps
    # generate a draw at true params
    m = WileE_model(θtrue)    
    # do full statistic MSM Bayesian estimation
    chain = MCMC(m, false, info)
    results_raw[mcrep,:] = Analyze(chain)
    # do NN statistic MSM Bayesian estimation
    chain = MCMC(m, true, info)
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

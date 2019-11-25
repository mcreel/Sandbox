# run Initiate (only one time!) before this. If this give a world age
# error message, just run it again.
include("Analyze.jl")

function RunProject()

# generate the raw training data
include("MakeData.jl")

# transform the raw statistics, and split out params and stats
include("Transform.jl")
## when this is done, can delete raw_data.bson

# train the net using the transformed training/testing data
include("Train.jl")

# when this is done, can delete cooked_data.bson
include("MCMC.jl")

mcreps = 1000
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
    println("____________________________")
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

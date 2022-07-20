using SimulatedNeuralMoments, DelimitedFiles
using Flux, Turing, MCMCChains, AdvancedMH
using StatsPlots, DelimitedFiles, LinearAlgebra
using BSON:@save
using BSON:@load

# the model-specific code
include("ABMlib.jl")
function main()

reps = 100
results = zeros(reps, 13)
# setting for sampling
names = ["a", "b", "σf"]
S = 20
covreps = 500
length = 250
nchains = 4
burnin = 50
tuning = 3.0 # increase to lower acceptance rate

# fill in the structure that defines the model
lb, ub = PriorSupport() # bounds of support
model = SNMmodel("ABM example", lb, ub, InSupport, PriorDraw, auxstat)

# train the net, and save it and the transformation info
transf = bijector(@Prior) # transforms draws from prior to draws from  ℛⁿ 
transformed_prior = transformed(@Prior, transf) # the transformed prior
@load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net

@model function MSM(m, S, model)
    θt ~ transformed_prior
    if !InSupport(invlink(@Prior, θt))
        Turing.@addlogprob! -Inf
        return
    end
    # sample from the model, at the trial parameter value, and compute statistics
    mbar, Σ = mΣ(invlink(@Prior,θt), S, model, nnmodel, nninfo)
    m ~ MvNormal(mbar, Symmetric(Σ))
end

# draw a sample at the design parameters, or use an existing data set
for rep = 1:reps
    y = ABMmodel(TrueParameters(), rand(1:Int64(1e12))) # draw a sample of 500 obsns. at design parameters
    m = NeuralMoments(auxstat(y), nnmodel, nninfo)
    θhat = invlink(@Prior, m)
    junk, Σp = mΣ(θhat, covreps, model, nnmodel, nninfo)
    chain = sample(MSM(m, S, model),
        MH(:θt => AdvancedMH.RandomWalkProposal(MvNormal(zeros(size(m,1)), tuning*Σp))),
        MCMCThreads(), length, nchains; init_params=Iterators.repeated(m), discard_initial=burnin)
    # transform back to original domain
    chain = Array(chain)
    acceptance = size(unique(chain[:,1]),1)[1] / size(chain,1)
    println("acceptance rate: $acceptance")
    Threads.@threads for i = 1:size(chain,1)
        chain[i,:] = invlink(@Prior, chain[i,:])
    end
    chain = Chains(chain, names)
    display(chain)
    q = quantile(chain)
    ql = q.nt[2] # 2.5% quantile
    qu = q.nt[end] # 97.5% quantile
    qm = q.nt[4] # median
    results[rep,:] = vcat(θhat, qm, ql, qu, acceptance)
end
end
results = main()
dlmwrite("results.txt", results)


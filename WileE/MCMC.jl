# This does MCMC, either using raw statistic, or using NN transform,
# depending on the argument usenn
using SV, Flux, Econometrics, LinearAlgebra, Statistics, DelimitedFiles
using BSON:@load
include("lib/transform.jl")
include("lib/lnL.jl")
global const info = readdlm("transformation_info")

function MCMC(m, usenn)
    # get the trained net
    @load "best.bson" model
    S = nSimulationDraws # number of simulations
    if usenn
        m = transform(m', info)
        m = model(m').data
        m = Float64.(m)
    end    
    # set up MCMC
    tuning = [0.01, 0.01, 0.01] # fix this somehow
    if usenn
        θinit = m # use the NN fit as initial θ
        lnL = θ -> LL(θ, m, S, model)
    else          # use the prior mean as initial θ 
        # use a rapid SAMIN to get good initialization values for chain
        θinit = (ub+lb)./2.0
        lnL = θ -> LL(θ, m, S)
        obj = θ -> -1.0*lnL(θ)
        θinit, junk, junk, junk = samin(obj, θinit, lb, ub; coverage_ok=1, maxevals=1000, ns = 10, verbosity = 0, rt = 0.25)
    end
    Prior = θ -> prior(θ, lb, ub) # uniform, doesn't matter
    # define things for MCMC
    verbosity = false
    ChainLength = 500
    Proposal = θ -> proposal1(θ, tuning, lb, ub)
    chain = mcmc(θinit, ChainLength, burnin, Prior, lnL, Proposal, verbosity)
    # now use a MVN random walk proposal with updates of covariance and longer chain
    # on final loop
    θinit = mean(chain[:,1:3],dims=1)[:]
    Σ = NeweyWest(chain[:,1:3])
    tuning = 1.0
    ChainLength = 500
    MC_loops = 4
    for j = 1:MC_loops
        P = try
            P = (cholesky(Σ)).U
        catch
            P = diagm(diag(Σ))
        end    
        Proposal = θ -> proposal2(θ,tuning*P, lb, ub)
        if j == MC_loops
            ChainLength = 2000
        end    
        θinit = mean(chain[:,1:3],dims=1)[:]
        chain = mcmc(θinit, ChainLength, 0, Prior, lnL, Proposal, verbosity)
        if j < MC_loops
            accept = mean(chain[:,end])
            if accept > 0.35
                tuning *= 2.0
            elseif accept < 0.25
                tuning *= 0.25
            end
            Σ = NeweyWest(chain[:,1:3])
        end    
    end
    # plain MCMC fit
    chain = chain[:,1:3]
    return chain
end

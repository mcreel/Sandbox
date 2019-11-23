# This does MCMC, but filtering the statistics through the trained net,
# to reduce dimension to the minimum needed for identification
using SV, Flux, Econometrics, LinearAlgebra, Statistics, DelimitedFiles
using BSON:@load
include("lib/transform.jl")
global const info = readdlm("transformation_info")

# specialized likelihood for MCMC using net
function logL_NN(θ, m, n, burnin, S, model, withdet=true)
    k = size(m,1)
    ms = zeros(S, k)
    Threads.@threads for s = 1:S
        ms[s,:] = Float64.(model(transform(WileE_model(θ[:])', info)').data)
    end
    mbar = mean(ms,dims=1)[:]
    Σ = cov(ms)
    x = (m .- mbar)[:]
    lnL = -0.5*log(det(Σ)) - 0.5*x'*inv(Σ)*x # for Bayesian
    #=
    if ~any(isnan.(mbar))
        Σ = cov(ms)
        x = (m .- mbar)[:]
        lnL = try
            if withdet
                lnL = -0.5*log(det(Σ)) - 0.5*x'*inv(Σ)*x # for Bayesian
            else    
                lnL = 0.5*x'*inv(Σ)*x # for classic indirect inference (note sign change)
            end    
        catch
            lnL = -Inf
        end
     else
         lnL = -Inf
     end
     =#
     return lnL
end

function MSM_MCMC_NN(m)
    # get the trained net
    @load "best.bson" model
    S = nSimulationDraws # number of simulations
    m = transform(m', info)
    m = model(m').data
    m = Float64.(m)
    # set up MCMC
    tuning = [0.01, 0.01, 0.01] # fix this somehow
    θinit = m # use the NN fit as initial θ
    lnL = θ -> logL_NN(θ, m, n, S, burnin, model)
    Prior = θ -> prior(θ, lb, ub) # uniform, doesn't matter
    # define things for MCMC
    verbosity = false
    ChainLength = 800
    Proposal = θ -> proposal1(θ, tuning, lb, ub)
    chain = mcmc(θinit, ChainLength, burnin, Prior, lnL, Proposal, verbosity)
    # now use a MVN random walk proposal with updates of covariance and longer chain
    # on final loop
    Σ = NeweyWest(chain[:,1:3])
    tuning = 1.0
    ChainLength = 800
    MC_loops = 5
    for j = 1:MC_loops
        P = (cholesky(Σ)).U
        Proposal = θ -> proposal2(θ,tuning*P, lb, ub)
        if j == MC_loops
            ChainLength = 1600
        end    
        chain = mcmc(θinit, ChainLength, 0, Prior, lnL, Proposal, verbosity)
        if j < MC_loops
            accept = mean(chain[:,end])
            if accept > 0.35
                tuning *= 2.0
            elseif accept < 0.25
                tuning *= 0.25
            end
            Σ = 0.5*Σ + 0.5*NeweyWest(chain[:,1:3])
        end    
    end
    # plain MCMC fit
    chain = chain[:,1:3]
    return chain
end

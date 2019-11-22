using SV, Flux, Econometrics, LinearAlgebra, Statistics
using BSON:@load
function main()
    # these are the true params that generated the data
    S = nSimulationDraws # number of simulations
    # generate statistic at the true param value
    m = WileE_model(θtrue)
    # set up MCMC
    shocks_u = randn(n+burnin,S) # fixed shocks for simulations
    shocks_e = randn(n+burnin,S) # fixed shocks for simulations
    tuning = [0.01, 0.01, 0.01] # fix this somehow
    θinit = (ub+lb)./2.0
    lnL = θ -> logL(θ, m, n, shocks_u, shocks_e, true)
    Prior = θ -> prior(θ, lb, ub) # uniform, doesn't matter
    # define things for MCMC
    verbosity = false
    # initial fast chain to tune covariance
    ChainLength = 800
    Proposal = θ -> proposal1(θ, tuning, lb, ub)
    chain = mcmc(θinit, ChainLength, burnin, Prior, lnL, Proposal, verbosity)
    # now use a MVN random walk proposal with updates of covariance and longer chain
    # on final loop
    Σ = NeweyWest(chain[:,1:3])
    tuning = 0.5
    ChainLength = 800
    MC_loops = 5
    for j = 1:MC_loops
        P = (cholesky(Σ)).U
        Proposal = θ -> proposal2(θ,tuning*P, lb, ub)
        θinit = vec(mean(chain[:,1:3],dims=1))
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
    posmean = vec(mean(chain,dims=1))
    inci = zeros(3)
    for i = 1:3
        lower = quantile(chain[:,i],0.05)
        upper = quantile(chain[:,i],0.95)
        inci[i] = θtrue[i] >= lower && θtrue[i] <= upper
    end
    prettyprint([posmean inci])
    return chain
end
main();

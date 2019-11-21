# This does MCMC, but filtering the statistics through the trained net,
# to reduce dimension to the minimum needed for identification
using SV, Flux, Econometrics, LinearAlgebra, Statistics
using BSON:@load
include("Transform.jl")

# specialized likelihood for MCMC using net
function logL_NN(θ, m, n, η, ϵ, model, withdet=true)
    S = size(η,2)
    k = size(m,1)
    ms = zeros(S, k)
    Threads.@threads for s = 1:S
        y, junk = SVmodel(θ, n, η[:,s], ϵ[:,s])
        stat = sqrt(n)*aux_stat(y)
        # the following two lines apply the net to the raw stat
        transform!(stat)
        ms[s,:] = Float64.(model(stat).data)
    end
    mbar = mean(ms,dims=1)[:]
    if ~any(isnan.(mbar))
        Σ = cov(ms)
        x = (m .- mbar)
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
     return lnL
end

function main()
    # get the trained net
    @load "best.bson" model
    # get the MC design inputs
    @load "simdata.bson" statistics nDrawsFromPrior
    # these are the true params that generated the data
    σe = exp(-0.736/2.0)
    ρ = 0.9
    σu = 0.363
    θtrue = [σe, ρ, σu] # true param values, on param space
    n = 1000 # sample size
    burnin = 100
    S = 100 # number of simulations
    # get the data from the MC design, with the NN dataprep
    m = statistics[nDrawsFromPrior+1,:]
    m = Float64.(model(m).data)
    @show m
    # set up MCMC
    shocks_u = randn(n+burnin,S) # fixed shocks for simulations
    shocks_e = randn(n+burnin,S) # fixed shocks for simulations
    tuning = [0.01, 0.01, 0.01] # fix this somehow
    θinit = m # use the NN fit as initial θ
    lnL = θ -> logL_NN(θ, m, n, shocks_u, shocks_e, model)
    lb = [0.0, 0.0, 0.0]
    ub = [2.0, 0.99, 1.0]
    Prior = θ -> prior(θ, lb, ub) # uniform, doesn't matter
    # define things for MCMC
    verbosity = true
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
            Σ = 0.8*Σ + 0.2*NeweyWest(chain[:,1:3])
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
    p1 = npdensity(chain[:,1]) # example of posterior plot
    p2 = npdensity(chain[:,2]) # example of posterior plot
    p3 = npdensity(chain[:,3]) # example of posterior plot
    display(plot(p1,p2,p3))
    prettyprint([posmean inci])
    return chain
end
main();

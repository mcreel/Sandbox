using SV, Econometrics, LinearAlgebra, Statistics, DelimitedFiles
function MCMC(n, θtrue; burnin=100, S=100, verbosity=false)
    #=
    # load the example data
    y = readdlm("svdata.txt")
    y = y[:]
    m =  sqrt(n)*aux_stat(y) # generate the sample and save the data
    =#
    verbosity = false
    # or generate some new data, if you prefer
    shocks_u = randn(n+burnin)
    shocks_e = randn(n+burnin)
    y, junk = SVmodel(θtrue, n, shocks_u, shocks_e, false)
    m = sqrt(n)*aux_stat(y)
    # set up MCMC
    # use antithetic random draws (negatively correlated)
    shocks_u = randn(n+burnin,Int(S/2)) # fixed shocks for simulations
    shocks_e = randn(n+burnin,Int(S/2)) # fixed shocks for simulations
    shocks_u = [shocks_u; -1.0*shocks_u]
    shocks_e = [shocks_e; -1.0*shocks_e]
    tuning = [0.01, 0.01, 0.01] # fix this somehow
    lb = [0.0, 0.0, 0.0]
    ub = [2.0, 0.99, 1.0]
    # start value is intelligent for α, prior mean for ρ and σ
    θinit = (ub+lb)./2.0
    αinit = sqrt(mean(y.^2.0))
    αinit = max(lb[1],αinit)
    αinit = min(ub[1],αinit)
    θinit[1] = αinit
    lnL = θ -> logL(θ, m, n, shocks_u, shocks_e, true)
    obj = θ -> -1.0*lnL(θ)
    Prior = θ -> prior(θ, lb, ub) # uniform, doesn't matter
    # use a rapid SAMIN to get good initialization values for chain
    θinit, junk, junk, junk = samin(obj, θinit, lb, ub; coverage_ok=1, maxevals=1000, ns = 10, verbosity = 0, rt = 0.25)
    # define things for MCMC
    burnin = 100
    # initial fast chain to tune covariance
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
        θinit = vec(mean(chain[:,1:3],dims=1))
        if j == MC_loops
            ChainLength = 1600
        end    
        chain = mcmc(θinit, ChainLength, 0, Prior, lnL, Proposal, verbosity)
        if j < MC_loops
            accept = mean(chain[:,end])
            #println("tuning: ", tuning, "  acceptance rate: ", accept)
            if accept > 0.35
                tuning *= 1.5
            elseif accept < 0.25
                tuning *= 0.25
            end
            Σ = 0.8*Σ + 0.2*NeweyWest(chain[:,1:3])
        end    
    end
    return chain, m
end

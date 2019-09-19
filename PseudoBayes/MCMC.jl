using SV, Econometrics, LinearAlgebra, Statistics, DelimitedFiles
function MCMC(n; burnin=100, S=100, verbosity=false)
    #=
    # load the example data
    y = readdlm("svdata.txt")
    y = y[:]
    m =  sqrt(n)*aux_stat(y) # generate the sample and save the data
    =#
    verbosity = true
    # these are the true params
    α = -7.36
    ρ = 0.9
    σ = 0.363
    θtrue = [α, ρ, σ] # true param values, on param space
    # or generate some new data, if you prefer
    shocks_u = randn(n+burnin)
    shocks_e = randn(n+burnin)
    y = SVmodel(θtrue, n, shocks_u, shocks_e, false)
    m = sqrt(n)*aux_stat(y)
    # set up MCMC
    # use antithetic random draws (negatively correlated)
    shocks_u = randn(n+burnin,Int(S/2)) # fixed shocks for simulations
    shocks_e = randn(n+burnin,Int(S/2)) # fixed shocks for simulations
    shocks_u = [shocks_u; -1.0*shocks_u]
    shocks_e = [shocks_e; -1.0*shocks_e]
    tuning = [0.01, 0.01, 0.01] # fix this somehow
    lb = [-20.0, 0.0, 0.0]
    ub = [0.0, 0.99, 3.0]
    # start value is intelligent for α, prior mean for ρ and σ
    θinit = (ub+lb)./2.0
    αinit = log(mean(y.^2.0))
    θinit[1] = αinit
    @show θinit
    lnL = θ -> logL(θ, m, n, shocks_u, shocks_e, true)
    obj = θ -> -1.0*lnL(θ)
    Prior = θ -> prior(θ, lb, ub) # uniform, doesn't matter
    # use a rapid SAMIN to get good initialization values for chain
    #θinit, junk, junk, junk = samin(obj, θinit, lb, ub; coverage_ok=1, maxevals=1000, ns = 10, verbosity = 2, rt = 0.25)
    # define things for MCMC
    burnin = 500
    # initial fast chain to tune covariance
    ChainLength = 1000
    Proposal = θ -> proposal1(θ, tuning, lb, ub)
    chain = mcmc(θinit, ChainLength, burnin, Prior, lnL, Proposal, verbosity)
    # now use a MVN random walk proposal 
    Σ = cov(chain[:,1:3])
    tuning = 1.0
    ChainLength = 1000
    MC_loops = 4
    for j = 1:MC_loops
        P = (cholesky(Σ)).U
        Proposal = θ -> proposal2(θ,tuning*P, lb, ub)
        θinit = vec(mean(chain[:,1:3],dims=1))
        if j == MC_loops
            ChainLength = 4000
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
            # keep every 4th
            i = 1:size(chain,1)
            keep = mod.(i,4.0).==0
            chain = chain[keep,:]
            Σ = 0.8*Σ + 0.2*cov(chain[:,1:3])
        end    
    end
    return chain, m
end

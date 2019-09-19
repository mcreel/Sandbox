include("MCMC.jl")
include("DoFit.jl")
function mc_rep()
    n = 1000
    chain, Z = MCMC(n; verbosity=false)
    chain = chain[:,1:3]
    #println("nonparametric fit results:")
    θnp = DoFit(Z, n, chain, 700)
    #println("ordinary MCMC posterior mean:")
    θmean = mean(chain, dims=1)
    θmed = median(chain, dims=1)
    #println("true parameters:")
    α = -7.36
    ρ = 0.9
    σ = 0.363
    θtrue = [α, ρ, σ] # true param values, on param space
    #@show θnp
    #@show θmean
    # first 3 are the two version of np (single chain, and replications)
    vcat(θnp[:]-θtrue, θmean[:]-θtrue, θmed[:]-θtrue)
end

include("MCMC.jl")
include("DoFit.jl")
function mc_rep()
    n = 1000
    # true parameters
    α = exp(-0.736/2.0)
    ρ = 0.9
    σ = 0.363
    θtrue = [α, ρ, σ] # true param values, on param space
    chain, Z = MCMC(n, θtrue; verbosity=false)
    chain = chain[:,1:3]
    θmean = mean(chain, dims=1)
    θmed = median(chain, dims=1)
    θnp = DoFit(Z, n, chain, 400)
    vcat(θnp[:]-θtrue, θmean[:]-θtrue, θmed[:]-θtrue)
end

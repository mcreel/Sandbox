include("MCMC.jl")
include("DoFit.jl")
include("func.jl")
include("SVmodel.jl")
function mc_rep()
n = 500
chain, Z = MCMC(n; verbosity=false)
chain = chain[:,1:3]
#println("nonparametric fit results:")
θnp = DoFit(Z, n, chain)
#println("ordinary MCMC posterior mean:")
θmean = mean(chain, dims=1)
θmed = median(chain, dims=1)
#println("true parameters:")
σe = exp(-0.736/2.0)
ρ = 0.9
σu = 0.363
θtrue = [σe, ρ, σu] # true param values, on param space
#@show θnp
#@show θmean
# first 3 are the two version of np (single chain, and replications)
vcat(θnp[:]-θtrue, θmean[:]-θtrue, θmed[:]-θtrue)
end

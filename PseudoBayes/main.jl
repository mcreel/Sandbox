include("MCMC.jl")
include("DoFit.jl")
include("func.jl")
include("SVmodel.jl")
#function main()
n = 500
chain, Z = MCMC(n; verbosity=true)
println("nonparametric fit results:")
@show DoFit(Z, n, chain)
println("ordinary MCMC posterior mean:")
@show mean(chain, dims=1)

println("true parameters:")
σe = exp(-0.736/2.0)
ρ = 0.9
σu = 0.363
θtrue = [σe, ρ, σu] # true param values, on param space
@show θtrue
#end
#main()

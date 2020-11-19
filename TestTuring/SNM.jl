using Turing
using StatsPlots
using LinearAlgebra:cholesky


@model function SNM(z)
    # priors for parameters
    m1 ~ Normal(0,5)
    m2 ~ Normal(0,5)
    n, S = 100, 100
    m = auxstat(m1, m2, n, S) # sample is S repetitions of a single sample
    z ~ MvNormal(m, Σ/n*(1+1/S))
end


function auxstat(m1, m2, n, S)
    μ = [m1, m2]
    P = cholesky(Σ).U
    mean(μ' .+ randn(n*S,2)*P, dims=1)[:]
end

m1, m2 = randn(2)
Σ = [2.0 1.0; 1.0 3.0] # true variance of z
z = auxstat(m1, m2, 100, 1)
chain = sample(SNM(z), HMC(0.1, 5), 1000)

# Summarise results
describe(chain)

# Plot and save results
p = plot(chain)
display(p)
@show m1, m2

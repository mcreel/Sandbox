# the first stage estimation using identity matrix is working
# ex3 will try to get the second stage to work
using Turing
using StatsPlots, Statistics, Optim
using LinearAlgebra:cholesky,I


@model function L(z, Σ)
    # priors for parameters
    m1 ~ Normal(0,5)
    m2 ~ Normal(0,5)
    n, S = 100, 10
    m = auxstat((m1, m2), n, S) # sample is S repetitions of a single sample
    z ~ MvNormal(m, Σ*(1+1/S))
end

# the dgp is just two MVN random vars
# the statistic is the quantiles of each, and the correlation
function auxstat(θ, P, S)
    m1, m2 = θ
    n = 100
    data = [m1 m2] .+ randn(n*S,2)*P
    q = vcat(quantile(data[:,1], [0.1, 0.5, 0.9]), quantile(data[:,2], [0.1, 0.5, 0.9]), cor(data[:,1], data[:,2])) 
end

# stuff for the dgp
θ⁰ = (randn(), randn())  # true parameters
Ω = [2.0 1.0; 1.0 3.0] # true variance of z
P = cholesky(Ω).U
# the real sample value of statistic
z = auxstat(θ⁰, P, 1)
# initial quasi-likelihood with identity weight
model = L(z,I)
chain = sample(model, HMC(0.1, 5), 10000)
# get the posterior point estimator
θbar = mean.((chain[:m1], chain[:m2]))
θ50 = median.((chain[:m1], chain[:m2]))
@show θ⁰
@show θbar
@show θ50
#=
# compute the estimator of covariance of statistic
S = 100
zs = zeros(S, size(z,1))
for s = 1:S
    zs[s,:] = auxstat(θbar, P, 1)
end
# second round with efficient weight
model2 = SNM(z,cov(zs))
chain = sample(model2, HMC(0.1, 5), 1000)
θbar = mean.((chain[:m1], chain[:m2]))
θ50 = median.((chain[:m1], chain[:m2]))
=#

# Summarise results
describe(chain)

# Plot and save results
p = plot(chain)
display(p)
@show θ⁰
@show θbar
@show θ50
nothing
# how to do MLE (probably want posterior median for first round estimate)
# optimize(model, MLE())

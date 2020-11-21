# the first stage estimation using identity matrix is working
# ex3 will try to get the second stage to work
using Turing
using StatsPlots, Statistics
using LinearAlgebra
@model function L(z, P, Σ)
    # priors for parameters
    μ1 ~ Normal(0,5)
    μ2 ~ Normal(0,5)
    # get the statistic from simulated sample, S times longer than real
    S = 10
    m = auxstat((μ1, μ2), P, S)
    # likelihood, taking Σ as fixed
    z ~ MvNormal(m, Σ*(1+1/S))
end
# the dgp is just two MVN random vars
# the statistic is the quantiles of each, and the correlation
function auxstat(θ, P, S)
    μ1, μ2 = θ
    n = 100
    data = [μ1 μ2] .+ randn(n*S,2)*P
    vcat(quantile(data[:,1], [0.1, 0.5, 0.9]), quantile(data[:,2], [0.1, 0.5, 0.9]), cor(data[:,1], data[:,2])) 
end

# stuff for the dgp
θ⁰ = (1.0, 1.0)  # true parameters
Ω = [2.0 1.0; 1.0 3.0] # true variance of z
P = cholesky(Ω).U
# the real sample value of statistic
z = auxstat(θ⁰, P, 1)
# initial quasi-likelihood with identity weight
model = L(z,P, I)
# do the sampling
#chain = sample(model, HMC(0.1, 5), 1000)
chain = sample(model, NUTS(0.65), 1000)
# get the posterior point estimators
θbar = mean.((chain[:μ1], chain[:μ2]))
θ50 = median.((chain[:μ1], chain[:μ2]))
# compute the estimator of covariance of statistic
S = 100
zs = zeros(S, size(z,1))
for s = 1:S
    zs[s,:] = auxstat(θbar, P, 1)
end
# second round with efficient weight
model = L(z,P, cov(zs))  # set the likehood
#chain = sample(model, HMC(0.1, 5), 1000)
chain = sample(model, NUTS(0.65), 1000)
θbar = mean.((chain[:μ1], chain[:μ2]))
θ50 = median.((chain[:μ1], chain[:μ2]))
# optimize(model, MLE(), LBFGS())
# Summarise results
@show chain
# Plot and save results
p = plot(chain)
display(p)
@show θ⁰
@show θbar
@show θ50
nothing


# the first stage estimation using identity matrix is working
# ex3 will try to get the second stage to work
using Turing
using StatsPlots, Statistics
using LinearAlgebra
include("MNlib.jl")
@model function L(z, Σ)
    # priors for parameters
    μ1 ~ Uniform(0.0, 3.0)
    μ2 ~ Uniform(0.0, 3.0)
    σ1 ~ Uniform(0.0, 1.0)
    σ2 ~ Uniform(0.0, 3.0)
    prob ~ Diriclet(2, 1)
    # get the statistic from simulated sample, S times longer than real
    S = 10
    m = auxstat((μ1, μ2, σ1, σ2, prob), S)
    # likelihood, taking Σ as fixed
    z ~ MvNormal(m, Σ*(1+1/S))
end
function auxstat(θ, reps)
    n = 1000*reps
    r = 0.0 : 0.1 : 1.0
    μ1, μ2, σ1, σ2, prob = θ

      # Draw assignments for each datum and generate it from a multivariate normal.
    k = Vector{Int}(undef, N)
    for i in 1:N
        k[i] ~ Categorical(prob)
        x[:,i] ~ MvNormal([μ[k[i]], μ[k[i]]], 1.)
    end
    return k


    d1=randn(n).*σ1 .+ μ1
    d2=randn(n).*(σ1+σ2) .+ (μ1 - μ2) # second component lower mean and higher variance
    ps=rand(n).<prob
    data=zeros(n)
    data[ps].=d1[ps]
    data[.!ps].=d2[.!ps]
    sqrt(n)*vcat(mean(data), std(data), skewness(data), kurtosis(data), quantile(data,r))
end    

# stuff for the dgp
θ⁰ = (1.0, 1.0, 0.2, 1.8, 0.4)  # true parameters
# the real sample value of statistic
z = auxstat(θ⁰, 1)
# initial quasi-likelihood with identity weight
model = L(z,I)
# do the sampling
#chain = sample(model, HMC(0.1, 5), 1000)
chain = sample(model, NUTS(0.65), 1000)
# get the posterior point estimators
θbar = mean.((chain[:μ1], chain[:μ2], chain[:σ1], chain[:σ2], chain[:prob]))
θ50 = median.((chain[:μ1], chain[:μ2], chain[:σ1], chain[:σ2], chain[:prob]))
# compute the estimator of covariance of statistic
S = 100
zs = zeros(S, size(z,1))
for s = 1:S
    zs[s,:] = auxstat(θbar, 1)
end
# second round with efficient weight
model = L(z, cov(zs))  # set the likehood
#chain = sample(model, HMC(0.1, 5), 1000)
chain = sample(model, NUTS(0.65), 1000)
θbar = mean.((chain[:μ1], chain[:μ2], chain[:σ1], chain[:σ2], chain[:prob]))
θ50 = median.((chain[:μ1], chain[:μ2], chain[:σ1], chain[:σ2], chain[:prob]))
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


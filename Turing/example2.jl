# second example, with a vector of parameters
using Turing, Statistics, Distributions, Random, LinearAlgebra, StatsPlots
using AdvancedMH

#@views function main()
θ = [2.; 3.]
n = 100 # sample size
S = 100 # number of simulation draws
# the data generating process
function dgp(θ, n)
    [rand.(Exponential(θ[1]),n) rand.(Poisson(θ[2]),n)] 
end
    # summary statistics for estimation
function moments(y)
    sqrt(n) .* [mean(y, dims=1)[:]; std(y, dims=1)[:]]
end
y = dgp(θ, n)
z = moments(y)
zs = zeros(S,size(z,1))

@model function abc(z)
    # create the prior: the product of the following array of marginal priors
    θ  ~ arraydist([LogNormal(1.,1.); LogNormal(1.,1.)])
    # sample from the model, at the trial parameter value, and compute statistics
    @inbounds for i = 1:S
        y .= dgp(θ, n)
        zs[i,:] .= moments(y) # simulated summary statistics
    end
    # the asymptotic Gaussian distribution of the statistics
    m = mean(zs, dims=1)[:]
    Σ = Symmetric((1. + 1/S)*cov(zs))
    z ~ MvNormal(m, Σ)
end;

# sample chains, 8 chains of length 1000, with 100 burnins dropped
chain = sample(abc(z), 
               MH(:θ => AdvancedMH.RandomWalkProposal(MvNormal(zeros(2), 0.25*I))), MCMCThreads(), 2100, 8)
chain = chain[101:end,:,:]
display(chain)
p = plot(chain)
display(p)
#end
#main()

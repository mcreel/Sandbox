# simple example of method of simulated moments estimated by
# NUTS MCMC

# load packages
using TransformVariables, LogDensityProblems, DynamicHMC, MCMCDiagnostics,
    Parameters, Statistics, Distributions, ForwardDiff


"""
DGP is``y ∼ μ + ϵ``, where ``ϵ ∼ N(0, 1)`` IID,
and the statistic is the sample mean.

Prior for μ is flat
"""
function dgp(μ)
    y = randn(100) .+ μ
    m = mean(y)
end

# generate data
μ_true = 3.0
m = dgp(μ_true)

# Define a structure for the problem
# Should hold the data and  the parameters of prior distributions.
struct MSM_Problem{Tm <: Real}
    "statistic"
    m::Tm
end

# Make the type callable with the parameters *as a single argument*.
function (problem::MSM_Problem)(θ)
    @unpack m = problem   # extract the data
    @unpack μ = θ         # extract parameters (only one here)
    S = 100
    ms = 0.0
    for s = 1:S
        ms += dgp(μ)/S
    end
    loglikelihood(Normal(0, 1), [m .- ms])
end

# original problem, without transformation of parameters
p = MSM_Problem(m)
# define the transformation of parameters (in this case, an identity)
problem_transformation(p::MSM_Problem) = as((μ=as(Array,1),))
# Wrap the problem with the transformation
t = problem_transformation(p)
P = TransformedLogDensity(t, p)
# use AD (Flux) for the gradient
∇P = ADgradient(:ForwardDiff, P);

# Sample from the posterior. `chain` holds the chain (positions and
# diagnostic information), while the second returned value is the tuned sampler
# which would allow continuation of sampling.
chain, NUTS_tuned = NUTS_init_tune_mcmc(∇P, 1000);

# We use the transformation to obtain the posterior from the chain.
posterior = transform.(Ref(t), get_position.(chain));

# Extract the parameter and plot the posterior
μhat = [i[1][1] for i in posterior]
dstats(μhat)
post_dens = npdensity(μhat)
gui()

# Effective sample sizes (of untransformed draws)
ess = mapslices(effective_sample_size, get_position_matrix(chain); dims = 1)
NUTS_statistics(chain)


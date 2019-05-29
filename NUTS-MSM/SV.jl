# simple example of method of simulated moments estimated by
# NUTS MCMC

# load packages
using TransformVariables, LogDensityProblems, DynamicHMC, MCMCDiagnostics,
    Parameters, Statistics, Distributions, ForwardDiff, LinearAlgebra

include("SVlib.jl")

# generate data
σu = 0.69
ρ = 0.9
σe = 0.363
n = 500 # sample size
burnin = 1000
S = 100 # number of simulations
shocks_u = randn(n+burnin,1)
shocks_e = randn(n+burnin,1)
m = SVmodel(σu, ρ, σe, shocks_u, shocks_e)
shocks_u = randn(n+burnin,S) # fixed shocks for simulations
shocks_e = randn(n+burnin,S) # fixed shocks for simulations

# Define a structure for the problem
# Should hold the data and  the parameters of prior distributions.
struct MSM_Problem{Tm <: Vector, Tshocks_u <: Array, Tshocks_e <: Array }
    "statistic"
    m::Tm
    "shocks"
    shocks_u::Tshocks_u
    shocks_e::Tshocks_e

end

# Make the type callable with the parameters *as a single argument*.
function (problem::MSM_Problem)(θ)
    @unpack m, shocks_u, shocks_e = problem   # extract the data
    @unpack σu, ρ, σe = θ         # extract parameters (only one here)
    S = size(shocks_u,2)
    k = size(m,1)
    mbar = zeros(k)
    Σ = zeros(k,k)
    for s = 1:S
        mm = SVmodel(σu, ρ, σe, shocks_u[:,s], shocks_e[:,s])
        mbar += mm/S
        Σ += mm*mm'/S
    end
    mbar = mbar[:]
    x = (m .- mbar)
    #Σ = eye(k)
    logL = try
        logL = -0.5*log(det(Σ)) - 0.5*x'*inv(Σ)*x
    catch
        logL = -Inf
    end    
end

# original problem, without transformation of parameters
p = MSM_Problem(m, shocks_u, shocks_e)
# define the transformation of parameters (in this case, an identity)
problem_transformation(p::MSM_Problem) = as((σu=as𝕀, ρ=as𝕀 σe=as𝕀))
# Wrap the problem with the transformation
t = problem_transformation(p)
P = TransformedLogDensity(t, p)
# use AD (Flux) for the gradient
∇P = ADgradient(:ForwardDiff, P)

# Sample from the posterior. `chain` holds the chain (positions and
# diagnostic information), while the second returned value is the tuned sampler
# which would allow continuation of sampling.
chain, NUTS_tuned = NUTS_init_tune_mcmc(∇P, 1000)


# We use the transformation to obtain the posterior from the chain.
posterior = transform.(Ref(t), get_position.(chain));

# Extract the parameter and plot the posterior
σu_hat = [i[1][1] for i in posterior]
dstats(σu_hat)
post_dens_sigu = npdensity(σu_hat)
# Extract the parameter and plot the posterior
ρhat = [i[2][1] for i in posterior]
dstats(ρhat)
post_dens_rho = npdensity(ρhat)
# Extract the parameter and plot the posterior
\sigmae_hat = [i[3][1] for i in posterior]
dstats(σe_hat)
post_dens_sige = npdensity(σe_hat)

# Effective sample sizes (of untransformed draws)
ess = mapslices(effective_sample_size, get_position_matrix(chain); dims = 1)
NUTS_statistics(chain)

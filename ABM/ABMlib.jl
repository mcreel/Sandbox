# Simulates model from
# Alfarano, S., Lux, T., & Wagner, F. (2008). Time variation 
# of higher moments in a financial market with heterogeneous
# agents: An analytical approach. Journal of Economic Dynamics
# and Control, 32(1), 101-136. 
#
# References to equations follow 
# Chen, Zhenxi, and Thomas Lux. "Estimation of sentiment 
# effects in financial markets: A simulated method of moments
# approach." Computational Economics 52, no. 3 (2018): 711-744.

using Random, Statistics

@views function ABMmodel(θ, rndseed=1234)
Random.seed!(rndseed)
a, b, σf = θ
T = 2000 # sample size (days)
burnin = 1000 # burn in period (days)
# numbers of agents
Nc = 100  # number of noise traders
n = 50    # initial number of traders in optimistic state
ns = zeros(T+1) # container for # optimistic agents in discrete time
t = -burnin # initial continuous time
dt = 1 # observations in discrete time start at t=1
while t < T+1
    # rates: from eq. 3, first expressiion, but dropping multiplication
    # by numbers of agents in states, as that is done in λ, below
    π⁺ = a + b*n # rate of positive switches, for a single agent
    π⁻ = a + b*(Nc - n) # rate of negative switches, for a single agent
    # wait time for state change (Julia uses scale, not rate)
    λ = n*π⁻  + (Nc - n)*π⁺ # rate of state change in population
    Δt = -log(rand())/λ # time elapsed until state change, draw from exp. dist.
    t += Δt # time of state change
    if t > dt # arrived to observation time?
        ns[dt] = n # record n right before dt
        dt +=1 # at new t, we are in next trading day
    end
    # change state randomly up or down, with prob. determined by rates
    rand() < n*π⁻/λ ? n -=1 : n += 1 # uses prob. that one of two exponentials occurs first
end
# sentiment index, defined in paragraph following eq. 2
x = (ns .- (Nc .- ns)) ./ Nc
# returns, defined in eq. 9 (note that NcVc/(NfVf) is set to 1,
# as discussed in paragraph following Table 1
rets = σf .* randn(T) + (x[2:end] - x[1:end-1]) 
return rets
end

# method that generates stats for a number of random samples
function auxstat(θ, reps)
    auxstat.([ABMmodel(θ, rand(1:Int64(1e12))) for i = 1:reps])  # reps draws of data
end

#= 
# Moments from Chen and Lux, 2018
# moments: method for a given sample
@views function auxstat(r)
    stats = sqrt(size(r,1)) .* [
    mean(r .^ 2.0),                             # m1
    mean(r[2:end] .* r[1:end-1]),               # m2
    mean((r[2:end] .* r[1:end-1]) .^ 2.0),      # m3
    mean(r .^ 4.0),                             # m4
    mean(abs.(r[2:end]) .* abs.(r[1:end-1])),   # m5
    mean((r[6:end] .* r[1:end-5]) .^ 2.0),      # m6
    mean(abs.(r[6:end]) .* abs.(r[1:end-5])),   # m7
    mean((r[11:end] .* r[1:end-10]) .^ 2.0),    # m8
    mean(abs.(r[11:end]) .* abs.(r[1:end-10])), # m9
    mean((r[16:end] .* r[1:end-15]) .^ 2.0),    # m10
    mean(abs.(r[16:end]) .* abs.(r[1:end-15])), # m11
    mean((r[21:end] .* r[1:end-20]) .^ 2.0),    # m12
    mean(abs.(r[21:end]) .* abs.(r[1:end-20])), # m13
    mean((r[26:end] .* r[1:end-25]) .^ 2.0),    # m14
    mean(abs.(r[26:end]) .* abs.(r[1:end-25]))  # m15
   ]
end    
=#
# method for a given sample
@views function auxstat(r)
	s = std(r)
    k = kurtosis(r)
	r = abs.(r)
	m = mean(r)
	s2 = std(r)
    r = r./s2
	c = cor(r[1:end-1],r[2:end])
	# ratios of quantiles of moving averages to detect clustering
	#q = try
	    q = quantile((ma(r,3)[3:end]), [0.25, 0.75])
	#catch
#	    q = [1.0, 1.0]
	#end
#@show q
c1 = log(q[2]) - log(q[1])
#    h = HAR(r)
    stats = sqrt(size(r,1)) .* [m, s, s2, k, c, c1]
end

function TrueParameters()
    [0.0003, 0.0014, 0.03]
end

# priors are from page 8,
# Lux, T. (2022). Approximate Bayesian inference for agent-based
# models in economics: a case study. Studies in Nonlinear Dynamics & Econometrics. 
# https://doi.org/10.1515/snde-2021-0052
function PriorSupport()
    lb = [0.0, 0.0, 0.0]
    ub = [0.05, 0.05, 5.0*0.038843]
    lb,ub
end    

# prior should be an array of distributions, one for each parameter
lb, ub = PriorSupport() # need these in Prior
macro Prior()
    return :( arraydist([Uniform(lb[i], ub[i]) for i = 1:size(lb,1)]) )
end
# check if parameter is in support. In this case, we require
# the bounds, and that the unconditional variance of the volatility
# shock be limited
function InSupport(θ)
    lb, ub = PriorSupport()
    all(θ .>= lb) & all(θ .<= ub) ? true : false
end

function PriorDraw()
    lb, ub = PriorSupport()
    θ = (ub-lb).*rand(size(lb,1)) + lb
end    


# taken from https://github.com/mcreel/Econometrics  
# returns the variable (or matrix), lagged p times,
# with the first p rows filled with ones (to avoid divide errors)
# remember to drop those rows before doing analysis
@views function lag(x,p)
	n = size(x,1)
    k = size(x,2)
	lagged_x = [ones(p,k); x[1:n-p,:]]
end

# returns the variable (or matrix), lagged from 1 to p times,
# with the first p rows filled with ones (to avoid divide errors)
# remember to drop those rows before doing analysis
@views function  lags(x,p)
	n = size(x,1)
	k = size(x,2)
	lagged_x = zeros(eltype(x),n,p*k)
	for i = 1:p
		lagged_x[:,i*k-k+1:i*k] = lag(x,i)
	end
    return lagged_x
end	
 
# compute moving average using p most recent values, including current value
@views function ma(x, p)
    m = zeros(size(x))
    for i = p:size(x,1)
        m[i] = mean(x[i-p+1:i])
    end
    return m
end

# auxiliary model: HAR-RV
# Corsi, Fulvio. "A simple approximate long-memory model
# of realized volatility." Journal of Financial Econometrics 7,
# no. 2 (2009): 174-196.
@views function HAR(y)
    ylags = lags(y,10)
    X = [ones(size(y,1)) ylags[:,1]  mean(ylags[:,2:5],dims=2) mean(ylags[:,6:10],dims=2)]
    # drop missings
    y = y[11:end]
    X = X[11:end,:]
    βhat = X\y
    σhat = std(y-X*βhat)     
    vcat(βhat,σhat)
end

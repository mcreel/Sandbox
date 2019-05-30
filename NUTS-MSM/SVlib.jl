using Statistics, Econometrics, Random, Distributions
include("lag.jl")
include("lags.jl")

function ma(x, p)
    m = similar(x)
    for i = p:size(x,1)
        m[i] = mean(x[i-p+1:i])
    end
    return m
end

# auxiliary model: HAR-RV
# Corsi, Fulvio. "A simple approximate long-memory model
# of realized volatility." Journal of Financial Econometrics 7,
# no. 2 (2009): 174-196.
include("lags.jl")
function HAR(y)
    ylags = lags(y,10)
    X = [ones(size(y,1)) ylags[:,1]  mean(ylags[:,1:4],dims=2) mean(ylags[:,1:10],dims=2)]
    # drop missings
    y = y[11:end]
    X = X[11:end,:]
    βhat = X\y
    pred = X*βhat
    σhat = sqrt(mean((y-pred).^2))     
    return vcat(βhat,σhat)
end

function aux_stat(y)
    # sig is good for sig_e
    IQR = quantile(y,0.75) - quantile(y,0.25)
    y = abs.(y)
    IQR2 = quantile(y,0.75) - quantile(y,0.25)
    # look for evidence of volatility clusters
    mm = ma(y,5)
    mm = mm[5:end]
    clusters5 = quantile(mm,0.75)/quantile(mm, 0.25)
    mm = ma(y,10)
    mm = mm[10:end]
    clusters10 = quantile(mm,0.75)/quantile(mm, 0.25)
    ϕ = HAR(y)
    vcat(IQR, IQR2, clusters5, clusters10, ϕ)[:]
end

# the dgp: simple discrete time SV model
function SVmodel(σu, ρ, σe, shocks_u, shocks_e)
    burnin = 1000
    n = 500
    hlag = 0.0
    t = 1
    h = ρ.*hlag .+ σu.*shocks_u[t] # log variance follows AR(1)
    y = σe.*exp(h./2.0).*shocks_e[t]
    ys = zeros(eltype(y),n)
    for t = 1:burnin+n
        h = ρ.*hlag .+ σu.*shocks_u[t] # log variance follows AR(1)
        y = σe.*exp(h./2.0).*shocks_e[t]
        if t > burnin 
            ys[t-burnin] = y
        end    
        hlag = h
    end
    #plot(ys)
    sqrt(n).*aux_stat(ys)
end



using Statistics, Random
include("func.jl")
# version which generates shock internally
function SVmodel(θ, n, burnin)
    shocks_u = randn(n+burnin)
    shocks_e = randn(n+burnin)
    SVmodel(θ, n, shocks_u, shocks_e, false)
end    

# the dgp: simple discrete time stochastic volatility (SV) model
function SVmodel(θ, n, shocks_u, shocks_e, savedata=false)
    σe = θ[1]
    α = 2.0*log(σe)
    ρ = θ[2]
    σu = θ[3]
    burnin = size(shocks_u,1) - n
    hlag = 0.0
    h = α + ρ*(hlag-α) + σu*shocks_u[1] # figure out type
    y = exp.(h./2.0).*shocks_e[1]
    ys = zeros(n,1)
    for t = 1:burnin+n
        h = α + ρ*(hlag-α) + σu*shocks_u[t]
        y = exp.(h./2.0).*shocks_e[t]
        if t > burnin 
            ys[t-burnin] = y
        end    
        hlag = h
    end
    if savedata == true
        plot(ys)
        gui()
        #writedlm("svdata.txt", ys)
    end    
    ys
end

using Pkg
Pkg.activate(".")
using SV, Econometrics, StatsBase
using BSON: @load
using BSON: @save
include("Transform.jl")    

function MakeData()
    n = 1000
    burnin = 1000
    S = Int(1e5) # size of training and testing
    SS = 1000 # size of design 
    # true parameters
    α = exp(-0.736/2.0)
    ρ = 0.9
    σ = 0.363
    θtrue = [α, ρ, σ] # true param values, on param space
    lb = [0.0, 0.0, 0.0]
    ub = [2.0, 0.99, 1.0]
    data = 0.0
    datadesign = 0.0
    # training and testing
    for s = 1:S
        θ = rand(size(lb,1)).*(ub-lb) + lb
        y, volatility = SVmodel(θ, n, burnin)
        m = sqrt(n)*aux_stat(y)
        if s == 1
            data = zeros(S, size(vcat(θ, m),1))
        end
        data[s,:] = vcat(θ, m)
    end
    # design
    for s = 1:SS
        y, volatility = SVmodel(θtrue, n, burnin)
        m = sqrt(n)*aux_stat(y)
        if s == 1
            datadesign = zeros(SS, size(vcat(θtrue, m),1))
        end
        datadesign[s,:] = vcat(θtrue, m)
    end
    # save needed items with standard format
    params = [data; datadesign][:,1:3]
    statistics = [data; datadesign][:,4:end]
    create_transformation(statistics)
    transform!(statistics)
    nDrawsFromPrior = S
    @save "simdata.bson" params statistics nDrawsFromPrior
    return nothing
end
MakeData()

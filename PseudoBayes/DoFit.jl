include("MakeZs.jl")
function DoFit(Z, n, chain)
    chain = [chain; chain; chain; chain; chain]
    Zs = MakeZs(n, chain)
    Z = reshape(Z, 1, size(Z,1))
    # do a nonparametric fit at Z using chain and Zs as data
    bandwidth = 1.0
    weights = kernelweights(Zs, Z, bandwidth, true, "knngaussian", 1000)
    θhat = npreg(chain[:,1:3], Zs, Z, weights, 1) 
end    

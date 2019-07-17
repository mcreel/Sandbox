include("MakeZs.jl")
function DoFit(Z, n, chain)
    Zs = MakeZs(n, chain)
    Z = reshape(Z, 1, size(Z,1))
    # do a nonparametric fit at Z using chain and Zs as data
    bandwidth = size(chain,1)^(-1.0/(4+size(chain,2))) # rull of thumb bandwidth (prewhitening means don't use std. devs.)
    weights = kernelweights(Zs, Z, bandwidth, true, "knngaussian", 500)
    θhat = npreg(chain, Zs, Z, weights) 
end    

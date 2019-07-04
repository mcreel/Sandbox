include("MakeZs.jl")
function DoFit(Z, chain)
    Zs = MakeZs(chain)
    Z = reshape(Z, 1, size(Z,1))
    # do a nonparametric fit at Z using chain and Zs as data
    bandwidth = 0.1
    weights = kernelweights(Zs, Z, bandwidth, true, "gaussian", 200)
    θhat = npreg(chain[:,1:3], Zs, Z, weights, 1) 
end    

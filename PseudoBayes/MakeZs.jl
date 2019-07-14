using SV
function MakeZs(n, chain)
    Zs = 0.0
    burnin = 100
    S = size(chain,1)
    for s = 1:S
        θ = chain[s,1:3]
        Z = sqrt(n)*aux_stat(SVmodel(θ, n, randn(n+burnin), randn(n+burnin)))
        if s == 1
            Zs = zeros(S,size(Z,1))
        end
        Zs[s,:]=Z
    end
    Zs
end    

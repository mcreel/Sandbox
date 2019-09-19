using SV, Econometrics, LinearAlgebra, Statistics, DelimitedFiles
    
n = 1000
# these are the true params
α = 7.36
ρ = 0.9
σ = 0.363
θtrue = [α, ρ, σ] # true param values, on param space
    shocks_u = randn(n+burnin)
    shocks_e = randn(n+burnin)
m = sqrt(n)*aux_stat(SVmodel(θtrue, n, shocks_u, shocks_e, false))
ms = zeros(1000,size(m,1))
burnin = 100
for i = 1:1000
    shocks_u = randn(n+burnin)
    shocks_e = randn(n+burnin)
    m = sqrt(n)*aux_stat(SVmodel(θtrue, n, shocks_u, shocks_e, false))
    ms[i,:] = m
end



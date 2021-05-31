using SolveDSGE, StatsPlots

filename = "CK.txt"
path = joinpath(@__DIR__,filename)
process_model(path)
processed_filename = "CK_processed.txt"
processed_path =  joinpath(@__DIR__,processed_filename)

dsge = retrieve_processed_model(processed_path)


# convert this to a function that computes SS given parameters
function params()
    α = 0.33
    β = 0.99
    δ = 0.025
    γ = 2.0
    ρ₁ = 0.9
    σ₁ = 0.02 
    ρ₂ = 0.7
    σ₂ = 0.01
    nss = 1.0/3.0
    [α, β, δ, γ, ρ₁, σ₁, ρ₂, σ₂, nss]
end   

function CKss()
    α, β, δ, γ, ρ₁, σ₁, ρ₂, σ₂, nss = params()
    c1 = ((1/β  + δ  - 1)/α )^(1/(1-α))
    kss = nss/c1
    yss = kss^α * nss^(1-α)
    css = yss - δ*kss
    MUCss = css^(-γ)
    rss = α * kss^(α-1) * nss^(1-α)
    wss = (1-α)* (kss)^α * nss^(-α)
    MULss = wss*MUCss
    [0.0, 0.0, kss, yss, css, nss, rss, wss, MUCss, MULss]
end

#= Use this to verify steady state
tol = 1e-8
maxiters = 1000
ss = compute_steady_state(dsge, CKss(), tol, maxiters)
=#
scheme = PerturbationScheme(CKss(), 1.0, "third")
solution = solve_model(dsge, scheme)
burnin = 1000
nobs = 160
data = simulate(solution,CKss()[1:3], burnin+nobs)
data = data[4:8, burnin+1:end]'
plot(data, label=["output" "cons" "hours" "r" "w"])



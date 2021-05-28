using SolveDSGE

filename = "CK.txt"
path = joinpath(@__DIR__,filename)

process_model(path)

processed_filename = "CK_processed.txt"
processed_path =  joinpath(@__DIR__,processed_filename)

dsge = retrieve_processed_model(processed_path)


# convert this to a function that computes SS given parameters
function CKss()
alppha = 0.33
betta = 0.99
delta = 0.025
gam = 2.0
rho1 = 0.9
sigma1 = 0.02 
rho2 = 0.7
sigma2 = 0.01
nss = 1.0/3.0
#alppha, betta, delta, gam, rho1, sigma1, rho2, sigma2, nss = params
c1 = ((1/betta + delta - 1)/alppha)^(1/(1-alppha))
kss = nss/c1
iss = delta*kss
yss = kss^alppha * nss^(1-alppha)
css = yss - iss;
MUCss = css^(-gam)
rss = alppha  * kss^(alppha-1) * nss^(1-alppha)
wss = (1-alppha)* (kss)^alppha * nss^(-alppha)
MULss = wss*MUCss
[0.0, 0.0, kss, kss, yss, css, nss, iss, MUCss, MULss, rss, wss]
end

#= Use this to verify steady state
tol = 1e-8
maxiters = 1000
ss = compute_steady_state(dsge, CKss(), tol, maxiters)
=#
NNN = PerturbationScheme(CKss(), 1.0, "third")
soln_to = solve_model(dsge,NNN)


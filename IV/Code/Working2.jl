
# this generates the covariance as a random
# positive definite matrix. It shows that the
# general idea offers an improvement, quite
# subtantial in some cases, at least.
using Plots, LinearAlgebra, Statistics
#function main()
reps = 10000 # number of Monte Carlo reps.
n = 100 # sample size
dim = 6 # 1 endog regressor, dim-2 instruments/exog regressors, and last col is the error
factors = 6 # 
included = 3 # number of included regressors (the first cols of generated vbls)
truebeta = [0.0; ones(included)] # true betas (none for the error)
betaols = zeros(reps,included+1)
betaiv = zeros(reps,included+1)
betaivR = zeros(reps,included+1)
for i = 1:reps
    # generate a PD covariance matrix for [x W ϵ]
    # x is endogenous regressor, in first col
    # W are dim-2 instruments
    # generate variables and errors
    a = false
    Σ = 0.0
    while !a
        W = randn(dim,factors)
        S = W*W' + diagm(rand(dim))
        T = diagm(1.0 ./sqrt.(diag(S)))
        S = T*S*T
        Σ = tril(S) + tril(S,-1)'
        Σ[end,:] .= 0.0
        Σ[:,end] .= 0.0
        Σ[1,end] = 0.8
        Σ[end,1] = 0.8
        Σ[end,end] = 1.0
        a = isposdef(Σ)
    end    
    p = cholesky(Σ).U
	XWE = randn(n,dim)*p
	e = XWE[:,dim:dim]
	w = [ones(n) XWE[:,2:dim-1]]
	x = [ones(n) XWE[:,1:included]]
	y = x*truebeta + e
	# OLS
	betaols[i,:] = (x\y)'
	# IV
    xhat = w*(w\x)
	betaiv[i,:] = (xhat\y)'
    # IV ridge
    #k = max(std(xhat))
	k = 0.5
    betaivR[i,:] = inv(xhat'xhat + k*eye(size(truebeta,1)))*xhat'y
end
p1 = npdensity(betaiv[:,2])
plot!(p1,title="IV")
p2 = npdensity(betaivR[:,2])
plot!(p2,title="IVR")
plot(p1,p2,layout=(2,1))
plot!(xlims=(0.0,2.0))
#savefig("olsiv.svg")
gui()
println("true betas are ", truebeta)
println("OLS results")
dstats(betaols, short=true)
println("IV results")
dstats(betaiv, short=true)
println("IV Ridge results")
dstats(betaivR, short=true)
#return
#end
#main()

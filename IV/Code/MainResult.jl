#= 
Generates a random covariance for x, w, e, with all variables
having var=1. Fairly general random covariance, by .
=#
# this generates the covariance as a random
# positive definite matrix. It shows that the
# general idea offers an improvement, quite
# subtantial in some cases, at least.
using Plots, LinearAlgebra, Statistics
function main()
reps = 10000 # number of Monte Carlo reps.
n = 100 # sample size
dim = 6 # 1 endog regressor, dim-2 instruments/exog regressors, and last col is the error
factors = 6 # same as dim, for now. Effects of reducing?
included = 3 # number of included regressors (the first cols of generated vbls)
truebeta = [0.0; ones(included)] # true betas (none for the error)
betaols = zeros(reps,included+1)
betaiv = zeros(reps,included+1)
betaivR = zeros(reps,included+1)
OtherInfo = zeros(reps,3) # 1st stage coefs, 1st stage F, and condition of OLS and Ridge regressors for second stage
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
    endogreg = XWE[:,1:1]
    error = XWE[:,end:end]
    w = [ones(n) XWE[:,2:end-1]]
	x = [ones(n) XWE[:,1:included]]
	y = x*truebeta + error
    # ordinary first stage F test
    r = rsq(endogreg, w) # regress on both inst
    r2 = rsq(endogreg, XWE[:,2]) # regress only on included inst
    F = (r - r2)/(1-r)*(size(w,1) - size(w,2))
    OtherInfo[i,1] = F
    # OLS
	betaols[i,:] = (x\y)'
	# IV
    πhat = w\endogreg # first stage coefficients
    r = rsq(w*πhat,[ones(n) XWE[:,2]]) 
    VIF = 1.0 / (1.0 - r) # VIF for second stage coef of endog vbl
    OtherInfo[i,2] = VIF
    xhat = x
    xhat[:,2] = w*πhat
	betaiv[i,:] = (xhat\y)'
    P = diag(xhat*inv(xhat'xhat)*xhat')
    P = P ./(1.0 .- P)
    OtherInfo[i,3] = maximum(P)/minimum(P)
    # IV ridge
	k = 0.5 # low values give very low bias, but higher variance
    betaivR[i,:] = inv(xhat'xhat + k*eye(size(xhat,2)))*xhat'y
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
return [betaols betaiv betaivR OtherInfo]
end
main()

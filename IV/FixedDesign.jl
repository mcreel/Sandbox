# use a particular covariance to explore importance of factors
using Plots, LinearAlgebra, Statistics
function main()
reps = 100000 # number of Monte Carlo reps.
betaols = zeros(reps,4)
betaiv = zeros(reps,4)
betaivR = zeros(reps,4)
Fs = zeros(reps)
n = 500 # sample size

truebeta = [0.0, 1.0, 1.0, 1.0] # true beta
# generate a PD covariance matrix for [x Z₁ Z₂ Z₃ ϵ]
# x is endogenous regressor
# Zᵢ are 3 instruments, first is excludes, second two are included in regression
# generate variables and errors
Σ = 0.5*eye(5) # will become ones on diag when we make it symmetric
Σ[1,2] = 0.8 # goodness of excluded instrument
Σ[1,3] = 0.5 # correlation of regressor with first included
Σ[1,4] = 0.5 # correlation of regressor with second included
Σ[1,5] = 0.5 # correlation of regressor with error: severity of endogeneity
Σ[2,3] = 0.5 # correlation of excluded inst with included
Σ[2,4] = 0.5 # correlation of excluded inst with included
Σ[3,4] = 0.5 # correlation of included instruments
Σ = Σ + Σ' # make symmetric
println("Σ")
prettyprint(Σ)
println("check pos def.:", isposdef(Σ))
p = cholesky(Σ).U

Threads.@threads for i = 1:reps
	XWE = randn(n,5)*p
	e = XWE[:,5:5]
	w = [ones(n) XWE[:,2:4]]
	x = [ones(n) XWE[:,1:3]]
	y = x*truebeta + e
    # ordinary first stage F test
    r = rsq(e, w)
    nn,k = size(w)
    F = r/(1.0-r)*(k-1)/(nn-k)
    Fs[i] = F
    # OLS
	betaols[i,:] = (x\y)'
	# IV
    xhat = w*(w\x)
	betaiv[i,:] = (xhat\y)'
    # IV ridge
    #k = max(std(xhat))
	k = 0.5
    xx = xhat'xhat
    @show cond(xx)
    betaivR[i,:] = inv(xx + k*eye(4))*xhat'y
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
end
main()

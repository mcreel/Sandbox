# This example give the proposal showing a good improvement. It's a
# good example of the possibility of improvement

# use a particular covariance to explore importance of factors
using Plots, LinearAlgebra, Statistics
function main()
reps = 100000 # number of Monte Carlo reps.
betaols = zeros(reps,3)
betaiv = zeros(reps,3)
betaivR = zeros(reps,3)
OtherInfo = zeros(reps,2) # 1st stage coefs, 1st stage F, and condition of OLS and Ridge regressors for second stage
n = 400 # sample size
truebeta = [0.0, 1.0, 1.0] # true beta
# generate a PD covariance matrix for [x Z₁ Z₂ ϵ]
# x is endogenous regressor
# Z₁ is the included instrument, Z₂ is the excluded instrument
# generate variables and errors
Σ = 0.5*eye(4) # will become ones on diag when we make it symmetric
Σ[1,2] = 0.5 # cor endog regressor with exog regressor
Σ[1,3] = 0.5 # cor endog regressor with excluded instr
Σ[1,4] = 0.5 # cor endog regressor with error; ENDOGENEITY
Σ[2,3] = 0.9 # correlation of incl. and excl. insts.
Σ = Σ + Σ' # make symmetric
println("Σ")
prettyprint(Σ)
println("check pos def.:", isposdef(Σ))
p = cholesky(Σ).U

@simd for i = 1:reps
	XWE = randn(n,4)*p
    endogreg = XWE[:,1:1]
    error = XWE[:,4:4]
	w = [ones(n) XWE[:,2:3]]
	x = [ones(n) XWE[:,1:2]]
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
    xhat = [ones(n) w*πhat XWE[:,2]]
	betaiv[i,:] = (xhat\y)'
    # IV ridge
	k = 0.5 # low values give very low bias, but higher variance
    betaivR[i,:] = inv(xhat'xhat + k*eye(3))*xhat'y
end
p1 = histogram(betaiv[:,2])
plot!(p1,title="IV")
p2 = histogram(betaivR[:,2])
plot!(p2,title="IVR")
plot(p1,p2,layout=(2,1))
plot!(xlims=(-2.0,3.0))
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

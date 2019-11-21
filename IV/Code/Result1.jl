# this shows that a little ridge can greatly reduce the variance, 
# and that the ridge IV estimator can have little bias (less than 2SLS)
## here is some typical output
#=
julia> bols,biv, otherinf = include("Result1.jl");
Σ
     1.00000     0.50000     0.50000     0.20000
     0.50000     1.00000     0.90000     0.00000
     0.50000     0.90000     1.00000     0.00000
     0.20000     0.00000     0.00000     1.00000
check pos def.:true
true betas are [0.0, 1.0, 1.0]
OLS results
                    mean      median         std         min         max
           1    -0.00031     0.00022     0.09812    -0.40126     0.43855
           2     1.26656     1.26680     0.11456     0.70502     1.83232
           3     0.86705     0.86674     0.11494     0.27593     1.33972
IV results
                    mean      median         std         min         max
           1     0.02077     0.00007     8.19359 -1366.69008   980.64978
           2     0.75258     1.06348   137.98882-14152.21771 26781.39862
           3     1.10062     0.96809    80.83523-16661.55412  8636.86137
IV Ridge results
                    mean      median         std         min         max
           1     0.00071     0.00001     0.27061    -6.40011     7.43082
           2     0.98917     1.06218     2.96738   -44.36954    48.89106
           3     1.00526     0.96907     1.49281   -23.73267    28.22875

=#
# use a particular covariance to explore importance of factors
using Plots, LinearAlgebra, Statistics
function main()
reps = 100000 # number of Monte Carlo reps.
betaols = zeros(reps,3)
betaiv = zeros(reps,3)
betaivR = zeros(reps,3)
OtherInfo = zeros(reps,6) # 1st stage coefs, 1st stage F, and condition of OLS and Ridge regressors for second stage
n = 100 # sample size
truebeta = [0.0, 1.0, 1.0] # true beta
# generate a PD covariance matrix for [x Z₁ Z₂ ϵ]
# x is endogenous regressor
# Z₁ is the included instrument, Z₂ is the excluded instrument
# generate variables and errors
Σ = 0.5*eye(4) # will become ones on diag when we make it symmetric
Σ[1,2] = 0.5 # cor endog regressor with exog regressor
Σ[1,3] = 0.5 # cor endog regressor with excluded instr
Σ[1,4] = 0.2 # cor endog regressor with error; ENDOGENEITY
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
    xhat = [ones(n) w*πhat XWE[:,2:2]]
    OtherInfo[i,2:4] = πhat
	betaiv[i,:] = (xhat\y)'
    # IV ridge
	k = 0.001 # low values give very low bias, but higher variance
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
return betaols, betaiv, OtherInfo
end
main()
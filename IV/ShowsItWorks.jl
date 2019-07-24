# this generates the covariance as a random
# positive definite matrix. It shows that the
# general idea offers an improvement, quite
# subtantial in some cases, at least.
using Plots, LinearAlgebra, Statistics
function main()
reps = 10000 # number of Monte Carlo reps.
betaols = zeros(reps,4)
betaiv = zeros(reps,4)
betaivR = zeros(reps,4)
n = 100 # sample size

truebeta = [0.0, 1.0, 1.0, 1.0] # true beta
for i = 1:reps
    # generate a PD covariance matrix for [x W ϵ]
    # x is endogenous regressor
    # W are 3 instruments, first 2 included in regression
    # generate variables and errors
    a = false
    Σ = 0.0
    while !a
        Σ = rand(4,4)
        Σ = Σ'Σ
        s = [0.8, 0.0, 0.0, 0.0]
        Σ = [Σ s]
        Σ = [Σ;[s' 1.0]]
        a = isposdef(Σ)
    end
    p = cholesky(Σ).U

	XWE = randn(n,5)*p
	e = XWE[:,5:5]
	w = [ones(n) XWE[:,2:4]]
	x = [ones(n) XWE[:,1:3]]
	y = x*truebeta + e
	# OLS
	betaols[i,:] = (x\y)'
	# IV
    xhat = w*(w\x)
	betaiv[i,:] = (xhat\y)'

    # IV ridge
    #k = max(std(xhat))
	k = 0.5
    betaivR[i,:] = inv(xhat'xhat + k*eye(4))*xhat'y
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
return
end
main()

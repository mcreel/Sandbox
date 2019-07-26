#= this shows the idea works, using a random 
positive definite covariance matrix. There are
two included exogs, and one excluded instrument.
Typical output follows

julia> include("Result0.jl")
true betas are [0.0, 1.0, 1.0, 1.0]
OLS results
                    mean      median         std         min         max
           1    -0.00146    -0.00024     0.04973    -0.30609     0.24845
           2     1.95365     1.97317     0.17272     1.33107     2.27117
           3     0.57820     0.59199     1.09740   -67.18752    26.19572
           4     0.55333     0.58263     0.98871   -22.62338    37.99164
IV results
                    mean      median         std         min         max
           1    -0.04644     0.00039     4.84244  -376.10085   187.00790
           2     0.38548     1.07098    88.73264 -8350.85447  1413.31823
           3     0.49940     0.96917    61.61567 -3925.93877  2497.19987
           4     1.80703     0.95965    89.28773 -1433.84150  6501.36409
IV Ridge results
                    mean      median         std         min         max
           1    -0.00132    -0.00012     0.10100    -0.77860     0.53225
           2     1.03301     1.04669     0.35285    -0.71240     2.77336
           3     0.97045     0.97353     0.29279    -0.73117     2.50478
           4     0.96592     0.96344     0.29696    -1.08947     2.52411
=#


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
	k = 2.0
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

using Econometrics, CSV
data = CSV.read("fish.csv")
data = convert(Matrix{Float64}, data)

# dep var
LPA = data[:,1]
LPW = data[:,2]
LQA = data[:,3] 
LQW = data[:,4]
DAYS = data[:,5:8]
WEATHER = data[:,9:12]
constant = ones(size(data,1))
Y = LPA
X = [constant LQA DAYS] # regressors
W = [constant DAYS WEATHER] # instruments
βols = X\Y
XHAT = W*(W\X) # fit from reduced form
names = ["constant" "logQ" "MON" "TUE" "WED" "THUR"]
# replicate the estimate in Table 3 of Graddy, last col. 
# this is just the 2SLS for the same model as table 2, col 1, but using
# nearc4 as instrument for educ.
β2sls, junk, junk, junk, junk = tsls(Y, X, XHAT, names=names) # 2SLS estimates (same as if W used as inst.) 
println("VIF: ", 1.0/(1.0 - rsq(XHAT[:,2], [XHAT[:,1] XHAT[:,3:end]])))
k = 0.01
@show sizeof(βols)
βivR = ridge(Y, XHAT, k)
prettyprint([βols β2sls βivR], "", names)
ess = sum((Y - XHAT*β2sls).^2.0)
essR = sum((Y - XHAT*βivR).^2.0)
println("ess/essR: ", round(ess/essR, digits=3))

# loop over prior precisions to see effect
expon = [-3 -2.5 -2 -1.5 -1 -0.5]
ks = 10.0 .^expon
βs = zeros(size(β2sls,1), 6)
Rel_ESS = zeros(6)
for i = 1:size(ks,2)
    k = ks[i]
    βs[:,i] = ridge(Y, XHAT, k)
    Rel_ESS[i] = sum((Y - XHAT*βs[:,i]).^2.0) / ess
end
rel = βs ./ β2sls
plot(expon', rel[2:6,:]',label=["logQ" "MON" "TUE" "WED" "THUR"])
plot!(xlabel="log-10 prior precision (k)")
savefig("ridge_trace.svg")
plot(expon', [βs[2,:] β2sls[2,:].*ones(6)] ,xlabel="log-10 prior precision (k)", label=["ridge" "2sls"])
savefig("logQ-2SLSandRidge.svg")
plot(expon', Rel_ESS ,xlabel="log-10 prior precision (k)", legend=false)
savefig("RelativeESS.svg")


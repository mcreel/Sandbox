using Econometrics, CSV
card = CSV.read("card.csv")
data = convert(Matrix{Float64}, card)

# dep var
LNW = log.(data[:,1])
NEARC4 = data[:,2]
EDUC = data[:,3] 
AGE = data[:,4]
BLACK = data[:,5]
SMSA = data[:,6]
SOUTH = data[:,7]
EXPER = data[:,8]
EXPSQ = (EXPER.^2.0)/100.0
constant = ones(size(LNW))
X = [constant EDUC EXPER EXPSQ BLACK SMSA SOUTH] # regressors
W = [constant NEARC4 AGE AGE.^2.0 BLACK SMSA SOUTH] # instruments
βols = X\LNW
XHAT = W*(W\X) # fit from reduced form
names = ["constant" "EDUC" "EXPER" "EXPERSQ" "BLACK" "SMSA" "SOUTH"]
# replicate the estimate in Table 3 of Card, col 5. 
# this is just the 2SLS for the same model as table 2, col 1, but using
# nearc4 as instrument for educ.
β2sls, junk, junk, junk, junk = tsls(LNW, X, XHAT, names=names) # 2SLS estimates (same as if W used as inst.) 
println("VIF EDUCHAT: ", 1.0/(1.0 - rsq(XHAT[:,2], [XHAT[:,1] XHAT[:,3:end]])))
k = 10
@show sizeof(βols)
βivR = ridge(LNW, XHAT, k)
prettyprint([βols β2sls βivR], "", names)
ess = sum((LNW - XHAT*β2sls).^2.0)
essR = sum((LNW - XHAT*βivR).^2.0)
println("ess/essR: ", round(ess/essR, digits=3))

# loop over prior precisions to see effect
expon = [-1 0 1 2 3 4]
ks = 10.0 .^expon
βs = zeros(size(β2sls,1), 6)
Rel_ESS = zeros(6)
for i = 1:6
    k = ks[i]
    βs[:,i] = ridge(LNW, XHAT, k)
    Rel_ESS[i] = sum((LNW - XHAT*βs[:,i]).^2.0) / ess
end
rel = βs ./ β2sls
plot(expon', rel[2:7,:]',label=["EDUC" "EXPER" "EXPERSQ" "BLACK" "SMSA" "SOUTH"])
plot!(xlabel="log-10 prior precision (k)")
savefig("ridge_trace.svg")
plot(expon', [βs[2,:] β2sls[2,:].*ones(6)] ,xlabel="log-10 prior precision (k)", label=["ridge" "2sls"])
savefig("EDUC-2SLSandRidge.svg")
plot(expon', Rel_ESS ,xlabel="log-10 prior precision (k)", legend=false)
savefig("RelativeESS.svg")


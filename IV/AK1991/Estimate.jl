# this generates the results used in the paper for the Angrist-Krueger data
# note: x is the matrix of fitted regressors from the first stage, it
# contains the regressors after purging endogenous component. (different from other examples)
using CSV, Statistics
ak = CSV.read("AKdata.csv")
data = convert(Matrix{Float64}, ak)
# first stage regression to get EDUCHAT
EDUC = data[:,2]
y = EDUC
x = data[:,3:end]
EDUCHAT = x*(x\y)
rfresid = EDUC - EDUCHAT
# second stage regression
LNW = data[:,1]
y = LNW
x = [ones(size(y,1)) EDUCHAT data[:,3:24]] # use an overall constant and drop last year dummy
names = ["constant" "EDUCHAT" "RACE" "SMSA" "MARRIED" "AGEQ" "AGEQSQ" "ENOCENT" "ESOCENT" "MIDATL" "MT" "NEWENG" "SOATL" "WNOCENT" "WSOCENT" "YB_30" "YB_31" "YB_32" "YB_33" "YB_34" "YB_35" "YB_36" "YB_37" "YB_38"]
β2sls = x\y
xx = copy(x) # to compute SF residuals
xx[:,2] = EDUC
sfresid = y -x*β2sls
println("VIF EDUCHAT: ", 1.0/(1.0 - rsq(x[:,2], [x[:,1] x[:,3:end]])))
yhat = x*β2sls
k = 10.0
βivR = ridge(y, x, k)
prettyprint([β2sls βivR], "", names)
ess = sum((y - x*β2sls).^2.0)
essR = sum((y - x*βivR).^2.0)
println("ess/essR: ", round(ess/essR, digits=3))

# loop over prior precisions to see effect
ks = 10 .^ [1 2 3 4 5 6]
βs = zeros(size(β2sls,1), 6)
Rel_ESS = zeros(6)
for i = 1:6
    k = ks[i]
    βs[:,i] = ridge(y, x, k)
    Rel_ESS[i] = sum((y - x*βs[:,i]).^2.0) / ess
end
rel = βs ./ β2sls
plot(βs[2:7,:]',label=["EDUC" "RACE" "SMSA" "MARRIED" "AGEQ" "AGEQSQ"])
plot!(xlabel="log-10 prior precision (k)")
savefig("ridge_trace.svg")
plot([βs[2,:] β2sls[2,:].*ones(6)] ,xlabel="log-10 prior precision (k)", label=["ridge" "ols"])
savefig("EDUC-OLSandRidge.svg")
plot(Rel_ESS ,xlabel="log-10 prior precision (k)", legend=false)
savefig("RelativeESS.svg")

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
β2sls = x\y
xx = copy(x) # to compute SF residuals
xx[:,2] = EDUC
sfresid = y -xx*β2sls

# generate bootstrap data
errors = bootstrap([rfresid sfresid])
EDUC = EDUCHAT + errors[:,1] # bootstrap resample rf resids
xx[:,2] = EDUC
LNW = xx*β2sls + errors[:,2]
βtrue = β2sls

# run procedure on simulated data
y = EDUC
x = data[:,3:end]
EDUCHAT = x*(x\y)
# second stage regression
y = LNW
x = [ones(size(y,1)) EDUCHAT data[:,3:24]] # use an overall constant and drop last year dummy
names = ["constant" "EDUCHAT" "RACE" "SMSA" "MARRIED" "AGEQ" "AGEQSQ" "ENOCENT" "ESOCENT" "MIDATL" "MT" "NEWENG" "SOATL" "WNOCENT" "WSOCENT" "YB_30" "YB_31" "YB_32" "YB_33" "YB_34" "YB_35" "YB_36" "YB_37" "YB_38"]
β2sls = x\y
println("VIF EDUCHAT: ", 1.0/(1.0 - rsq(x[:,2], [x[:,1] x[:,3:end]])))
k = 1000.0
βivR = ridge(y, x, k)
prettyprint([βtrue β2sls βivR], "", names)
ess = sum((y - x*β2sls).^2.0)
essR = sum((y - x*βivR).^2.0)
println("ess/essR: ", round(ess/essR, digits=3))



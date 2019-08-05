using CSV, DataFrames, Statistics
ak = CSV.read("SecondStage.csv")
data = convert(Matrix{Float64}, ak)
y = data[:,1]
x = [ones(size(y,1)) data[:,2:end-1]] # use an overall constant, and drop last dummy
names = ["constant" "EDUCHAT" "RACE" "SMSA" "MARRIED" "AGEQ" "AGEQSQ" "ENOCENT" "ESOCENT" "MIDATL" "MT" "NEWENG" "SOATL" "WNOCENT" "WSOCENT" "YB_30" "YB_31" "YB_32" "YB_33" "YB_34" "YB_35" "YB_36" "YB_37" "YB_38"]
βols, junk, errors, ess, junk = ols(y,x, names=names, silent=true)
yhat = x*βols
k = 1000.0
βivR = ridge(y, x, k)
prettyprint([βols βivR], "", names)
yhatIVR = x*βivR
u = y - yhatIVR 
essR = sum(u .^ 2.0)
println("ess/essR: ", round(ess/essR, digits=3))
prettyprint([mean(y) mean(yhat) mean(yhatIVR)])
prettyprint([std(y) std(yhat) std(yhatIVR)])



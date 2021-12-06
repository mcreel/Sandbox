using DelimitedFiles, Plots, Statistics

model = "dsge"
model = "jd"
model = "sv"

chain = readdlm(model*"_chain.txt")
println("std. errors from MCMC chain")
display(std(chain,dims=1))

P = readdlm(model*"_P")
tuning = readdlm(model*"_tuning")
s = tuning.*randn(10000,size(P,1))*P'
println("std. errors from sample from proposal")
display(std(s,dims=1))



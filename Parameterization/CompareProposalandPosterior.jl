using DelimitedFiles, Plots, Statistics

model = "dsge"
#model = "jd"
#model = "sv"

chain = readdlm(model*"_chain.txt")
println("std. errors from MCMC chain")
display(std(chain,dims=1))

P = readdlm(model*"_P")
tuning = readdlm(model*"_tuning")
s = chain .+ tuning.*randn(size(chain,1),size(P,1))*P'
println("std. errors from sample from proposal")
display(std(s,dims=1))

scatter(s[:,1], s[:,7])
scatter!(chain[:,1], chain[:,7])


using StatsPlots, DelimitedFiles

chain = readdlm("jd_chain.txt")
p = []
for i = 1:7
    for j = i+1:8
        push!(p, scatter(chain[:,i],chain[:,j]))
    end
end
plot(p)

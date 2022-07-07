using StatsPlots, DelimitedFiles

#chain = readdlm("jd_chain.txt")
chain = readdlm("dsge_chain.txt")
#chain = readdlm("sv_chain.txt")
k = size(chain,2)
for i = 1:k-1
    for j = i+1:k
    display(scatter(chain[:,i],chain[:,j], title="$i vs $j"))
    sleep(2)
    end
end

using SolveDSGE, StatsPlots
include("CKlib.jl")

#p, ss = ParamsAndSS(TrueParameters())
p, ss = ParamsAndSS(PriorDraw())
dsge = assign_parameters(dsge, p)
scheme = PerturbationScheme(ss, 1.0, "third")
solution = solve_model(dsge, scheme)
burnin = 1000
nobs = 160
data = simulate(solution, ss[1:3], burnin+nobs; rndseed = 1234) # generated dsgedata.txt and figure
data = simulate(solution, ss[1:3], burnin+nobs; rndseed = rand(1:Int64(1e12)))
data = data[4:8, burnin+1:end]'
plot(data, legend=:outertopright, label=["output" "cons" "hours" "r" "w"])
#savefig("dsgedata.svg")
#writedlm("dsgedata.txt", data)



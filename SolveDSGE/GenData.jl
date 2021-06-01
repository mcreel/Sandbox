using SolveDSGE, StatsPlots
include("CKlib.jl")
filename = "CK.txt"
path = joinpath(@__DIR__,filename)
process_model(path)
processed_filename = "CK_processed.txt"
processed_path =  joinpath(@__DIR__,processed_filename)
dsge = retrieve_processed_model(processed_path)

p, ss = ParamsAndSS(TrueParameters())
dsge = assign_parameters(dsge, p)
scheme = PerturbationScheme(ss, 1.0, "third")
solution = solve_model(dsge, scheme)
burnin = 1000
nobs = 160
data = simulate(solution, ss[1:3], burnin+nobs; rndseed = 1234)
data = data[4:8, burnin+1:end]'
plot(data, legend=:outertopright, label=["output" "cons" "hours" "r" "w"])
#savefig("dsgedata.svg")
#writedlm("dsgedata.txt", data)



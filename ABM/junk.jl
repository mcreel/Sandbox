using Plots, Statistics
include("ABMlib.jl")
function main()
r, ns = ABMmodel([0.05, 0.05, 0.03], rand(1:1000000))
r2, ns2 = ABMmodel(PriorDraw(), rand(1:1000000))
p = histogram(ns)
p2 = histogram(ns2)
@show std(r)
@show std(r2)
plot(p, p2)
end
main()

using SimulatedNeuralMoments, Flux, StatsPlots, KernelDensity
using BSON:@load


# get the things to define the structure for the model
include("SVlib.jl") # contains the functions for the DSGE model
function main()
lb, ub = PriorSupport()
# fill in the structure that defines the model
model = SNMmodel("SV example", lb, ub, InSupport, Prior, PriorDraw, auxstat)
@load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net
# draw a sample at the design parameters, from the prior, or use the official "real" data
data = SVmodel(TrueParameters(), rand(1:Int64(1e6)))
# define the neural moments using the data
θhat = NeuralMoments(auxstat(data), model, nnmodel, nninfo)
# draw many samples at the estimate, and use them to bootstrap estimator
reps = 1000
data = auxstat(θhat, reps)
θs = zeros(reps,3)
for i = 1:reps
    θs[i,:] = NeuralMoments(data[i], model, nnmodel, nninfo)
end    
p1 = scatter(θs[:,1], θs[:,2], markersize=2, legend=false, title="1 vs 2")
k = kde((θs[:,1], θs[:,2]))
p1 = contour!(k)
p2 = scatter(θs[:,1], θs[:,3], markersize=2, legend=false, title="1 vs 3")
k = kde((θs[:,1], θs[:,3]))
p2 = contour!(k)
p3 = scatter(θs[:,2], θs[:,3], markersize=2, legend=false, title="2 vs 3")
k = kde((θs[:,2], θs[:,3]))
p3 = contour!(k)
p = plot(p1,p2,p3)
#savefig("bootstrap.png")
return p
end




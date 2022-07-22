#using SimulatedNeuralMoments
using Flux, Turing, MCMCChains, AdvancedMH
using StatsPlots, DelimitedFiles, LinearAlgebra
using BSON:@save
using BSON:@load

include("SimulatedNeuralMoments.jl")

# the model-specific code
include("ABMlib.jl")
function main()

# fill in the structure that defines the model
lb, ub = PriorSupport() # bounds of support
model = SNMmodel("ABM example", lb, ub, InSupport, PriorDraw, auxstat)

# train the net, and save it and the transformation info
transf = bijector(@Prior) # transforms draws from prior to draws from  ℛⁿ 
transformed_prior = transformed(@Prior, transf) # the transformed prior
nnmodel, nninfo = MakeNeuralMoments(model, transf)
@save "neuralmodel.bson" nnmodel nninfo  # use this line to save the trained neural net 
end
main()


using BSON:@save
include("Functions.jl")
include("JDlib.jl")
lb, ub = PriorSupport()
# fill in the structure that defines the model
model = SNMmodel("SP500 estimation", lb, ub, InSupport, Prior, PriorDraw, auxstat)
# make the training data for the nets
params, statistics = MakeData(model)
@save "data.bson" params statistics


using BSON:@save
using BSON:@load
using Statistics, LinRegOutliers, DataFrames
include("Functions.jl")
include("JDlib.jl")
lb, ub = PriorSupport()
# fill in the structure that defines the model
model = SNMmodel("SP500 estimation", lb, ub, InSupport, Prior, PriorDraw, auxstat)

# make the training data for the nets
params, statistics = MakeData(model)
@save "data.bson" params statistics
# @load "data.bson" params statistics
# df = convert(DataFrame, statistics)
# params = (params .- mean(params,dims=1)) ./std(params,dims=1)
# statistics = (statistics .- mean(statistics,dims=1)) ./std(statistics,dims=1)

# train nets
#StoPnet = TrainNet(statistics, params)
#PtoSnet = TrainNet(params, statistics)

#@save "nets.bson" PtoSnet StoPnet
#@load "nets.bson" PtoSnet StoPnet  # these were trained with standardized and normalized data


# reg = createRegressionSetting(@formula(x14 ~ 1), df)


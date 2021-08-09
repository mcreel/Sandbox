using Statistics

# the type that holds the model specifics
struct SNMmodel
    modelname::String # name of model
    lb::Vector # vector of lower bounds. Can be -Inf, if desired
    ub::Vector # vector of upper bounds. Can be inf, if desired
    insupport::Function # function that checks if a draw is valid
    prior::Function # function that evaluates the prior at draw
    priordraw::Function # function that returns a draw from prior
    auxstat::Function # function that returns an array of draws of statistic
end

# bounds by quantiles, and standardizes and normalizes around median
function TransformStats(data, info)
    q01,q50,q99,iqr = info
    data = max.(data, q01')
    data = min.(data, q99')
    data = (data .- q50') ./ iqr'
    return data
end

function MakeData(model::SNMmodel; TrainTestSize=1)
    data = 0.0
    datadesign = 0.0
    nParams = size(model.lb,1)
    # training and testing
    if (TrainTestSize == 1) TrainTestSize = Int64(2*nParams*1e4); end # use a default size if none provided
    params = zeros(TrainTestSize,nParams)
    statistics = zeros(TrainTestSize,size(model.auxstat(model.lb,1)[1],1))
    Threads.@threads for s = 1:TrainTestSize
        ok = false
        θ = model.priordraw()
        W = (model.auxstat(θ,1))[1]
        # repeat draw if necessary
        while any(isnan.(W))
            θ = model.priordraw()
            W = model.auxstat(θ,1)[1]
        end    
        params[s,:] = θ
        statistics[s,:] = W
    end
    # transform stats to robustify against outliers
    q50 = zeros(size(statistics,2))
    q01 = similar(q50)
    q99 = similar(q50)
    iqr = similar(q50)
    for i = 1:size(statistics,2)
        q = quantile(statistics[:,i],[0.01, 0.25, 0.5, 0.75, 0.99])
        q01[i] = q[1]
        q50[i] = q[3]
        q99[i] = q[5]
        iqr[i] = q[4] - q[2]
    end
    nninfo = (q01, q50, q99, iqr) 
    statistics = TransformStats(statistics, nninfo)
    params, statistics
end    
     

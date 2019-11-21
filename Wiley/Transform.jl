using Pkg
Pkg.activate(".")
using SV, Econometrics, StatsBase
using BSON: @load
using BSON: @save

# bounds by quantiles, and standardizes and normalizes around median
function transform!(data)
    @load "transformation_info.bson" q005 q50 q995 
    data = max.(data, q005')
    data = min.(data, q995')
    data = (data .- q50') ./ abs.(q50')
end

function create_transformation(statistics)
    q005 = zeros(size(statistics,2))
    q995 = similar(q005)
    q50 = similar(q005)
    for i = 1:size(statistics,2)
        q005[i] = quantile(statistics[:,i],0.005)
        q50[i] = quantile(statistics[:,i],0.5)
        q995[i] = quantile(statistics[:,i],0.995)
    end    
    @save "transformation_info.bson" q005 q50 q995
end  

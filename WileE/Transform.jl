using SV, Econometrics, StatsBase, DelimitedFiles
using BSON: @load
using BSON: @save

include("lib/transform.jl")
include("lib/create_transformation.jl")

function main()
    @load "raw_data.bson" data
    params = data[:,1:nParams]
    statistics = data[:,nParams+1:end]
    info = create_transformation(statistics)
    statistics = transform(statistics, info)
    @save "cooked_data.bson" params statistics
end    
main()

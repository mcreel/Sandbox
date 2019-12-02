using Econometrics, Statistics, Flux, Random, LinearAlgebra
using BSON: @load
using BSON: @save

include("lib/define_iterator.jl")

function Train()
    @load "cooked_data.bson" params statistics
    params = Float32.(params)
    statistics = Float32.(statistics)
    S = TrainingTestingSize # number of draws from prior
    trainsize = Int(TrainingProportion*S)
    yin = params[1:trainsize, :]'
    yout = params[trainsize+1:end, :]'
    xin = statistics[1:trainsize, :]'
    xout = statistics[trainsize+1:end, :]'
    # model
    nStats = size(xin,1)
    model = Chain(
        Dense(nStats,LayerConfig*nStats, tanh),
        Dense(LayerConfig*nStats,LayerConfig*nParams, tanh),
        Dense(LayerConfig*nParams,nParams)
    )
    θ = Flux.params(model)
    opt = AdaMax()
    loss(x,y) = Flux.mse(model(x),y)
    function monitor(e)
        println("epoch $(lpad(e, 4)): (training) loss = $(round(loss(xin,yin); digits=4)) (testing) loss = $(round(loss(xout,yout); digits=4))| ")
    end
    bestsofar = 1.0e10
    pred = 0.0 # define is here to have it outside the for loop
    inbatch = 0
    for i = 1:TrainingIters
        inbatch = rand(size(xin,2)) .< BatchProportion
        batch = DataIterator(xin[:,inbatch],yin[:,inbatch])
        Flux.train!(loss, θ, batch, opt)
        current = loss(xout,yout)
        if current < bestsofar
            bestsofar = current
            @save "best.bson" model
            xx = xout
            yy = yout
            println("________________________________________________________________________________________________")
            monitor(i)
            pred = model(xx) # map pred to param space
            error = yy .- pred
            results = [pred;error]
            rmse = sqrt.(mean(error.^Float32(2.0),dims=2))
            println(" ")
            println("RMSE for model parameters ")
            prettyprint(reshape(round.(rmse,digits=3),1,3))
            println(" ")
            println("dstats prediction of parameters:")
            dstats(pred');
            println(" ")
            println("dstats prediction error:")
            dstats(error');
        end
    end
    return nothing
end 

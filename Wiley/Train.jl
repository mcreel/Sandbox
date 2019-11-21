using Pkg
Pkg.activate(".")
using Econometrics, Statistics, Flux, Random, LinearAlgebra
using BSON: @load
using BSON: @save

#-------------------- define iterator ---------------------
struct DataIterator
   X
   Y
end
Base.length(xy::DataIterator) = min(size(xy.X, 2), size(xy.Y,2))
function Base.iterate(xy::DataIterator, idx=1)
   # Return `nothing` to end iteration
   if idx > length(xy)
       return nothing
   end
   # Pull out the observation and ground truth at this index
   result = (xy.X[:,idx], xy.Y[:,idx])
   # step forward
   idx += 1
   # return result and state
   return (result, idx)
end
#-------------------- end define iterator ---------------------

function main()
    @load "simdata.bson" params statistics nDrawsFromPrior
    whichdep = 1:3
    S = nDrawsFromPrior
    trainsize = Int(0.5*S)
    yin = params[1:trainsize, whichdep]'
    yout = params[trainsize+1:end, whichdep]'
    xin = statistics[1:trainsize, :]'
    xout = statistics[trainsize+1:end, :]'
    ydesign = params[S+1:end, whichdep]'
    xdesign = (statistics[S+1:end, :])'
    # model
    model = Chain(
        Dense(size(xin,1),100, tanh),
        Dense(100,9, tanh),
        Dense(9,3)
    )
    θ = Flux.params(model)
    opt = AdaMax()
    loss(x,y) = sqrt.(Flux.mse(model(x),y)) #+ 0.01*L2penalty(θ)
    function monitor(e)
        println("epoch $(lpad(e, 4)): (training) loss = $(round(loss(xin,yin).data; digits=4)) (testing) loss = $(round(loss(xout,yout).data; digits=4))| ")
    end
    bestsofar = 1.0e10
    pred = 0.0 # define is here to have it outside the for loop
    inbatch = 0
    for i = 1:500
        inbatch = rand(size(xin,2)) .< 1000.0/size(xin,2)
        batch = DataIterator(xin[:,inbatch],yin[:,inbatch])
        Flux.train!(loss, θ, batch, opt)
        current = loss(xout,yout).data
        if current < bestsofar
            bestsofar = current
            @save "best.bson" model
            xx = xdesign
            yy = ydesign
            println("________________________________________________________________________________________________")
            monitor(i)
            pred = model(xx).data # map pred to param space
            error = yy .- pred
            results = [pred;error]
            rmse = sqrt.(mean(error.^2.0,dims=2))
            println(" ")
            println("True values α, ρ, σ: ")
            prettyprint(reshape(round.(yy[1:3,1],digits=3),1,3))
            println(" ")
            println("RMSE for α, ρ, σ: ")
            prettyprint(reshape(round.(rmse,digits=3),1,3))
            println(" ")
            println("dstats prediction:")
            dstats(pred');
            println(" ")
            println("dstats prediction error:")
            dstats(error');
        end
    end
    return pred
end 
pred = main();

# simulates data from prior, trains and tests the net, and returns
# the trained net and the information for transforming the inputs

using Statistics, Flux
using Base.Iterators

function MakeNeuralMoments(model::SMImodel;TrainTestSize=1, Epochs=1000)
    data = 0.0
    datadesign = 0.0
    nParams = size(model.lb,1)
    # training and testing
    if (TrainTestSize == 1) TrainTestSize = Int64(2*nParams*1e4); end # use a default size if none provided
    params = zeros(TrainTestSize,nParams)
    data = zeros(TrainTestSize,size(model.dgp(model.lb),1)) 
    
    
    params = [model.priordraw() for i in 1:TrainTestSize]
    data = model.dgp.(params)
    @inbounds Threads.@threads for s = 1:TrainTestSize
        ok = false
        θ = model.priordraw()
        y = model.dgp(θ)
        # repeat draw if necessary
        while any(isnan.(y))
            θ = model.priordraw()
            y = model.dgp(θ)
        end    
        params[s,:] = θ
        data[s,:] = y
    end
#=
# transform stats to robustify against outliers
    q50 = zeros(size(data,2))
    q01 = similar(q50)
    q99 = similar(q50)
    iqr = similar(q50)
    for i = 1:size(data,2)
        q = quantile(data[:,i],[0.01, 0.25, 0.5, 0.75, 0.99])
        q01[i] = q[1]
        q50[i] = q[3]
        q99[i] = q[5]
        iqr[i] = q[4] - q[2]
    end
    nninfo = (q01, q50, q99, iqr)
    data = TransformStats(data, nninfo)
=#
    # train net
    TrainingProportion = 0.5 # size of training/testing
#    params = Float32.(params)
    s = std(params, dims=1)'
#    data = Float32.(data)
    trainsize = Int(TrainingProportion*TrainTestSize)
    yin = params[1:trainsize, :]'
    yout = params[trainsize+1:end, :]'
    xin = data[1:trainsize, :]'
    xout = data[trainsize+1:end, :]'
    # define the neural net
    nStats = size(xin,1)
    NNmodel = Chain(
        RNN(1, 10*nParams, tanh),
        Dense(10*nParams, 3*nParams, tanh),
        Dense(3*nParams, nParams)
    )
    loss(x,y) = Flux.huber_loss(NNmodel(x)./s, y./s; δ=0.1) # Define the loss function
    # monitor training
    function monitor(e)
        println("epoch $(lpad(e, 4)): (training) loss = $(round(loss(xin,yin); digits=4)) (testing) loss = $(round(loss(xout,yout); digits=4))| ")
    end
    # do the training
    bestsofar = 1.0e10
    pred = 0.0 # define it here to have it outside the for loop
    batches = [(xin[:,ind],yin[:,ind])  for ind in partition(1:size(yin,2), 50)]
    bestmodel = 0.0
    for i = 1:Epochs
        if i < 20
            opt = Momentum() # the optimizer
        else
            opt = ADAMW() # the optimizer
        end 
        Flux.train!(loss, Flux.params(NNmodel), batches, opt)
        current = loss(xout,yout)
        if current < bestsofar
            bestsofar = current
            bestmodel = NNmodel
            xx = xout
            yy = yout
            println("________________________________________________________________________________________________")
            monitor(i)
            pred = NNmodel(xx)
            error = yy .- pred
            results = [pred;error]
            rmse = sqrt.(mean(error.^Float32(2.0),dims=2))
            println(" ")
            println("RMSE for model parameters ")
            prettyprint(reshape(round.(rmse,digits=3),1,nParams))
            println(" ")
            println("dstats prediction of parameters:")
            dstats(pred'; short=true);
            println(" ")
            println("dstats prediction error:")
            dstats(error'; short=true);
        end
    end
    bestmodel, nninfo
end

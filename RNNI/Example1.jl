using ProgressMeter, Plots

TrainTestSize = 1000
TrainingProportion = 0.5
batchsize = 128

# the type that holds the model specifics
struct SMImodel
    modelname::String # name of model
    lb::Vector # vector of lower bounds. Can be -Inf, if desired
    ub::Vector # vector of upper bounds. Can be inf, if desired
    insupport::Function # function that checks if a draw is valid
    prior::Function # function that evaluates the prior at draw
    priordraw::Function # function that returns a draw from prior
    dgp::Function # function that returns data from model
end
#=
# bounds by quantiles, and standardizes and normalizes around median
function TransformStats(data, info)
    q01,q50,q99,iqr = info
    data = max.(data, q01')
    data = min.(data, q99')
    data = (data .- q50') ./ iqr'
    return data
end
=#

# get the things to define the structure for the model
# For your own models, you will need to supply the functions
# found in MNlib.jl, using the same formats
include("MakeNeuralMoments.jl")
include("SV/SVlib.jl")
lb, ub = PriorSupport()
nParams = size(lb,1)

# fill in the structure that defines the model
model = SMImodel("Stochastic Volatility example", lb, ub, InSupport, Prior, PriorDraw, SVmodel)

yin = [Float32.(model.priordraw()) for i in 1:TrainTestSize]
xin = (model.dgp.(yin))
xin = [Float32.(xin[i].^2.0) for i = 1:size(xin,1)]
yout = [Float32.(model.priordraw()) for i in 1:TrainTestSize]
xout = model.dgp.(yout)
xout = [Float32.(xout[i].^2.0) for i = 1:size(xout,1)]

data_loader = Flux.Data.DataLoader((xin, yin), batchsize=batchsize, shuffle=true, partial=false);
# create rnn
x_size = 1
num_labels = 3
num_hidden = 64
learning_rate = 0.001
num_epochs = 10

rnn = Chain(LSTM(1, num_hidden), BatchNorm(64, relu), Dense(num_hidden, num_hidden, tanh), Dense(num_hidden, num_labels))
pred = (x)->rnn(x)

# create optimizer
opt = RMSProp(learning_rate)

ps = params(rnn)
losses = Float32[]
for epoch ∈ 1:num_epochs
    sum_ℒ = 0.0f0
    prg = Progress(length(data_loader), 1, "Epoch $(epoch): ")
    for datum ∈ data_loader
        (X, Y) = datum
        x = reshape.((Flux.batchseq(X)), 1, :)
        y = Flux.batch([Y[i] for i in 1:batchsize])
        Flux.reset!(rnn)
        ℒ = 0.0f0        
        gs = gradient(ps) do
            ℒ = Flux.Losses.huber_loss(map(rnn, x)[end], y)
            sum_ℒ += ℒ
            return ℒ
        end
#         func()
        Flux.update!(opt, ps, gs)
        next!(prg, showvalues=Dict(:ℒ => ℒ))
    end
    push!(losses, sum_ℒ)
    print("epoch: $(epoch), loss: $(sum_ℒ)")
end


plot(losses./length(data_loader))

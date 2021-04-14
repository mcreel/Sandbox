# this tries to learn the MA(1) parameter given a sample of size n,
# using a recurrent NN.

using Flux
using Base.Iterators

# define the model
L1 = RNN(1, 10)   # number of vars in sample by number of learned stats
L2 = Dense(10, 5, tanh)
L3 = Dense(5, 1)
m = Chain(L1, L2, L3)

# Data generating process: returns sample of size n from MA(1) model, and the parameter that generated it
function dgp(reps)
    n = 100  # for future: make this random?
    ys = zeros(Float32, reps, n)
    θs = zeros(Float32, reps)
    for i = 1:reps
        ϕ = rand(Float32)
        e = randn(Float32, n+1)
        ys[i,:] = e[2:end] .+ ϕ*e[1:end-1] # MA1
        θs[i] = ϕ
    end
    return ys, θs
end    

# make the data for the net: x is input, θ is output 
nsamples = 1000
x, θ = dgp(nsamples)  # these are a nsamples X 100 matrix, and an nsamples vector

xx = [[x[j,i] for i = 1:100] for j = 1:nsamples]'
m.(xx)


# train
function loss(x,y)
    Flux.reset!(m)
    sum(m.(x)[end][:] .-y)) # Define the loss function
opt = ADAM()
evalcb() = @show(loss(x, θ))
Flux.@epochs 1000 Flux.train!(loss, Flux.params(m), zip(xx,θ), opt, cb = Flux.throttle(evalcb, 1))


# this is a 3 layer model
# the first layer is recursive and nonlinear, getting the states from the raw inputs
# the second layer is a dense nonlinear
# the third layer is dense linear

using Optim, Statistics

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


# recursive layer to compute the "statistics"
function R1(h, x, Wh, Wx, b)
    h = tanh.(Wx*x .+ Wh*h .+ b)
end

# dense layer, tanh activation
function DT(x,W,b)
    tanh.(W*x .+ b)
end

# dense layer, linear activation
function DL(x,W,b)
    W*x .+ b
end

# get the state from inputs
function state!(h, x, Wh, Wx, b)
   h = R1(h,x,Wh,Wx,b)
end

# net: returns output given state
function net(h, W1, b1, W2, b2)
    y = DL(DT(h,W1, b1), W2, b2)
end

# initialize parameters of needed sizes
function params(n_inputs, n_states, nodes, n_outputs)
    Wx = randn(n_states, n_inputs)
    Wh = randn(n_states, n_states)
    br = zeros(n_states)
    W1 = randn(nodes, n_states)
    b1 = zeros(nodes)
    W2 = randn(n_outputs, nodes)
    b2 = zeros(n_outputs)
    params = 0.1 .* vcat(Wx[:], Wh[:], br, W1[:], b1, W2[:], b2)
end

# split up parameters
function params(ϕ, n_inputs, n_states, nodes, n_outputs)
    start = 1
    stop = n_states*n_inputs
    Wx = reshape(ϕ[start:stop], n_states, n_inputs)
    start = stop + 1
    stop = stop + n_states^2
    Wh = reshape(ϕ[start:stop], n_states, n_states)
    start = stop + 1
    stop = stop + n_states
    br = ϕ[start:stop]
    start = stop + 1
    stop = stop + nodes*n_states
    W1 = reshape(ϕ[start:stop], nodes, n_states)
    start = stop + 1
    stop = stop + nodes
    b1 = ϕ[start:stop]
    start = stop + 1
    stop = stop + n_outputs*nodes
    W2 = reshape(ϕ[start:stop], n_outputs, nodes)
    start = stop + 1
    stop = stop + n_outputs
    b2 = ϕ[start:stop]
    Wx, Wh, br, W1, b1, W2, b2
end

function splitParams(params, n_inputs, n_states, nodes, n_outputs)

end    
function pred(samples, ϕ)
    n = size(samples,1)
    yhat = zeros(n, n_outputs)
    # break out items from parameter vector
    Wx, Wh, br, W1, b1, W2, b2 = params(ϕ, n_inputs, n_states, nodes, n_outputs)
    for i = 1:n
        h = zeros(4) # reset the state with every new sample
        y = 0.0
        for j = 1:size(samples,2) # iterate through the obsn. in the sample to get the final state
            h = state!(h, samples[i,j], Wh, Wx, br) # get stats from the sample, recursively
        end    
        yhat[i,:] = net(h, W1, b1, W2, b2)
    end
    yhat
end

# make the data for the net: x is input, θ is output 
nsamples = 1000
samples, θ = dgp(nsamples)  # these are a nsamples X 100 matrix, and an nsamples vector
n_inputs = 1
n_states = 4
nodes = 10
n_outputs = size(θ, 2)
ϕ = params(n_inputs, n_states, nodes, n_outputs) # initial params

function rmse(samples, θ, ϕ)
    e = θ - pred(samples, ϕ)
    sqrt(mean(e.*e, dims=1))[1]
end

obj = ϕ -> rmse(samples, θ, ϕ)
lb = -2.0 .* ones(size(ϕ))
ub = 2.0 .* ones(size(ϕ))
θsa = (Optim.optimize(obj, lb, ub, ϕ, SAMIN(rt=0.5, coverage_ok = true, verbosity=3),Optim.Options(iterations=10^6))).minimizer



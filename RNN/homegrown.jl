# this is a 3 layer model
# the first layer is recursive and nonlinear, getting the states from the raw inputs
# the second layer is a dense nonlinear
# the third layer is dense linear

using Statistics
include("PrintDivider.jl")
include("samin.jl")
# Data generating process: returns sample of size n from MA(1) model, and the parameter that generated it
function dgp(reps)
    n = 100  # for future: make this random?
    ys = zeros(n, reps)
    θs = zeros(reps)
    Threads.@threads for i = 1:reps
        ϕ = rand()
        e = randn(n+1)
        ys[:,i] = e[2:end] .+ ϕ*e[1:end-1] # MA1
        θs[i] = ϕ
    end
    return ys, θs
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
    h = tanh.(Wx*x .+ Wh*h .+ b)
end

# net: returns output given state
function net(h, W1, b1, W2, b2)
    y = DL(DT(h, W1, b1), W2, b2)
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

function main()
# make the data for the net: x is input, θ is output 
nsamples = 100
n_inputs = 1
n_states = 2
nodes = 20
samples, θ = dgp(nsamples)  # these are a nsamples X 100 matrix, and an nsamples vector
n_outputs = size(θ, 2)
ϕ = params(n_inputs, n_states, nodes, n_outputs) # initial params

    function pred(sample, ϕ)
        # break out items from parameter vector
        Wx, Wh, br, W1, b1, W2, b2 = params(ϕ, n_inputs, n_states, nodes, n_outputs)
        h = zeros(n_states) # reset the state with every new sample
        for i = 1:size(sample,1) # iterate through the obsn. in the sample to get the final state
            h = state!(h, sample[i], Wh, Wx, br) # get stats from the sample, recursively
        end    
        yhat = net(h, W1, b1, W2, b2)[1]
    end

    function rmse(samples, θ, ϕ)
        sse = 0.0
        Threads.@threads for j = 1:size(samples,2)
            sse += abs2.(θ[j] .- pred(samples[:,j], ϕ))
        end
        sse
    end    

for rep = 1:10 # renew sample each time
    obj = ϕ -> rmse(samples, θ, ϕ)
    if rep == 1
        lb = -2.0 .* ones(size(ϕ))
        ub = 2.0 .* ones(size(ϕ))
    else
        lb = ϕ .- 0.1
        ub = ϕ .+ 0.1
    end    
    ϕ, junk, junk, junk = samin(obj, ϕ, lb, ub, rt=0.25, paramtol = 1.0, functol = 1e-5, coverage_ok = true, verbosity=2, maxevals = 5*size(ϕ,1))
    samples, θ = dgp(nsamples)  # these are a nsamples X 100 matrix, and an nsamples vector
end
p = [pred(samples[:,i], ϕ )[] for i = 1:nsamples]
return [θ p]
end
main()

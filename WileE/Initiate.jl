using Pkg
Pkg.activate(".")

# the following defines a function that returns a draw of the statistic, given the parameter
global const WileE_model = θ -> sqrt(n)*aux_stat(SVmodel(θ::Array{Float64,1}, n::Int64, burnin::Int64)[1])

# controls and characteristics of model
global const n = 1000
global const burnin = 100

global const θtrue = [exp(-0.736/2.0), 0.9, 0.363]
global const lb = [0.0, 0.0, 0.0]
global const ub = [2.0, 0.99, 1.0]
global const nParams = 3

# controls of raw draws from prior, for training net
global const TrainingTestingSize = Int64(1e5)
global const TrainingProportion = 0.5
global const LayerConfig = 3
global const BatchProportion = 0.05

# controls of the MSM simulations
global const nSimulationDraws = 100

println("Ready to run the SV project!");


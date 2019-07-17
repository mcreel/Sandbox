#=
Simple example of use of montecarlo.jl
From the system prompt, execute
mpirun -np X julia pi_mpi.jl
setting X to the number of ranks you'd
like to use, subject to  X-1 being an even divisor
of 1e6, e.g., set X=5.
=#

using LinearAlgebra, Statistics, DelimitedFiles

include("montecarlo.jl")
include("mc_rep.jl")

# this function reports intermediate results during MC runs
function monitor(sofar, results)
    # Examine results every 12.5*10^5 draws
    if mod(sofar, 20) == 0
        dstats(results[1:sofar,:])
        writedlm("results.txt", results[1:sofar,:])
    end
end

# do the monte carlo: 10^6 reps of single draws
function main()
    MPI.Init()
    n_evals = 1000 # desired number of MC reps
    n_returns = 9
    batchsize = 1
    montecarlo(mc_rep, monitor,
               MPI.COMM_WORLD, n_evals, n_returns, batchsize)
    MPI.Finalize()
end

main()

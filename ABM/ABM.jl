using Distributions, StatsPlots

@views function alw(a=0.0003, b=0.0014, σf=0.03; T=20000, burnin=5000)
# numbers of agents
Nc = 100  # number of noise traders
n = 50    # initial number of traders in optimistic state
ns = zeros(T+1) # container
t = -burnin # initial continuous time
dt = 1 # discrete time at start of recording
while t < T+1
    # rates
    π⁺ = (Nc - n)*(a + b*n) # rate of positive switches
    π⁻ = n*(a + b*(Nc - n)) # rate of negative switches
    # wait time for state change
    λ = n*π⁻  + (Nc - n)*π⁺ # rate of state change
    Δt = rand(Exponential(1.0/λ)) # random wait time for state change
    t += Δt # current time
    if t > dt
        ns[dt] = n # record n before changing it
        dt +=1 # increment trading day
    end
    # change state
    rand() < n*π⁻/λ ? n -=1 : n +=1
end
x = 2.0 .* ns./Nc .- 1.0  # defined in paragraph following eq. 2
# returns, defined in CL2018 eq. 9 (note that NcVc/(NfVf) is set to 1,
# as discussed in paragraph following Table 1
rets = σf .* randn(T) + (x[2:end] - x[1:end-1]) 
return rets
end
r = alw()
plot(r)

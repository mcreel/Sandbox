# these are the true params
α = -7.36
ρ = 0.9
σ = 0.363
θtrue = [α, ρ, σ] # true param values, on param space

# plain MCMC fit
posmean = vec(mean(chain[:,1:3],dims=1))

# CIs
inci = zeros(3)
for i = 1:3
    lower = quantile(chain[:,i],0.05)
    upper = quantile(chain[:,i],0.95)
    inci[i] = θtrue[i] >= lower && θtrue[i] <= upper
end

p1 = npdensity(chain[:,1]) # example of posterior plot
p2 = npdensity(chain[:,2]) # example of posterior plot
p3 = npdensity(chain[:,3]) # example of posterior plot
display(plot(p1,p2,p3))
prettyprint([posmean inci])


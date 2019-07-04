# these are the true params
σe = exp(-0.736/2.0)
ρ = 0.9
σu = 0.363
θtrue = [σe, ρ, σu] # true param values, on param space

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


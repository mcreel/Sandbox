include("SV.jl")
posterior, chain, NUTS_tuned = main()
# Extract the parameter and plot the posterior
σu_hat = [i[1][1] for i in posterior]
#σu_hat = 2.0*log.(σu_hat)
dstats(σu_hat)
post_dens_sigu = npdensity(σu_hat)
# Extract the parameter and plot the posterior
ρhat = [i[2][1] for i in posterior]
dstats(ρhat)
post_dens_rho = npdensity(ρhat)
# Extract the parameter and plot the posterior
σe_hat = [i[3][1] for i in posterior]
dstats(σe_hat)
post_dens_sige = npdensity(σe_hat)

# Effective sample sizes (of untransformed draws)
ess = mapslices(effective_sample_size, get_position_matrix(chain); dims = 1)
NUTS_statistics(chain)

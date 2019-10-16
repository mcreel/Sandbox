function RawMoment(θ, m)
    α = θ[1]
    ρ = θ[2]
    σ = θ[3]
    exp(m*α/2.0 + (m^2.0)*σ/(1.0 - ρ^2.0)/4.0) # Shepphard GMM notes, page 384
end    

function UnconditionalMoments(θ, y)
    α = θ[1]
    ρ = θ[2]
    σ = θ[3]
    vcat(
        mean(abs.(y)) - sqrt(2.0/pi)*RawMoment(θ,1),
        mean(y.^2.0) - RawMoment(θ,2),
        mean(abs.(y.^3.0)) - 2.0*sqrt(2.0/pi)*RawMoment(θ,3),
        mean(y.^4.0) - 3.0*RawMoment(θ,4)
        )
end        




using LogDensityProblems

function LogDensityProblems.logdensity(vgb::LogDensityProblems.ValueGradientBuffer,
                                       ℓ::YourProblem, x::AbstractVector{T})
    gradient = vgb.buffer
    v0 = logdensity(Value, ℓ, x)
    ϵ = 1e-6
    x′ = copy(x)
    for i in axes(x, 1)
        x′[i] = x[i] + ϵ
        gradient[i] = (logdensity(Value, ℓ, x′) - v0) / ϵ
        x′[i] = x[i]
    end
    ValueGradient(oftype(eltype(gradient), v0), gradient)
end

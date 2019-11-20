# transforms R̨^k to parameter space, or the inverse if
# second arg is true
function transform(v, inverse=false)
    v = v[:]
    lb = [0.0, 0.0, 0.0]
    ub = [2.0, 0.99, 1.0]
    if !inverse
        ret = (1.0 ./(1.0 .+ exp.(-v))) .* (ub .- lb) .+ lb
    else
        ret = -1.0.*log.((ub .- lb) ./(v .- lb) .- 1.0)
    end
    ret
end



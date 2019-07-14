function func(x)
    if x < 0.0
        y = exp(x)
    else
        y = 1.0 + x + x^2
    end
    return y
end    

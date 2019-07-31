using LinearAlgebra, Statistics
d = 20;    #// number of dimensions
k = 5;      #// number of factors

W = randn(d,k)
S = W*W' + diagm(rand(d))
T = diagm(1.0 ./sqrt.(diag(S)))
S = T*S*T
S = tril(S) + tril(S,-1)'
isposdef(S)

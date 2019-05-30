# returns the variable (or matrix), lagged p times,
# with the first p rows filled with ones (to avoid divide errors)
# remember to drop those rows before doing analysis
function lag(x,p::Int64)
	n = size(x,1)
	lagged_x = zeros(eltype(x),n,p)
	lagged_x = [ones(p); x[1:n-p]]
end	 

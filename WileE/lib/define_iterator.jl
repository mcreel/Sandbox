struct DataIterator
   X
   Y
end

Base.length(xy::DataIterator) = min(size(xy.X, 2), size(xy.Y,2))

function Base.iterate(xy::DataIterator, idx=1)
   # Return `nothing` to end iteration
   if idx > length(xy)
       return nothing
   end
   # Pull out the observation and ground truth at this index
   result = (xy.X[:,idx], xy.Y[:,idx])
   # step forward
   idx += 1
   # return result and state
   return (result, idx)
end

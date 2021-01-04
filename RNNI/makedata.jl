function makedata(nsamples)
    data = zeros(1,500, 1, nsamples)
    for i=nsamples
        params[:,:,:,i] = model.priordraw()	
        data[:,:,:,i] = model.dgp(params[:,:,:,i])
    end
    return params, data
end

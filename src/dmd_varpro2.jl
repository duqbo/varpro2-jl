
function phidmd!(phimat,alpha,t)
    c0 = zero(Complex{Float64});
    c1 = one(Complex{Float64});
    tc = t + 0.0*im
    BLAS.gemm!('N','T',c1,tc,alpha,c0,phimat);
    for I in eachindex(phimat)
        phimat[I] = exp(phimat[I]);
    end
end

function dphidmd_slab!(dphislab,islab,alpha,i,t)
    l = 1
    dphislab[:,1] = t.*exp(alpha[i]*t)
    islab[1] = i
    
    return l
end


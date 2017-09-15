include("levmarq.jl")

type VarPro2Params

    xdata::Array{Complex{Float64},2}
    m::Int64
    n::Int64
    is::Int64
    ia::Int64    
    lmax::Int64
    
end

function VarPro2Params1(xdata,n,ia,lmax)

    m,is = size(xdata)
    return VarPro2Params(xdata,m,n,is,ia,lmax)
    
end

type VarPro2Vars

    b::Array{Complex{Float64},2}
    res::Array{Complex{Float64},2}
    btslab::Array{Complex{Float64},2}    
    phimat::Array{Complex{Float64},2}
    u::Array{Complex{Float64},2}
    sd::Array{Float64,1}
    vct::Array{Complex{Float64},2}
    r::Int64
    vctslab::Array{Complex{Float64},2}
    dphislab::Array{Complex{Float64},2}
    lastalpha::Array{Complex{Float64},1}
    islab::Array{Int64,1}

end

function VarPro2Vars(m,n,is,ia,lmax)

    b = zeros(Complex{Float64},n,is)
    res = zeros(Complex{Float64},m,is)

    phimat = zeros(Complex{Float64},m,n)
    u = zeros(Complex{Float64},m,min(m,n))
    sd = zeros(Float64,min(m,n))
    vct = zeros(Complex{Float64},min(m,n),n)
    
    r = 0

    dphislab = zeros(Complex{Float64},m,lmax)
    btslab = zeros(Complex{Float64},is,lmax)
    vctslab = zeros(Complex{Float64},min(m,n),lmax)
    
    lastalpha = zeros(Complex{Float64},ia)
    islab = zeros(Int64,lmax)

    return VarPro2Vars(b,res,btslab,phimat,
                       u,sd,vct,r,vctslab,dphislab,
                       lastalpha,islab)
    
end

type VarPro2Result

    # type struct for result of varpro2 function

    phi::Array{Complex{Float64},2}
    b::Array{Complex{Float64},2}
    alpha::Array{Complex{Float64},1}
    niter::Int64
    err::Array{eltype(real(Complex{Float64})),1}
    imode::Int64
    alphas::Array{Complex{Float64},2}

end

function VarPro2ResultLM(rsltlm::LevMarqResult,pars::VarPro2Params,
                       vars::VarPro2Vars,phi!)
    m = pars.m
    n = pars.n
    is = pars.is
    ia = pars.ia
    lmax = pars.lmax

    alpha = rsltlm.minimizer

    if (alpha != vars.lastalpha)
        varpro2_update!(pars,vars,alpha,phi!)
    end

    return VarPro2Result(vars.phimat,vars.b,rsltlm.minimizer,
           rsltlm.niter,rsltlm.err,rsltlm.imode,rsltlm.min_history)

end

    
    

function varpro2_update!(pars::VarPro2Params,vars::VarPro2Vars,
                         alpha::Array{Complex{Float64},1},phi!)

    xdata = pars.xdata
    m = pars.m

    phimat = vars.phimat
    u = vars.u
    sd = vars.sd
    vct = vars.vct
    b = vars.b
    res = vars.res
    lastalpha = vars.lastalpha

    phi!(phimat,alpha)

    f = svdfact(phimat,thin=true)
    copy!(u,f[:U])
    copy!(sd,f[:S])
    copy!(vct,f[:Vt])
    epsmachine = eps(eltype(real(phimat[1,1])))
    tolrank = m*epsmachine
    vars.r = max(sum(sd .> tolrank*sd[1]),1)

    copy!(b,vct'*(Diagonal(sd)\(u'*xdata)))
    copy!(res,xdata - phimat*b)
    
    copy!(lastalpha,alpha)

end
            
function res_varpro2!(res,alpha,pars::VarPro2Params,
                      vars::VarPro2Vars,phi!)

    # check that phimat, u, sd, vd, etc. are up to date

    if (alpha != vars.lastalpha)
        varpro2_update!(pars,vars,alpha,phi!)
    end

    # reshape and copy over current residual

    m = pars.m
    is = pars.is
    copy!(res,reshape(vars.res,m*is,1))

end

function jac_varpro2_slab!(jacmat,alpha,pars::VarPro2Params,
                           vars::VarPro2Vars,phi!,dphi!,
                           iffulljac=true)

    c0 = zero(Complex{Float64})
    c1 = one(Complex{Float64})

    # check that phimat, u, sd, vd, etc. are up to date

    if (alpha != vars.lastalpha)
        varpro2_update!(pars,vars,alpha,phi!)
    end

    m = pars.m
    ia = pars.ia
    is = pars.is

    dphislab = vars.dphislab
    vctslab = vars.vctslab
    b = vars.b
    btslab = vars.btslab
    res = vars.res

    r = vars.r
    vct = view(vars.vct,1:r,:)
    u = view(vars.u,:,1:r)
    sd = view(vars.sd,1:r)

    islab = vars.islab

    jacmattemp = zeros(Complex{Float64},m,is)

    for i = 1:ia

        l = dphi!(dphislab,islab,alpha,i)
        dphislabv = view(dphislab,:,1:l); islabv = view(islab,1:l)
        btslabv = view(btslab,:,1:l)
        vctslabv = view(vctslab,:,1:l)
        copy!(btslabv,transpose(b[islabv,:]))
        copy!(vctslabv,vct[:,islabv])

        if (iffulljac)
            tempmat = BLAS.gemm('C','N',c1,dphislabv,res)
            jacmatmatv = view(jacmattemp,1:r,:)
            BLAS.gemm!('N','N',c1,vctslabv,tempmat,c0,jacmatmatv)
            tempmat = Diagonal(sd)\jacmatmatv
            BLAS.gemm!('N','N',c1,u,tempmat,c0,jacmattemp)
        else
            fill!(jacmattemp,c0)
        end

        tempmat = BLAS.gemm('C','N',c1,u,dphislabv)
        BLAS.gemm!('N','N',-c1,u,tempmat,c1,dphislabv)
        BLAS.gemm!('N','T',c1,dphislabv,btslabv,c1,jacmattemp)

        jacmat[:,i] = jacmattemp[:]

    end

end
            
    

function varpro2(pars,alpha_init,phi!,dphi!;
                 opts::LevMarqOpts = LevMarqOpts(),
                 iffulljac::Bool = true)
    
    #VARPRO2 Variable projection algorithm for multivariate data
    #
    # Attempts a fit of the columns of xdata as linear combinations
    # of the columns of phi(alpha,t), i.e.
    #
    # xdata_k = sum_j=1^n b_jk phi_j(alpha,t)
    #
    # Note that phi(alpha,t) is a matrix of dimension
    # m x n where m is length (t) and n is number of columns.
    #
    # phi_j(alpha,t) is the jth column
    # xdata_k is the kth column of the data
    #
    # Input:
    #
    # xdata - M x IS matrix of data
    # t - M vector of sample times
    # phi(alpha,t) - M x N matrix (or sparse matrix) valued 
    #              function with input alpha
    # dphi(alpha,t,i) - M x N matrix (or sparse matrix) valued
    #                 function of alpha: returns the derivative 
    #                 of the entries of phi with respect to the 
    #                 ith component of alpha
    # m - integer, number of rows of data/number of sample times
    # n - integer, number of columns of phi
    # is - integer, number of columns of data .. number of 
    #      functions to fit
    # ia - integer, dimension of alpha
    # alpha_init - initial guess for alpha
    # opts - options structure. See varpro_opts.m for details. Can
    #   be created with default values via 
    #       opts = varpro_opts();
    #
    # Output:
    
    #
    # rslt.b - N x IS matrix of coefficients .. each column gives
    #     the coefficients for one of the functions (columns
    #     of data) corresponding to the best fit
    # rslt.alpha - N vector of values of alpha for best fit
    # rslt.niter - number of iterations of the Marquardt algorithm
    # rslt.err - the error for each iteration of the algorithm
    # rslt.imode - failure mode
    #            imode = 0, normal execution, tolerance reached
    #            imode = 1, maxiter reached before tolerance
    #            imode = 4, failed to find new search direction
    #                       at step niter
    #            imode = 8, stall detected
    #            imode = 16, size of phi function return type
    #                       not compatible with xdata  
    #            imode = 32, phi function return type
    #                       not compatible with xdata  
    #
    # Example:
    #
    #   >> opts = varpro2opts() # default values
    #   >> [b,alpha,niter,err,imode,alphas] = varpro2(xdata,alpha_init,
    #                                   phi,dphi,pars,opts)
    #
    #
    # Copyright (c) 2017 Travis Askham
    #
    # Available under the MIT license
    #
    # References: 
    # - Extensions and Uses of the Variable Projection 
    # Algorithm for Solving Nonlinear Least Squares Problems by 
    # G. H. Golub and R. J. LeVeque ARO Report 79-3, Proceedings 
    # of the 1979 Army Numerical Analsysis and Computers Conference
    # - "Variable projection for nonlinear least squares problems." 
    # Computational Optimization and Applications 54.3 (2013):
    # 579-593. by Dianne P. O’Leary and Bert W. Rust. 
    #

    ifprint = opts.ifprint
    ifsavealpha = opts.ifsavex

    # get dimensions

    m = pars.m
    n = pars.n
    is = pars.is
    ia = pars.ia
    lmax = pars.lmax

    # allocate and initialize varpro2 storage

    vars = VarPro2Vars(m,n,is,ia,lmax)
    varpro2_update!(pars,vars,alpha_init,phi!)

    # wrap up varpro2 problem as standard Levenberg Marquardt

    y = reshape(pars.xdata,m*is,1)


    res! = (res,alpha) -> res_varpro2!(res,alpha,pars,vars,phi!)
    jac! = (jacmat,alpha) -> jac_varpro2_slab!(jacmat,alpha,pars,
                                               vars,phi!,dphi!,
                                               iffulljac)

    jactemp = zeros(Complex{Float64},is*m,ia)
    jac!(jactemp,alpha_init)
    println("norm ",norm(jactemp))

    # call Levenberg Marquardt routine

    rsltlm = levmarq(alpha_init,y,res!,jac!,opts)

    # translate result back to varpro2 setting

    return VarPro2ResultLM(rsltlm,pars,vars,phi!)

end

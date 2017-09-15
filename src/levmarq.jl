type LevMarqOpts

    # Optimization options for the
    # varpro2 function
    
    lambda0::Float64
    lammax::Float64
    lamup::Float64
    lamdown::Float64
    ifmarq::Bool
    maxiter::Integer
    rel_tol::Float64
    abs_tol::Float64
    eps_stall::Float64
    ifsavex::Bool
    ifprint::Bool
    iprintevery::Integer
    
end

function LevMarqOpts(;lambda0 = 1.0,
    lammax = 1.0e16,
    lamup = 2.0,
    lamdown = 2.0,
    ifmarq = true,
    maxiter = 30,
    rel_tol = 1.0e-6,
    abs_tol = 0.0,
    eps_stall = 1.0e-9,
    ifsavex = false,
    ifprint = true,
    iprintevery = 1)
    
    # Default value constructor
    # assigns reasonable values to the optimization
    # parameters

    return LevMarqOpts(lambda0,lammax,lamup,lamdown,
                ifmarq,maxiter,rel_tol,abs_tol,eps_stall,
                ifsavex,ifprint,iprintevery)
    
end

type LevMarqResult

    # Return type for Levenberg Marquardt algorithm
    
    minimizer::Array{Complex{Float64}}
    min_history::Array{Complex{Float64},2}
    niter::Integer
    err::Array{Float64,1}
    imode::Integer

end

function LevMarqResultInit(x_init,maxiter::Integer,
                       ifsavex::Bool)

    minimizer = copy(x_init)
    niter = 0
    err = zeros(Float64,maxiter)
    imode = 0
    if (ifsavex)
        min_history = zeros(Float64,length(minimizer),
                       maxiter)
    else
        min_history = zeros(Float64,1,1)
    end

    return LevMarqResult(minimizer,min_history,niter,err,
                         imode)
    
end

function levmarq(x_init,y,fun!,jac!,
                 opts::LevMarqOpts=LevMarqOpts())

    #
    # This function applies the Levenberg Marquardt
    # iteration to the initial value x_init for the
    # problem specified by y, fun! and jac!.
    #
    # The iteration finds a local solution to the
    # problem:
    #
    #    min_x vecnorm(y-x)
    #
    # Input:
    #
    # x_init - array, initial guess
    # y - array, target output
    # fun! - function of two parameters (f,x)
    #        for input x, writes output to vector f
    # jac! - function of two parameters (jac,x)
    #        for input x, writes output to matrix jac
    #        such that the (i,j)th entry of jac is the
    #        derivative of the ith output of fun! with
    #        respect to the jth component of x
    # opts - structure containing the optimization
    #        options. See LevMarqOpts type for
    #        details
    #
    # Output:
    #
    # rslt - LevMarqResult structure which stores
    #        minimizer and some statistics of the
    #        optimization (depending on options chosen)
    #        See LevMarqResult type for details

    maxscal = 1.0
    minscal = 1.0e-6
    maxinneriter = 1000

    # get optimization options from opts structure
    
    lambda = opts.lambda0
    lammax = opts.lammax
    lamup = opts.lamup
    lamdown = opts.lamdown
    ifmarq = opts.ifmarq
    maxiter = opts.maxiter
    rel_tol = opts.rel_tol
    abs_tol = opts.abs_tol
    eps_stall = opts.eps_stall
    ifsavex = opts.ifsavex
    ifprint = opts.ifprint
    iprintevery = opts.iprintevery
    
    # initialize result

    rslt = LevMarqResultInit(x_init,maxiter,ifsavex)
    rslt.imode = 0; rslt.niter = 1

    # rename things

    x = rslt.minimizer
    x_hist = rslt.min_history
    xs = [view(x_hist,:,i) for i = 1:size(x_hist,2)]
    
    # allocate things

    x0 = copy(x); x1 = copy(x)
    m = length(y); n = length(x)
    jacmat = zeros(Complex{Float64},m,n)
    fvec = zeros(Complex{Float64},m)
    res = zeros(Complex{Float64},m); res0 = copy(res)
    res1 = copy(res)
    normy = vecnorm(y)
    tau = zeros(Complex{Float64},min(m,n))
    scales = zeros(Float64,n)
    rjac = zeros(Complex{Float64},m,n)
    jacmod = zeros(Complex{Float64},m+n,n)
    jacmodtop = view(jacmod,1:m,1:n)
    jacmodbot = view(jacmod,m+1:m+n,1:n)
    rhs = zeros(Complex{Float64},m+n)
    rhstop = view(rhs,1:m)
    rhsbot = view(rhs,m+1:m+n)
    delta = copy(rhs)

    # get residual
    
    fun!(fvec,x)
    res = y-fvec
    normres = vecnorm(res)
    errlast = normres/normy

    if (normres < abs_tol || errlast < rel_tol)
        if (ifsavex)
            copy!(xs[1],x)
        end
        rslt.err[1] = errlast
        rslt.imode = 8
        rslt.niter = 0
        return rslt
    end

    for iter = 1:maxiter

        # grab Jacobian

        jac!(jacmat,x)
        fill!(scales,1.0)

        if (ifmarq)
            for j = 1:n
                scales[j] = vecnorm(jacmat[:,j])
                scales[j] = min(scales[j],maxscal)
                scales[j] = max(scales[j],minscal)
            end
        end
        
        # loop to determine lambda (lambda gives the "levenberg" part)

        # pre-compute components that don't depend on 
        # step-size parameter (lambda)
        
        # get pivots and lapack style qr for jacobian matrix

        jpvt = collect(1:n)
        LAPACK.geqp3!(jacmat,jpvt,tau)
        copy!(rjac,jacmat)
        triu!(rjac)
        copy!(rhstop,res)
        LAPACK.ormqr!('L','C',jacmat,tau,rhstop) # Q'*res
        fill!(rhsbot,0.0) # rhs = [Q'*res; 0]
        
        # check if current step size or shrunk version works
        
        # get step
        
        copy!(delta,rhs)
        copy!(jacmodtop,rjac)
        copy!(jacmodbot,lambda*diagm(scales[jpvt[1:n]]))
        levmarq_solve_special!(jacmod,delta)
        
        # new guess = x  - delta (be sure to unscramble delta)

        x0[jpvt[1:n]] = delta[1:n]

        x0 = x0 + x
        
        # corresponding residual
        
        fun!(res0,x0)
        res0 = y-res0
        normres0 = vecnorm(res0)
        err0 = normres0/normy
        
        # check if this is an improvement
        
        if (err0 < errlast) 

            # see if a smaller lambda is better
            
            lambda1 = lambda/lamdown

            # get step
            
            copy!(delta,rhs)
            copy!(jacmodtop,rjac)
            copy!(jacmodbot,lambda1*diagm(scales[jpvt[1:n]]))
            levmarq_solve_special!(jacmod,delta)
            
            # new guess = x  - delta (be sure to unscramble delta)

            x1[jpvt[1:n]] = delta[1:n]

            x1 = x1 + x

            # corresponding residual
            
            fun!(res1,x1)
            res1 = y-res1
            normres1 = vecnorm(res1)
            err1 = normres1/normy
            
            if (err1 < err0)
                lambda = lambda1
                copy!(x,x1)
                errlast = err1
                copy!(res,res1)
                normres = normres1
            else
                copy!(x,x0)
                errlast = err0
                copy!(res,res0)
                normres = normres0
            end
        else
            # if not, increase lambda until something works
            # this makes the algorithm more and more like gradient descent

            inneriter = 1

            while lambda < lammax && inneriter < maxinneriter

                lambda = lambda*lamup

                # get step
                
                copy!(delta,rhs)
                copy!(jacmodtop,rjac)
                copy!(jacmodbot,lambda*diagm(scales[jpvt[1:n]]))
                levmarq_solve_special!(jacmod,delta)
                
                # new guess = x  - delta (be sure to unscramble delta)

                x0[jpvt[1:n]] = delta[1:n]
                
                x0 = x0 + x

                # corresponding residual
                
                fun!(res0,x0)
                res0 = y-res0
                normres0 = vecnorm(res0)
                err0 = normres0/normy
                
                if (err0 < errlast) 
                    break
                end

                inneriter += 1

            end
            
            if (err0 < errlast) 
                copy!(x,x0)
                errlast = err0
                normres = normres0
                copy!(res,res0)
            else
                
                # no appropriate step length found
                
                rslt.niter = iter
                rslt.err[rslt.niter] = errlast
                if (ifsavex)
                    copy!(xs[rslt.niter],x)
                end
                rslt.imode = 4
                if (ifprint)
                    @printf "levmarq: step %d failed to find appropriate step length\n" iter
                    @printf "current residual %e\n" errlast
                end
                return rslt
            end
        end

        if (ifprint && mod(iter,iprintevery) == 0)
            @printf "levmarq: step %d current residual %e\n" iter errlast
        end
        
        if (ifsavex)
            copy!(xs[iter],x)
        end
        
        rslt.err[iter] = errlast
        if (errlast < rel_tol || normres < abs_tol)
            
            # tolerance met
            
            rslt.niter = iter
            return rslt
        end
        
        if (iter > 1)
            if (abs(rslt.err[iter-1]-rslt.err[iter])
                < eps_stall*rslt.err[iter-1])
                
                # stall detected
                
                rslt.niter = iter
                rslt.imode = 2
                if (ifprint)
                    @printf "levmarq: step %d stall detected\n" iter
                    @printf "residual reduced by less than %e times\n residual at previous step. current residual %e\n" eps_stall errlast
                end
                return rslt
            end
        end
        
end 

# failed to meet tolerance in maxiter steps

rslt.niter = maxiter
rslt.imode = 1
if (ifprint)
    @printf "levmarq: failed to reach tolerance after maxiter = %d steps\n" maxiter
    @printf "current residual %e\n" errlast
end

return rslt

end


function levmarq_solve_special!(A,b)
    # VARPRO2_SOLVE_SPECIAL Solves a system of the form 
    # 
    #     [ R ] 
    #     [---] x = b   
    #     [ D ] 
    # 
    # Where R is upper triangular and D is 
    # diagonal, using orthogonal reflectors as 
    # described in:
    #
    # Gene H. Golub and V. Pereyra, 'The Differentiation of 
    #   Pseudo-inverses and Nonlinear Least Squares Problems 
    #   Whose Variables Separate,' SIAM J. Numer. Analysis 10, 
    #   413-432 (1973).
    #
    # Fill-in is reduced for such a system. This
    # routine does not pivot
    #
    # This routine overwrites A and b
    #       

    mpn, n = size(A)
    m = mpn-n

    for i = 1:n
        ind = vcat(i, m+1:m+i)
        u = copy(A[ind,i])
        sigma =  vecnorm(u)
        beta = 1/(sigma*(sigma+abs(u[1])))
        u[1] = sign(u[1])*(sigma+abs(u[1]))
        A[ind,i:end] = A[ind,i:end]-beta*u*(u'*A[ind,i:end])
        b[ind] = b[ind]-beta*u*dot(u,b[ind])
    end

    Atop = view(A,1:n,1:n)
    triu!(Atop)
    btop = view(b,1:n)

    LAPACK.trtrs!('U','N','N',Atop,btop)

    return

end

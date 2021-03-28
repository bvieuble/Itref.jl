"""
File: itref.jl
Author: Alfredo Buttari and Bastien Vieublé
Email: bastien.vieuble@irit.fr
Github: https://github.com/bvieuble
Description: Module containing LU-based and GMRES-based iterative refinement 
for the solution of square linear system. It implements both LU-IR3 and 
GMRES-IR5 and allows a highly versatile choice of arithmetic precision 
combinations. This code has been made for research purposes and is numerical 
oriented, meaning we did not consider performance issues.
"""

module Itref

export itref, gmres

using Printf
using LinearAlgebra

import Base: isinf, isnan


function isinf(x::AbstractVector)
    found=false
    for i = 1:size(x,1)
        if(isinf(x[i]))
            found=true
            break
        end
    end
    found
end

function isnan(x::AbstractVector)
    found=false
    for i = 1:size(x,1)
        if(isnan(x[i]))
            found=true
            break
        end
    end
    found
end

function gmres(A::AbstractMatrix,
               b::AbstractVector,
               F::LU;
               xexact   =nothing,
               restrt   =nothing, 
               m        =nothing,
               tol      =nothing,
               bstop    =nothing,
               fstop    =nothing,
               verbose  =true,
               ug       =Float64,
               up       =Float64)
    """A left preconditioned restarted MGS-GMRES in two precisions. The 
    operations are applied in precision ug, except the preconditioning with
    the LU factors applied in precision up. Initial guess is zero. It was made
    to be used by the function itref, but can be used as a standalone. 

    Args:
        A (AbstractMatrix) : The matrix A of the linear system.
        b (AbstractVector) : The right-hand side.
        F (AbstractLU) : The LU factors of A for preconditioning.
        ug (DataType, Optional) : The precision at which we apply the GMRES 
            operations except the preconditioning.
        up (DataType, Optional) : The precision at which we apply the 
            preconditioning.
        restrt (Int, Optional) : Parameter defining the maximum number of 
            restart to do.
        tol (Int, Optional) : The tolerance stopping criterion condition on the
            solution precision. 
        xexact (AbstractVector, Optional) : The true solution.
        bstop (Float64, Optional) : Condition of convergence on the backward 
            error.
        fstop (Float64, Optional) : Condition of convergence on the forward 
            error.
        verbose (Bool, Optional) : "true" print the stat per iteration, 
            "false" does not print the stat.

    Returns:
        (AbstractVector) : The computed solution of the linear system in 
            working precision.
        (Vector{Float64}) : The evolution of the backward error for each 
            iteration of restart.
        (Vector{Float64}) : The evolution of the forward error for each 
            iteration of restart.
        (Int) : Number of iteration done in GMRES.
        (Bool) : true if it converges (bstop or fstop met at some iteration),
            false otherwise.
    """ 
    if(restrt == nothing)
        max_it = 1;
    else
        max_it = restrt;
    end
    
    if(tol == nothing)
        tol = sqrt(eps(ug));
    end

    if(bstop == nothing)
        bstop = tol;
    end

    if(fstop == nothing)
        fstop = tol;
    end

    if(m == nothing)
        m = size(b,1);
    end

    if(eltype(b) != up)
        bp = convert(Array{up,1},b); 
    else
        bp = b;
    end

    if(eltype(A) != up)
        Ap = convert(Array{up,2},A); 
    else
        Ap = A;
    end

    if(eltype(F) != up)
        Fp = convert(LU{up,Array{up,2}}, F); 
    else
        Fp = F;
    end

    n       = size(Ap,1);                 
    Vg      = zeros(ug, n, m+1);
    Hg      = zeros(ug, m+1,m);
    Gg      = Array{Tuple{LinearAlgebra.Givens{ug},ug},1}(undef,n);
    xg      = zeros(ug, size(b,1));
    e1      = zeros(ug, n+1);
    e1[1]   = 1.0;
    nit     = 0;
    tmp     = 0;
    cvg     = false;
    error   = AbstractFloat[];
    bkw     = AbstractFloat[];
    fwd     = AbstractFloat[];
    bnrm2   = convert(ug,norm(b,2));
 
    for iter = 1:max_it                  
        xp = convert(Array{up,1},xg);       
        rp = Fp\(bp - Ap*xp);
        rg = convert(Array{ug,1},rp);
        rnrm2 = norm(rg, 2);
        push!(error, rnrm2/bnrm2);

        if(xexact!=nothing)
            push!(fwd, norm(xexact-xp, 2)/norm(xexact, 2));
            push!(bkw, error[end]);
            if (verbose)
                @printf("it: %2d --- bkw = %.5e --- fwd = %.5e ", 
                        iter, bkw[end], fwd[end]);
                @printf("--- gmresits = %d\n", nit-tmp);
                tmp = nit;
            end
            if(bkw[end] < bstop && fwd[end] < fstop)
                cvg = true;
                break;
            end
        else
            push!(bkw, error[end]);
            if (verbose)
                @printf("it: %2d --- bkw = %.5e --- gmresits = %d\n", 
                        iter, bkw[end], nit-tmp);
                tmp = nit;
            end
            if(bkw[end] <= bstop)
                cvg = true;
                break;
            end
        end
 
        Vg[:,1] = rg./norm(rg);
        sg = norm(rg)*e1;

        for i = 1:m           
            nit = nit+1;
            wp = Fp\(Ap*convert(Array{up,1}, Vg[:,i]));
            wg = convert(Array{ug,1}, wp);
            
            for k = 1:i
                Hg[k,i]= wg'*Vg[:,k];
                wg = wg - Hg[k,i]*Vg[:,k];
            end
            Hg[i+1,i] = norm( wg );
            Vg[:,i+1] = wg / Hg[i+1,i];
            for k = 1:i-1                           
                Hg[:,i] = (Gg[k])[1]*Hg[:,i];
            end
            Gg[i]     = givens(Hg[:,i],i,i+1);
            sg[:]     = (Gg[i])[1] * sg;
            Hg[i,i]   = (Gg[i])[2];
            Hg[i+1,i] = 0.0;
            error     = push!(error, convert(ug, abs(sg[i+1]) / rnrm2));
            if(error[end] <= tol)
                yg = Hg[1:i,1:i] \ sg[1:i];
                addvec = Vg[:,1:i]*yg;
                xg[:] = xg + addvec;
                break;
            end
        end
        if(error[end] > tol)
            yg = Hg[1:m,1:m] \ sg[1:m];
            addvec = Vg[:,1:m]*yg;
            xg[:] = xg + addvec;
        end
    end
    xg, bkw, fwd, nit, cvg;
end
        
function itref(A::AbstractMatrix{TA}, 
               b::AbstractVector{TB};
               xexact   = nothing,
               F        = nothing,
               nitmax   = 20, 
               bstop    = nothing, 
               fstop    = nothing, 
               verbose  = true, 
               tol      = nothing,
               isgmres  = false,
               uf       = Float64, 
               uw       = Float64, 
               ur       = Float64, 
               ug       = nothing, 
               up       = nothing) where 
        {TA<:AbstractFloat,TB<:AbstractFloat,TX<:AbstractFloat} 
    """Implementation of LU-IR3 and GMRES-IR5.

    The choice of precision combination is free and manageable through the 
    function arguments: uf, uw, ur, ug, and up. The choice of solver (LU or 
    double precisions preconditioned MGS-GMRES) is set with the argument gmres.
    Precomputed factors can be provided with the argument F, the true solution
    can be provided with the argument xexact (if provided the forward error is 
    computed). There are three stopping criterions: backward error tolerance 
    (bstop), forward error tolerance (fstop), and the maximum number of 
    iterations for iterative refinement (nitmax). If verbose is chosen to be 
    true, the error(s) at each iterations is(are) displayed.

    The algorithm automatically adapt the precision of the matrix and vector 
    structure in argument, it tries to avoid useless copies as well.

    Args:
        A (AbstractMatrix) : The matrix A of the linear system.
        b (AbstractVector) : The right-hand side b.
        xexact (AbstractVector, Optional) : The true solution.
        F (AbstractLU, Optional): Already computed LU factors of A.
        nitmax (Int, Optional) : Maximum number of iteration in the iterative 
            refinement.
        bstop (Float64, Optional) : Condition of convergence on the backward 
            error.
        fstop (Float64, Optional) : Condition of convergence on the forward 
            error.
        verbose (Bool, Optional) : "true" print the stat per iteration, 
            "false" does not print the stat.
        tol (Float64, Optional) : The GMRES tolerance condition. 
        isgmres (Bool, Optional) : If "false" the LU solver is used, if "true"
            the preconitioned MGS-GMRES solver is used.
        uf (DataType, Optional) : Precision used for the factorization.
        uw (DataType, Optional) : Targeted precision for the solution.
        ur (DataType, Optional) : Precision used for the residual computation.
        ug (DataType, Optional) : The precision used for the GMRES 
            operations except the preconditioning.
        up (DataType, Optional) : The precision used for the GMRES for the 
            application of the preconditioning.

    Returns:
        (AbstractVector) : The computed solution of the linear system in 
            working precision.
        (Vector{Float64}) : The evolution of the backward error for each 
            iteration of iterative refinement.
        (Vector{Float64}) : The evolution of the forward error for each 
            iteration of iterative refinement.
        (Int) : Number of iterative refinement iteration.
        (Int) : The sum of the number of GMRES iteration over the iterative 
            refinement iteration.
        (Bool) : true if it converges (bstop or fstop met at some iteration),
            false otherwise.
    """
    if(eltype(A) == uf)
        Af = A;
    else
        Af = convert(Array{uf,2}, A);
    end
    
    if(eltype(A) == ur)
        Ar = A;
    else
        Ar = convert(Array{ur,2}, A);
    end
    
    if(eltype(A) == uw)
        Aw = A;
    else
        Aw = convert(Array{uw,2}, A);
    end

    if(eltype(b) == uf)
        bf = b;
    else
        bf = convert(Array{uf,1}, b);
    end
    
    if(eltype(b) == ur)
        br = b;
    else
        br = convert(Array{ur,1}, b);
    end
    
    if(eltype(b) == uw)
        bw = b;
    else
        bw = convert(Array{uw,1}, b);
    end      
                            
    if(isgmres)
        if(up == nothing)
            up = ur;
            Ap = Ar;
        elseif(up == eltype(A))
            Ap = A;
        else
            Ap = convert(Array{up,2}, A);
        end
        if(ug == nothing)
            ug = uw;
        end
    end

    if(xexact!=nothing)
        nrmxex = norm(xexact,2)
    end
    nrma = norm(Ar,2);
    nrmb = norm(br,2);
 
    if(bstop == nothing); bstop=eps(uw); end
    if(fstop == nothing); fstop=1/eps(uw); end
    
    if F != nothing
        if(eltype(F) != uf)
            Ff = convert(LU{uf,Array{uf,2}},F);
        else
            Ff = F;
        end
    else
        Ff = lu(Af);
    end
    xf = Ff\bf;
    xw = convert(Array{uw,1},xf);
                                
    if(isinf(xw) || isnan(xw))
        @printf("Reinitializing x to zero\n");
        xw=zeros(uw,size(A,1));
    end
    
    if(isgmres)
        if(up == eltype(Ff));
            Fp = Ff;
        else
            Fp = convert(LU{up,Array{up,2}},Ff);
        end
    end
                                    
    bkw = Float64[];
    fwd = Float64[];
    
    nit         = 0;
    gmresits    = 0;
    its         = 0;
    nrmr        = 0;
    cvg         = false;

    for it=1:nitmax
        nit=it;
        xr      = convert(Array{ur,1},xw);
        rr      = Ar*xr-br;
        nrmr    = norm(rr,2);
        nrmx    = norm(xr,2);
        push!(bkw, nrmr/(nrma*nrmx+nrmb));
        breakout = bkw[end]<=bstop;
        if(xexact==nothing)
            if(isgmres && verbose)
                @printf("it: %2d --- bkw = %.5e --- gmresits = %d\n",
                        it, bkw[it], its);
            elseif(verbose)
                @printf("it: %2d --- bkw = %.5e\n",
                        it, bkw[it]);
            end
        else
            push!(fwd, norm(xr-xexact,2)/nrmxex);
            if(isgmres && verbose)
                @printf("it: %2d --- bkw = %.5e --- fwd = %.5e ",
                        it, bkw[it], fwd[it]);
                @printf("--- gmresits = %d\n",
                        its);
            elseif(verbose)
                @printf("it: %2d --- bkw = %.5e --- fwd = %.5e\n",
                        it, bkw[it], fwd[it]);
            end
            breakout = breakout && (fwd[end]<=fstop);
        end
        if(breakout)
            cvg = true;
            break;
        end
        rrnrm = norm(rr, 2);
        rrs = rr/rrnrm;
        if(isgmres)
            rp     = convert(Array{up,1}, rrs);
            dg, _, _, its,_ = gmres(Ap, rp, Fp, restrt=1, tol=tol, 
                                    verbose=false, ug=ug, up=up);
            gmresits+=its;
            dw      = convert(Array{uw,1},dg);
        else
            rf = convert(Array{uf,1},rrs);
            df = Ff\rf;
            dw = convert(Array{uw,1},df);
        end
        xw = xw - convert(uw,rrnrm)*dw;
    end 
    xw, bkw, fwd, nit, gmresits, cvg;
end        

end

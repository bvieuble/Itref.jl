[![Build Status](https://travis-ci.com/bvieuble/Itref.jl.svg?branch=master)](https://travis-ci.com/bvieuble/Itref.jl)
[![codecov](https://codecov.io/gh/bvieuble/Itref.jl/branch/master/graph/badge.svg?token=8MJY6KCSS5)](https://codecov.io/gh/bvieuble/Itref.jl)

# Itref.jl: mixed-precision iterative refinement algorithms for Julia

## Implemented algorithms

This Julia package contains LU-based and GMRES-based iterative refinement 
for the solution of square linear system

<p align=center> Ax=b,&nbsp;&nbsp;&nbsp;&nbsp;A∈ℝ<sup>n ✕ n</sup>,&nbsp;&nbsp;&nbsp;&nbsp;b∈ℝ<sup>n</sup>. </p>

It implements both LU-IR3 and GMRES-IR5 and allows a highly versatile choice of 
arithmetic precision combinations. This code has been made for research purposes 
and is numerical oriented, meaning we did not consider performance issues.

* **LU-IR3**  
   LU-based iterative refinement is the most known form of iterative refinement. It has been extended to up to three precision in *"Accelerating the Solution of Linear Systems by Iterative Refinement in Three Precisions"* (Carson and Higham 2018)<sup>[1](#myfootnote1)</sup>. The goal being to accelerate a direct solving by processing the factorization in low precision and then recover a good solution precision by mean of refinement steps.

   > **Algorithm:** LU-based iterative refinement in three precisions  
   > Compute the LU factorization A = LU in precision (u<sub>f</sub>)  
   > Initialize x<sub>0</sub>  
   > **while not** converged **do**  
   > &nbsp;&nbsp;&nbsp;&nbsp;Compute r<sub>i</sub> = b-Ax<sub>i</sub> in precision (u<sub>r</sub>)  
   > &nbsp;&nbsp;&nbsp;&nbsp;Solve Ad<sub>i</sub> = r<sub>i</sub> by d<sub>i</sub> = U<sup>-1</sup>L<sup>-1</sup>r<sub>i</sub> in precision (u<sub>f</sub>)  
   > &nbsp;&nbsp;&nbsp;&nbsp;Compute x<sub>i+1</sub> = x<sub>i</sub> + d<sub>i</sub> in precision (u)  
   > **end while**  

   u<sub>f</sub>, u<sub>r</sub>, and u are the arithmetic precisions to set up. In Julia we can use the following arithmetic precisions: bfloat16 (bfloat, `using BFloat16s`<sup>[4](#myfootnote4)</sup>, type:`BFloat16`), fp16 (half, type:`Float16`), fp32 (single, type:`Float32`), fp64 (double, type:`Float64`), fp128 (quadruple, `using Quadmath`<sup>[5](#myfootnote5)</sup>, type:`Float128`). It should be clear that bfloat16, fp16, and fp128 are simulated, this code does not use accelerators.


* **GMRES-IR5**  
   GMRES-based iterative refinement has been first introduced in *"A New Analysis of Iterative Refinement and Its Application to Accurate Solution of Ill-Conditioned Sparse Linear Systems"* (Carson and Higham 2017)<sup>[2](#myfootnote2)</sup> in two precisions to process ill-conditioned systems, LU-IR3 being more sensitive to the conditioning. This algorithm has been extended to five precisions in *"Five-precision GMRES-based Iterative Refinement"* (Amestoy et al. 2021)<sup>[3](#myfootnote3)</sup>

   > **Algorithm:** GMRES-based iterative refinement in five precisions  
   > Compute the LU factorization A = LU in precision (u<sub>f</sub>)  
   > Initialize x<sub>0</sub>  
   > **while not** converged **do**  
   > &nbsp;&nbsp;&nbsp;&nbsp;Compute r<sub>i</sub> = b-Ax<sub>i</sub> in precision (u<sub>r</sub>)  
   > &nbsp;&nbsp;&nbsp;&nbsp;Solve U<sup>-1</sup>L<sup>-1</sup>Ad<sub>i</sub> = r<sub>i</sub> by GMRES at precision (u<sub>g</sub>) with matrix-vector products with Ã = U<sup>-1</sup>L<sup>-1</sup>A computed at precision (u<sub>p</sub>)  
   > &nbsp;&nbsp;&nbsp;&nbsp;Compute x<sub>i+1</sub> = x<sub>i</sub> + d<sub>i</sub> in precision (u)  
   > **end while**  

## How to use

This package provides the function `itref` implementing LU-IR3 and GMRES-IR5. Here is the header of the function, except for `A` and `b` the other arguments are optional 

    function itref(A, b;                     # Matrix and right-hand side of the system
                   xexact   = nothing,       # Exact solution for err computation
                   F        = nothing,       # LU factors if already computed               
                   nitmax   = 20,            # Nb max of IR iteration
                   bstop    = eps(uw),       # Stopping cond on the backward
                   fstop    = nothing,       # Stopping cond on the forward
                   verbose  = true,          # 'true': print errs; 'false': no print
                   tol      = sqrt(eps(ug)), # Stopping cond in GMRES
                   isgmres  = false,         # 'true': GMRES; 'false': LU
                   uf       = Float64,       # Precision uf
                   uw       = Float64,       # Precision u
                   ur       = Float64,       # Precision ur
                   ug       = uw,            # Precision ug
                   up       = ur             # Precision up
                  ) 

**Example 1:** setup for GMRES-IR5 for computing a solution with a forward/backward errors of order 10<sup>-15</sup>, with GMRES solver in double precision except the preconditioning applied in quadruple precision, and a low tolerance. This is a good configuration for precision and robstness.

    itref(A, b;            
          isgmres=true,
          xexact=xexact,
          nitmax=50, 
          bstop=1e-15,
          fstop=1e-15,
          verbose=false,
          tol=1e-12,
          uf=fp16,
          uw=fp64,
          ur=fp128,
          ug=fp64,
          up=fp128
         );

**Example 2:** setup for GMRES-IR5 for computing a solution with a forward/backward errors of order 10<sup>-15</sup>, with a full single precision GMRES solver. This is a good configuration for precision and tradeoff robustness/performance. When `F` is provided, `itref` does not compute the factorization itself and uses instead the LU factors provided. When `xexact` and `fstop` are not provided, the stopping condition of the algorithm is then only based on the backward error.

    itref(A, b;            
          isgmres=true,
          nitmax=50, 
          F = LUcomp,
          bstop=1e-15,
          verbose=true,
          tol=1e-6,
          uw=fp64,
          ur=fp128,
          ug=fp32,
          up=fp32
         );

**Example 3:** setup for LU-IR3 for computing a solution with a forward/backward errors of order 10<sup>-15</sup>, with a LU solver. This is a good configuration for precision and performance. Since `tol`, `up`, and `ug` are parameters of the GMRES solver, when using LU it is useless to set them up.

    itref(A, b;            
          xexact=xexact,
          bstop=1e-15,
          fstop=1e-15,
          verbose=true,
          uf=fp16, 
          uw=fp64,
          ur=fp128
         );

**Example 4:** Calling `itref(A, b)` applies a fixed double precision LU-based iterative refinement.

The function `itref` returns  the computed solution in precision (u), the backward and forward errors at each iteration, the total number of iterative refinement iterations for convergence, the cumulated number of GMRES iterations over the iterative refinement iterations, a boolean stating if the algorithm converges or not.

    xw, bkws, fwds, nit, ngmresit, cvg = itref(A, b; ...);


<a name="myfootnote1">1</a>: [Accelerating the Solution of Linear Systems by Iterative Refinement in Three Precisions](https://epubs.siam.org/doi/abs/10.1137/17M1140819)  
<a name="myfootnote2">2</a>: [A New Analysis of Iterative Refinement and Its Application to Accurate Solution of Ill-Conditioned Sparse Linear Systems](https://epubs.siam.org/doi/abs/10.1137/17M1122918)  
<a name="myfootnote3">3</a>: Coming soon  
<a name="myfootnote4">4</a>: [BFloat16s Julia package](https://github.com/JuliaMath/BFloat16s.jl)  
<a name="myfootnote5">5</a>: [Quadmath Julia package](https://github.com/JuliaMath/Quadmath.jl)

# To do
---------
## Code writing
* Fit defect positions to a square root function (or parabola or whatever). 
* Figure out why eigenvector function isn't working with Numba compilation
  - I think it's because numba doesn't do boolean array indexing.
* (Figure out whether @jit works for scipy functions)
* (Rewrite compiled version of biharmonic solver)
  - This is actually probably not going to provide a significant speedup
    since basically all of the legwork is done by pre-compiled functions (FFT, cg, etc.).
  - Would want to rewrite this in C++/Cuda and then wrap in a python function. 
* ~~Write driving function for biharmonic equation (i.e. $f_1(Q)$ and $f_2(Q)$).~~
* ~~Add flag to finite difference operators to ignore boundaries.~~
* ~~Write the rest of the higher order finite difference schemes.~~
* ~~Add Laplacian operator (and maybe other sums) to lower number of calculations~~
* ~~Write `.pyf` file for the `dbihar` subroutine to make it more pythonic.~~
* ~~Write some compilation scripts for that.~~
* ~~Rewrite finite difference schemes to include "ghost point" method.~~
* ~~Write higher order finite difference scheme for dx^4, dy^4, dx^2dy^2~~
* ~~Write out a biharmonic solver in Python.~~
---------
## Testing
* Introduce sympy into the testing scheme.
* Figure out how to effectively test the methods in the `LiquidCrystalHelper` package.
* ~~Rewrite test files for the compiled versions of each of the packages.~~
  - ~~Rewrite tests for compiled FiniteDifference methods~~
  - ~~Rewrite test for compile ForwardEuler~~
* ~~Test dx^4, dy^4, and dx^2dy^2 finite difference schemes.~~
* ~~Write test script for the modified biharmonic solver~~
---------
## Documentation
* Document where I got the stuff relevant for the biharmonic solver
* Include link to documentation of Fortran stuff, as well as my own repo
* Explicitly show that $\mathbb{F}$ diagonalizes $\Lambda_2$ and $\Lambda_4^c$
* (Explain how the conjugate gradients method works)
* ~~Nondimensionalize the hydrodynamic equations.~~
* ~~Clean up explanation of the theory (and get rid of Fourier Transform stuff)~~
* ~~Explicitly write down the effect of $B$ on $U$~~
* ~~Figure out how to think of a horizontal concatenation of a tensor product~~
* ~~Explicitly find the effect of $V$ and $V^\top$ on $U$~~
* ~~Show explicitly that $VV^\top = B$~~
* ~~Write out the Woodbury identity, show how it works in this case~~
* ~~Update the readme to reflect the code organization~~
-----------
## Code refactoring
* Rewrite finite difference scheme so it works with Dirichlet or Neumann BC
  - Neumann conditions should include a whole array for the boundary.
  - This will probably be significantly less efficient, just because there
    are more pieces to keep track of. 
* ~~Rewrite simulation scripts, have them save things in a specific file.~~
-----------
## Finite Element packages to check out:
* goma
* deal.ii
* fenics
* open foam
* petsi
* petsc

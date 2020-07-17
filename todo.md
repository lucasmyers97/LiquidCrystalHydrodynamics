# Miles to go before I sleep...
---------
## Code writing
* Write `.pyf` file for the `dbihar` subroutine to make it more pythonic.
* Write some compilation scripts for that.
* Write higher order finite difference schemes.
* Write driving function for biharmonic equation (i.e. $f_1(Q)$ and $f_2(Q)$).
* Fit defect positions to a square root function (or parabola or whatever). 
---------
## Testing
* Figure out how to effectively test the methods in the `LiquidCrystalHelper` package.
* Rewrite test files for the compiled versions of each of the packages.
* Write test script for the biharmonic solver
* Introduce sympy into the testing scheme.
* Need to test higher order finite difference schemes.
---------
## Documentation
* Clean up explanation of the theory (and get rid of Fourier Transform stuff)
* Document where I got the stuff relevant for the biharmonic solver
* Include link to documentation of Fortran stuff, as well as my own repo
-----------
## Code refactoring
* Rewrite finite difference scheme so it works with Dirichlet or Neumann BC
  - Neumann conditions should include a whole array for the boundary.
* Rewrite simulation scripts, have them save things in a specific file.
-----------
## Finite Element packages to check out:
* goma
* deal.ii
* fenics
* open foam
* petsi
* petsc

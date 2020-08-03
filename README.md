# Liquid-Crystal-Hydrodynamics
Summer 2020 work at the University of Minnesota simulating nematic liquid crystals including hydrodynamic effects.

## Code Structure

* `src` holds all of the lower level numerical-type source code. This includes `FiniteDifference.py` which handles the finite difference differentiation scheme and `LiquidCrystalHelper.py` which holds methods to plot, generate, etc. $Q$-tensor configurations. Additionally `LiquidCrystalHelper.py` holds the relevant equations of motion for $\eta$, $\mu$, and $\nu$ using the `FiniteDifference.py` library. The code in both of these are copied into so-called "Compiled" scripts, whereby the `numba` `jit` operator is applied to enable just-in-time compilation for faster processing. Finally, there's a modified biharmonic equation solver `biharm.py` which uses the `scipy` package's FFT and Conjugate Gradient methods.

* `tests` holds tests for several of the methods in these packages. By the end I hope to have testing scripts for all of the `FiniteDifference.py` methods, the `biharm.py` methods, and the `FiniteDifferenceCompiled.py` methods. The `LiquidCrystalHelper.py` methods are more difficult to automatically test. These are set up using the `unittest` package in Python, so that one may run `python -m unittest` from the command line to automatically run all of the testing scripts.
* `simulations` hold all of the LC simulations. These are run in Jupyter Notebooks calling the methods from the scripts in the `src` folder. Each simulation has a sub-folder in which the relevant figures are saved. 

* `helper-scripts` is just a collection of notebooks which have helped in writing the source code. So far the only file helps generate stencils for the finite difference method.

* `junk` is a collection of things that are no longer relevant to the project, but might be worth not throwing away. This includes the Fortran-compiled biharmonic solver, along with the relevant python scripts to compile it. 

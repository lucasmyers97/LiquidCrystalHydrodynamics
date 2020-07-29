"""

"""

import unittest

import numpy as np
import FiniteDifference as fd
import biharm as bh
import matplotlib.pyplot as plt

class TestBiharmonicSolver(unittest.TestCase):
    
    def test_biharmSolver(self):
        """
        This function tests the biharmonic equation solver found in the
        `biharm.py` package. It uses sin(X - Y)*cos(X + Y) as the source 
        function and then checks numerically applies the modified biharmonic
        equation to the solution to see if the result matches the source.
        """

origin = [-2, -4]
L = [3, 5]
shape = [400, 600]
alpha = 3
maxiter = 500

Nx = shape[0]
Ny = shape[1]
hx = L[0]/shape[0]
hy = L[1]/shape[1]
        
def applyBc(U):
    Uex = np.zeros((Nx+3, Ny+3))
    Uex[2:-2, 2:-2] = U
    Uex[0, :] = Uex[2, :]
    Uex[-1, :] = Uex[-3, :]
    Uex[:, 0] = Uex[:, 2]
    Uex[:, -1] = Uex[:, -3]
    return Uex

def d2x(X):
    return (X[2:, 1:-1] - 2*X[1:-1, 1:-1] + X[:-2, 1:-1]) / hx**2
def d2y(X):
    return (X[1:-1, 2:] - 2*X[1:-1, 1:-1] + X[1:-1, :-2]) / hy**2
def lap(X):
    return d2x(X) + d2y(X)
def bih(X):
    return lap(lap(X))

x = np.linspace(origin[0], origin[0] + L[0], num=shape[0] - 1)
y = np.linspace(origin[1], origin[1] + L[1], num=shape[1] - 1)
X, Y = np.meshgrid(x, y, indexing='ij')
F = np.sin(X - Y)*np.cos(X + Y)

biharm_solv = bh.Biharm(L, shape, alpha, cg_maxiter=maxiter)
U, info, calls = biharm_solv.solve(F)

Uext = applyBc(U)
res = bih(Uext) + alpha * d2x(d2y(Uext)) - F

plt.matshow(res)
plt.colorbar()

# U_bds = biharm_solv.applyBCs(U)

# dx = biharm_solv.hx
# dy = biharm_solv.hy
# result = ( fd.dx2(fd.dx2(U_bds, dx), dx) + fd.dy2(fd.dy2(U_bds, dy), dy)
#            + (2 + alpha)*fd.dx2(fd.dy2(U_bds, dy), dx) )
# residual = np.sum( np.abs( result[1:-1, 1:-1] - F ) )
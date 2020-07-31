"""

"""

import unittest

import numpy as np
import src.FiniteDifference as fd
import src.biharm as bh

class TestBiharmonicSolver(unittest.TestCase):
    
    def test_biharmSolver(self):
        """
        This function tests the biharmonic equation solver found in the
        `biharm.py` package. It uses sin(X - Y)*cos(X + Y) as the source 
        function and then checks numerically applies the modified biharmonic
        equation to the solution to see if the result matches the source.
        """
        
        origin = [0, 0]
        L = [3, 5]
        shape = [300, 500]
        alpha = 1e-100
        maxiter = 500
        
        x = np.linspace(origin[0], origin[0] + L[0], num=(shape[0]-1))
        y = np.linspace(origin[1], origin[1] + L[1], num=(shape[1]-1))
        X, Y = np.meshgrid(x, y, indexing='ij')
        F = np.sin(X - Y)*np.cos(X + Y)
        
        biharm_solv = bh.Biharm(L, shape, alpha, cg_maxiter=maxiter)
        U, info, calls = biharm_solv.solve(F)
        
        U_bds = biharm_solv.applyBCs(U)
        
        dx = biharm_solv.hx
        dy = biharm_solv.hy
        result = ( fd.dx4(U_bds, dx) + fd.dy4(U_bds, dy) 
                    + (2 + alpha)*fd.dx2(fd.dy2(U_bds, dy), dx)[1:-1, 1:-1] )
        residual = np.sum( np.abs( result - F ) )
        
        self.assertTrue(residual < np.sqrt(dx**2 + dy**2))
        
if __name__ == '__main__':
    unittest.main()
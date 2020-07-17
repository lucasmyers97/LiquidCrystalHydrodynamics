"""
This program tests the forward euler method in the `FiniteDifference.py`
module by using an exponential function and checking the error.

Lucas Myers
Written: July 16, 2020
"""

import unittest

import numpy as np
import src.FiniteDifference as fd

class TestFiniteDifferenceMethods(unittest.TestCase):
    
    def test_forwardEuler(self):
        """
        This function tests the `forwardEuler` function in the
        `FiniteDifference.py` package by using the differential equation
        y' = exp(t). Of course, the solution to this is y = exp(t) so we
        compare the analytic solution to the forwardEuler numerical solution.
        """
        
        def S(f, t):
            return np.exp(t)
        
        error_tol = 0.5
        dt_list = [1e-5, 1e-3, .1, 1, 2]
        n_list = [1, 3, 5, 10]
        
        for dt in dt_list:
            for n in n_list:
                t = 0
                f = np.exp(t)
                for i in range(n):
                    f = fd.forwardEuler(f, dt, S, t)
                    t += dt
                    
                error = np.abs(f - np.exp(t))
                rel_error = error/np.exp(t)
                
                self.assertTrue(rel_error/dt < error_tol)
                
if __name__ == '__main__':
    unittest.main()
"""
This function tests the `findMinima` function in the LiquidCrystalHelper
package.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.abspath('src')))

import numpy as np
import LiquidCrystalHelper as lch
import unittest

class TestFindMinima(unittest.TestCase):
    
    def test_findMinima(self):
        """
        This function tests the `findMinima` function in the
        LiquidCrystalHelper package by generating a series of Gaussians at
        various points and then seeing if the peaks are found.

        """
        
        def gaussian(x, y, a, x0, y0, c):
            
            return a*np.exp(-( (x - x0)**2 + (y - y0)**2 )/(2*c**2))
        
        # Define the domain
        shape = (500, 700)
        length = (20, 30)
        x = np.linspace(-length[0], length[0], num=shape[0])
        y = np.linspace(-length[1], length[1], num=shape[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Generate gaussian parameters
        ctrs = ( np.array([23, 79, 254, 425, 300]),
                 np.array([41, 623, 500, 235, 602]) )
        x0 = X[ctrs]
        y0 = Y[ctrs]
        a = np.array([2, 1, 3, 1, 3])
        c = np.array([2, 3, 1, 1, 1.5])
        
        f = np.zeros(shape)
        for i in range(ctrs[0].shape[0]):
            f -= gaussian(X, Y, a[i], x0[i], y0[i], c[i])
            
        minima = lch.findMinima(f)
        
        # Make sets of center coordinates and minima coordinates
        minima_coords = list(zip(minima[0], minima[1]))
        ctrs_coords = list(zip(ctrs[0], ctrs[1]))
        minima_set = set(minima_coords)
        ctrs_set = set(ctrs_coords)
        
        # Test whether sets are equal
        self.assertTrue(minima_set >= ctrs_set and minima_set <= ctrs_set)
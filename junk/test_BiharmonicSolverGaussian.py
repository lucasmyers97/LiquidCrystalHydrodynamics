"""
This script tests whether the biharmonic solver `bihar.f` compiled into the
python module `bihar.cp37-win_amd64.pyd` is able to solve the biharmonic eq.
for a Gaussian on a square.

Lucas Myers
Written July 20, 2020
"""

import unittest
import numpy as np

class TestBiharmonicSolver(unittest.TestCase):
    
    def test_biharWithGaussian(self):
        """
        Here we just generate the effect of a squared Laplacian on a gaussian.
        If the gaussian is:
            
            A*exp((x - x0)**2/(2*sigx**2))*exp((y - y0)**2/(2*sigy**2))
            
        so that the derivative is
        """
        
        def gauss(x, y, A=1, x0=0, y0=0, 
                  sigx=1/np.sqrt(2), sigy=1/np.sqrt(2)):
            
            return A * np.exp( (x - x0)**2/(2*sigx**2) )\
                     * np.exp( (y - y0)**2/(2*sigy**2) )
            
        def bihar_gauss(x, y, A=1, x0=0, y0=0, 
                        sigx=1/np.sqrt(2), sigy=1/np.sqrt(2)):
            
            return ( sigx**8 * (3*sigy**4 + 6*sigy**2*(y - y0)**2 + \
                                (y - y0)**4) \
                    + 2*sigx**4*sigy**4 * (sigx**2 + (x - x0)**2) \
                                        * (sigy**2 + (y - y0)**2) \
                    + sigy**8 * (3*sigx**4 + 6*sigx**2*(x - x0)**2 + \
                                 (x - x0)**4) ) / (sigx**8*sigy**8) \
                    * gauss(x, y, A, x0, y0, sigx, sigy)
                    
        
"""
This program holds all of the algorithms necessary to take finite difference
spatial derivatives. Additionally, has a forward Euler scheme which accepts
functions. This version uses the numba package to compile the functions to
run quicker (hopefully on a parallel computing device).

Lucas Myers
Created: July 4, 2020
Updated: July 8, 2020
"""
import numpy as np
from numba import jit

@jit(nopython=True, parallel=True, cache=True)
def dx2(f, dx):
    """
    Returns second order finite difference approximation of d^2 f/dx^2 for
    a 2D field f.

    Parameters
    ----------
    f : ndarray
        An nxn array representing a smooth function f: R^2 -> R. 
    dx: float
        Specifies the x-direction spacing of adjacent gridpoints. 

    Returns
    -------
    dx2f : ndarray
        nxn array representing the second partial of f with respect to x. 
        Neumann boundary conditions used with normal derivatives at the
        boundaries set to zero. 

    """
    
    dx2f = np.zeros(f.shape)
    dx2f[1:-1, :] = (f[2:, :] - 2*f[1:-1, :] + f[0:-2, :]) / dx**2
    
    dx2f[0, :] = (8*f[1, :] - f[2, :] - 7*f[0, :]) / (2*dx**2)
    dx2f[-1, :] = (8*f[-2, :] - 7*f[-1, :] - f[-3, :]) / (2*dx**2)
    
    
    return dx2f

@jit(nopython=True, parallel=True, cache=True)
def dy2(f, dy):
    """
    Returns second order finite difference approximation of d^2 f/dx^2 for
    a 2D field f.

    Parameters
    ----------
    f : ndarray
        An nxn array representing a smooth function f: R^2 -> R. 
    dy: float
        Specifies the y-direction spacing of adjacent gridpoints. 

    Returns
    -------
    dx2f : ndarray
        nxn array representing the second partial of f with respect to y. 
        Neumann boundary conditions used with normal derivatives at the
        boundaries set to zero. 

    """
    
    dy2f = np.zeros(f.shape)
    dy2f[:, 1:-1] = (f[:, 2:] - 2*f[:, 1:-1] + f[:, 0:-2]) / dy**2
    
    dy2f[:, 0] = (8*f[:, 1] - f[:, 2] - 7*f[:, 0]) / (2*dy**2)
    dy2f[:, -1] = (8*f[:, -2] - 7*f[:, -1] - f[:, -3]) / (2*dy**2)
    
    
    return dy2f

def makeForwardEuler(S):  

    @jit(nopython=True, parallel=True)
    def forwardEuler(f, dt, a, b, dx, dy):
        """
        Takes one step of size `dt` for a forward euler scheme characterized by
        the equation f_{n + 1} = f + dt*S(f, *args) where *args are any other
        arguments that the time evolution might depend on.
    
        Parameters
        ----------
        f : ndarray
            mxn array representing some smooth function f: R^2 -> R which is
            parameterized by time t.
        dt : double
            Finite time interval of a step in this method
        S : function
            Function which gives \partial f/\partial t. It takes f and any number
            of other arguments as parameters.
        *args : arguments of S
            The arguments fed into S used to calculate \partial f/\partial t for
            the forward euler method.
    
        Returns
        -------
        f_n1 : ndarray
            mxn array representing f: R^2 -> R at some later time t + dt as
            approximated by the forward euler method.
    
        """
        
        f_n1 = f + dt*S(f, a, b, dx, dy)
        
        return f_n1
    
    return forwardEuler

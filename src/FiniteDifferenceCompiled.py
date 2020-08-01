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
        An mxn array representing a smooth function f: R^2 -> R. 
    dx: float
        Specifies the x-direction spacing of adjacent gridpoints. 

    Returns
    -------
    dx2f : ndarray
        mxn array representing the second partial of f with respect to x. 
        Neumann boundary conditions used with normal derivatives at the
        boundaries set to zero. 
        
    Notes
    -----
    We use the "ghost point" method, because we assume the normal derivatives 
    are zero at the boundaries. This consists of defining new points outside 
    the domain and setting their values equal to the set of first interior
    points -- this is consistent with the second order central difference
    approximation for first derivatives.
    """
    
    dx2f = np.zeros(f.shape)
    dx2f[1:-1, :] = (f[2:, :] - 2*f[1:-1, :] + f[0:-2, :]) / dx**2
    dx2f[0, :] = 2*(f[1, :] - f[0, :]) / dx**2
    dx2f[-1, :] = 2*(f[-2, :] - f[-1, :]) / dx**2

@jit(nopython=True, parallel=True, cache=True)
def dy2(f, dy):
    """
    Returns second order finite difference approximation of d^2 f/dx^2 for
    a 2D field f.

    Parameters
    ----------
    f : ndarray
        An mxn array representing a smooth function f: R^2 -> R. 
    dy: float
        Specifies the y-direction spacing of adjacent gridpoints. 

    Returns
    -------
    dy2f : ndarray
        mxn array representing the second partial of f with respect to y. 
        Neumann boundary conditions used with normal derivatives at the
        boundaries set to zero. 
        
    Notes
    -----
    We use the "ghost point" method, because we assume the normal derivatives 
    are zero at the boundaries. This consists of defining new points outside 
    the domain and setting their values equal to the set of first interior
    points -- this is consistent with the second order central difference
    approximation for first derivatives.

    """
    
    dy2f = np.zeros(f.shape)
    dy2f[:, 1:-1] = (f[:, 2:] - 2*f[:, 1:-1] + f[:, 0:-2]) / dy**2
    dy2f[:, 0] = 2*(f[:, 1] - f[:, 0]) / dy**2
    dy2f[:, -1] = 2*(f[:, -2] - f[:, -1]) / dy**2
    
    
    return dy2f

@jit(nopython=True, parallel=True, cache=True)
def dx4(f, dx):
    """
    Returns second order finite difference approximation of d^4 f/dx^4 for
    a 2D field f.

    Parameters
    ----------
    f : ndarray
        An mxn array representing a smooth function f: R^2 -> R. 
    dx: float
        Specifies the x-direction spacing of adjacent gridpoints. 

    Returns
    -------
    dx4f : ndarray
        (m - 2)x(n - 2) array representing the fourth partial of f with 
        respect to x. Neumann boundary conditions used with normal derivatives
        at the boundaries set to zero. 
        
    Notes
    -----
    We use the "ghost point" method, because we assume the normal derivatives 
    are zero at the boundaries. This consists of defining new points outside 
    the domain and setting their values equal to the set of first interior
    points -- this is consistent with the second order central difference
    approximation for first derivatives.

    """
    m, n = f.shape
    dx4f = np.zeros((m - 2, n - 2))
    dx4f[1:-1, :] = ( f[4:, 1:-1] + f[:-4, 1:-1] + 6*f[2:-2, 1:-1] 
                      -4*(f[3:-1, 1:-1] + f[1:-3, 1:-1])  ) / dx**4
    dx4f[0, :] = (  7*f[1, 1:-1] + f[3, 1:-1]
                   - 4*(f[2, 1:-1] + f[0, 1:-1]) ) / dx**4
    dx4f[-1, :] = ( 7*f[-2, 1:-1] + f[-4, 1:-1]
                    - 4*(f[-3, 1:-1] + f[-1, 1:-1]) ) / dx**4
    
    return dx4f

@jit(nopython=True, parallel=True, cache=True)
def dy4(f, dy):
    """
    Returns second order finite difference approximation of d^4 f/dy^4 for
    a 2D field f.

    Parameters
    ----------
    f : ndarray
        An mxn array representing a smooth function f: R^2 -> R.
    dy: float
        Specifies the y-direction spacing of adjacent gridpoints. 

    Returns
    -------
    dy4f : ndarray
        (m - 1)x(n - 1) array representing the fourth partial of f with 
        respect to y. Neumann boundary conditions used with normal derivatives
        at the boundaries set to zero. 
        
    Notes
    -----
    We use the "ghost point" method, because we assume the normal derivatives 
    are zero at the boundaries. This consists of defining new points outside 
    the domain and setting their values equal to the set of first interior
    points -- this is consistent with the second order central difference
    approximation for first derivatives.

    """
    
    m, n = f.shape
    dy4f = np.zeros((m - 2, n - 2))
    dy4f[:, 1:-1] = ( f[1:-1, 4:] + f[1:-1, :-4] + 6*f[1:-1, 2:-2] 
                      - 4*(f[1:-1, 1:-3] + f[1:-1, 3:-1]) ) / dy**4
    dy4f[:, 0] = ( 7*f[1:-1, 1] + f[1:-1, 3]
                   - 4*(f[1:-1, 2] + f[1:-1, 0]) ) / dy**4
    dy4f[:, -1] = ( 7*f[1:-1, -2] + f[1:-1, -4]
                    - 4*(f[1:-1, -3] + f[1:-1, -1]) ) / dy**4
    
    return dy4f

@jit(nopython=True, parallel=True, cache=True)
def dx2dy2(f, h):
    """
    Returns second order finite difference approximation of d^4 f/dx^2 dy^2
    for a 2D field f.

    Parameters
    ----------
    f : ndarray
        An mxn array representing a smooth function f: R^2 -> R. 
    h : float
        Specifies grid spacing. Note that this requires the x- and y- grid
        spacing to be the same. If you have different grid spacing, use
        dx2(dy2(f, dy), dx). 

    Returns
    -------
    dx2dy2f : ndarray
        An mxn array representing d^4 f/dx^2 dy^2. Neumann boundary conditions
        used with normal derivatives at boundaries set to zero.
        
    Notes
    -----
    We use the "ghost point" method, because we assume the normal derivatives 
    are zero at the boundaries. This consists of defining new points outside 
    the domain and setting their values equal to the set of first interior
    points -- this is consistent with the second order central difference
    approximation for first derivatives.

    """
    
    dx2dy2f = np.zeros(f.shape)
    
    # Interior
    dx2dy2f[1:-1, 1:-1] = ( -2*(f[:-2, 1:-1] + f[2:, 1:-1] + f[1:-1, :-2] 
                            + f[1:-1, 2:]) + f[:-2, :-2] + f[:-2, 2:]
                            + f[2:, :-2] + f[2:, 2:] + 4*f[1:-1, 1:-1] ) / h**4
     
    # Edges
    dx2dy2f[0, 1:-1] = 2*( f[1, :-2] + f[1, 2:] - f[0, :-2] - f[0, 2:]
                            + 2*(f[0, 1:-1] - f[1, 1:-1]) ) / h**4
    dx2dy2f[-1, 1:-1] = 2*( f[-2, :-2] + f[-2, 2:] - f[-1, :-2] - f[-1, 2:]
                            + 2*(f[-1, 1:-1] - f[-2, 1:-1]) ) / h**4
    dx2dy2f[1:-1, 0] = 2*( f[:-2, 1] + f[2:, 1] - f[:-2, 0] - f[2:, 0]
                            + 2*(f[1:-1, 0] - f[1:-1, 1]) ) / h**4
    dx2dy2f[1:-1, -1] = 2*( f[:-2, -2] + f[2:, -2] - f[:-2, -1] - f[2:, -1]
                            + 2*(f[1:-1, -1] - f[1:-1, -2]) ) / h**4
     
    # Corners
    dx2dy2f[0, 0] = 4*( f[1, 1] - f[0, 1] + f[0, 0] - f[1, 0] ) / h**4
    dx2dy2f[-1, 0] = 4*( f[-2, 1] - f[-1, 1] + f[-1, 0] - f[-2, 0] ) / h**4
    dx2dy2f[0, -1] = 4*( f[1, -2] - f[1, -1] + f[0, -1] - f[0, -2] ) / h**4
    dx2dy2f[-1, -1] = 4*( f[-2, -2] - f[-2, -1] 
                          + f[-1, -1] - f[-1, -2] ) / h**4
    
    return dx2dy2f

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
            Function which gives \partial f/\partial t. It takes f and any 
            number of other arguments as parameters.
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

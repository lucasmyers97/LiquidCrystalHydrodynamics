"""
This program holds all of the algorithms necessary to take finite difference
spatial derivatives. Additionally, there is a symbolic program which helps
generate stencils for higher degree and higher order derivatives, even at the
boundaries.

Lucas Myers
July 4, 2020
"""
import numpy as np
import sympy as sy

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

def forwardEuler(f, dt, S, *args, **kwargs):
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
    
    f_n1 = f + dt*S(f, *args, **kwargs)
    
    return f_n1

def genFDStencil(template, diff, order, boundary_diff=None):
    """
    This function generates a finite difference stencil from a template matrix
    which consists of 1's at gridpoints that you might need to sample during
    the finite difference scheme, a -1 at the point at which you are trying to
    caluclate the given derivative and 0's everywhere else. It returns a 
    stencil matrix. Specify the particular differential you want to calculate 
    with `diff` (x_partial, y_partial) and specify the order you want to 
    calculate it to with `order`. If you have Neumann boundary conditions, 
    specify which derivative is fixed with the boundary_diff parameter.

    Parameters
    ----------
    template : sympy Matrix
        nxn sympy Matrix where n is odd. Represents a template of gridpoints
        that you may need to sample to approximate the derivative. Should have
        1 at gridpoints you might need to sample, -1 at the point where you are
        approximating the derivative, and 0 everywhere else.
    diff : tuple of ints
        2-element tuple representing the x-degree and y-degree of the 
        derivative you are approximating.
    order : int
        The order to which the derivative approximation should be made.
    boundary_diff : tuple of ints, optional
        Tuple of ints representing which derivative is known at the boundary if
        Neumann boundary conditions are being considered. If `None` then it is
        assumed there are no boundary conditions necessary. 
        The default is None.

    Returns
    -------
    stencil : sympy Matrix
        Stencil corresponding to the finite difference scheme.
    boundary_val : number
        Coefficient in front of the fixed boundary value that you need to
        subtract from the finite difference scheme. 
        
    Examples
    --------
    >>> import FiniteDifference as fd
    >>> import sympy as sy
    >>> sy.init_printing(pretty_print=False)
    >>> template = sy.Matrix([[1, 1, 1, 1, 1],
    ...                       [1, 1, 1, 1, 1],
    ...                       [1, 1, -1, 1, 1],
    ...                       [1, 1, 1, 1, 1],
    ...                       [1, 1, 1, 1, 1]])
    >>> diff = (3, 1)
    >>> order = 2
    >>> stencil, boundary_val = fd.genFDStencil(template, diff, order)
    >>> stencil
    Matrix([
    [-1/10,   1/5, 0,  -1/5,  1/10],
    [-1/20,  1/10, 0, -1/10,  1/20],
    [    0,     0, 0,     0,     0],
    [ 1/20, -1/10, 0,  1/10, -1/20],
    [ 1/10,  -1/5, 0,   1/5, -1/10]])
    >>> boundary_val
    0
    
    `genFDStencil` can also deal with Neumann boundary conditions -- here you
    need to stipulate which derivative is normal to the boundary using the
    optional `boundary_diff` parameter.
    
    >>> import FiniteDifference as fd
    >>> import sympy as sy
    >>> sy.init_printing(pretty_print=False)
    >>> template = sy.Matrix([[0, 0, 0],
    ...                       [-1, 1, 1],
    ...                       [0, 0, 0]])
    >>> diff = (2, 0)
    >>> order = 2
    >>> boundary_diff = (1, 0)
    >>> stencil, boundary_val = fd.genFDStencil(template, diff, 
    ...                                         order, boundary_diff)
    >>> stencil
    Matrix([
    [   0, 0,    0],
    [-7/2, 4, -1/2],
    [   0, 0,    0]])
    >>> boundary_val
    -3
    """
    
    diff_degree = diff[0] + diff[1]
    m, n = template.shape
    k = diff_degree + order # total expansion degree
    x, y = sy.symbols('x y')
    
    # if not m % 2 or not n % 2:
    #     raise ValueError("Template dimensions must be odd")     
    # if m is not n:
    #     raise ValueError("Template matrix must be square")
        
    # Find center of template, marked by -1
    center = None
    for i in range(m):
        for j in range(n):
            if template[i, j] == -1:
                center = (i, j)
    if not center:
        raise ValueError("Did not set center of template")
    
    # Make list of gridpoints measured from center in the template
    # Note that it's in ij index convention
    gridpoints = []
    for i in range(m):
        for j in range(n):
            if template[i, j] == 1:
                gridpoints.append((j - center[1], center[0] - i))
           
    # Generate standard series to degree k to compare other expansions against
    f = sy.exp(x)*sy.exp(y)
    q = sy.expand( f.series(x, 0, k).removeO().series(y, 0, k).removeO() )
    q = sum(term for term in q.args 
            if sy.degree(term, x) + sy.degree(term, y) < k
            and sy.degree(term, x) + sy.degree(term, y) > 0)
    q = sy.poly(q, x, y)
    
    # Find place of desired differential, and terms excluded by b.c.
    diff_place = q.monoms(order='grlex').index(diff)
    if boundary_diff:
        excluded_place = q.monoms(order='grlex').index(boundary_diff)
    else:
        excluded_place = None
        
    # Populate linear system matrix and inhomogeneous vector w/expansion terms
    M = sy.zeros(len(gridpoints), q.length())
    boundary_diff_vec = sy.zeros(len(gridpoints), 1)
    gridpoint_idx = 0
    for (i, j) in gridpoints:
        f = sy.exp(i*x)*sy.exp(j*y)
        p = sy.expand( f.series(x, 0, k).removeO().series(y, 0, k).removeO() )
        p = sum(term for term in p.args 
                if sy.degree(term, x) + sy.degree(term, y) < k 
                and sy.degree(term, x) + sy.degree(term, y) > 0)
        p = sy.poly(p, x, y)
    
        taylor_term_idx = 0
        for deg in q.monoms(order='grlex'):
            if taylor_term_idx == excluded_place:
                M[gridpoint_idx, taylor_term_idx] = 0
                boundary_diff_vec[gridpoint_idx] = -p.coeff_monomial(deg)
            else:
                M[gridpoint_idx, taylor_term_idx] = p.coeff_monomial(deg)
            taylor_term_idx += 1
        gridpoint_idx += 1
        
    # Calc pseudoinverse, check that it inverts desired vector calc stencil
    M_inv = M.pinv()
    diff_vec = sy.zeros(q.length(), 1)
    diff_vec[diff_place] = 1
    if M_inv*M*diff_vec == diff_vec:
        stencil = template
        gridpoint_idx = 0
        centerpoint_val = 0
        for i in range(m):
            for j in range(n):
                if template[i, j] == 1:
                    stencil[i, j] = M_inv[diff_place, gridpoint_idx]
                    centerpoint_val += M_inv[diff_place, gridpoint_idx]
                    gridpoint_idx += 1
        stencil[center[0], center[1]] = -centerpoint_val
        boundary_val = (M_inv*boundary_diff_vec)[diff_place]
    else:
        print("Could not make a stencil")
        stencil = None
        boundary_val = None
        
    return stencil, boundary_val
        
    

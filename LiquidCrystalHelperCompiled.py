"""
Contains a collection of helper functions associated with generating, 
manipulating, and plotting nematic liquid crystal configurations.

Lucas Myers
Created: July 4, 2020
Updated: July 8, 2020
"""
import numpy as np
from numba import jit
from scipy.signal import find_peaks
import FiniteDifferenceCompiled as fd

# Default dimensionless A, B, C values given in Svensek and Zumer
A = -0.064
B=-1.57
C=1.29

def uniaxialQ(S, phi):
    """
    Generate a Q-tensor corresponding to a uniaxial configuration
    (i.e. specified by some S and phi). The Q-matrix is generated as
    follows:
        
        Q = S/2 [[3 cos^2 phi - 1, (3/2) sin 2phi,  0],
                 [(3/2) sin 2phi,  3 sin^2 phi - 1, 0],
                 [0,               0,              -1]]

    Parameters
    ----------
    S : ndarray
        mxn numpy array representing the value of S at each position 
        (x, y)
    phi : ndarray
        mxn numpy array representing the value of phi at each position
        (x, y)

    Returns
    -------
    Q : ndarray
        3x3xmxn matrix representing the Q-tensor at each position (x,y).
        To find Q_ij at every point in space, index as Q[i, j, :, :]. To
        find Q as a matrix at position (x_k, y_l), index as Q[:, :, k, l].

    """
    
    # allocates 3x3xmxn matrix
    Q = np.zeros((3, 3) + phi.shape)
    
    Q[0, 0] = (S/2) * (3*np.cos(phi)**2 - 1)
    Q[0, 1] = (S/2) * ((3/2)*np.sin(2*phi))
    Q[0, 2] = 0
    
    Q[1, 0] = (S/2) * ((3/2)*np.sin(2*phi))
    Q[1, 1] = (S/2) * (3*np.sin(phi)**2 - 1)
    Q[1, 2] = 0
    
    Q[2, 0] = 0
    Q[2, 1] = 0
    Q[2, 2] = -(S/2)
    
    return Q

def auxVars(Q):
    """
    Generate auxiliary variables eta, mu, nu from the Q-tensor.

    Parameters
    ----------
    Q : ndarray
        3x3xmxn numpy matrix representing the Q-tensor field.

    Returns
    -------
    eta : ndarray
        mxn numpy matrix representing the auxiliary variable eta as a
        field
    mu : ndarray
        mxn numpy matrix representing the auxiliary variable mu as a field
    nu : ndarray
        mxn numpy matrix representing the auxiliary variable nu as a field
        
    """
    
    eta = (3/2)*Q[0, 0]
    mu = Q[1, 1] + (1/2)*Q[0, 0]
    nu = Q[0, 1]
    
    return eta, mu, nu

def sparseIdx(shape, sparse_shape):
    """
    Generate an array which indexes the locations of sparse matrix
    elements. Typically one would like to pick out some evenly-spaced
    subset of matrix elements in order to plot results (e.g. director
    angles) without the plot getting crowded. This gives an index array
    which can be used to pick out that subset of elements. 
    

    Parameters
    ----------
    shape : tuple of ints
        (m, n) where m and n are the dimensions of the domain on which
        the Q-tensor is defined
    sparse_shape : tuple of ints
        (m, n) where m and n are the dimension of the new sparse matrix
        which is being generated from the original Q-tensor.

    Returns
    -------
    sparse_idx : tuple of ndarrays
        Tuple of ndarrays that can be used to index the sparse matrix
        elements from the original matrix. For example, if `X` is the
        matrix representing the x-positions of the domain, to get the
        sparse matrix of x-positions one would use `X[sparse_idx]`.
    """
    
    # Index spacing between sparse gridpoints
    dx_idx = np.floor(shape[0]/sparse_shape[0]).astype('int')
    dy_idx = np.floor(shape[1]/sparse_shape[1]).astype('int')
    
    # x and y index locations for sparse gridpoints
    x_idx = np.arange(0, shape[0], dx_idx)
    y_idx = np.arange(0, shape[1], dy_idx)
    
    # shift over gridpoints so they are more centered
    x_idx = x_idx + np.floor((shape[0] - x_idx[-1]) / 2).astype('int')
    y_idx = y_idx + np.floor((shape[1] - y_idx[-1]) / 2).astype('int')
    
    # Create array which indexes selected sparse matrix elements
    sparse_idx = np.ix_(x_idx, y_idx)
    
    return sparse_idx
    
@jit(nopython=True, parallel=True, cache=True)
def calcQEigenvals(eta, mu, nu):
    """
    Calculate maximal eigenvalues of the Q-tensor across the domain. 
    
    Parameters
    ----------
    eta : ndarray
        mxn array holding value of the auxiliary variable eta across the
        whole domain.
    mu : ndarray
        mxn array holding value of the auxiliary variable mu across the
        whole domain.
    nu : ndarray
        mxn array holding value of the auxiliary variable nu across the
        whole domain.

    Returns
    -------
    lambda_max : ndarray
        mxn array holding the largest eigenvalue of the Q-tensor
        evaluated across the domain.
    """
    
    # Collecting terms to calculate plus and minus values easier
    a = (1/6)*eta + (1/2)*mu
    b = (1/2)*np.sqrt( (eta - mu)**2 + 4*nu**2 )
    
    lambda_p = a + b
    lambda_m = a - b
    
    # Greater of the two eigenvalues corresponds to S
    lambda_max = np.maximum(lambda_p, lambda_m)
    
    return lambda_max

@jit(nopython=True, parallel=True, cache=True)
def calcQEigenvecs(eta, mu, nu, lambda_pm, S_cutoff):
    """
    Calculate the x- and y-components of the maximal eigenvectors of the 
    Q-tensor across the domain. For an S-value under S_cutoff, the director 
    angles are not calculated (director angle isn't meaningful in these 
    regions).
    
    Parameters
    ----------
    eta : ndarray
        mxn array holding value of the auxiliary variable eta across the
        whole domain.
    mu : ndarray
        mxn array holding value of the auxiliary variable mu across the
        whole domain.
    nu : ndarray
        mxn array holding value of the auxiliary variable nu across the
        whole domain.
    lambda_pm : ndarray
        mxn array holding maximal eigenvalue of the Q-tensor across the whole
        domain.
    S_cutoff : double
        Value of S under which the director angle will not be calculated.

    Returns
    -------
    Um : MaskedArray
        mxn masked array which contains the normalized x-value of the director
        field vector across the domain. Masked values correspond to places
        where S < S_cutoff.
    Vm : MaskedArray
        mxn masked array which contains the normalized y-value of the director
        field vector across the domain. Masked values correspond to places
        where S < S_cutoff.
    """
    
    # Calculate Q_11 - lambda and Q_22 - lambda values
    Q11_lambda = (2/3)*eta - lambda_pm
    Q22_lambda = -(1/3)*eta + mu - lambda_pm
    
    # Masks for where to use first/second row of Q to calculate eigenvecs
    # first row mask --> frm; second row mask --> srm
    frm = np.abs(Q11_lambda) >= np.abs(Q22_lambda)
    srm= np.logical_not(frm)
    
    # Makes sure S is sufficiently large for calculating eigenvecs
    S_big_enough = (3/2)*lambda_pm > S_cutoff
    frm = np.logical_and(frm, S_big_enough)
    srm = np.logical_and(srm, S_big_enough)
    
    # Calculate X and Y components of eigenvecs
    U = np.zeros(lambda_pm.shape)
    V = np.zeros(lambda_pm.shape)
    
    V[frm] = 1/np.sqrt(( nu[frm]/Q11_lambda[frm] )**2 + 1)
    U[frm] = -nu[frm]*V[frm]/Q11_lambda[frm]
    
    U[srm] = 1/np.sqrt(( nu[srm]/Q22_lambda[srm] )**2 + 1)
    V[srm] = -nu[srm]*U[srm]/Q22_lambda[srm]
    
    # Mask U and V where S < cutoff for plotting
    Um = np.ma.masked_where(np.logical_not(S_big_enough), U)
    Vm = np.ma.masked_where(np.logical_not(S_big_enough), V)
    
    return Um, Vm

def makeTactoid(X, Y, S_val=1, ctr=[0, 0], r=1, m=1):
    """
    Generate the `S` and `phi` values for a tactoid centered at `ctr` with
    radius `r` and winding number `m`.

    Parameters
    ----------
    X : ndarray
        mxn array representing the x-values of the domain.
    Y : ndarray
        mxn array representing the y-values of the domain.
    S_val : double
        Value of S in the nematic region.
    ctr : double
        Center of the tactoid
    r : double
        Radius of the tactoid
    m : int
        Winding number of the tactoid.

    Returns
    -------
    S : ndarray
        mxn array representing the uniaxial order parameter S at every point.
    phi : ndarray
        mxn array representing the director angle at every point.
    """
    
    S = np.zeros(X.shape)
    S[np.sqrt((X - ctr[0])**2 + (Y - ctr[1])**2) > r] = S_val
    phi = m*np.arctan2(Y - ctr[1], X - ctr[0])
    
    return S, phi

def makeRightDisclination(X, Y, S_val=1, ctr=[0, 0], r=1, m=1):
    """
    Generate the `S` and `phi` values for a disclination centered at `ctr` with
    radius `r` and winding number `m`. For half-integer disclinations, there
    is always an axis of symmetry pointing to the right.

    Parameters
    ----------
    X : ndarray
        mxn array representing the x-values of the domain.
    Y : ndarray
        mxn array representing the y-values of the domain.
    S_val : double
        Value of S in the nematic region.
    ctr : double
        Center of the disclination
    r : double
        Radius of the disclination
    m : int
        Winding number of the disclination.

    Returns
    -------
    S : ndarray
        mxn array representing the uniaxial order parameter S at every point.
    phi : ndarray
        mxn array representing the director angle at every point.
    """
    
    S = np.zeros(X.shape)
    S = S_val*( 1 - np.exp(-np.sqrt( (X - ctr[0])**2 + (Y - ctr[1])**2 )/r) )
    phi = m*np.arctan2(Y - ctr[1], X - ctr[0])
    
    return S, phi

def makeLeftDisclination(X, Y, S_val=1, ctr=[0, 0], r=1, m=1):
    """
    Generate the `S` and `phi` values for a disclination centered at `ctr` with
    radius `r` and winding number `m`. For half-integer disclinations, there
    is always an axis of symmetry pointing to the left.

    Parameters
    ----------
    X : ndarray
        mxn array representing the x-values of the domain.
    Y : ndarray
        mxn array representing the y-values of the domain.
    S_val : double
        Value of S in the nematic region.
    ctr : double
        Center of the disclination
    r : double
        Radius of the disclination
    m : int
        Winding number of the disclination

    Returns
    -------
    S : ndarray
        mxn array representing the uniaxial order parameter S at every point.
    phi : ndarray
        mxn array representing the director angle at every point.
    """
    
    S = np.zeros(X.shape)
    S = S_val*( 1 - np.exp(-np.sqrt( (X - ctr[0])**2 + (Y - ctr[1])**2 )/r) )
    phi = m*np.arctan2(Y - ctr[1], -(X - ctr[0]))
    phi = np.fliplr(phi)
    
    return S, phi

def makeMultiDisclination(X, Y, S_val=1, 
                          ctr=[[0, 0]], r=[1], m=[1/2], decay=True):
    """
    Generate the `S` and `phi` values for a disclination centered at `ctr` with
    radius `r` and winding number `m`. For half-integer disclinations, there
    is always an axis of symmetry pointing to the left.

    Parameters
    ----------
    X : ndarray
        mxn array representing the x-values of the domain.
    Y : ndarray
        mxn array representing the y-values of the domain.
    S_val : double
        Value of S in the nematic region.
    ctr : double
        Center of the disclination
    r : double
        Radius of the disclination
    m : int
        Winding number of the disclination
    decay : bool
        Determine whether the S-field should exponentially decay around the
        disclinations. 

    Returns
    -------
    S : ndarray
        mxn array representing the uniaxial order parameter S at every point.
    phi : ndarray
        mxn array representing the director angle at every point.
    """
    
    S = np.zeros(X.shape)
    phi = np.zeros(X.shape)
    
    if not decay:
        S = S + S_val
    
    for i in range(0, len(ctr)):
        if decay:
            S = S + S_val*( 1 - np.exp(-np.sqrt( (X - ctr[i][0])**2 \
                                                + (Y - ctr[i][1])**2 )/r[i]) )
        
        phi = phi + m[i]*np.arctan2(Y - ctr[i][1], -(X - ctr[i][0]))
        
    return S, phi
    
    
@jit(nopython=True, parallel=True, cache=True)
def etaEOM(eta, mu, nu, dx, dy, A=A, B=B, C=C):
    """
    Equation of motion for eta. Returns LdG equation for 
    \partial \eta/\partial t.

    Parameters
    ----------
    eta : ndarray
        mxn array holding value of the auxiliary variable eta across the
        whole domain.
    mu : ndarray
        mxn array holding value of the auxiliary variable mu across the
        whole domain.
    nu : ndarray
        mxn array holding value of the auxiliary variable nu across the
        whole domain.
    dx : double
        Spacing between gridpoints in the x-direction.
    dy : double
        Spacing between gridpoints in the y-direction.
    A : double, optional
        Dimensionless LdG free energy coefficient A_bar. Set to -0.064 by
        default.
    B : double, optional
        Dimensionless LdG free energy coefficient B_bar. Set to -1.57 by
        default.
    C : TYPE, optional
        Dimensionless LdG free energy coefficient C_bar. Set to 1.29 by
        default.

    Returns
    -------
    deta_dt : ndarray
        mxn array holding the value of \partial \eta/\partial t as calculated
        from the LdG free energy.

    """
    
    deta_dt = fd.dx2(eta, dx) + fd.dy2(eta, dy) - A*eta \
                - B*( (2/3)*eta**2 + (3/2)*nu**2 )\
                    - C*eta*( (2/3)*eta**2 + 2*nu**2 + 2*mu**2 )
                    
    return deta_dt

@jit(nopython=True, parallel=True, cache=True)
def muEOM(mu, eta, nu, dx, dy, A=A, B=B, C=C):
    """
    Equation of motion for mu. Returns LdG equation for 
    \partial \mu/\partial t.

    Parameters
    ----------
    mu : ndarray
        mxn array holding value of the auxiliary variable mu across the
        whole domain.
    eta : ndarray
        mxn array holding value of the auxiliary variable eta across the
        whole domain.
    nu : ndarray
        mxn array holding value of the auxiliary variable nu across the
        whole domain.
    dx : double
        Spacing between gridpoints in the x-direction.
    dy : double
        Spacing between gridpoints in the y-direction.
    A : double, optional
        Dimensionless LdG free energy coefficient A_bar. Set to -0.064 by
        default.
    B : double, optional
        Dimensionless LdG free energy coefficient B_bar. Set to -1.57 by
        default.
    C : TYPE, optional
        Dimensionless LdG free energy coefficient C_bar. Set to 1.29 by
        default.

    Returns
    -------
    dmu_dt : ndarray
        mxn array holding the value of \partial \mu/\partial t as calculated
        from the LdG free energy.

    """
    
    dmu_dt = fd.dx2(mu, dx) + fd.dy2(mu, dy) - A*mu \
                - B*( (1/3)*eta**2 + mu**2 + (3/2)*nu**2 - (2/3)*eta*mu )\
                    - C*mu*( (2/3)*eta**2 + 2*nu**2 + 2*mu**2 )
                    
    return dmu_dt

@jit(nopython=True, parallel=True, cache=True)
def nuEOM(nu, eta, mu, dx, dy, A=A, B=B, C=C):
    """
    Equation of motion for nu. Returns LdG equation for 
    \partial \nu/\partial t.

    Parameters
    ----------
    nu : ndarray
        mxn array holding value of the auxiliary variable nu across the
        whole domain.
    eta : ndarray
        mxn array holding value of the auxiliary variable eta across the
        whole domain.
    mu : ndarray
        mxn array holding value of the auxiliary variable mu across the
        whole domain.
    dx : double
        Spacing between gridpoints in the x-direction.
    dy : double
        Spacing between gridpoints in the y-direction.
    A : double, optional
        Dimensionless LdG free energy coefficient A_bar. Set to -0.064 by
        default.
    B : double, optional
        Dimensionless LdG free energy coefficient B_bar. Set to -1.57 by
        default.
    C : TYPE, optional
        Dimensionless LdG free energy coefficient C_bar. Set to 1.29 by
        default.

    Returns
    -------
    dnu_dt : ndarray
        mxn array holding the value of \partial \nu/\partial t as calculated
        from the LdG free energy.

    """
    
    dnu_dt = fd.dx2(nu, dx) + fd.dy2(nu, dy) - A*nu \
                - B*( (1/3)*eta*nu + mu*nu )\
                    - C*nu*( (2/3)*eta**2 + 2*nu**2 + 2*mu**2 )
                    
    return dnu_dt

@jit(nopython=True, parallel=True, cache=True)
def findDefectCenters(lambda_max, X, Y, n_defects=2):
    """
    Finds the (x, y) coordinates of the defect centers in a nematic liquid
    crystal based on the maximum eigenvalues (proportional to S). It does this
    by (1) assuming that the defects lie along some constant y = C line, (2)
    going through each 1D slice and using the SciPy `find_peaks` method. Then
    it (3) finds the max eigenvalues associated with each peak at each slice,
    and then (4) finds the slice at which each peak is maximized. This gives
    the y-index at which each defect is located which we can plug into the
    peak locations array to get the x-index. Plugging these indices into the
    X and Y domain arrays gives the x- and y-locations of the defects.

    Parameters
    ----------
    lambda_max : ndarray
        mxn array holding the maximal eigenvalue of the Q-tensor evaluated at
        each point on the domain.
    X : ndarray
        mxn array holding the x-value of the domain at each index.
    Y : ndarray
        mxn array holding the y-value of the domain at each index.
    n_defects : int, optional
        Number of defects. Default is 2 because we typically look at defect
        annihilation.

    Returns
    -------
    x_locs : ndarray
        n_defects length array holding x-locations of the defects.
    y_locs : ndarray
        n_defects length array holding y-locations of the defects.

    """

    # (i, j) entry of `peaks` holds index of x-direction of the jth y = C slice
    m, n = lambda_max.shape
    peaks = np.zeros((n_defects, n), dtype='int')
    for j in range(n):
        peaks[:, j], _ = find_peaks(-lambda_max[:, j])
        
    # Find values of those peaks, find index in y-direction of maximal peaks
    peak_vals = np.zeros((n_defects, n))
    peak_args = np.zeros((n_defects), dtype='int')
    for i in range(n_defects):
        peak_vals[i, :] = np.diagonal(lambda_max[peaks[i, :], :])
        peak_args[i] = np.argmin(peak_vals[i, :])
    
    x_idx = np.diagonal(peaks[:, peak_args[:]])
    y_idx = peak_args
    
    x_locs = X[x_idx, 0]
    y_locs = Y[0, y_idx]
    
    return x_locs, y_locs
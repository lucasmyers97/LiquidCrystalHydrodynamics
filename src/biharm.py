import numpy as np
from scipy.fft import dst
from scipy.sparse.linalg import LinearOperator, cg

class Biharm:
    """
    Implements the modified Biharmonic equation solver described here:
    https://math.stackexchange.com/a/3763462/683123
    It uses the Woodbury formula along with the method of Conjugate Gradients 
    in order to invert a slightly modified biharmonic equation with Dirichlet
    boundary conditions (zero at the boundaries, zero normal derivatives at
    the boundaries).
    
    The class `Biharm` holds all of the information about the specific problem
    being solved (size and shape of the domain, eigenvalues of the modified
    Biharmonic operator, etc.). 

    Parameters
    ----------
    L : tuple of ints
        Length-2 tuple which holds the x- and y-lengths of the domain.
    shape : tuple of ints
        Length-2 tuple which holds the x- and y-dimensions of the domain.
    alpha : double
        Alpha parameter of the modified biharmonic equation.
    cg_tol : double, optional
        Convergence tolerance used when calling the scipy Conjugate Gradient
        method. The default is 1e-14.
    cg_maxiter : int, optional
        Maximum interation parameter used when calling the scipy Conjugate
        Gradient method. The default is 500.

    Returns
    -------
    None

    """
    
    
    def __init__(self, L, shape, alpha, cg_tol=1e-14, cg_maxiter=500):
        
        self.Nx = shape[0]
        self.Ny = shape[1]
        self.hx = L[0]/shape[0]
        self.hy = L[1]/shape[1]
        self.alpha = alpha
        self.d = self.calcdMatrix()
        self.cg_tol = cg_tol
        self.cg_maxiter = cg_maxiter
        
    def calcdMatrix(self):
        """
        Calculate the d_{ij} matrix, which represents the eigenvalues of the
        A matrix. For details on what the A operator is, see 
        https://math.stackexchange.com/a/3763462/683123 or the accompanying
        theory documentation.

        Returns
        -------
        d : ndarray
            (Nx - 1)x(Ny - 1) array which contains the eigenvalues of the A
            matrix.
            
        """
        
        # Create local instances for increased readability
        Nx = self.Nx
        Ny = self.Ny
        hx = self.hx
        hy = self.hy
        alpha = self.alpha
        
        # Reshape specifies them as column and row matrices respectively
        lambda_x = self.lambda_k(Nx - 1).reshape(-1, 1)
        lambda_y = self.lambda_k(Ny - 1).reshape(1, -1)
        
        # vectors are broadcast appropriately to return (Nx - 1)x(Ny - 1) grid
        return ( lambda_x**2/hx**4 
                + (2 + alpha)/(hx**2 * hy**2)*lambda_x*lambda_y
                + lambda_y**2/hy**4 )
    
    def lambda_k(self, N):
        """
        Calculate an array of the eigenvalues for the a length-N Lambda_2 
        matrix. For more details on what the Lambda_2 matrix is, see
        https://math.stackexchange.com/a/3763462/683123 or the accompanying
        theory documentation.

        Parameters
        ----------
        N : int
            One dimension of the square Lambda_2 matrix

        Returns
        -------
        lambda_k : ndarray
            Length-N array containing eigenvalues lambda_1, lambda_2,...,
            lambda_N of the Lambda_2 matrix.

        """
        
        k = np.arange(N)
        return -4*np.sin( np.pi*(k + 1) / (2*(N + 1)) )**2
    
    def dst2D(self, M):
        """
        Calculate the 2-dimensional Discrete Sine Transform of the first kind
        for a matrix `M`. Based off of the 1D DST from scipy. Note that this
        will be more efficient for matrices of side-lengths
        (2^i * 3^i * 5^i - 1). Scipy has a function called 
        `scipy.fftpack.next_fast_len` which helps pick a size. More info here:
        https://stackoverflow.com/a/54726284/7736506

        Parameters
        ----------
        M : ndarray
            (Nx)x(Ny) array to be transformed via a 2D DST. More efficient if
            Nx and Ny are of size (2^i * 3^i * 5^i - 1) for some integer i.

        Returns
        -------
        F_M : ndarray
            `M` transformed by the 2D DST of the first kind. Will be the same
            size as `M`

        """
        
        return dst( dst(M, axis=0, type=1, norm='ortho'), 
                    axis=1, type=1, norm='ortho' )
    
    def A_Inv(self, M):
        """
        Apply A^{-1} to a matrix `M`. It does this by applying a 2D DST to `M`,
        dividing by the eigenvalues of A, and then doing an inverse 2D DST. For
        more info on the A matrix see here:
        https://math.stackexchange.com/a/3763462/683123 or the associated
        theory documentation.

        Parameters
        ----------
        M : ndarray
            (Nx)x(Ny) array to which A^{-1} will be applied.

        Returns
        -------
        A_Inv_M : ndarray
            A^{-1} applied to the matrix `M`. Same size as `M`.

        """
        
        return self.dst2D( self.dst2D(M)/self.d )
    
    def VT(self, X):
        """
        Apply V^\top to a matrix `X`. Essentially takes the +x, -x, +y, -y
        boundaries from `X` (in index convention) and then concatenates them
        in a vector. Since `X` is size (Nx - 1)x(Ny - 1) by convention, this
        vector will be length 2(Ny - 1) + 2(Nx - 1).

        Parameters
        ----------
        X : ndarray
            Size (Nx - 1)x(Ny - 1) matrix from which you want to apply V^\top
            (equivalent to taking out the borders and concatenating them in a
            vector). 

        Returns
        -------
        VTX : ndarray
            Size 2(Ny - 1) + 2(Nx - 1) vector which is the result of applying
            V^\top to `X`.

        """
        
        # Create local instances for increased readability
        Nx = self.Nx
        Ny = self.Ny
        hx = self.hx
        hy = self.hy
        
        # Size of VTX as specified by documentation
        VTX = np.zeros(2*(Ny - 1) + 2*(Nx - 1))
        s2 = np.sqrt(2)
        
        # Fill in VTX vector with +x, -x, +y, -y boundaries
        VTX[0:Ny - 1] = (s2/hx**2) * X[0, :]
        VTX[Ny - 1:2*(Ny - 1)] = (s2/hx**2) * X[-1, :]
        VTX[2*(Ny - 1):2*(Ny - 1) + (Nx - 1)] = (s2/hy**2) * X[:, 0]
        VTX[2*(Ny - 1) + (Nx - 1):] = (s2/hy**2) * X[:, -1]
        
        return VTX
    
    def V(self, X):
        """
        Apply V to a vector X. Takes the first two sets of (Ny - 1) entries and
        arranges them along the +x and -x edges of an (Nx - 1)x(Ny - 1) size
        matrix, then takes the last two sets of (Nx - 1) entries and arranges
        then along the +y and -y edges. Corners are the sum of the two.

        Parameters
        ----------
        X : ndarray
            Length 2(Ny - 1) + 2(Nx - 1) vector containing the edges of some
            matrix that V^\top acted on.

        Returns
        -------
        VX : ndarray
            The result of applying V to `X`. Will be of size (Nx - 1)x(Ny - 1)
            -- the edges will be filled with entries from `X` (or sum of
            entries in the case of the corners). The interior will be all 0's.

        """
        
        # Create local instances for increased readability
        Nx = self.Nx
        Ny = self.Ny
        hx = self.hx
        hy = self.hy
        
        # VX is size of original domain
        VX = np.zeros((Nx - 1, Ny - 1))
        s2 = np.sqrt(2)
        
        # Fit sections of X around border of VX
        VX[0, :] += (s2/hx**2) * X[0:Ny - 1]
        VX[-1, :] += (s2/hx**2) * X[Ny - 1:2*(Ny - 1)]
        VX[:, 0] += (s2/hy**2) * X[2*(Ny - 1):2*(Ny - 1) + (Nx - 1)]
        VX[:, -1] += (s2/hy**2) * X[2*(Ny - 1) + (Nx - 1):]
        
        return VX
    
    def C(self, X):
        """
        Applies the capacitance matrix C to some vector `X`. The capacitance
        matrix is given by (I + V^\top A^{-1} V) -- each of these individual
        matrices are applied via methods in the `Biharm` class.

        Parameters
        ----------
        X : ndarray
            Vector of size 2(Ny - 1) + 2(Nx - 1) to which you will apply C.

        Returns
        -------
        CX : ndarray
            Vector of size 2(Ny - 1) + 2(Nx - 1) which is the result of
            applying C to `X`.

        """
        
        return X + self.VT( self.A_Inv( self.V(X) ) )
    
    class CapacitanceMatrix(LinearOperator):
        """
        This class packages the application of the capacitance matrix (here
        implemented by the `C` method) into a LinearOperator object from scipy
        so that we may pass the `C` method into the `cg` scipy method which
        carries out a matrix solution via the method of Conjugate Gradients.
        Additionally, the number of calls are recorded.
        
        Attributes
        ----------
        calls : int
            Records the number of times the capacitance matrix method `C` was
            called since instantiation (presumably by the conjugate gradients
            method).
        biharm_obj : Biharm object
            The instance of Biharm which is creating this CapacitanceMatrix
            object.
        
        Parameters
        ----------
        biharm_obj : Biharm object
            The instance of Biharm which is creating this CapacitanceMatrix
            object. It holds necessary information about the size of the
            domain, and also calls the `C` capacitance matrix method.
            
        """
        
        def __init__(self, biharm_obj):
            
            Nx = biharm_obj.Nx
            Ny = biharm_obj.Ny
            
            super().__init__(dtype=np.double, 
                             shape=(2*(Ny - 1) + 2*(Nx - 1), 
                                    2*(Ny - 1) + 2*(Nx - 1)))
            self.calls = 0
            self.biharm_obj = biharm_obj
            
        def _matvec(self, X):
            """
            Calls the capacitance matrix `C` method and adds 1 to the number of
            calls. Meant to be called by the `cg` scipy conjugate gradient
            method.

            Parameters
            ----------
            X : ndarray
                Vector of size 2(Ny - 1) + 2(Nx - 1) to which C will be 
                applied.

            Returns
            -------
            CX : ndarray
                Vector of size 2(Ny - 1) + 2(Nx - 1) which is the result of
                operating on X with C.

            """
            
            self.calls += 1
            return self.biharm_obj.C(X)
        
    def solve(self, F):
        """
        Solve a biharmonic equation of the form (A + VV^\top)U = F. This takes
        advantage of the Woodbury identity to recast the equation into the
        form U = A^{-1}F - A^{-1}V(I + V^\top A^{-1} V)^{-1}V^\top A^{-1} F. 
        The second term is found by using the Conjugate Gradients method to
        solve (I + V^\top A^{-1} V)^{-1} r as a solution to
        (I + V^\top A^{-1} V)s = r. The rest is solved by applying efficient
        implementations of V, V^\top, and A^{-1}.

        Parameters
        ----------
        F : ndarray
            (Nx - 1)x(Ny - 1) array which represents the source term in the
            modified biharmonic equation.

        Returns
        -------
        solution : ndarray
            (Nx - 1)x(Ny - 1) array which represents the solution to the
            modified biharmonic equation.
        info : int
            Provides convergence information:
                0 : successful exit 
                >0 : convergence to tolerance not achieved, number of 
                     iterations 
               <0 : illegal input or breakdown
        calls : int
            Number of calls the Conjugate Gradients method made of the
            CapacitanceMatrix class.

        """
        n, m = F.shape
        if n != self.Nx - 1 or m != self.Ny - 1:
            raise ValueError("F does not have the correct shape")
        
        CM = self.CapacitanceMatrix(self)
        
        r = self.A_Inv(F)
        s, info = cg(CM, self.VT(r), tol=self.cg_tol, 
                     atol='legacy', maxiter=self.cg_maxiter)
        
        return r - self.A_Inv( self.V(s) ), info, CM.calls
    
    def applyBCs(self, U):
        """
        Adds a border around the solution to the modified biharmonice equation
        which is just zeros -- this enforces the first part of the dirichlet
        boundary conditions.

        Parameters
        ----------
        U : ndarray
            (Nx - 1)x(Ny - 1) matrix representing the solution to the modified
            biharmonic equation.

        Returns
        -------
        U_bds : ndarray
            (Nx)x(Ny) matrix which is the modified biharmonic equation solution
            that has the exterior values set to zero to match the first part
            of the Dirichlet boundary conditions.

        """
        
        m, n = U.shape
        U_bds = np.zeros((m + 2, n + 2))
        U_bds[1:-1, 1:-1] = U
        
        return U_bds
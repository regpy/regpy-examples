import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import lpmv
import scipy.sparse as sp

from regpy.vecsps import NumPyVectorSpace
from regpy.hilbert import HilbertSpace, L2
from regpy.operators import MatrixMultiplication, SuperLUInverse
from regpy import util


def integrated_legendre_basis(p, xi):
    p_extended = np.arange(p+1)
    xx, pp = np.meshgrid(xi, p_extended)
    legendrefunc = lpmv(0, pp, xx)
    # Construction of phip
    b = np.zeros((p+1, p+1))
    b[0,0] = -1
    b[-1,0] = 1
    b[np.arange(1,p),np.arange(1,p)] = 1

    scaling = np.zeros((p+1,1))
    scaling[0,:] = 0.5
    scaling[-1,:] = 0.5
    scaling[1:-1,:] = np.sqrt((2*np.arange(1,p).reshape((p-1,1))+1)/2)
    b = scaling * b 

    Phi_der = b @ legendrefunc

    b = np.zeros((p+1, p+1))
    b[0, 1] = -1
    b[0,0] = 1
    b[-1, 0] = 1
    b[-1,1] = 1
    b[np.arange(1,p),np.arange(p-1)] = -1
    b[np.arange(1,p),np.arange(2,p+1)] = 1

    scaling = np.zeros((p+1,1))
    scaling[0,:] = 0.5
    scaling[-1,:] = 0.5
    scaling[1:-1,:] = 1/np.sqrt(2*(2*np.arange(1,p).reshape((p-1,1))+1))
    b = scaling * b 

    Phi = b @ legendrefunc
    return Phi, Phi_der

class OneDimensionalFEM(NumPyVectorSpace):
    """
    1D Finite Element Method space for polynomial degree p.
    """
    def __init__(self, p=3, n_nodes = 50, a=-1, b=1):
        if b<a:
            raise ValueError("Right endpoint must be greater than left endpoint.")
        if not isinstance(n_nodes, int) or n_nodes < 2:
            raise ValueError("Number of nodes n_nodes must be an integer >= 2.")
        self.nodes = np.linspace(a, b, n_nodes+1, endpoint=True)
        self.hj = np.diff(self.nodes)
        self.n_nodes = n_nodes
        self.start_end = (a, b)
        if not isinstance(p, int) or p < 1: 
            raise ValueError("Polynomial degree p must be a positive integer greater or equal to 1.")
        self.p = p

        self.initialize_FEM_data()
        super().__init__((self.nr_dofs,))

    @property
    def is_finite_dimensional(self):
        return True

    def initialize_FEM_data(self):
        self.quad_pts, self.quad_weights = leggauss(self.p+1)
        self.phi, self.phip = integrated_legendre_basis(self.p,self.quad_pts)

        self.nr_quad_pts = self.n_nodes*len(self.quad_pts)
        self.nr_dofs = self.n_nodes * self.p + 1

        self.hj_long = np.repeat(self.hj, len(self.quad_pts))
        left_ends = np.tile(self.nodes[:-1], (len(self.quad_pts), 1))
        offsets   = (0.5 + 0.5 * self.quad_pts[:, None]) * self.hj
        self.global_quad_pts = (left_ends + offsets).T.flatten()
        self.global_quad_weights = np.tile(self.quad_weights, self.n_nodes)* self.hj_long / 2
        # Build DOF2pts and DOF2pts_der matrices
        DOF2pts = sp.lil_matrix((self.nr_quad_pts,self. nr_dofs))
        DOF2pts_der = sp.lil_matrix((self.nr_quad_pts, self.nr_dofs))

        for j in range(self.n_nodes):
            Ij = np.arange(j * self.p, (j+1) * self.p + 1)  # global DOF indices
            I2j = np.arange(j * len(self.quad_weights), (j + 1) * len(self.quad_pts))  # quad point indices
            DOF2pts[I2j[:, None], Ij] = self.phi.T
            DOF2pts_der[I2j[:, None], Ij] = self.phip.T

        # Convert to CSR for efficient matrix operations
        self.DOF2pts = DOF2pts.tocsr()
        self.DOF2pts_der = DOF2pts_der.tocsr()

        # Assemble mass matrix M
        W_diag = sp.diags(self.global_quad_weights)
        self.M = DOF2pts.T @ W_diag @ DOF2pts
        return None
    
    def get_fem_from_pts(self, x):
        if not isinstance(x, np.ndarray) and x.ndim != 1 and len(x) != len(self.global_quad_pts):
            raise TypeError("Input x must be a numpy array with the size of number of global quadrature points.")
        # Compute projection RHS
        b = self.DOF2pts.T @ (self.global_quad_weights * x)

        # Solve for coefficients
        return sp.linalg.spsolve(self.M, b)
    

class LegendreSpace(NumPyVectorSpace):
    """
    Represents a Legendre series in the 1D FEM space.
    """
    def __init__(self, degree=30, x = None, n_nodes = None, w = None):
        if not isinstance(degree, int) and degree < 1:
            raise TypeError("Degree must be a positive integer.")
        self.degree = degree
        if x is None and n_nodes is None:
            raise ValueError("Either x or n_nodes must be specified")
        elif x is None:
            if not isinstance(n_nodes, int) or n_nodes <= 0:
                raise ValueError("n_node must be a positive float.")
            self.n_nodes = n_nodes
            self.x = np.linspace(-1, 1, n_nodes+1, endpoint=True)
            self.w = np.ones_like(self.x)*(self.x[1]-self.x[0])
        else:
            self.x = x/(x[-1]-x[0])*2
            self.w = np.ones_like(self.x) if w is None else w 
            self.n_nodes = len(x)-1
        self.basis = sp.diags(self.w)@self.legendre_basis(degree, self.x)@(np.sqrt(self.n_nodes)/2*sp.eye(self.degree+1))
        super().__init__((degree+1,))

    def coeff2pts(self, coeffs):
        if not coeffs in self:
            raise TypeError("Coefficients must be a 1D numpy array of length degree + 1.")
        return (self.basis @ (coeffs))
    
    def pts2coeff(self, pts):
        if not isinstance(pts, np.ndarray) or pts.ndim != 1 or len(pts) != self.n_nodes + 1:
            raise TypeError("Points must be a 1D numpy array with the size of number of nodes.")
        return self.basis.T @ pts
    
    def ones(self):
        """
        Returns the constant function 1 in the Legendre series space.
        """
        coeffs = self.zeros()
        coeffs[0] = 1
        return coeffs
    
    def legendre_basis(self,p,x):
        p_extended = np.arange(p+1)
        xx, pp = np.meshgrid(x, p_extended)
        legendrefunc = lpmv(0, pp, xx)
        scale = np.sqrt((2*(p_extended)+1)/2)
        return (legendrefunc * scale.reshape((p+1,1))).T
    
class L2_Legendre(HilbertSpace):
    """L2 Hilbert space on the Legendre discretized space.

    Parameters
    ----------
    vecsp : LegendreSpace
        The Legendre space on which the L2 space is defined.
    """

    def __init__(self, vecsp):
        if not isinstance(vecsp, LegendreSpace):
            raise ValueError(f"The vecsps is not a LegendreSpace but a {type(vecsp)}")
        super().__init__(vecsp)

    @util.memoized_property
    def gram(self):
        return self.vecsp.identity
        


class L2_OneDimensionalFEM(HilbertSpace):
    
    def __init__(self, vecsp):
        if not isinstance(vecsp, OneDimensionalFEM):
            raise ValueError(f"The vecsps is not a OneDimensionalFEM but a {type(vecsp)}")
        self.mat = vecsp.M
        super().__init__(vecsp)

    @util.memoized_property
    def gram(self):
        return MatrixMultiplication(self.mat, domain=self.vecsp,codomain=self.vecsp)
    
    @util.memoized_property
    def gram_inv(self):
        return SuperLUInverse(self.gram)

L2.register(LegendreSpace,L2_Legendre)
L2.register(OneDimensionalFEM,L2_OneDimensionalFEM)
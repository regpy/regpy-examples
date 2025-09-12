from regpy.operators import Exponential, FourierTransform, RealPart, SquaredModulus

import numpy as np
import logging

from regpy.operators import Operator
from regpy.vecsps import UniformGridFcts,VectorSpace, DirectSum
    
class Corr(Operator):
    
    """Maps: A -> 2* D A Cov(u) A^* D^*
        input:
        A: matrix

        output:
        matrix

        parameters:
        D:  matrix
        cov_u: matrix
    """
    
    def __init__(self, domain, codomain, D, cov_u):
        self.cov_u=cov_u
        self.D=D
        self.N=cov_u.shape[0]
        self.M=int(np.sqrt(self.N))
        super().__init__(domain, codomain, linear=False)

    def _eval(self, x, differentiate=False):
        mat=self.D*x.T.flatten()
        #mat=self.D*x.flatten()
        if differentiate==True:
            self.mat=mat.copy()
        return 2*mat.dot(self.cov_u).dot(mat.T.conj())
    
    def _derivative(self, x):
        res=(self.D*x.T.flatten()).dot(self.cov_u).dot(self.mat.T.conj())
        return 2*(res+res.T.conj())

    def _adjoint(self, y):
        y_adj=y+y.T.conj()
        adj=np.diagonal(self.D.T.conj().dot(y_adj).dot(self.mat).dot(self.cov_u))
        return 2*adj.reshape(self.M, self.M).T
            
class Tau(Operator):
    
    """This operator allows to rewrite the pointwise squared modulus of a 
    low rank operator as a matrix product of two low rank matrices. 
    The factor matrices are written as 3-tensors and their rank as matrices is 
    the square of the rank of the product matrix. This operator maps a matrix 
    to the corresponding rank-3-tensor

        Mapping of (A_{ij})_{i,j} -> (A_{ip} A^*_{iq})_{i,p,q} 
    """
    
    def __init__(self, domain, codomain):
        self.N=domain.shape[0]
        self.k=domain.shape[-1]
        super().__init__(domain, codomain, linear=False)
        
    def _eval(self, A, differentiate=True):
        if differentiate:
            self.A=A
        return A.reshape(self.N, self.N, self.k, 1)*A.conj().reshape(self.N, self.N, 1, self.k)
    
    def _derivative(self, B):
        first=self.A.reshape(self.N, self.N, self.k, 1)*B.conj().reshape(self.N, self.N, 1, self.k)
        second=B.reshape(self.N, self.N, self.k, 1)*self.A.conj().reshape(self.N, self.N, 1, self.k)
        return first+second
    
    def _adjoint(self, C):
        second_adj=np.sum(C*self.A.reshape(self.N, self.N, 1, self.k), axis=-1)
        first_adj=np.sum(C*self.A.conj().reshape(self.N, self.N, self.k, 1), axis=-2).conj()
        return first_adj+second_adj
        
class MatrixProductOp(Operator):
    
    """Operator mapping a pair of rectangluar matrices (D,E) of the same size to the matrix product D*E^*.
    D and E may be represent mapping from a tensor space of shape shape_columns to a tensor 
    space of shape shape_rows such that D and E are tensor of shape shape_rows+shape_columns (+ in the sense of concatenation of tuples).
    Input:
        DE:  numpy array with first dimension =2: D=DE[0,:] and E=DE[1,:] 
        ndim_cols :  integer [default 1] number of dimensions of the domain of D,E as linear mappings
    Output: matrix product
        D @ E^T, a tensor of shape shape_col+shape_col
    This operator is particularly useful if the product of the last ndim_cols dimensions is much smaller than 
    the product of the other dimensions.
    _deriv_adjoint and _eval_adjoint can be evaluated even if the matrix product does not fit into storage.
    """
    
    def __init__(self, MatrixSpace,ndim_col=1):
        if not isinstance(MatrixSpace,VectorSpace):
            raise TypeError(f'First argument must be a VectorSpace. Was given {MatrixSpace1}')
        else:
            self.shape = MatrixSpace.shape
            dtype = MatrixSpace.dtype
        if not isinstance(ndim_col,int):
            raise TypeError('ndim_col must be integer.')
            self.s = s
        self.ndim_col = ndim_col
        self.ndim_row = len(self.shape) - ndim_col
        self.shape_columns = self.shape[:-ndim_col]
        self.shape_rows = self.shape[-ndim_col:]

        shape_domain = (2,)+self.shape
        shape_codomain = self.shape_columns*2      
        super().__init__(VectorSpace(shape_domain,dtype=dtype), 
                         VectorSpace(shape_codomain,dtype=dtype), 
                         linear=False)

    def prod_A_BT(self,A,B):
        # returns matrix product A * B^T 
        # (sums over the "column axes" of A and B, which are assumed to be the last ones both in A and B)
        ax = [np.arange(-self.ndim_col,0,1),np.arange(-self.ndim_col,0,1)]  
        return np.tensordot(A,B,ax)
    
    def prod_GT_A(self,G,A):
        # returns matrix product G^T * A 
        # (sums over "row axes", which are assumed to be the first ones both in G and A)
        ax = [np.arange(0,self.ndim_row),np.arange(0,self.ndim_row)]
        return np.tensordot(G,A,ax)
 
    def prod_G_A(self,G,A):
        # returns matrix product G * A 
        # (sums over "row axes", which are assumed to the  last ones of G and the first ones of A )
        ax = [np.arange(-self.ndim_row,0,1),np.arange(0,self.ndim_row)]      
        return np.tensordot(G,A,ax)

    def _eval(self, DE, differentiate=False, adjoint_derivative=True):
        if differentiate==True:
            self.DE = DE
        return self.prod_A_BT(DE[0],DE[1])

    def _derivative(self, dDE):
        return (self.prod_A_BT(self.DE[0],dDE[1]) 
              + self.prod_A_BT(dDE[0],    self.DE[1]))

    def _adjoint(self, G):
        return np.stack((self.prod_G_A(G,self.DE[1]),
                        self.prod_GT_A(G,self.DE[0])),
                        axis=0)

    def _eval_adjoint(self, DE):
        self.DE = DE
        return np.stack((self.prod_G_A(DE[0], self.prod_GT_A(DE[1],DE[1])),
                        self.prod_G_A(DE[1], self.prod_GT_A(DE[0],DE[0]))),
                        axis=0)

    def _deriv_adjoint(self, dDE):
        return np.stack(self.prod_G_A(self.DE[0], self.prod_GT_A(dDE[1],self.DE[1])) 
                      + self.prod_G_A(dDE[0],  self.proj_GT_A(self.DE[1],self.DE[1])),
                        self.prod_G_A(self.DE[1], self.prod_GT_A(dDE[0],self.DE[0])) 
                      + self.prod_G_A(dDE[1],  self.proj_GT_A(self.DE[0],self.DE[0])),
                        axis=0)
 
class Theta:
    
    """Implements the standard matrix product DxE -> D E^* as an operator. 
    Its intended for use of matrices where the inner dimension is much smaller 
    than the outer dimension, and the matrix product should never be actually computed
    as it may not fit into storage.  
    Therefore, we just apply _deriv_adjoint and _eval_adjoint.
    """
    
    def __init__(self, N, k):
        self.N=N
        self.k=k
        
    """def _deriv_adjoint(self, D, E, dD, dE):
        D=D.reshape(self.N**2, self.k*2)
        E=E.reshape(self.N**2, self.k*2)
        dD=dD.reshape(self.N**2, self.k*2)
        dE=dE.reshape(self.N**2, self.k*2)
        first=np.dot(D, dE.T.conj().dot(E))+np.dot(dD, E.T.conj().dot(E))
        second=np.dot(dE, D.T.conj().dot(D))+np.dot(E, dD.T.conj().dot(D))
        return first.reshape(self.N, self.N, self.k, self.k), second.reshape(self.N, self.N, self.k, self.k)"""
    
    def _deriv_adjoint(self, D, E, dDE):
        dD=dDE[0, ...]
        dE=dDE[1, ...]
        first=self._backprop(D, E, D, dE)
        second=self._backprop(D, E, dD, E)
        return first+second
    
    def _eval_adjoint(self, D, E, C, k_C=None):
        C_1=C[0, ...]
        C_2=C[1, ...]
        return self._backprop(D, E, C_1, C_2, k_G=k_C)
    
    def _backprop(self, D, E, G_1, G_2, k_G=None):
        if k_G is None:
            k_G=self.k**2
        D=D.reshape(self.N**2, self.k**2)
        E=E.reshape(self.N**2, self.k**2)
        G_1=G_1.reshape(self.N**2, k_G)
        G_2=G_2.reshape(self.N**2, k_G)
        first=np.dot(G_1, G_2.T.conj().dot(E))
        second=np.dot(G_2, G_1.T.conj().dot(D))
        res=np.zeros((2, self.N, self.N, self.k, self.k), dtype=complex)
        res[0, ...]=first.reshape(self.N, self.N, self.k, self.k)
        res[1, ...]=second.reshape(self.N, self.N, self.k, self.k)
        return res
    
    
class Mat(Operator):
    
    """
    Maps x-> D e^f V
    """
    
    def __init__(self, domain, codomain, Vcov, fp):
        self.N=domain.shape[0]
        self.k=codomain.shape[-1]
        self.Vcov=Vcov #[N, k]
        self.fp=fp
        super().__init__(domain, codomain, linear=False)

    def _eval(self, x, differentiate=True):
        if differentiate:
            self.x=x
        mat=self.codomain.zeros()
        for i in range(0, self.k):
            mat[:, :,  i]=self.fp(np.exp(x)*self.Vcov[:, i].reshape(self.N, self.N))
        return mat
    
    def _derivative(self, h):
        mat=self.codomain.zeros()
        for i in range(0, self.k):
            mat[:, :, i]=self.fp(np.exp(self.x)*h*self.Vcov[:, i].reshape(self.N, self.N))
        return mat
    
    def _adjoint(self, y):
        res=self.domain.zeros()
        for i in range(0, self.k):
            right=self.Vcov.T.conj()[i, :].reshape(self.N, self.N)
            left=self.fp._adjoint(y[:, :, i])
            res+=right*left*np.exp(self.x.conj())
        return res

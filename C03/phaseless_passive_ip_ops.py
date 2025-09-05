from regpy.operators import Exponential, FourierTransform, RealPart, SquaredModulus

import numpy as np
import logging

from regpy.operators import Operator
from regpy.vecsps import UniformGridFcts
    
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
    
    
class Theta_2(Operator):
    
    """Mapping of DxE -> D E^*, we just apply _deriv_adjoint and _eval_adjoint
    """
    
    def __init__(self, domain, codomain, N, k):
        self.N=N
        self.k=k
        super().__init__(domain, codomain, linear=False)
    
    def _deriv_adjoint(self, DE, dDE):
        D=DE[0, ...]
        E=DE[1, ...]
        dD=dDE[0, ...]
        dE=dDE[1, ...]
        first=self._backprop(D, E, D, dE)
        second=self._backprop(D, E, dD, E)
        return first+second
    
    def _eval_adjoint(self, DE, C, k_C=None):
        D=DE[0, ...]
        E=DE[1, ...]
        C_1=C[0, ...]
        C_2=C[1, ...]
        return self._backprop(D, E, C_1, C_2, k_G=k_C)
    
    def _backprop(self, DE, G_1, G_2, k_G=None):
        if k_G is None:
            k_G=self.k**2
        D=DE[0, ...]
        E=DE[1, ...]
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

from regpy.operators import Exponential, FourierTransform, RealPart, SquaredModulus

import numpy as np
import logging

from regpy.operators import Operator
from regpy.vecsps import UniformGridFcts,NumPyVectorSpace, DirectSum
    
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
    
    """This operator allows to rewrite the elementwise squared modulus  |A|^2 of a 
    complex low rank matrix A as a matrix product of two low rank matrices:
           |A|^2 = Tau(A) * Tau(A)^* 
    The linear mapping corresponding to A may be a mapping from tensor spaces 
    of shapes shape_rows and shape_cols such that A is a tensor of shape shape_rows + shape_cols.
    Then Tau(A) has shape shape_rows + shape_cols*2.

    Initialization:
        domain: complex NumPyVectorSpace of shape shape_cols
        codomain: complex NumPyVectorSpace of shape shape_rows
    Input: 
        A: complex numpy array of shape shape_rows+shape_cols
    Output:
        Tau(A): complex numpy array of shape shape_rows+shape_cols*2 given by 
            Tau(A)[i,p,q] = A[i,p] * np.conj(A[i,q]) for i in shape_rows, p,q in shape_cols
    """
    
    def __init__(self, domain, codomain):
        if not domain.dtype == complex:
            raise ValueError('domain must be complex')
        if not codomain.dtype == complex:
            raise ValueError('codomain must be complex')         
        self.shape_cols=domain.shape
        self.ncol = len(self.shape_cols)
        self.shape_rows=codomain.shape
        self.nrow = len(self.shape_rows)
        self.aux_shape = self.shape_rows + (1,)*len(self.shape_cols) + self.shape_cols
        super().__init__(domain=NumPyVectorSpace(shape=self.shape_rows+self.shape_cols,dtype=complex),
                         codomain=NumPyVectorSpace(shape=self.shape_rows+self.shape_cols*2,dtype=complex),
                         linear=False)
        
    def _eval(self, A, differentiate=True):
        if differentiate:
            self.A=A
        return A[(...,) + (None,)*self.ncol]*A.conj().reshape(self.aux_shape)
    
    def _derivative(self, B):
        first=self.A[(...,) + (None,)*self.ncol] * B.conj().reshape(self.aux_shape)
        second=B[(...,) + (None,)*self.ncol]*self.A.conj().reshape(self.aux_shape)
        return first+second
    
    def _adjoint(self, C):
        second_adj=np.sum(C*self.A.reshape(self.aux_shape), axis=tuple(np.arange(-self.ncol,0,1)))
        first_adj=np.sum(C*self.A[(...,) + (None,)*self.ncol].conj(), 
                         axis=tuple(np.arange(-2*self.ncol,-self.ncol,1))).conj()
        return first_adj+second_adj

class MatrixAutoProductOp(Operator):
    """Operator mapping a rectangluar matrix E to the matrix product E*E^*.
    E may be represent mapping from a tensor space of shape shape_columns to a tensor 
    space of shape shape_rows such that E is a tensor of shape shape_rows+shape_columns (+ in the sense of concatenation of tuples).
    Input:
        E:  numpy array  
        ndim_cols :  integer [default 1] number of dimensions of the domain of E as linear mapping
    Output: matrix product
        E @ E^*, a tensor of shape shape_rows+shape_columns
    This operator is particularly useful if the product of the last ndim_cols dimensions is much smaller than 
    the product of the other dimensions.
    _deriv_adjoint and _eval_adjoint can be evaluated even if the matrix product does not fit into storage.
    """
    
    def __init__(self, MatrixSpace,ndim_col=1):
        if not isinstance(MatrixSpace,NumPyVectorSpace):
            raise TypeError(f'First argument must be a NumPyVectorSpace. Was given {MatrixSpace1}')
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

        shape_domain = self.shape
        shape_codomain = self.shape_columns*2      
        super().__init__(NumPyVectorSpace(shape_domain,dtype=dtype), 
                         NumPyVectorSpace(shape_codomain,dtype=dtype), 
                         linear=False)

    def prod_A_Bs(self,A,B):
        # returns matrix product A * B^* 
        # (sums over the "column axes" of A and B, which are assumed to be the last ones both in A and B)
        ax = [np.arange(-self.ndim_col,0,1),np.arange(-self.ndim_col,0,1)]  
        return np.tensordot(A,np.conj(B),ax)

    def prod_A_B(self,A,B):
        # returns matrix product A * B 
        # (sums over the "column axes" of A and B, which are assumed to be the last ones in A and the first ones in B)
        ax = [np.arange(-self.ndim_col,0,1),np.arange(0,self.ndim_col,1)]  
        return np.tensordot(A,B,ax)
 
    def prod_G_A(self,G,A):
        # returns matrix product G * A 
        # (sums over "row axes", which are assumed to be the last ones in G and the first ones A)
        ax = [np.arange(-self.ndim_row,0,1),np.arange(0,self.ndim_row)]
        return np.tensordot(G,A,ax)
    
    def prod_As_G(self,A,G):
        # returns matrix product A^* * G 
        # (sums over "row axes", which are assumed to be the first ones both in G and A)
        ax = [np.arange(0,self.ndim_row),np.arange(0,self.ndim_row)]
        return np.tensordot(np.conj(A),G,ax)

    def _eval(self, E, differentiate=False, return_adjoint_eval=False):
        if differentiate==True:
            self.E = E
        if return_adjoint_eval == False:
            return self.prod_A_Bs(E,E)
        else:
            return self._adjoint_eval(E)

    def _derivative(self, dE):
        return self.prod_A_Bs(self.E,dE) + self.prod_A_Bs(dE,self.E)

    def _adjoint(self, G):
        return self.prod_G_A(G,self.E) + self.prod_As_G(G,self.E)

    def _adjoint_eval(self, E):
        self.E = E
        return 2*self.prod_A_B(E, self.prod_As_G(E,E)) 

    def _adjoint_data(self, data):
        """expects *centered* intensities as data"""
        return np.tensordot(data,self.prod_G_A(data,(2./data.shape[0])*self.E),[(0,),(0,)])

    def _adjoint_derivative(self, dE):
        return 2*(self.prod_A_B(self.E, self.prod_As_G(dE,self.E))
                  +self.prod_A_B(dE,     self.prod_As_G(self.E,self.E)))

class CovarianceModGaussian(Operator):
    """ Operator that takes a complex matrix V as input and yields the covariance operator of the Cox process whose intensity 
    is given by the squared modulus of the centered circular Gaussian field with covariance operator VV^*. 
    This operator is given explicitly by 
        V |-> |V^*V|^2 + diag(|Diag (VV^*)|^2)
    The associated linear mapping might act between tensor space such that V can be a tensor. 

    Parameters:
    MatrixSpace: NumPyVectorSpace of the input tensors V
    StateSpace: NumPyVectorSpace of the outputs 
    """

    def __init__(self,MatrixSpace,StateSpace):
        if not isinstance(MatrixSpace,NumPyVectorSpace) or not isinstance(StateSpace,NumPyVectorSpace):
            raise TypeError(f"domain and codomain must be NumPyVectorSpaces, got {domain} and {codomain}")
        sh = StateSpace.shape
        sh_long = MatrixSpace.shape
        if not sh_long[:len(sh)] == sh:
            raise ValueError(f"Last dimensions of domain must agree with those of codomain. Shape of given domain and codmain are {sh_long}, {sh} ")
        sh_diff = sh_long[len(sh):]

        Tau_op = Tau(NumPyVectorSpace(shape=sh_diff,dtype=complex),StateSpace.complex_space())
        MatMul = MatrixAutoProductOp(Tau_op.codomain,ndim_col=len(sh_diff)*2)
        self.Cov = MatMul * Tau_op
        super().__init__(MatrixSpace, MatMul.codomain,linear=False)

    def _eval(self,tau,differentiate=False,return_adjoint_eval=False):
        if differentiate==False:
            return self.Cov(tau)
        else:
            res, self.derivCov = self.Cov.linearize(tau,differentiate=True,return_adjoint_eval=return_adjoint_eval)
            return res

    def _derivative(self,tau):
        return self.derivCov(tau)

    def _adjoint(self,G):
        return self.derivCov.adjoint(G)

    def _adjoint_eval(self,tau):
        res, self.derivCov = self.Cov.linearize(tau,return_adjoint_eval=True)
        return res
    
    def _adjoint_data(self,data):
        return self.Cov.adjoint_data(data)

    def _adjoint_derivative(self,tau):
        return self.derivCov.adjoint_eval(tau)

class CovarianceCoxModGaussian(CovarianceModGaussian):
    def __init__(self,MatrixSpace,StateSpace,T):
        self.Exp = (1/T)*ExpectationCoxModGaussian(MatrixSpace,StateSpace)
        super().__init__(MatrixSpace,StateSpace)

    def _eval(self,tau,differentiate=False,return_adjoint_eval=False):
        if differentiate==False:
            return self.Cov(tau)
        else:
            resCox, self.derivCov = self.Cov.linearize(tau,differentiate=True,return_adjoint_eval=return_adjoint_eval)
            resExp, self.derivExp = self.Exp.linearize(tau,differentiate=True,return_adjoint_eval=return_adjoint_eval)
            sh = resExp.sh
            eye = np.eye(np.prod(sh), dtype=resExp.dtype).reshape(sh + sh)
            # Note that the second summand below is "diag(resExp)"!
            return resCox + eye * resExp.reshape(sh + (1,) * len(sh))

    def _derivative(self,tau):
        dExp = self.derivExp(tau)
        sh = dExp.sh
        eye = np.eye(np.prod(sh), dtype=dExp.dtype).reshape(sh + sh)
        # Again, second term is "diag(dExp)"
        return self.derivCov(tau) + eye*dExp.reshape(sh + (1,) * len(sh))

    def _adjoint(self,G):
        diagG = np.zeros_like(self.Exp)
        for i in np.ndindex(self.Exp):
            diagG[i]=G[i+i] 
        return self.derivCov.adjoint(G) + self.derivExp.adjoint(diagG)

    def _adjoint_eval(self,tau):
        resCov, self.derivCov = self.Cov.linearize(tau,return_adjoint_eval=True)
        resExp, self.derivExp = self.Exp.linearize(tau,return_adjoint_eval=True)
        return resCov + resExp
    
    def _adjoint_derivative(self,tau):
        return self.derivCov.adjoint_eval(tau) + self.derivExp.adjoint_eval(tau)
    
class summation(Operator):
    """
    Operator that wraps the numpy.sum over (any number of) last axes of a numpy array.
    Parameters:
        domain: NumPyVectorSpace of input array
        codomain: NumPyVectorSpace of output array
    """
    def __init__(self, domain, codomain):
        if not isinstance(domain,NumPyVectorSpace) or not isinstance(codomain,NumPyVectorSpace):
            raise TypeError(f"domain and codomain must be NumPyVectorSpaces, got {domain} and {codomain}")
        sh = codomain.shape
        if not domain.shape[:len(sh)] == sh:
            raise ValueError(f"Last dimensions of domain must agree with those of codomain. Shape of given domain and codmain are {domain.shape}, {codomain.shape} ")
        if not domain.dtype == codomain.dtype:
            raise ValueError(f"Data types of domain and codomain must agree. Given: {domain.dtype} and {codomain.dtype}")
        self.sum_axes = tuple(np.arange(len(sh),len(domain.shape)))
        super().__init__(domain, codomain, linear=True)

    def _eval(self, x):
        return np.sum(x,axis=self.sum_axes)
    
    def _adjoint(self, y):
        return np.broadcast_to(y[(...,)+(None,)*len(self.sum_axes)],self.domain.shape)

def ExpectationCoxModGaussian(MatrixSpace,StateSpace):
    """ Yields an operator that takes a complex matrix V as input and yields the expectation of the Cox process whose intensity 
    is given by the squared modulus of the centered circular Gaussian field with covariance operator VV^*. 
    This operator is given explicitly by 
        V |-> \sum_i |V[:,i]|^2
    The associated linear mapping might act between tensor space such that V can be a tensor. 

    Parameters:
    MatrixSpace: NumPyVectorSpace of the input tensors V
    StateSpace: NumPyVectorSpace of the outputs 
    """
    return summation(MatrixSpace.real_space(),StateSpace.real_space()) * SquaredModulus(MatrixSpace)

from copy import deepcopy
class Contrast2FactorPhasedCovOp(Operator):    
    """
    Operator mapping 
        f |-> D e^f V
    Parameters:
        Vcov = factor of the covariance matrix VV^* of the incident beam
        Dfresnel: Fresnel propagator
    Input: contrast
    Output: A factor W = D e^f V of the covariance matrix WW^* of the phased measurement data
    """
    
    def __init__(self, Vcov, Fresnel_prop):
        self.k = Vcov.shape[-1]  # rank of covariance matrix of incident beam
        assert Vcov.shape == Fresnel_prop.domain.shape+(self.k,)
        self.Vcov=Vcov 
        self.Fresnel=Fresnel_prop
        if not Fresnel_prop.linear: # may be only affinely linear if padding by 1 is used
            Fresnel2 = deepcopy(Fresnel_prop)
            _, self.DFresnel = Fresnel2.linearize(Fresnel2.domain.zeros())
        else:
            self.DFresnel = Fresnel_prop

        super().__init__(Fresnel_prop.domain, 
                         NumPyVectorSpace(Fresnel_prop.codomain.shape+(self.k,),dtype=complex), 
                         linear=False)

    def _eval(self, f, differentiate=True):
        if differentiate:
            self.f=f
        mat=self.codomain.zeros()
        for i in range(0, self.k):
            mat[:, :,  i]=self.Fresnel(np.exp(f)*self.Vcov[:,:, i])
        return mat
    
    def _derivative(self, h):
        mat=self.codomain.zeros()
        for i in range(0, self.k):
            mat[:, :, i]=self.DFresnel(np.exp(self.f)*h*self.Vcov[:,:, i])
        return mat
    
    def _adjoint(self, y):
        res=self.domain.zeros()
        for i in range(0, self.k):
            right=self.Vcov[:,:,i].conj()
            left=self.DFresnel.adjoint(y[:, :, i])
            res+=right*left*np.exp(self.f.conj())
        return res
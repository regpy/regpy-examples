from regpy.operators import Exponential, FourierTransform, RealPart, SquaredModulus

import numpy as np
from regpy.vecsps import NumPyVectorSpace
import logging

from regpy.operators import Operator
from regpy.vecsps import UniformGridFcts

def _build_fresnel_2(domain, number=complex(0,1)/80):
    
    """
    Returns the Fresnel-propagator
    """
    
    N=domain.shape[0]
    ft = FourierTransform(domain, centered=True)
    frqs = ft.codomain.coords*0.5*np.pi*N
    propagation_factor = np.exp((-number * (frqs[0]**2 + frqs[1]**2)))
    fresnel_multiplier = Ptw_Multiplication(ft.codomain, propagation_factor)
    return ft.adjoint * fresnel_multiplier * ft


class fresnel_prop(Operator):
    
    """
    Evaluation of Fresnelpropagator
    """
    
    def __init__(self, domain, number=complex(0,1)/80):
        self.N=domain.shape[0]
        self.number=number
        frqs = domain.coords*0.5*np.pi*self.N
        self.propagation_factor = np.fft.fftshift(np.exp((-number * (frqs[0]**2 + frqs[1]**2))))
        super().__init__(domain, domain, linear=True)
        
    def _eval(self, x):
        x=x.reshape(self.domain.shape)
        return np.fft.fftshift(np.fft.ifft2(self.propagation_factor*np.fft.fft2(np.fft.fftshift(x))))
    
    def _adjoint(self, y):
        return np.fft.ifftshift(np.fft.ifft2(self.propagation_factor.T.conj()*np.fft.fft2(np.fft.fftshift(y))))


class Ptw_Multiplication(Operator):
    """A multiplication operator by a constant factor.

    Parameters
    ----------
    domain : regpy.discrs.Discretization
        The underlying discretization
    factor : array-like
        The factor by which to multiply. Can be anything that can be broadcast to `domain.shape`.
    """
    def __init__(self, domain, factor):
        factor = np.asarray(factor)
        # Check that factor can broadcast against domain elements without
        # increasing their size.
        if domain:
            factor = np.broadcast_to(factor, domain.shape)
            assert factor in domain
        self.factor = factor
        super().__init__(domain, domain, linear=True)

    def _eval(self, x):
        return self.factor * x

    def _adjoint(self, x):
        if self.domain.is_complex:
            return np.conj(self.factor) * x
        else:
            return self.factor * x


class Reshape(Operator):
    
    """Reshaping an operator"""
    
    def __init__(self, domain, codomain):
        assert np.prod(domain.shape)==np.prod(codomain.shape)
        
        super().__init__(domain, codomain, linear=True)
        
    def _eval(self, x):
        return x.reshape(self.codomain.shape)
    
    def _adjoint(self, y):
        return y.reshape(self.domain.shape)
    
class Real_to_complex(Operator):
    
    def __init__(self, domain, codomain):
        assert codomain.dtype==complex
        assert domain.dtype!=complex
        super().__init__(domain, codomain, linear=True)
        
    def _eval(self, x):
        return x[0]+complex(0,1)*x[1]
    
    def _adjoint(self, y):
        res=self.domain.zeros()
        res[0]=y.real
        res[1]=y.imag
        return res
        
class Proj(Operator):
    
    def __init__(self, domain, codomain):
        self.N=domain.shape[0]
        self.k=domain.shape[-1]
        super().__init__(domain, codomain, linear=True)
        
    def _eval(self, x):
        res=self.codomain.zeros()
        res[0, ...]=x
        res[1, ...]=x
        return res
    
    def _adjoint(self, y):
        return y[0, ...]+y[1, ...]
    
class power(Operator):
    
    def __init__(self, domain, codomain):
        self.N=domain.shape[0]
        super().__init__(self, domain, codomain, linear=False)
        
    def _eval(self, x, differentiate=True):
        return np.sum(abs(x)**2, axis=-1)
    
    def _derivative(self, h):
        return 2*np.sum(self.x*h.conj(), axis=-1).real
    
    def _adjoint(self, y):
        return 2*y.real.reshape(self.N, self.N, 1)*self.x.conj()
  
from regpy.operators import RealPart, ImaginaryPart
    
class ReIm(Operator):
    """
    Splits a complex numpy array into its real and imaginary part.
    The image space has leading dimension 2, the first component of the result being the real part and the second component the imaginary part. 
    """    
    def __init__(self, domain):
        if not isinstance(domain,UniformGridFcts):
            raise TypeError(f'domain has to be UniformGridFcts. Got {domain}.')
        codomain = UniformGridFcts(np.array([-1,1]),*domain.axes)
        super().__init__(domain, codomain, linear=True)
        self.Re=RealPart(domain)
        self.Im=ImaginaryPart(domain)        
        
    def _eval(self, x):
        res=np.zeros(self.codomain.shape)
        res[0]=self.Re(x)
        res[1]=self.Im(x)
        return res
    
    def _adjoint(self, x):
        return self.Re.adjoint(x[0])+self.Im.adjoint(x[1])

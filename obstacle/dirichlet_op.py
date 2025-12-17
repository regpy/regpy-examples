import numpy as np
import scipy.linalg as scla
import os
import sys
sys.path.append(os.path.dirname(__file__))
from functions.operator import op_S, op_K
from functions.farfield_matrix import farfield_matrix
from functions.setup_iop_data import setup_iop_data
from regpy.operators import Operator
from regpy.vecsps import GridFcts
from regpy.vecsps.curve import GenTrigDiscr





class DirichletOp(Operator):
    r"""Operator that maps the shape of a sound-soft obstacle to the far-field measurements. 
    The scattering problem is described by

	.. math::
        \begin{cases}
            \Delta u +\kappa^2 u = 0 & \text{ in } \mathbb{R}^2\backslash\overline{D}\\
             u = 0  & \text{ on } \partial D\\
            \displaystyle{\lim_{r\to\infty}}r^{\frac{1}{2}}(\frac{\partial u^s}{\partial r}-i\kappa u^s)=0 & \text{ for } r=|x|,
        \end{cases}


    where \(u=u^s+u^i)\ is the total field and \(D)\ is a bounded obstacle in \(\mathbb{R}^2)\ with \(\partial D\in\mathcal{C}^2)\.

    Rather than directly differentiating the forward operator, the Fréchet derivative is evaluated based on 
    an independent discretization of the continuous characterization of this derivative. The adjoint of the 
    Fréchet derivative is the discrete adjoint of derivative.

    Attributes
    ----------
    kappa : complex
        Wave number.
    N_ieq : int
        Number of discrete boundary points.
    N_inc : int
        Number of incident direction.
    N_meas : int
        Number of measurement direction.
    N_FK : int
        Number of Fourier coefficients.

    References
    ----------
    - T. Hohage "Logarithmic convergence rates of the iteratively regularized
      Gauss–Newton method for an inverse potential and an inverse scattering problem", Inverse
      Problems, 13 (1997) 1279–1299.
    """

    def __init__(self, kappa, N_ieq=128, N_inc=4, N_meas=64, N_FK=64):   
        self.kappa = kappa 
        """Wave number."""          
        self.N_ieq = N_ieq
        """(2*self.N_ieq) is the number of discrete boundary points."""
        if isinstance(N_inc, int) and N_inc > 0:
            self.N_inc = N_inc
            """Number of incident direction."""
            t=2*np.pi*np.arange(0, self.N_inc)/self.N_inc
            self.inc_directions=[np.array([np.cos(s), np.sin(s)]) for s in t]
            """Incident direction."""
        elif isinstance(N_inc, list) and all([dir.shape == (2,) for dir in N_inc]):
            self.N_inc = len(N_inc)
            """Number of incident direction."""
            self.inc_directions = N_inc 
            """Incident direction."""
        else: 
            raise ValueError("Incident direction neither an arry of direction nor an positiv integer")

        if isinstance(N_meas, int) and N_meas > 0:
            self.N_meas = N_meas
            """Number of measurement direction."""
            t=2*np.pi*np.arange(0, self.N_meas)/self.N_meas
            self.meas_directions=[np.array([np.cos(s), np.sin(s)]) for s in t]
            """Measurement direction."""
        elif isinstance(N_meas, list) and all([meas.shape == (2,) for meas in N_meas]):
            self.N_meas = len(N_meas)
            """Number of Measurement direction."""
            self.meas_directions = N_meas 
            """Measurement direction."""
        else: 
            raise ValueError("Measurement direction neither an arry of direction nor an positiv integer")

        self.N_FK = N_FK
        """Number of Fourier coefficients."""
        self.domain_curve = None
        self.w_sl=-1*complex(0,1)*self.kappa
        self.w_dl=1
        """Weights of single and double layer potentials. Use a mixed single and double layer potential ansatz with
        weights w_sl and w_dl."""

        meas_dir=np.linspace(0, 2*np.pi, self.N_meas, endpoint=False)
        inc_dir=np.linspace(0, 2*np.pi, self.N_inc, endpoint=False)
        codomain=GridFcts(meas_dir, inc_dir, dtype=complex)

        super().__init__(
            domain=GenTrigDiscr(2*self.N_FK),
            codomain=codomain,
            linear=False
        )

    def _eval(self, coeff, differentiate=True):

        self.domain_curve = self.domain.bd_eval(coeff, 2*self.N_ieq, 3)
        Iop_data = setup_iop_data(self.domain_curve, self.kappa)

        # Assemble integral operator matrix
        if self.w_sl!=0:
            Iop = self.w_sl*op_S(self.domain_curve, Iop_data)
        else:
            Iop = np.zeros(np.size(self.domain.curve,1),np.size(self.domain.curve,1))
        if self.w_dl!=0:
            Iop = Iop + self.w_dl*(np.diag(self.domain_curve.zpabs)+op_K(self.domain_curve,Iop_data))
        # LU-factorization of integral operator matrix
        self.lu, self.piv = scla.lu_factor(Iop)

        # Assemble far field operator matrix for operator application
        FF_SL = farfield_matrix(self.domain_curve,self.meas_directions,self.kappa,-1.,0.)
        
        if differentiate:
            # Assemble far field operator matrix
            self.FF_combined = farfield_matrix(self.domain_curve,self.meas_directions,self.kappa, \
                                               self.w_sl,self.w_dl)        
   
        # Assemble right hand sides
        rhs = np.zeros((2*self.N_ieq, self.N_inc), dtype=complex)
        for l, dir in enumerate(self.inc_directions):
            rhs[:,l] = 2*np.exp(complex(0,1)*self.kappa*dir.dot(self.domain_curve.z))*  \
                (self.w_dl*complex(0,1)*self.kappa*dir.dot(self.domain_curve.normal) \
                                         +self.w_sl*self.domain_curve.zpabs)
            
        self.dudn = scla.lu_solve((self.lu,self.piv), rhs,trans=1)
        """Normal derivative of total field at boundary."""

        return np.dot(FF_SL, self.dudn)

    def _derivative(self, h):
        rhs = -2*self.domain_curve.zpabs[:,None] * np.ones(self.N_inc) 
        rhs *= self.domain_curve.der_normal(h)[:,np.newaxis]
        rhs = rhs * self.dudn
        phi =  scla.lu_solve((self.lu,self.piv), rhs)

        return self.FF_combined @ phi

    def _adjoint(self, g):
        phi = self.FF_combined.T.conj() @ g
        rhs = scla.lu_solve((self.lu,self.piv), phi, trans=2)
        res = np.sum((rhs*np.conjugate(self.dudn)).real,axis=1)
        res *= -2.*self.domain_curve.zpabs

        return self.domain_curve.adjoint_der_normal(res)

    def create_synthetic_data(self, true_curve, N_ieq_synth=64):
        """
        This alternative operator evaluation method is included to avoid inverse crimes. 
        In contrast to the _eval, a potential ansatz is chosen in the boundary integral equation method 
        rather than a direct ansatz. Moreover, a different discretization may be chosen.
        """
        bd_ex = true_curve(2*N_ieq_synth,2)
        wdlTmp=1*self.w_dl
        self.w_dl=0

        Iop_data = setup_iop_data(bd_ex, self.kappa)
        if self.w_sl!=0:
            Iop = self.w_sl*op_S(bd_ex, Iop_data)
        else:
            Iop = np.zeros(2*N_ieq_synth,2*N_ieq_synth)
        if self.w_dl!=0:
            Iop = Iop + self.w_dl*(np.diag(bd_ex.zpabs) + op_K(bd_ex, Iop_data))
            
        FF_combined = farfield_matrix(bd_ex, self.meas_directions, self.kappa, self.w_sl, self.w_dl)

        farfield = np.zeros((self.N_meas, self.N_inc),dtype = complex)
        for l, dir in enumerate(self.inc_directions):
            rhs = -2*np.exp(complex(0,1)*self.kappa*dir.dot(bd_ex.z))*bd_ex.zpabs
            rhs=rhs.flatten()
            phi = scla.solve(Iop, rhs)
            farfield[:,l]=FF_combined.dot(phi)

        self.w_dl=wdlTmp
        return farfield, bd_ex
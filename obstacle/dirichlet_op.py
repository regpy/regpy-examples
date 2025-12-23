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
from regpy.vecsps.curve import GenTrigSpc, StarTrigRadialFcts


def check_scattering_parameters(kappa,N_ieq,inc_waves,meas_dir):
    if not isinstance(kappa,(float,int,complex)) or not kappa.real>0:
        raise ValueError('kappa must be positive.')          
    if not isinstance(N_ieq,int):
        raise ValueError('N_ieq must be integer.')
    if isinstance(inc_waves, int) and inc_waves > 0:
        N_inc = inc_waves
        t=2*np.pi*(np.arange(0, N_inc)/N_inc-0.5)
        inc_directions=[np.array([np.cos(s), np.sin(s)]) for s in t]
    elif isinstance(inc_waves, list) and all([dir.shape == (2,) and dir[0]**2+dir[1]**2==1. \
                                                for dir in inc_waves]):
        N_inc = len(inc_waves)
        inc_directions = inc_waves 
    else: 
        raise ValueError("Incident direction neither an array of direction nor a positive integer")

    if isinstance(meas_dir, int) and meas_dir > 0:
        N_meas = meas_dir
        t=2*np.pi*(np.arange(0, N_meas)/N_meas-0.5)
        meas_directions=[np.array([np.cos(s), np.sin(s)]) for s in t]
    elif isinstance(meas_dir, list) and all([meas.shape == (2,) and meas[0]**2 + meas[1]**2==1 \
                                                for meas in meas_dir]):
        N_meas = len(meas_dir)
        meas_directions = meas_dir 
    else: 
        raise ValueError("Measurement direction neither an arry of direction nor a positive integer")
    meas_dir = [np.angle(complex(x,y)) for (x,y) in meas_directions]
    inc_dir = [np.angle(complex(x,y)) for (x,y) in inc_directions]
    codomain=GridFcts(meas_dir, inc_dir, dtype=complex,use_cell_measure=True)

    return kappa, N_ieq, N_inc, inc_directions, N_meas, meas_directions, codomain


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

    Paramters:
    ----------
    kappa: complex
        wave number
    N_ieq: int [default: 128]
        boundary integral equation matrix is of size (2*N_ieq)^2
    inc_waves: int or list of tuples of reals [default: 4]
        directions of incident waves. If int these are chosen equidistant on the unit circle. 
        If a list of tuples, each tuple must lie on the unit circle
    meas_dir: int or list of tuples of reals [default: 64]
        farfield measurement directions. Same format as inc_waves.
        kappa : complex
        Wave number.
    N_FK : int
        Number of Fourier coefficients.

    References
    ----------
    - T. Hohage "Logarithmic convergence rates of the iteratively regularized
      Gauss–Newton method for an inverse potential and an inverse scattering problem", Inverse
      Problems, 13 (1997) 1279–1299.
    """

    def __init__(self, kappa, N_ieq=128, N_inc=4, N_meas=64, domain = None):   
        self.kappa,  self.N_ieq, self.N_inc, self.inc_directions, self.N_meas, self.meas_directions,codomain \
            = check_scattering_parameters(kappa,N_ieq,N_inc,N_meas)

        if domain is None:
            domain = StarTrigRadialFcts(dim=2*self.N_ieq,n=2*self.N_ieq)
        else:
            domain.n = 2*self.N_ieq
        self.w_sl=-1*complex(0,1)*self.kappa
        self.w_dl=1
        """Weights of single and double layer potentials. Use a mixed single and double layer potential ansatz with
        weights w_sl and w_dl."""

        super().__init__(
            domain=domain,
            codomain=codomain,
            linear=False
        )

    def _eval(self, coeff, differentiate=True):

        self.curve = self.domain.coeff2curve(coeff, 2)
        Iop_data = setup_iop_data(self.curve, self.kappa)

        # Assemble integral operator matrix
        Iop = self.w_sl*op_S(self.curve, Iop_data) if self.w_sl!=0 \
              else np.zeros(np.size(self.domain.curve,1),np.size(self.domain.curve,1))
        if self.w_dl!=0:
            Iop += self.w_dl*(np.diag(self.curve.zpabs)+op_K(self.curve,Iop_data))
        # LU-factorization of integral operator matrix
        self.lu, self.piv = scla.lu_factor(Iop)

        # Assemble far field operator matrix for operator application
        FF_SL = farfield_matrix(self.curve,self.meas_directions,self.kappa,-1.,0.)
        
        if differentiate:
            # Assemble far field operator matrix
            self.FF_combined = farfield_matrix(self.curve,self.meas_directions,self.kappa, \
                                               self.w_sl,self.w_dl)        
   
        # Assemble right hand sides
        rhs = np.zeros((2*self.N_ieq, self.N_inc), dtype=complex)
        for l, dir in enumerate(self.inc_directions):
            rhs[:,l] = 2*np.exp(complex(0,1)*self.kappa*dir.dot(self.curve.z))*  \
                (self.w_dl*complex(0,1)*self.kappa*dir.dot(self.curve.normal) \
                                         +self.w_sl*self.curve.zpabs)
            
        self.dudn = scla.lu_solve((self.lu,self.piv), rhs,trans=1)
        """Normal derivative of total field at boundary."""

        return FF_SL @ self.dudn

    def _derivative(self, h):
        rhs = -2*self.curve.zpabs[:,None] * np.ones(self.N_inc) 
        rhs *= self.curve.der_normal(h)[:,np.newaxis]
        rhs = rhs * self.dudn
        phi =  scla.lu_solve((self.lu,self.piv), rhs)

        return self.FF_combined @ phi

    def _adjoint(self, g):
        phi = self.FF_combined.T.conj() @ g
        rhs = scla.lu_solve((self.lu,self.piv), phi, trans=2)
        res = np.sum((rhs*np.conjugate(self.dudn)).real,axis=1)
        res *= -2.*self.curve.zpabs

        return self.curve.adjoint_der_normal(res)

    def create_synthetic_data(self, true_curve, N_ieq_synth=128):
        """
        This alternative operator evaluation method is included to avoid inverse crimes. 
        In contrast to the _eval, a potential ansatz is chosen in the boundary integral equation method 
        rather than a direct ansatz. Moreover, a different discretization may be chosen.
        """
        bd_ex = true_curve(2*N_ieq_synth,2)
        wdlTmp=1*self.w_dl
        self.w_dl=0

        Iop_data = setup_iop_data(bd_ex, self.kappa)
        Iop = self.w_sl*op_S(bd_ex, Iop_data) if self.w_sl!=0. \
           else np.zeros(2*N_ieq_synth,2*N_ieq_synth)
        if self.w_dl!=0:
            Iop += self.w_dl*(np.diag(bd_ex.zpabs) + op_K(bd_ex, Iop_data))
            
        FF_combined = farfield_matrix(bd_ex, self.meas_directions, self.kappa, self.w_sl, self.w_dl)
        self.w_dl=wdlTmp

        rhs = np.zeros((2*N_ieq_synth,self.N_inc),dtype=complex)
        for l, dir in enumerate(self.inc_directions):
            rhs[:,l] = -2*np.exp(complex(0,1)*self.kappa*dir.dot(bd_ex.z))*bd_ex.zpabs
        phi = scla.solve(Iop, rhs)
        farfield = FF_combined @ phi

        return farfield, bd_ex
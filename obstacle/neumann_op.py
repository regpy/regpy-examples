import numpy as np
import scipy.linalg as scla

from functions.operator import op_T
from functions.operator import op_K
from dirichlet_op import check_scattering_parameters

from functions.farfield_matrix import farfield_matrix
from functions.setup_iop_data import setup_iop_data
from regpy.operators import Operator

#from regpy.vecsps.curve import StarCurveDiscr
from regpy.vecsps import GridFcts
from regpy.vecsps.curve import GenTrigSpc

class NeumannOp(Operator):
    r"""Operator that maps the shape of a sound-hard obstacle to the far-field measurements. 
    The scattering problem is modeled by

	.. math::
        \begin{cases}
            \Delta u +\kappa^2 u = 0 & \text{ in }\mathbb{R}^2\backslash\overline{D}\\
            \frac{\partial u}{\partial\nu}=0  & \text{ on }\partial D\\
            \displaystyle{\lim_{r\to\infty}}r^{\frac{1}{2}}(\frac{\partial u^s}{\partial r}-i\kappa u^s)=0 &\text{ for } r=|x|.
        \end{cases}


    where \(u=u^s+u^i\) is the total field and \(D\) is the bounded obstacle in \mathbb{R}^2 with \(\partial D\in\mathcal{C}^2\).

    Parameters: 
    kappa, N_ieq, inc_waves, meas_dir: see DirichletOp
 
    References
    ----------
    - T. Hohage. "Convergence rates of a regularized Newton method in sound-hard inverse scattering", 
    SIAM journal on numerical analysis, 36 (1998): 125-142."""

    def __init__(self, kappa, N_ieq=128, inc_waves=4, meas_dir=64, N_FK=64):
        self.kappa,  self.N_ieq, self.N_inc, self.inc_directions, self.N_meas, self.meas_directions, codomain \
            = check_scattering_parameters(kappa,N_ieq,inc_waves,meas_dir)
        self.N_FK = N_FK
        """Number of Fourier coefficients."""
        self.w_sl = -complex(0,1)*self.kappa
        self.w_dl = 1
        """Weights of single and double layer potentials."""

        super().__init__(
            domain=GenTrigSpc(2*self.N_FK),
            codomain=codomain,
            linear=False
        )

    def _eval(self, coeff, differentiate=True): 

        # assemble boundary integral operator
        self.curve = self.domain.bd_eval(coeff, 2*self.N_ieq, 3)
        Iop_data = setup_iop_data(self.curve, self.kappa)
        Iop = self.w_dl*op_T(self.curve, Iop_data) if self.w_dl!=0 \
            else np.zeros((2*self.N_ieq, 2*self.N_ieq),dtype=complex)
        if self.w_sl!=0:
            Iop += self.w_sl*(op_K(self.curve, Iop_data).T - np.diag(self.curve.zpabs))

        # LU-factorization of integral operator matrix
        self.lu, self.piv = scla.lu_factor(Iop)

        # assemble far-field matrices
        FF_DL = farfield_matrix(self.curve, self.meas_directions, self.kappa, 0, 1)
        if differentiate:        
           self.FF_combined = farfield_matrix(self.curve, self.meas_directions, self.kappa,\
                                           self.w_sl, self.w_dl)

        # assemble right-hand sides    
        rhs = np.zeros((2*self.N_ieq,self.N_inc),dtype=complex)
        for l, dir in enumerate(self.inc_directions):
            rhs[:,l] = -2*np.exp(complex(0,1)*self.kappa*dir.dot(self.curve.z))*\
                (self.w_dl*complex(0,1)*self.kappa*dir.dot(self.curve.normal)\
                                         +self.w_sl*self.curve.zpabs)
        self.u =  scla.lu_solve((self.lu,self.piv), rhs,trans=1)
        """total field at boundary."""

        if differentiate:
            self.duds = np.zeros((2*self.N_ieq,self.N_inc),dtype=complex)
            """arc-length derivative of total field at the boundary""" 
            for l in range(0, self.N_inc):
               self.duds[:,l] = self.curve.arc_length_der(self.u[:,l])
        else:
            del self.u

        return FF_DL @ self.u
    
    def _derivative(self, h):
        hn = self.curve.der_normal(h)
        rhs = self.kappa**2* hn[:,None]*self.u
        for l in range(0,self.N_inc):
            rhs[:,l] += self.curve.arc_length_der(hn*self.duds[:,l])
        
        rhs *= 2*self.curve.zpabs[:,None]
        
        return self.FF_combined @ scla.lu_solve((self.lu,self.piv), rhs)

    def _adjoint(self, g):
        v =  scla.lu_solve((self.lu,self.piv),  self.FF_combined.T.conj() @ g, trans=2)

        rhs = self.kappa**2*(v.conj() * self.u).real

        for l in range(0, self.N_inc):
            dvds = self.curve.arc_length_der(v[:,l])
            rhs[:,l] -= (dvds.conj() * self.duds[:,l]).real

        res = np.sum(rhs,axis=1)
        res *= 2.*self.curve.zpabs

        return self.curve.adjoint_der_normal(res)

    def create_synthetic_data(self, true_curve, N_ieq_synth=128):
        """
        This alternative operator evaluation method is included to avoid inverse crimes. 
        In contrast to the _eval, a potential ansatz is chosen in the boundary integral equation method 
        rather than a direct ansatz. Moreover, a different discretization may be chosen.
        """
        bd_ex = true_curve(2*N_ieq_synth,3)
        
        Iop_data = setup_iop_data(bd_ex, self.kappa)

        Iop = self.w_dl*(op_T(bd_ex, Iop_data)) if self.w_dl!=0 \
          else np.zeros(2*N_ieq_synth,2*N_ieq_synth)
        if self.w_sl!=0:
            Iop += self.w_sl*(op_K(bd_ex, Iop_data).T-np.diag(bd_ex.zpabs))
    
        FF_combined = farfield_matrix(bd_ex, self.meas_directions, self.kappa, self.w_sl, self.w_dl)

        rhs = np.zeros((2*N_ieq_synth,self.N_inc),dtype=complex)
        for l, dir in enumerate(self.inc_directions):
            rhs[:,l] = -2*np.exp(complex(0,1)*self.kappa*dir.dot(bd_ex.z))*(complex(0,1)*self.kappa*dir.dot(bd_ex.normal))
        
        phi = scla.solve(Iop, rhs)
        farfield=FF_combined @ phi 

        return farfield, bd_ex
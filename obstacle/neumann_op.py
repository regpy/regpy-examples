import numpy as np
import scipy.linalg as scla

from functions.operator import op_T, op_K
from functions.farfield_matrix import farfield_matrix, nearfield_matrix
from functions.setup_iop_data import setup_iop_data
from dirichlet_op import check_scattering_parameters, check_create_synthetic_data_params
from regpy.operators import Operator
from regpy.vecsps.curve import ParameterizedCurveSpc,GenCurve,Peanut

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
    kappa, N_ieq, inc_waves, R_inc, meas_dir,R_meas,domain: see DirichletOp
 
    References
    ----------
    - T. Hohage. "Convergence rates of a regularized Newton method in sound-hard inverse scattering", 
    SIAM journal on numerical analysis, 36 (1998): 125-142."""

    def __init__(self, kappa:complex, 
                 N_ieq:int=128, 
                 inc_waves: int | list[tuple[float]]=4,
                 R_inc: float = np.inf, 
                 meas:  int | list[tuple[float]]=64, 
                 R_meas: float = np.inf,
                 domain: ParameterizedCurveSpc|None = None):   
        self.kappa,  self.N_ieq, self.N_inc, self.inc_pts, self.R_inc, \
            self.N_meas, self.meas_pts, self.R_meas, \
            domain, codomain \
            = check_scattering_parameters(kappa,N_ieq,inc_waves,R_inc,meas,R_meas,domain)
        if R_inc!=np.inf:
            raise ValueError('Only the case of plane incident wave is implemented so far, no point sources.')

        self.w_sl = -complex(0,1)*self.kappa
        self.w_dl = 1
        """Weights of single and double layer potentials."""

        super().__init__(
            domain=domain,
            codomain=codomain,
            linear=False
        )

    def _eval(self, coeff, differentiate=True): 

        # assemble boundary integral operator
        self.curve = self.domain.coeff2curve(coeff,3)
        Iop_data = setup_iop_data(self.curve, self.kappa)
        Iop = self.w_dl*op_T(self.curve, Iop_data) if self.w_dl!=0 \
            else np.zeros((2*self.N_ieq, 2*self.N_ieq),dtype=complex)
        if self.w_sl!=0:
            Iop += self.w_sl*(op_K(self.curve, Iop_data).T - np.diag(self.curve.zpabs))

        # LU-factorization of integral operator matrix
        self.lu, self.piv = scla.lu_factor(Iop)

        # assemble far- or near-field matrices
        matrix_fct = farfield_matrix if self.R_meas == np.inf else nearfield_matrix
        FF_DL = matrix_fct(self.curve, self.meas_pts, self.kappa, 0, 1)
        if differentiate:        
           self.FF_combined = matrix_fct(self.curve, self.meas_pts, self.kappa,\
                                           self.w_sl, self.w_dl)

        # assemble right-hand sides    
        rhs = np.zeros((2*self.N_ieq,self.N_inc),dtype=complex)
        for l, dir in enumerate(self.inc_pts):
            rhs[:,l] = -2*np.exp(1j*self.kappa*dir.dot(self.curve.z))*\
                (self.w_dl*1j*self.kappa*dir.dot(self.curve.normal)\
                                         +self.w_sl*self.curve.zpabs)
        self.u =  scla.lu_solve((self.lu,self.piv), rhs,trans=1)
        """total field at boundary."""

        farfield = FF_DL @ self.u

        if differentiate:
            self.duds =  self.curve.arc_length_der(self.u)
            """arc-length derivative of total field at the boundary""" 
        else:
            del self.u

        return farfield
    
    def _derivative(self, h):
        hn = self.curve.der_normal(h)

        rhs = self.kappa**2* hn[:,None]*self.u
        rhs += self.curve.arc_length_der(hn[:,None]*self.duds)        
        rhs *= 2*self.curve.zpabs[:,None]
        
        return self.FF_combined @ scla.lu_solve((self.lu,self.piv), rhs)

    def _adjoint(self, g):
        v =  scla.lu_solve((self.lu,self.piv),  self.FF_combined.T.conj() @ g, trans=2)

        rhs = self.kappa**2*(v.conj() * self.u).real       
        dvds = self.curve.arc_length_der(v)
        rhs -= (dvds.conj() * self.duds).real

        res = np.sum(rhs,axis=1)
        res *= 2.*self.curve.zpabs

        return self.curve.der_normal.adjoint(res)

    def create_synthetic_data(self, 
                              true_curve:GenCurve = Peanut, 
                              N_ieq_synth:int|None=None):
        """
        This alternative operator evaluation method is included to avoid inverse crimes. 
        In contrast to the _eval, a potential ansatz is chosen in the boundary integral equation method 
        rather than a direct ansatz. Moreover, a different discretization may be chosen.
        """
        bd_ex,N_ieq_synth = check_create_synthetic_data_params(true_curve,N_ieq_synth,nderivs=3,op=self)
              
        Iop_data = setup_iop_data(bd_ex, self.kappa)

        Iop = self.w_dl*(op_T(bd_ex, Iop_data)) if self.w_dl!=0 \
          else np.zeros(2*N_ieq_synth,2*N_ieq_synth)
        if self.w_sl!=0:
            Iop += self.w_sl*(op_K(bd_ex, Iop_data).T-np.diag(bd_ex.zpabs))
    
        matrix_fct = farfield_matrix if self.R_meas == np.inf else nearfield_matrix
        FF_combined = matrix_fct(bd_ex, self.meas_pts, self.kappa, self.w_sl, self.w_dl)

        rhs = np.zeros((2*N_ieq_synth,self.N_inc),dtype=complex)
        for l, dir in enumerate(self.inc_pts):
            rhs[:,l] = -2*np.exp(complex(0,1)*self.kappa*dir.dot(bd_ex.z))*(complex(0,1)*self.kappa*dir.dot(bd_ex.normal))
        
        phi = scla.solve(Iop, rhs)
        farfield=FF_combined @ phi 

        return farfield, bd_ex
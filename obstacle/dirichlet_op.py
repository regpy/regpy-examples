import numpy as np
import scipy.linalg as scla
from scipy.special import hankel1
import os
import sys
sys.path.append(os.path.dirname(__file__))
from functions.operator import op_S, op_K
from functions.farfield_matrix import farfield_matrix,nearfield_matrix
from functions.setup_iop_data import setup_iop_data
from regpy.operators import Operator
from regpy.vecsps import GridFcts
from regpy.vecsps.curve import ParameterizedCurveSpc, StarTrigRadialFcts,GenCurve, Peanut
from regpy.util import Errors


def check_scattering_parameters(kappa,N_ieq,inc_waves,R_inc,meas_dir,R_meas,domain):
    if not isinstance(kappa,(float,int,complex)) or not kappa.real>0:
        raise ValueError('kappa must be positive.')          
    if not isinstance(N_ieq,int):
        raise ValueError('N_ieq must be integer.')
    if not isinstance(R_inc,(float,int)):
        raise TypeError(f'R_inc must be a float. Got {R_inc}.')
    if isinstance(inc_waves, int) and inc_waves > 0:
        N_inc = inc_waves
        t=2*np.pi*(np.arange(0, N_inc)/N_inc-0.5)
        R = R_inc if R_inc<np.inf else 1.
        inc_pts=R*np.vstack((np.cos(t), np.sin(t))).T
    elif isinstance(inc_waves, list) and all([dir.shape == (2,) and dir[0]**2+dir[1]**2==1. \
                                                for dir in inc_waves]):
        N_inc = len(inc_waves)
        inc_pts = inc_waves 
    else: 
        raise ValueError(f"Incident direction neither an array of direction nor a positive integer. Got {inc_waves}")

    if not isinstance(R_meas,(float,int)):
        raise TypeError(f'R_meas must be a float. Got {R_meas}.')
    if isinstance(meas_dir, int) and meas_dir > 0:
        N_meas = meas_dir
        t=2*np.pi*(np.arange(0, N_meas)/N_meas-0.5)
        R = R_meas if R_meas<np.inf else 1.
        meas_pts=R*np.vstack((np.cos(t), np.sin(t))).T
    elif isinstance(meas_dir, list) and all([meas.shape == (2,) and meas[0]**2 + meas[1]**2==1 \
                                                for meas in meas_dir]):
        N_meas = len(meas_dir)
        meas_pts = meas_dir 
    else: 
        raise ValueError("Measurement direction neither an arry of direction nor a positive integer")
    meas_grid = np.angle(meas_pts[:,0]+1j*meas_pts[:,1]) if isinstance(meas_dir,int) else np.arange(meas_dir.shape[0])
    inc_grid = np.angle(inc_pts[:,0]+1j*inc_pts[:,1]) if isinstance(inc_waves,int) else np.arange(inc_waves.shape[0])
    codomain=GridFcts(meas_grid, inc_grid, dtype=complex,use_cell_measure=True)

    if domain is None:
        domain = StarTrigRadialFcts(dim=2*N_ieq,n=2*N_ieq)
    else:
        if not isinstance(domain,ParameterizedCurveSpc):
            raise TypeError(Errors.not_instance(domain,ParameterizedCurveSpc))
        domain.n = 2*N_ieq
    return kappa, N_ieq, N_inc, inc_pts, R_inc, N_meas, meas_pts, R_meas, domain, codomain

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
    inc_waves: int or np.ndarray, optional
        Determines the incident waves. Defaults to 4. If an integer, the incident directions 
        (for R_inc=np.inf) or the source points of the incident waves (for R_inc<np.inf) are chosen equidistant, in the latter case on a circle of radiaus R_inc.
        If inc_waves is a 2xN_inc matrix, it determines the directions of the incident waves
        (if R_inc==np.inf -- in this case each rows must lie on the unit circle) or the locations 
        of the source points (if R_inc<np.inf -- in this case the value of R_inc does not matter). 
    R_inc: float, optional
        Determines if plane incident waves (R_inc = np.inf) are used or point sources. Defaults to np.inf. 
    meas: int or list of tuples of floats, optional
        If an integer, it determines the number of measurement directions (R_meas==np.inf) or 
        measurement points, which are then chosen equidistant, for R_meas<np.inf on the circle 
        of radius R_meas. Defaults to 64.
        If a 2xN_meas matrix, its rows contain the measurement directions (R_meas==np.inf) 
        or the measurement points (R_meas<np.inf -- in this case the value of R_meas does not matter).
    R_meas: float, optional
        Determines if farfield data (R_meas==np.inf) or near field data are used. Defaults to np.inf.
    kappa : complex
        Wave number.
    domain: regpy.vecsps.curve.ParameterizedCurveSpc|None, optional
        Domain of the operator. If None [default], it is chose as StarTrigRadialFcts with dim=2*N_ieq

    References
    ----------
    - T. Hohage "Logarithmic convergence rates of the iteratively regularized
      Gauss–Newton method for an inverse potential and an inverse scattering problem", Inverse
      Problems, 13 (1997) 1279–1299.
    """

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
        if self.R_inc!=np.inf:
            raise ValueError('Case of point sources does not work, yet!')

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

        # Assemble far or near field operator matrix for operator application
        matrix_fct = farfield_matrix if self.R_meas == np.inf else nearfield_matrix
        FF_SL = matrix_fct(self.curve,self.meas_pts,self.kappa,-1.,0.)
        
        if differentiate:
            # Assemble far or near field operator matrix for the combined potential
            self.FF_combined = matrix_fct(self.curve,self.meas_pts,self.kappa, \
                                               self.w_sl,self.w_dl)        
   
        # Assemble right hand sides
        rhs = np.zeros((2*self.N_ieq, self.N_inc), dtype=complex)
        for l, dir in enumerate(self.inc_pts):
            if self.R_inc == np.inf:
                rhs[:,l] = 2*np.exp(1j*self.kappa*dir.dot(self.curve.z))*  \
                    (self.w_dl*1j*self.kappa*dir.dot(self.curve.normal) \
                                         +self.w_sl*self.curve.zpabs)
            else:
                kdiff = self.kappa*(self.curve.z - dir[:,None])
                kdist = np.linalg.norm(kdiff,axis=0)
                rhs[:,l] = 0.5j*self.w_sl*self.curve.zpabs*hankel1(0,kdist) \
                    - 0.5j*self.w_dl*np.sum(kdiff*self.curve.normal,axis=0) * hankel1(1,kdist) 
            
        self.dudn = scla.lu_solve((self.lu,self.piv), rhs,trans=1)
        """Normal derivative of total field at boundary."""

        return FF_SL @ self.dudn

    def _derivative(self, h):
        hn = self.curve.der_normal(h)
        rhs = -2*self.curve.zpabs[:,None] * np.ones(self.N_inc) 
        rhs *= hn[:,np.newaxis]
        rhs = rhs * self.dudn
        phi =  scla.lu_solve((self.lu,self.piv), rhs)

        return self.FF_combined @ phi

    def _adjoint(self, g):
        phi = self.FF_combined.T.conj() @ g
        rhs = scla.lu_solve((self.lu,self.piv), phi, trans=2)
        res = np.sum((rhs*np.conjugate(self.dudn)).real,axis=1)
        res *= -2.*self.curve.zpabs

        return self.curve.der_normal.adjoint(res)

    def create_synthetic_data(self, true_curve :GenCurve | type = Peanut, N_ieq_synth:int | None =None):
        """
        This alternative operator evaluation method is included to avoid inverse crimes. 
        In contrast to the _eval, a potential ansatz is chosen in the boundary integral equation method 
        rather than a direct ansatz. Moreover, a different discretization may be chosen.
        """
        bd_ex,N_ieq_synth = check_create_synthetic_data_params(true_curve,N_ieq_synth,nderivs=2,op=self)
        wdlTmp=1*self.w_dl
        self.w_dl=0

        Iop_data = setup_iop_data(bd_ex, self.kappa)
        Iop = self.w_sl*op_S(bd_ex, Iop_data) if self.w_sl!=0. \
           else np.zeros(2*N_ieq_synth,2*N_ieq_synth)
        if self.w_dl!=0:
            Iop += self.w_dl*(np.diag(bd_ex.zpabs) + op_K(bd_ex, Iop_data))
            
        matrix_fct = farfield_matrix if self.R_meas == np.inf else nearfield_matrix
        FF_combined = matrix_fct(bd_ex, self.meas_pts, self.kappa, self.w_sl, self.w_dl)
        self.w_dl=wdlTmp

        rhs = np.zeros((2*N_ieq_synth,self.N_inc),dtype=complex)
        for l, dir in enumerate(self.inc_pts):
            if self.R_inc == np.inf:
                rhs[:,l] = -2*np.exp(complex(0,1)*self.kappa*dir.dot(bd_ex.z))*bd_ex.zpabs
            else:
                kdiff = self.kappa*(bd_ex.z - dir[:,None])
                kdist = np.linalg.norm(kdiff,axis=0)
                rhs[:,l] = -0.5j*self.w_sl*bd_ex.zpabs*hankel1(0,kdist) 

        phi = scla.solve(Iop, rhs)
        farfield = FF_combined @ phi

        return farfield, bd_ex
    
def check_create_synthetic_data_params(true_curve,N_ieq_synth,nderivs,op):
    """
    """
    if not isinstance(nderivs,int):
        raise TypeError(Errors.not_instance(nderivs,int))
    if not isinstance(N_ieq_synth,int) and N_ieq_synth is not None:
        raise TypeError(Errors.not_instance(N_ieq_synth,int))
    if isinstance(true_curve,type):
        N_ieq_synth=N_ieq_synth if N_ieq_synth is not None else op.N_ieq
        curve = true_curve(2*N_ieq_synth,nderivs)
    elif isinstance(true_curve,GenCurve):
        curve =true_curve
        if N_ieq_synth is not None:
            curve.n = 2*N_ieq_synth
        else:
            N_ieq_synth = curve.n//2
    else:
        raise TypeError(Errors.generic_message(f"true_curve has unsuitable type {type(true_curve)}"))
    return curve, N_ieq_synth
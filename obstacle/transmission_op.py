import numpy as np
import scipy.linalg as scla
from scipy.special import hankel1

from functions.operator import op_S, op_K, op_T
from functions.farfield_matrix import farfield_matrix_trans, farfield_matrix,nearfield_matrix
from functions.setup_iop_data import setup_iop_data
from dirichlet_op import check_scattering_parameters, check_create_synthetic_data_params
from regpy.operators import Operator
from regpy.vecsps.curve import ParameterizedCurveSpc,GenCurve,Peanut

class TransmissionOp(Operator):
    r"""Operator that maps an admissible boundary \partial D onto the corresponding far field pattern. 
    The related transmission problem for the Helmholtz equation is modeled by

	.. math::
        \begin{cases}
            \Delta u_i +\kappa_i^2 u_i = 0 & \text{ in } D \\
            \Delta u_e +\kappa_e^2 u_e = 0 & \text{ in } \mathbb{R}^2\backslash\overline{D}\\
             u_i=u & \text{ on } \partial D\\
            \frac{\partial u_i}{\partial\nu}=\rho\frac{\partial u}{\partial\nu} & \text{ on }\partial D\\
            \displaystyle{\lim_{r\to\infty}}r^{\frac{1}{2}}(\frac{\partial u_e}{\partial r}-i\kappa u_e)=0 &\text{ for } r=|x|.
        \end{cases}

    where \rho\in\mathbb{C}\backslash 0, \(u=u_e+u^{inc}\) is the total field in \mathbb{R}^2\backslash\overline{D}.

    Parameter:
    ----------
    kappa_in,kappa_ex: complex
        interior and exterior wave number
    rho: complex, optional
        parameter in second transmission condition. Default: 2.
    N_ieq,inc_waves,R_inc,meas_dir,R_meas,domain: 
        see DiricheletOp

    References
    ----------
    see T. Hohage & C. Schormann. "A Newton-type method for a transmission
    problem in inverse scattering", Inverse Problems, 14 (1998), 1207-1227."""
    
    def __init__(self, kappa_in:complex, kappa_ex:complex, rho:complex=2., 
                 N_ieq:int =64, 
                 inc_waves:int | np.ndarray=4, 
                 R_inc:float = np.inf,
                 meas:int |np.ndarray=64, 
                 R_meas:float = np.inf,
                 domain:ParameterizedCurveSpc|None =None):

        self.kappa_ex,  self.N_ieq, self.N_inc, self.inc_pts, self.R_inc,\
            self.N_meas, self.meas_pts,self.R_meas,domain,codomain \
            = check_scattering_parameters(kappa_ex,N_ieq,inc_waves,R_inc,meas,R_meas,domain)            

        self.kappa_in = kappa_in         
        """Interior wave number."""
        self.rho = rho
        """Density ratio."""

        self.w_sl_ex = -1
        self.w_dl_ex = 1
        self.w_sl_in = rho 
        self.w_dl_in = -1
        """Weights of single and double layer potentials."""

        super().__init__(
            domain = domain,
            codomain = codomain,
            linear = False
        )


    def _eval(self, coeff, differentiate=False):
        self.curve = self.domain.coeff2curve(coeff,3)

        # assemble system of boundary integral operators
        Iop_data_ex = setup_iop_data(self.curve, self.kappa_ex)
        Iop_data_in = setup_iop_data(self.curve, self.kappa_in)

        Iop1 = self.w_dl_ex*op_K(self.curve, Iop_data_ex)+\
                        self.w_dl_in*op_K(self.curve, Iop_data_in)+(self.w_dl_ex-self.w_dl_in-4)*np.diag(self.curve.zpabs)  
        Iop2 = self.w_sl_ex*op_S(self.curve, Iop_data_ex)+self.w_sl_in*op_S(self.curve, Iop_data_in)
        Iop3 = self.w_dl_ex*op_T(self.curve, Iop_data_ex)+self.w_dl_in*op_T(self.curve, Iop_data_in)
        Iop4 = self.w_sl_ex*op_K(self.curve, Iop_data_ex).T+self.w_sl_in*op_K(self.curve, Iop_data_in).T+\
                        (self.w_sl_in-2*self.rho-self.w_sl_ex-2)*np.diag(self.curve.zpabs)
    
        Iop = np.block([[Iop1, Iop2], \
                        [Iop3, Iop4]])
        
        R1  = -self.w_dl_in*op_K(self.curve, Iop_data_in)+(self.w_dl_in+2)*np.diag(self.curve.zpabs) 
        R2  = -self.w_sl_in*op_S(self.curve, Iop_data_in) 
        R3  = -self.w_dl_in*op_T(self.curve, Iop_data_in) 
        R4  = -self.w_sl_in*op_K(self.curve, Iop_data_in).T+(2*self.rho-self.w_sl_in)*np.diag(self.curve.zpabs)
       
        R = np.block([[R1, R2],\
                      [R3, R4]])

        self.Iop = np.linalg.inv(Iop).dot(R)

        matrix_fct = farfield_matrix if self.R_meas == np.inf else nearfield_matrix
        field_mat_a = matrix_fct(self.curve, self.meas_pts, self.kappa_ex, 0., self.w_dl_ex)
        field_mat_b = matrix_fct(self.curve, self.meas_pts, self.kappa_ex, self.w_sl_ex, 0.)
        self.FF_combined = np.hstack((field_mat_a,field_mat_b))

        uinc     = np.zeros((2*self.N_ieq, self.N_inc), dtype=complex)
        duincdnu = np.zeros((2*self.N_ieq, self.N_inc), dtype=complex)

        for l, dir in enumerate(self.inc_pts):
            if self.R_inc== np.inf:
                uinc[:, l] = np.exp(1j*self.kappa_ex*dir.dot(self.curve.z))
                duincdnu[:, l] = np.exp(1*1j*self.kappa_ex*dir.dot(self.curve.z))*\
                     (1j*self.kappa_ex*dir.dot(self.curve.normal))/self.curve.zpabs
            else:
                kdiff = self.kappa_ex*(self.curve.z - dir[:,None])
                kdist = np.linalg.norm(kdiff,axis=0) 
                uinc[:, l] = 0.25j*self.curve.zpabs*hankel1(0,kdist)
                duincdnu[:, l] = -0.25j*self.kappa_ex*np.sum(kdiff*self.curve.normal,axis=0) \
                    * hankel1(1,kdist)/(kdist*self.curve.zpabs)       
        rhs = np.vstack((uinc, duincdnu))
            
        self.dudn = self.Iop @ rhs
        farfield = self.FF_combined @ self.dudn
      
        if differentiate:
            ue = self.dudn[0:2*self.N_ieq, :]
            duednu = self.dudn[2*self.N_ieq:4*self.N_ieq, :]
            self.ui = ue + uinc 
            self.duidnu= self.rho*(duednu + duincdnu)
            u = (1-self.rho)*self.ui
            self.duds  = self.curve.arc_length_der(u)
        else:
            del self.dudn

        return farfield

    def _derivative(self, h): 
        hn  = self.curve.der_normal(h)
        rhs_a = (1/self.rho-1.)*hn[:,None]*self.duidnu      
        rhs_b = self.curve.arc_length_der(hn[:,None]*self.duds)+self.kappa_in**2*hn[:,None]\
                        *self.ui-self.rho*self.kappa_ex**2*hn[:,None]*(self.ui)
        rhs_b /= self.rho
        rhs = np.vstack((rhs_a, rhs_b))
        phi = self.Iop @ rhs
        return self.FF_combined @ phi

    def _adjoint(self, g):
        res = 1j*np.zeros(2*self.N_ieq)
        phi = self.FF_combined.T.conj() @ g 
        rhs   = self.Iop.T.conj() @ phi

        rhs_a = rhs[0:2*self.N_ieq,:]
        rhs_b = rhs[2*self.N_ieq:4*self.N_ieq,:]
        
        res = np.real(np.conjugate((1/self.rho-1.)*self.duidnu)*rhs_a) 
        res -= np.real(np.conjugate(self.duds/self.rho)*\
                    self.curve.arc_length_der(rhs_b/self.curve.zpabs[:,None])*self.curve.zpabs[:,None]) 
        res += np.real(np.conjugate(self.kappa_in**2*self.ui/self.rho - self.kappa_ex**2*self.ui)*rhs_b)
            
        adj = self.curve.der_normal.adjoint(np.sum(res,axis=1))

        return adj
    

    def create_synthetic_data(self, 
                              true_curve:GenCurve | type = Peanut, 
                              N_ieq_synth:int|None=None
                              ):
        bd_ex,N_ieq_synth = check_create_synthetic_data_params(true_curve,N_ieq_synth,nderivs=3,op=self)

        Iop_data_ex = setup_iop_data(bd_ex, self.kappa_ex)
        Iop_data_in = setup_iop_data(bd_ex, self.kappa_in)
        
        Iop1 = self.w_dl_ex*op_K(bd_ex, Iop_data_ex)+self.w_dl_in*op_K(bd_ex, Iop_data_in)+(self.w_dl_ex-self.w_dl_in-4)*np.diag(bd_ex.zpabs)
        Iop2 = self.w_sl_ex*op_S(bd_ex, Iop_data_ex)+self.w_sl_in*op_S(bd_ex, Iop_data_in)
        Iop3 = self.w_dl_ex*op_T(bd_ex, Iop_data_ex)+self.w_dl_in*op_T(bd_ex, Iop_data_in)
        Iop4 = self.w_sl_ex*op_K(bd_ex, Iop_data_ex).T+self.w_sl_in*op_K(bd_ex, Iop_data_in).T+(self.w_sl_in-2*self.rho-self.w_sl_ex-2)*np.diag(bd_ex.zpabs)
        
        Iop = np.block([[Iop1, Iop2],\
                         [Iop3, Iop4]])
    
        R1 = -self.w_dl_in*op_K(bd_ex, Iop_data_in)+(self.w_dl_in+2)*np.diag(bd_ex.zpabs)
        R2 = -self.w_sl_in*op_S(bd_ex, Iop_data_in)
        R3 = -self.w_dl_in*op_T(bd_ex, Iop_data_in)
        R4 = -self.w_sl_in*op_K(bd_ex, Iop_data_in).T+(2*self.rho-self.w_sl_in)*np.diag(bd_ex.zpabs)

        R = np.block([[R1, R2], \
                      [R3, R4]])

        Iop = np.linalg.inv(Iop).dot(R)

        matrix_fct = farfield_matrix if self.R_meas == np.inf else nearfield_matrix
        field_mat_a = matrix_fct(bd_ex, self.meas_pts, self.kappa_ex, 0., self.w_dl_ex)
        field_mat_b = matrix_fct(bd_ex, self.meas_pts, self.kappa_ex, self.w_sl_ex, 0.)
        FF_combined = np.hstack((field_mat_a,field_mat_b))

        rhs_a  = np.zeros((2*N_ieq_synth, self.N_inc), dtype=complex)
        rhs_b = np.zeros((2*N_ieq_synth, self.N_inc), dtype=complex)

        for l, dir in enumerate(self.inc_pts):
            if self.R_inc==np.inf:
                rhs_a[:,l] = np.exp(1j*self.kappa_ex*dir.dot(bd_ex.z))
                rhs_b[:,l] = np.exp(1j*self.kappa_ex*dir.dot(bd_ex.z))\
                    *(1j*self.kappa_ex*(dir.dot(bd_ex.normal)))/bd_ex.zpabs
            else:
                kdiff = self.kappa_ex*(bd_ex.z - dir[:,None])
                kdist = np.linalg.norm(kdiff,axis=0) 
                rhs_a[:, l] = 0.25j*bd_ex.zpabs*hankel1(0,kdist)
                rhs_b[:, l] = -0.25j*self.kappa_ex*np.sum(kdiff*bd_ex.normal,axis=0) \
                    * hankel1(1,kdist)/(kdist*bd_ex.zpabs)                      
        rhs = np.vstack((rhs_a, rhs_b))

        return FF_combined @ (Iop @ rhs), bd_ex
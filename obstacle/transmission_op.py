import numpy as np
import scipy.linalg as scla

from functions.operator import op_S
from functions.operator import op_T
from functions.operator import op_K
from dirichlet_op import check_scattering_parameters

from functions.farfield_matrix import farfield_matrix_trans
from functions.setup_iop_data import setup_iop_data
from regpy.operators import Operator

from regpy.vecsps import GridFcts
from regpy.vecsps.curve import GenTrigSpc


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

    References
    ----------
    see T. Hohage & C. Schormann. "A Newton-type method for a transmission
    problem in inverse scattering", Inverse Problems, 14 (1998), 1207-1227."""
    
    def __init__(self, kappa_in, kappa_ex, rho=4.3-6*complex(0,1), N_ieq=128, N_inc=4, N_meas=64, N_FK=64):

        self.kappa_ex,  self.N_ieq, self.N_inc, self.inc_directions, self.N_meas, self.meas_directions,codomain \
            = check_scattering_parameters(kappa_ex,N_ieq,N_inc,N_meas)

        self.kappa_in = kappa_in         
        """Interior wave number."""
        self.rho = rho
        """Density ratio."""

        self.w_sl_ex = -1
        self.w_dl_ex = 1
        self.w_sl_in = rho 
        self.w_dl_in = -1
        """Weights of single and double layer potentials."""
        
        self.N_FK = N_FK
        """Number of Fourier coefficients."""

        super().__init__(
            domain = GenTrigSpc(2*self.N_FK),
            codomain = codomain,
            linear = False
        )


    def _eval(self, coeff, differentiate=False):
        self.curve = self.domain.bd_eval(coeff, 2*self.N_ieq, 3)

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

        self.FF_combined = farfield_matrix_trans(self.curve, self.meas_directions,\
                                    self.kappa_ex, self.w_sl_ex, self.w_dl_ex)

        uinc     = np.zeros((2*self.N_ieq, self.N_inc), dtype=complex)
        duincdnu = np.zeros((2*self.N_ieq, self.N_inc), dtype=complex)

        for l, dir in enumerate(self.inc_directions):
            uinc[:, l] = (np.exp(1*complex(0,1)*self.kappa_ex*dir.dot(self.curve.z))).T

            duincdnu[:, l] = np.exp(1*complex(0,1)*self.kappa_ex*dir.dot(self.curve.z))*\
                     (complex(0,1)*self.kappa_ex*dir.dot(self.curve.normal))/self.curve.zpabs
            
        rhs = np.vstack((uinc, duincdnu))
            
        self.dudn = self.Iop @ rhs
        farfield = self.FF_combined @ self.dudn
      
        if differentiate:
            ue = self.dudn[0:2*self.N_ieq, :]
            duednu = self.dudn[2*self.N_ieq:4*self.N_ieq, :]
            self.ui = ue + uinc 
            self.duidnu= self.rho*(duednu + duincdnu)
            u = (1-self.rho)*self.ui
            self.duds = np.zeros_like(u)
            for l in range(0, self.N_inc):
                self.duds[:,l]  = self.curve.arc_length_der(u[:,l])

        return farfield

    def _derivative(self, h): 
        hn  = self.curve.der_normal(h)
        rhs_a = (1/self.rho-1.)*hn[:,None]*self.duidnu
        rhs_b = np.zeros_like(self.duds)    
        for l in range(0, self.N_inc):            
            rhs_b[:,l] = self.curve.arc_length_der(hn*self.duds[:,l])+self.kappa_in**2*hn\
                        *self.ui[:,l]-self.rho*self.kappa_ex**2*hn*(self.ui[:,l])
        rhs_b /= self.rho
        rhs = np.vstack((rhs_a, rhs_b))
        phi = self.Iop.dot(rhs)
        return self.FF_combined @ phi

    def _adjoint(self, g):
        res = complex(0,1)*np.zeros(2*self.N_ieq)
        phi = self.FF_combined.T.conj() @ g 
        rhs   = self.Iop.T.conj() @ phi

        rhs_a = rhs[0:2*self.N_ieq,:]
        rhs_b = rhs[2*self.N_ieq:4*self.N_ieq,:]
        
        res = np.real(np.conjugate((1/self.rho-1.)*self.duidnu)*rhs_a) 
        sres = np.sum(res,axis=1)
        for l in range(self.N_inc):
            sres -= np.real(np.conjugate(self.duds[:,l]/self.rho)*\
                    self.curve.arc_length_der(rhs_b[:,l]/self.curve.zpabs.T)*self.curve.zpabs.T) 
            sres += np.real(np.conjugate(self.kappa_in**2*self.ui[:,l]/self.rho-\
                                  self.kappa_ex**2*(self.ui[:,l]))*rhs_b[:,l])
            
        adj = self.curve.adjoint_der_normal(sres)

        return adj
    

    def create_synthetic_data(self, true_curve, N_ieq_synth=64):
        bd_ex = true_curve(2*N_ieq_synth,3)

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

        FF_combined = farfield_matrix_trans(bd_ex, self.meas_directions, self.kappa_ex, self.w_sl_ex, self.w_dl_ex)

        rhs_a  = np.zeros((2*N_ieq_synth, self.N_inc), dtype=complex)
        rhs_b = np.zeros((2*N_ieq_synth, self.N_inc), dtype=complex)

        for l, dir in enumerate(self.inc_directions):
            rhs_a[:,l] = np.exp(complex(0,1)*self.kappa_ex*dir.dot(bd_ex.z))
            rhs_b[:,l] = np.exp(complex(0,1)*self.kappa_ex*dir.dot(bd_ex.z))\
                *(complex(0,1)*self.kappa_ex*(dir.dot(bd_ex.normal)))/bd_ex.zpabs
        rhs = np.vstack((rhs_a, rhs_b))

        return FF_combined @ (Iop @ rhs), bd_ex
from regpy.operators import Operator
from regpy.vecsps.ngsolve import NgsVectorSpace,NgsBaseVector
from regpy.vecsps import DirectSum
import ngsolve as ngs
from ngsolve.meshes import Make1DMesh

class FokkerPlanckOp(Operator):
    def __init__(self, sigma, delta_t, domain, codomain):
        super().__init__(domain=domain, codomain=codomain)
        self.sigma = sigma
        self.delta_t = delta_t

        self.gf_w=ngs.GridFunction(self.domain[0].fes)
        self.gf_drift=ngs.GridFunction(self.domain[1].fes)
        self.gfu_eval=ngs.GridFunction(self.codomain.fes)
        self.gfu_deriv=ngs.GridFunction(self.codomain.fes)
        self.gfu_help_adj=ngs.GridFunction(self.codomain.fes)


    @staticmethod
    def create_shared(sigma=0.5, delta_t=0.01, p_fem=1, p_legendre=2, n_nodes=51, a=-1, b=1):
        mesh=Make1DMesh(n_nodes,mapping=lambda x:a+(b-a)*x)
        fes_H1=ngs.H1(mesh, order=p_fem, dirichlet=".*")
        fes_L2=ngs.L2(mesh,order=p_legendre)
        vspace_H1=NgsVectorSpace(fes_H1)
        vspace_L2=NgsVectorSpace(fes_L2)
        domain=DirectSum(vspace_H1,vspace_L2)
        codomain=vspace_H1
        return {
            "sigma" : sigma,
            "delta_t" : delta_t,
            "domain": domain,
            "codomain": codomain
        }
    

    def _eval(self, x, differentiate = False):
        """
        Evaluate one time step in the Fokker-Planck operator on the input x.
        """
        w, drift = self.domain.split(x)
        self.gf_w.vec.data=w.vec
        self.gf_drift.vec.data=drift.vec
        system_bf=ngs.BilinearForm(self.codomain.fes)
        rhs=ngs.LinearForm(self.codomain.fes)
        u,v=self.codomain.fes.TnT()
        system_bf+=u*v*ngs.dx+self.delta_t*(self.sigma**2*ngs.grad(u)*ngs.grad(v)*ngs.dx-self.gf_drift*u*ngs.grad(v)*ngs.dx)
        rhs+=self.gf_w*v*ngs.dx
        system_bf.Assemble()
        rhs.Assemble()
        system_bf_inv=system_bf.mat.Inverse(self.codomain.fes.FreeDofs())
        self.gfu_eval.vec.data = system_bf_inv * rhs.vec
        w_next=NgsBaseVector(self.gfu_eval.vec,make_copy=True)
        self.system_bf_inv=system_bf_inv
        if(differentiate==True):
            self.system_bf_inv=system_bf_inv
            self.system_bf_adj=system_bf.mat.CreateTranspose()
            self.gfu_eval=self.gfu_eval#TODO check if this can be deleted or if copy is needed
        return w_next


    def _derivative(self, h):
        h_w, h_drift = self.domain.split(h)
        self.gf_w.vec.data=h_w.vec
        self.gf_drift.vec.data=h_drift.vec
        rhs=ngs.LinearForm(self.codomain.fes)
        v=self.codomain.fes.TestFunction() 
        rhs+=self.gf_w*v*ngs.dx+self.delta_t*self.gf_drift*self.gfu_eval*ngs.grad(v)*ngs.dx
        rhs.Assemble()
        self.gfu_deriv.vec.data = self.system_bf_inv * rhs.vec
        return NgsBaseVector(self.gfu_deriv.vec,make_copy=True)
    

    def _adjoint(self, y):
        self.gfu_help_adj.vec.data=self.system_bf_adj.Inverse(self.codomain.fes.FreeDofs())*y.vec
        lf_w=ngs.LinearForm(self.domain[0].fes)
        v0=self.domain[0].fes.TestFunction()
        lf_w+=self.gfu_help_adj*v0*ngs.dx
        lf_w.Assemble()
        lf_drift=ngs.LinearForm(self.domain[1].fes)
        v1=self.domain[1].fes.TestFunction()
        lf_drift+=self.delta_t*self.gfu_eval*ngs.grad(self.gfu_help_adj)*v1*ngs.dx(bonus_intorder=20) #TODO adapt quadrature order
        lf_drift.Assemble()
        return self.domain.join(NgsBaseVector(lf_w.vec,make_copy=True),NgsBaseVector(lf_drift.vec,make_copy=True))


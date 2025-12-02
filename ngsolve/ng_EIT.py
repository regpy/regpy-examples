import os
import sys

# If your project layout requires adding the package root to sys.path, uncomment and adjust the line below
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import logging

import ngsolve as ngs
from netgen.occ import *
from ngsolve.webgui import Draw 
from netgen.geom2d import SplineGeometry
import netgen.gui

import numpy as np

import regpy.stoprules as rules
from regpy.solvers import RegularizationSetting
from regpy.solvers.nonlinear.landweber import Landweber
from regpy.hilbert import Sobolev, L2Boundary
from regpy.vecsps.ngsolve import NgsVectorSpace
from regpy.operators.ngsolve import SecondOrderEllipticCoefficientPDE, ProjectToBoundary
from regpy.vecsps import TupleVector
from regpy.operators import VectorOfOperators



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

#Defining the Second order elliptic PDE for the EIT-problem
class EIT_inner(SecondOrderEllipticCoefficientPDE):

    def __init__(self, domain, sol_domain, g, alpha=0.01, im_type=0):
        self.g=g
        self.im_type=im_type
        if im_type ==0:
            self.alpha=alpha
        elif im_type==1:
            self.Number = ngs.NumberSpace(sol_domain.fes.mesh)
            self.r, self.s = self.Number.TnT()
        else: 
            raise SystemExit('Implementation type is not defined')
        super().__init__(domain, sol_domain)

    def _bf(self,a,u,v):
        return a*ngs.grad(u)*ngs.grad(v) * ngs.dx

    def _bf_0(self): 
        if self.im_type==0:
            return self.alpha*self.u*self.v*ngs.dx
        else:
            #return ngs.SymbolicBFI(self.u * self.s + self.v * self.r, definedon=self.codomain.fes.mesh.Boundaries("cyc"))
            return (self.u * self.s + self.v * self.r)*ngs.ds(self.codomain.bdr)

    def _lf(self):
        lf = ngs.LinearForm(self.codomain.fes)
        lf += self.g*self.v*ngs.ds(self.codomain.bdr)
        return lf.Assemble() 
    
    
    
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

geo = SplineGeometry()
bc = "cyc"
geo.AddCircle((0, 0), r=1, bc=bc, maxh=0.05)
mesh = ngs.Mesh(geo.GenerateMesh())

#a has to be defined on this mesh
fes_domain = ngs.H1(mesh, order=3)
domain = NgsVectorSpace(fes_domain, bdr=bc)

fes_codomain = ngs.H1(mesh, order=3, dirichlet=bc)
codomain = NgsVectorSpace(fes_codomain, bdr=bc)

#Define different current distirbutions g. The current distributions should satsify \int_{\partial \Omega} g=0.
const=0.01
x_val=np.pi*ngs.x
y_val=np.pi*ngs.y
summ=np.pi*(ngs.x+ngs.y)/ngs.sqrt(2)
diff=np.pi*(ngs.x-ngs.y)/ngs.sqrt(2)
g_list=[]
for i in range(1, 7):
    g_list.extend([const*ngs.sin(i*x_val), const*ngs.sin(i*y_val), const*ngs.cos(i*x_val), const*ngs.cos(i*y_val)])
    g_list.extend([const*ngs.sin(i*summ), const*ngs.sin(i*diff), const*ngs.cos(i*summ), const*ngs.cos(i*diff)])

alpha=0.01

N_g=len(g_list)

op_list=[]
for i in range(0, N_g):
    op_inner=EIT_inner(domain=domain, sol_domain=domain, g=g_list[i], im_type=1)
    proj = ProjectToBoundary(domain, codomain)
    op_list.append(proj * op_inner)

op=VectorOfOperators(op_list)

exact_solution_coeff = 1+0.5*ngs.exp(-2*(ngs.x)**2-2*(ngs.y-0.9)**2)+0.5*ngs.exp(-2*(ngs.x-0.9)**2-2*(ngs.y)**2)
exact_solution = domain.from_ngs(exact_solution_coeff)
exact_data=op(exact_solution)

#Create the noise
noise_list=[]
for i in range(0, N_g):
    noise_domain=1e-5*domain.rand()
    noise_list.append(proj(noise_domain))
    
noise=TupleVector(noise_list)
data=exact_data+noise

setting = RegularizationSetting(op=op, penalty=Sobolev, data_fid=L2Boundary)

init = domain.from_ngs(1)
init_data = op(init)

#Discrepancy Principle usually stops very early
landweber = Landweber(setting, data, init, stepsize=100)
stoprule = (
        rules.CountIterations(500) +
        rules.Discrepancy(setting.h_codomain.norm, data, noiselevel=setting.h_codomain.norm(noise), tau=1.1)
)

reco, reco_data = landweber.run(stoprule)

# Draw reconstructed solution
print("reconstructed solution")
Draw(domain.to_gf(reco), mesh,"reco")


#Check the error in the reconstruction error
error=setting.h_domain.norm(reco-exact_solution)
error_init=setting.h_domain.norm(init-exact_solution)
error_changed=setting.h_domain.norm(init-reco)

print('error', error)
print('error_init', error_init)
print('error_changed', error_changed)


# Draw data space
print("noisy data")
Draw(op_inner.codomain.to_gf(data[5]),mesh, "data")
print("reconstructed data")
Draw(op_inner.codomain.to_gf(reco_data[5]),mesh, "reco_data")



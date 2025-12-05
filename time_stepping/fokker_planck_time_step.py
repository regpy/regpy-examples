from basic_operator_chain import time_dependent_operator_chain
from fokker_planck_op import FokkerPlanckOp
import ngsolve as ngs
import numpy as np
from matplotlib import pyplot as plt
from regpy.vecsps.ngsolve import NgsBaseVector
from regpy.hilbert import L2
from regpy.solvers.nonlinear import IrgnmCG
from regpy.solvers import Setting
from regpy.stoprules import CountIterations

#Currently not fully functional as inversion seems to be non-unique

a=10
b=-10
xs=np.linspace(a,b,100)
delta_t=0.1
step_N=50
alpha=0.1


params=FokkerPlanckOp.create_shared(delta_t=delta_t,a=-10,b=10,n_nodes=101)
operators=[FokkerPlanckOp(**params) for _ in range(step_N)]
op=time_dependent_operator_chain(operators,separate_parameter_inputs=False,output_intermediate_solutions=False)


sol_exact=ngs.GridFunction(op.domain[0].fes)
drift=ngs.GridFunction(op.domain[1].fes)
sol=ngs.GridFunction(op.domain[0].fes)
sol_exact.Set(ngs.exp(-0.05*(ngs.x-1)*(ngs.x-1)))
drift.Set(ngs.CoefficientFunction(0.1))

x=op.domain.join(NgsBaseVector(sol_exact.vec),NgsBaseVector(drift.vec))
exact_data=op(x)
setting=Setting(op,L2,L2,data=exact_data,regpar=alpha)
solver=IrgnmCG(setting,init=0.5*op.domain.ones())
stoprule=CountIterations(20)

data_fits=[]
for reco, reco_data in solver.until(stoprule):
    data_fits.append(setting.h_codomain.norm(reco_data-exact_data))

plt.plot(data_fits)
plt.show()
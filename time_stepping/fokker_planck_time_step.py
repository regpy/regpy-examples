from basic_operator_chain import time_dependent_operator_chain
from fokker_planck_op import FokkerPlanckOp
import ngsolve as ngs
import numpy as np
from matplotlib import pyplot as plt
from regpy.vecsps.ngsolve import NgsBaseVector
from regpy.hilbert import L2
from regpy.solvers.nonlinear import IrgnmCG
from regpy.solvers import RegularizationSetting
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
sol=ngs.GridFunction(op.domain[1].fes)
# sol_exact.Set(ngs.exp(-0.05*(ngs.x-1)*(ngs.x-1)))
sol_exact.Set(-0.01*(ngs.x-a)*(ngs.x-b))
# drift.Set(-0.001*ngs.x*ngs.x*ngs.x)
drift.Set(ngs.CoefficientFunction(0.1))

xsmesh=op.codomain.fes.mesh(xs)

x=op.domain.join(NgsBaseVector(sol_exact.vec),NgsBaseVector(drift.vec))
exact_data=op(x)
setting=RegularizationSetting(op,L2,L2)
solver=IrgnmCG(setting,exact_data,alpha,init=0.1*op.domain.ones())
stoprule=CountIterations(200)
plt.ion()
# sol.vec.data=sol_exact.vec
# plt.plot(xs,sol(xsmesh))
for reco, reco_data in solver.until(stoprule):
    sol.vec.data=reco[1].vec
    plt.plot(xs,sol(xsmesh))
    plt.pause(0.01)
plt.show()
plt.ioff()

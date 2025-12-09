from regpy.solvers.nonlinear import IterativelyRegularizedNewton
from regpy.solvers.linear import FISTA,TikhonovCG

from xray_phase_contrast_operator import get_xray_phase_contrast
from regpy.hilbert import L2
from regpy.functionals import QuadNonneg
from regpy.vecsps import UniformGridFcts
from regpy.solvers import Setting
import regpy.stoprules as rules

import numpy as np
from scipy.datasets import ascent
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-20s :: %(message)s'
)


# Example parameters
fresnel_number = 5e-4    # Fresnel-number of the simulated imaging system, associated with the unit-lengthscale
                        # in grid (i.e. with the size of one pixel for the above choice of grid)
noise_level = 0.01      # Noise level in the simulated data


# Uniform grid of unit-spacing
grid = UniformGridFcts(np.arange(1024), np.arange(1024))

# Forward operator
op = get_xray_phase_contrast(grid, fresnel_number)

# Create phantom phase-image (= padded example-image)
exact_solution = ascent().astype(np.float64)
exact_solution /= exact_solution.max()
pad_amount = tuple([(grid.shape[0] - exact_solution.shape[0])//2,
                    (grid.shape[1] - exact_solution.shape[1])//2])
exact_solution = np.pad(exact_solution, pad_amount, 'constant', constant_values=0)

# Create exact and noisy data
exact_data = op(exact_solution)
noise = noise_level * op.codomain.randn()
data = exact_data + noise

# Image-reconstruction using the combined linearization method
setting = Setting(op=op, penalty=L2, data_fid=L2,data=data)
inner_stoprule=lambda setting:rules.DualityGapStopping(tol=1e-3)
solver = IterativelyRegularizedNewton(setting,inner_solver=FISTA,inner_solver_stoprule=inner_stoprule,regpar=10)
# inner_stoprule=rules.CountIterations(100)
# solver = IterativelyRegularizedNewton(setting,inner_solver=TikhonovCG,inner_solver_stoprule=inner_stoprule,regpar=10)
stoprule = (
    rules.CountIterations(max_iterations=10) +
    rules.Discrepancy(
        setting.h_codomain.norm,
        data,
        noiselevel=setting.h_codomain.norm(noise),
        tau=1.1
    )
)

reco, reco_data = solver.run(stoprule)

# Plot results
plt.figure()
plt.title('Exact solution (phase-image)')
plt.imshow(exact_solution)
plt.colorbar()

plt.figure()
plt.title('Simulated data (hologram)')
plt.imshow(data)
plt.colorbar()

plt.figure()
plt.title('Reconstruction (phase-image)')
plt.imshow(reco)
plt.colorbar()

plt.show()

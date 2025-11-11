import enum
import numpy as np
import scipy.sparse.linalg as spla
from scipy.special import hankel1, jv as besselj

from regpy.operators import Operator
from regpy.operators.convolution import PeriodizedHelmholtzVolumePotential
from regpy.operators import PtwMultiplication, SciPyLinearOperator, Identity
from regpy import util, vecsps




class MediumScatteringBase(Operator):
    """Acoustic scattering problem for inhomogeneous medium.

    The forward problem is solved Vainikko's fast solver of the Lippmann
    Schwinger equation.

    This is an abstract base class that computes the total field, but delegates
    farfield computation to subclasses, to allow implementing different
    measurement geometries. Child classes need to establish the `codomain` attribute
    in the initialization to the appropriate farfield space and pass it to the 
    super class, and overwrite the `_compute_farfield` and `_compute_farfield_adoint` 
    methods.

    Parameters
    ----------
    codomain : vecsps.VectorSpace
        The codomain discribing the Farfield values. 
    gridshape : tuple
        Tuple determining the size of the grid on which the total field is
        computed. Should have 2 or 3 elements depending on the dimension of the
        problem. The domain always is taken to range from `-2*radius` to
        `2*radius` along each axis.
    radius : float
        An a-priori estimate for the radius of a circle or sphere covering the
        entire unknown object.
    wave_number : float
        The wave number of the incident waves.
    inc_directions : array-like
        Directions of the incident waves. Should be of shape `(n, 2)` or
        `(n, 3)`, depending on the dimension. Each of the `n` directions needs
        to be normalized.
    support : array-like, callable or None
        Mask determining the subset of the grid on which the object is
        supported. Will be converted to a boolean array. If `None`, a circle of
        `radius` given by the radius argument will be assumed. A callable will
        be called with arguments `grid` and `radius` and should return a
        boolean array.
    gmres_args : dict
        Arguments passed to [`scipy.sparse.linalg.gmres`][1] for solving the
        Lippmann Schwinger equation. Default values are `restart=10`,
        `rtol=1e-14`, `maxiter=100` and `atol=0.0`.
    normalization : 'helmholtz' or 'schroedinger'
        How to normalize the kernel and farfield matrix.

    [1]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html

    References
    ----------
    T. Hohage: On the numerical solution of a 3D inverse medium scattering
    problem. Inverse Problems, 17:1743-1763, 2001.

    G. Vainikko: Fast solvers of the Lippmann-Schwinger equation in: Direct and
    inverse problems of mathematical physics edited by R.P.Gilbert, J.Kajiwara,
    and S.Xu, Kluwer, 2000.
    """

    def __init__(self, 
                 codomain,
                 gridshape, 
                 radius, 
                 wave_number, 
                 inc_directions,
                 support=None,
                 gmres_args=None,
                 normalization='helmholtz'):

        if len(gridshape) not in (2, 3):
            raise ValueError(f"The grid shape has to be either 2 or 3, was given {gridshape}")
        if any(not isinstance(s, int) for s in gridshape):
            raise ValueError(f"Each dimensional shape has to be an integer, was given {(type(s) for s in gridshape)}")
        grid = vecsps.UniformGridFcts(
            *(np.linspace(-2*radius, 2*radius, s, endpoint=False)
              for s in gridshape),
            dtype=complex
        )
        super().__init__(domain = grid, codomain=codomain)

        if support is None:
            support = (grid.coord_distances() <= radius)
        elif callable(support):
            support = np.asarray(support(grid, radius), dtype=bool)
        else:
            support = np.asarray(support, dtype=bool)
        if support.shape != self.domain.shape:
            raise RuntimeError(f"The constructed `support` has shape {support.shape} not matching the shape of the domain {self.domain.shape}")
        if (support > (self.domain.coord_distances() <= radius)).all():
            raise RuntimeError(f"The constructed `support` lies outside the ball of `radius`.")

        self.support = support
        """Boolean array for the support constraint"""

        self.wave_number = wave_number
        """The wave number of the incident waves"""

        inc_directions = np.asarray(inc_directions)
        if inc_directions.ndim != 2:
            raise ValueError(f"The incident directions have to be a two-dimensional array.")
        if inc_directions.shape[1] != self.domain.ndim:
            raise ValueError(f"The incident directions vectors are of dimension {inc_directions.shape[1]} mismatching the domain dimension of {self.domain.ndim}")
        if not np.allclose(np.linalg.norm(inc_directions, axis=1), 1):
            raise ValueError("The incident directions have to be normed vectors.")

        self.inc_directions = inc_directions
        """Array of incident directions"""

        self.inc_matrix = np.exp(1j * wave_number * (inc_directions @ np.asarray(grid.coords)[:, support]))

        if normalization not in {'helmholtz', 'schroedinger'}:
            raise ValueError(f"`normalization` has be either `helmholtz` or `schroedinger` was given {normalization}.")
        self.normalization = normalization
        """The normalization"""

        if grid.ndim == 2:
            if self.normalization == 'helmholtz':
                ls_fac = 1.
                normalization_factor = grid.volume_elem * self.wave_number**2*np.exp(1j*np.pi/4)/np.sqrt(8*np.pi*self.wave_number)

            elif self.normalization == 'schroedinger':
                ls_fac = 1/wave_number**2
                normalization_factor = grid.volume_elem / (2*np.pi)**2

        elif grid.ndim == 3:
            if self.normalization == 'helmholtz':
                normalization_factor = -grid.volume_elem * self.wave_number**2 / (4*np.pi)

            elif self.normalization == 'schroedinger':
                raise NotImplementedError('Schrödinger-Equation not implemented in 3d')

        self.normalization_factor = normalization_factor
        """The normalization factor of the farfield matrix, to be used by subclasses."""

        self.gmres_args = util.set_defaults(
            gmres_args, restart=10, rtol=1e-14, maxiter=100, atol=0.0
        )

        # all attributes defined above are constants
        self._consts.update(self.attrs)

        # pre-allocate to save time in _eval
        self._totalfield = np.empty((np.sum(self.support), self.inc_matrix.shape[0]),
                                    dtype=complex)
        # noinspection PyArgumentList
        """self._lippmann_schwinger = spla.LinearOperator(
            (np.prod(self.domain.shape),) * 2,
            matvec=self._lippmann_schwinger_op,
            rmatvec=self._lippmann_schwinger_adjoint,
            dtype=complex
        )"""

        self._volume_pot = ls_fac*PeriodizedHelmholtzVolumePotential(grid,wave_number)

    def _compute_farfield(self, farfield, inc_idx, v):
        """Abstract method, needs to be implemented by child classes.

        Compute the farfield for incident wave `inc_idx` (an index into
        `regpy.operators.mediumscattering.MediumScatteringBase.inc_directions`),
        where `v` is the contrast multiplied by the computed total field,
        supported on
        `regpy.operators.mediumscattering.MediumScatteringBase.support`. The
        result should be stored into `farfield` in-place. The return value is
        ignored. `farfield` will be initialized to zero before computing the
        first incident wave. The final `farfield` is the return value of the
        operator evaluation.
        """
        raise NotImplementedError

    def _compute_farfield_adjoint(self, farfield, inc_idx, v):
        """Abstract method, needs to be implemented by child classes.

        Compute the adjoint of the above method for a given `farfield`, storing
        the result into `v`, which should only be modified on `self.support`.
        """
        raise NotImplementedError

    def _eval(self, contrast, differentiate=False, adjoint_derivative=False):
        contrast = contrast.copy()
        contrast[~self.support] = 0
        farfield = self.codomain.empty()
        rhs = self.domain.zeros()
        self._lippmann_schwinger =  SciPyLinearOperator(Identity(self.domain)
                                               + PtwMultiplication(self.domain,contrast) 
                                                  * self._volume_pot)
        for j in range(self.inc_matrix.shape[0]):
            # Solve Lippmann-Schwinger equation v + a*conv(k, v) = a*u_inc for
            # the unknown v = a u_total. The Fourier coefficients of the
            # periodic convolution kernel k are precomputed.
            rhs[self.support] = self.inc_matrix[j, :] * contrast[self.support]
            v = self._gmres(self._lippmann_schwinger, rhs)
            self._compute_farfield(farfield, j, v)
            # The total field can be recovered from v in a stable manner by the formula
            # u_total = u_inc - conv(k, v)
            if differentiate or adjoint_derivative:
                self._totalfield[:, j] = (
                    self.inc_matrix[j, :] - self._volume_pot(v)[self.support]
                )
        return farfield

    def _derivative(self, contrast):
        contrast = contrast.copy()
        contrast = contrast[self.support]
        farfield = self.codomain.empty()
        rhs = self.domain.zeros()
        for j in range(self.inc_matrix.shape[0]):
            rhs[self.support] = self._totalfield[:, j] * contrast
            v = self._gmres(self._lippmann_schwinger, rhs)
            self._compute_farfield(farfield, j, v)
        return farfield

    def _adjoint(self, farfield):
        v = self.domain.zeros()
        contrast = self.domain.zeros()
        for j in range(self.inc_matrix.shape[0]):
            self._compute_farfield_adjoint(farfield, j, v)
            rhs = self._gmres(self._lippmann_schwinger.adjoint(), v)
            aux = self._totalfield[:, j].conj() * rhs[self.support]
            contrast[self.support] += aux
        return contrast

    def _gmres(self, op, rhs):
        result, info = spla.gmres(op, self.domain.flatten(rhs), **self.gmres_args)
        if info > 0:
            self.log.warn('Gmres failed to converge')
        elif info < 0:
            self.log.warn('Illegal Gmres input or breakdown')
        return self.domain.fromflat(result)


class MediumScatteringFixed(MediumScatteringBase):
    """Acoustic medium scattering with fixed measurement directions.

    Parameters
    ----------
    farfield_directions : array-like
        Array of measurement directions of the farfield, shape `(n, 2)` or `(n, 3)` depending on
        the problem dimension. All directions must be normalized.
    gridshape : tuple, optional
        Tuple determining the size of the grid on which the total field is
        computed. Should have 2 or 3 elements depending on the dimension of the
        problem. The domain always is taken to range from `-2*radius` to
        `2*radius` along each axis. Default values is (64,64).
    radius : float, optional
        An a-priori estimate for the radius of a circle or sphere covering the
        entire unknown object. Default value is 1.
    wave_number : float, optional
        The wave number of the incident waves. Default value is 1.
    inc_directions : array-like, optional
        Directions of the incident waves. Should be of shape `(n, 2)` or
        `(n, 3)`, depending on the dimension. Each of the `n` directions needs
        to be normalized. Default value is given by util.linspace_circle(16).
    **kwargs
        All other (keyword-only) arguments are passed to the base class, which
        see.
    """

    def __init__(self, *, 
            farfield_directions,
            gridshape=(64, 64), 
            radius=1,
            wave_number=1, 
            inc_directions = util.linspace_circle(16), 
            **kwargs):
        farfield_directions = np.asarray(farfield_directions)
        if farfield_directions.ndim != 2:
            raise ValueError(f"Farfield has to be 2 dimensional array.")
        if farfield_directions.shape[1] != len(gridshape):
            raise ValueError(f"The dimension of each farfield direction has to match the gridshap dimension.")
        if not np.allclose(np.linalg.norm(farfield_directions, axis=-1), 1):
            raise ValueError(f"The farfield directions have to be normed vectors.")
        self.farfield_directions = farfield_directions
        """The farfield directions."""

        codomain = vecsps.UniformGridFcts(
            axisdata=(farfield_directions, inc_directions),
            dtype=complex
        )
        super().__init__(
            codomain,
            gridshape, 
            radius, 
            wave_number, 
            inc_directions, 
            **kwargs
        )

        self.farfield_matrix = self.normalization_factor * np.exp(
            -1j * self.wave_number * (farfield_directions @ np.asarray(self.domain.coords)[:, self.support])
        )
        """The farfield matrix."""

    def _compute_farfield(self, farfield, inc_idx, v):
        farfield[:, inc_idx] = self.farfield_matrix @ v[self.support]

    def _compute_farfield_adjoint(self, farfield, inc_idx, v):
        v[self.support] = farfield[:, inc_idx] @ self.farfield_matrix.conj()


class MediumScatteringOneToMany(MediumScatteringBase):
    """Acoustic medium scattering with measurement directions depending on incident direction.

    Parameters
    ----------
    farfield_directions : array-like
        Array of measurement directions of the farfield, shape `(n_inc, n, 2)`, where `n_inc` is
        the number of incident directions. All directions must be normalized.
    inc_directions : array-like,
        Directions of the incident waves. Should be of shape `(n, 2)` or
        `(n, 3)`, depending on the dimension. Each of the `n` directions needs
        to be normalized.
    gridshape : tuple, optional
        Tuple determining the size of the grid on which the total field is
        computed. Should have 2 or 3 elements depending on the dimension of the
        problem. The domain always is taken to range from `-2*radius` to
        `2*radius` along each axis. Default values is (64,64).
    radius : float, optional
        An a-priori estimate for the radius of a circle or sphere covering the
        entire unknown object. Default value is 1.
    wave_number : float, optional
        The wave number of the incident waves. Default value is 1.
    **kwargs
        All other (keyword-only) arguments are passed to the base class, which
        see.
    """

    def __init__(self, *, 
            farfield_directions, 
            inc_directions,
            gridshape=(64, 64), 
            radius=1,
            wave_number=1,  
            **kwargs):

        farfield_directions = np.asarray(farfield_directions)
        if farfield_directions.ndim != 3:
            raise ValueError(f"`farfield_directions` has to be 3 dimensional array.")
        if farfield_directions.shape[0] != inc_directions.shape[0]:
            raise ValueError(f"The first dimension of the `farfield_directions` has to match the number of incedent directions.")
        if farfield_directions.shape[2] != len(gridshape):
            raise ValueError(f"The dimension of each farfield direction has to match the gridshap dimension.")
        if not np.allclose(np.linalg.norm(farfield_directions, axis=-1), 1):
            raise ValueError(f"The farfield directions have to be normed vectors.")

        ninc, nfarfield = farfield_directions.shape[:2]
        codomain = vecsps.VectorSpace(
            shape=(nfarfield, ninc),
            dtype=complex
        )
        super().__init__(
            codomain,
            gridshape, 
            radius, 
            wave_number, 
            inc_directions, 
            **kwargs
        )

        self.farfield_directions = farfield_directions
        """The farfield directions."""        
        self.farfield_matrix = self.normalization_factor * np.exp(
            -1j * self.wave_number * (farfield_directions @ np.asarray(self.domain.coords)[:, self.support])
        )
        """The farfield matrix."""

        

    def _compute_farfield(self, farfield, inc_idx, v):
        farfield[:, inc_idx] = self.farfield_matrix[inc_idx] @ v[self.support]

    def _compute_farfield_adjoint(self, farfield, inc_idx, v):
        v[self.support] = farfield[:, inc_idx] @ self.farfield_matrix[inc_idx].conj()

    @staticmethod
    def generate_directions(ninc, nfarfield, angle=np.pi):
        """Computes the measuring directions for the experiment around the incident direction in 2d.

        Parameters
        ----------
        ninc : int
            Number of equispaced incident directions between 0 and `2pi`.
        nfarfield : int
            Number of measured farfield directions per incident directions.
        angle : float
            The maximum angle between incident and farfield direction. For each incident direction
            `phi`, `nfarfield` measurement directions between `phi - angle` and `phi + angle` will
            be generated.

        Returns
        -------
        tuple of arrays
            The array of incident directions (shape `(ninc, 2)`) and the array of farfield
            directions (shape `(ninc, nfarfield, 2)`).
        """

        phi = np.linspace(0, 2*np.pi, ninc, endpoint=False)
        dphi = np.linspace(-angle, angle, nfarfield)

        inc = util.complex2real(np.exp(1j * phi))
        farfield = util.complex2real(np.exp(1j * (phi[:, np.newaxis] + dphi)))

        return inc, farfield

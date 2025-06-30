import enum
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import scipy.sparse.linalg as spla
from scipy.special import hankel1, jv as besselj

from regpy.operators import Operator
from regpy import util, vecsps


class MediumScatteringBase(Operator):
    """Acoustic scattering problem for inhomogeneous medium.

    The forward problem is solved Vainikko's fast solver of the Lippmann
    Schwinger equation.

    This is an abstract base class that computes the total field, but delegates
    farfield computation to subclasses, to allow implementing different
    measurement geometries. Child classes need to set the `codomain` attribute
    to the appropriate farfield space, and overwrite the `_compute_farfield` and
    `_compute_farfield_adoint` methods.

    Parameters
    ----------
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

    def __init__(self, gridshape, radius, wave_number, inc_directions,
                 support=None,
                 gmres_args=None,
                 normalization='helmholtz'):
        super().__init__()
        assert len(gridshape) in (2, 3)
        assert all(isinstance(s, int) for s in gridshape)
        grid = vecsps.UniformGridFcts(
            *(np.linspace(-2*radius, 2*radius, s, endpoint=False)
              for s in gridshape),
            dtype=complex
        )
        self.domain = grid

        if support is None:
            support = (np.linalg.norm(grid.coords, axis=0) <= radius)
        elif callable(support):
            support = np.asarray(support(grid, radius), dtype=bool)
        else:
            support = np.asarray(support, dtype=bool)
        assert support.shape == grid.shape
        assert (support <= (np.linalg.norm(grid.coords, axis=0) <= radius)).all()
        """assert support is contained in radius"""

        self.support = support
        """Boolean array for the support constraint"""

        self.wave_number = wave_number
        """The wave number of the incident waves"""

        inc_directions = np.asarray(inc_directions)
        assert inc_directions.ndim == 2
        assert inc_directions.shape[1] == grid.ndim
        assert np.allclose(np.linalg.norm(inc_directions, axis=1), 1)

        self.inc_directions = inc_directions
        """Array of incident directions"""

        self.inc_matrix = np.exp(1j * wave_number * (inc_directions @ grid.coords[:, support]))

        assert normalization in {'helmholtz', 'schroedinger'}
        self.normalization = normalization
        """The normalization"""

        compute_kernel = None  # Silence linter
        if grid.ndim == 2:
            if self.normalization == 'helmholtz':
                compute_kernel = _compute_kernel_2d
                normalization_factor = grid.volume_elem * self.wave_number**2*-np.exp(1j*np.pi/4)/np.sqrt(8*np.pi*self.wave_number)

            elif self.normalization == 'schroedinger':
                def compute_kernel(*args):
                    return _compute_kernel_2d(*args) / wave_number**2
                normalization_factor = grid.volume_elem / (2*np.pi)**2

        elif grid.ndim == 3:
            if self.normalization == 'helmholtz':
                compute_kernel = _compute_kernel_3d
                normalization_factor = -grid.volume_elem * self.wave_number**2 / (4*np.pi)

            elif self.normalization == 'schroedinger':
                raise NotImplementedError('Schrödinger-Equation not implemented in 3d')

        self.normalization_factor = normalization_factor
        """The normalization factor of the farfield matrix, to be used by subclasses."""

        self.kernel = compute_kernel(2*wave_number*radius, grid.shape)
        """The Lippmann-Schwinger kernel in Fourier space."""

        self.gmres_args = util.set_defaults(
            gmres_args, restart=10, rtol=1e-14, maxiter=100, atol=0.0
        )

        # all attributes defined above are constants
        self._consts.update(self.attrs)

        # pre-allocate to save time in _eval
        self._totalfield = np.empty((np.sum(self.support), self.inc_matrix.shape[0]),
                                    dtype=complex)
        # noinspection PyArgumentList
        self._lippmann_schwinger = spla.LinearOperator(
            (np.prod(self.domain.shape),) * 2,
            matvec=self._lippmann_schwinger_op,
            rmatvec=self._lippmann_schwinger_adjoint,
            dtype=complex
        )

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
        self._contrast = contrast
        farfield = self.codomain.empty()
        rhs = self.domain.zeros()
        for j in range(self.inc_matrix.shape[0]):
            # Solve Lippmann-Schwinger equation v + a*conv(k, v) = a*u_inc for
            # the unknown v = a u_total. The Fourier coefficients of the
            # periodic convolution kernel k are precomputed.
            rhs[self.support] = self.inc_matrix[j, :] * contrast[self.support]
            v = self._gmres(self._lippmann_schwinger, rhs).reshape(self.domain.shape)
            self._compute_farfield(farfield, j, v)
            # The total field can be recovered from v in a stable manner by the formula
            # u_total = u_inc - conv(k, v)
            if differentiate or adjoint_derivative:
                self._totalfield[:, j] = (
                    self.inc_matrix[j, :] - ifftn(self.kernel * fftn(v))[self.support]
                )
        return farfield

    def _derivative(self, contrast):
        contrast = contrast.copy()
        contrast = contrast[self.support]
        farfield = self.codomain.empty()
        rhs = self.domain.zeros()
        for j in range(self.inc_matrix.shape[0]):
            rhs[self.support] = self._totalfield[:, j] * contrast
            v = self._gmres(self._lippmann_schwinger, rhs).reshape(self.domain.shape)
            self._compute_farfield(farfield, j, v)
        return farfield

    def _adjoint(self, farfield):
        v = self.domain.zeros()
        contrast = self.domain.zeros()
        for j in range(self.inc_matrix.shape[0]):
            self._compute_farfield_adjoint(farfield, j, v)
            rhs = self._gmres(self._lippmann_schwinger.adjoint(), v).reshape(self.domain.shape)
            aux = self._totalfield[:, j].conj() * rhs[self.support]
            contrast[self.support] += aux
        return contrast

    def _gmres(self, op, rhs):
        result, info = spla.gmres(op, rhs.ravel(), **self.gmres_args)
        if info > 0:
            self.log.warn('Gmres failed to converge')
        elif info < 0:
            self.log.warn('Illegal Gmres input or breakdown')
        return result

    def _lippmann_schwinger_op(self, v):
        """Lippmann-Schwinger operator in spatial domain on fine grid
        """
        v = v.reshape(self.domain.shape)
        v = v + self._contrast * ifftn(self.kernel * fftn(v))
        return v.ravel()

    def _lippmann_schwinger_adjoint(self, v):
        """Adjoint Lippmann-Schwinger operator in spatial domain on fine grid
        """
        v = v.reshape(self.domain.shape)
        v = v + ifftn(np.conj(self.kernel) * fftn(np.conj(self._contrast) * v))
        return v.ravel()


# noinspection PyPep8Naming
def _compute_kernel_2d(R, shape):
    J = np.mgrid[[slice(-(s//2), (s+1)//2) for s in shape]]
    piabsJ = np.pi * np.linalg.norm(J, axis=0)
    Jzero = tuple(s//2 for s in shape)

    K_hat =  R**2 / (piabsJ**2 - R**2) * (
        1 + 1j*np.pi/2 * (
            piabsJ * besselj(1, piabsJ) * hankel1(0, R) -
            R * besselj(0, piabsJ) * hankel1(1, R)
        )
    )
    K_hat[Jzero] = -1/(2*R) + 1j*np.pi/4 * hankel1(1, R)
    K_hat[piabsJ == R] = 1j*np.pi*R/8 * (
        besselj(0, R) * hankel1(0, R) + besselj(1, R) * hankel1(1, R)
    )
    return 2 * R * fftshift(K_hat)


# noinspection PyPep8Naming
def _compute_kernel_3d(R, shape):
    J = np.mgrid[[slice(-(s//2), (s+1)//2) for s in shape]]
    piabsJ = np.pi * np.linalg.norm(J, axis=0)
    Jzero = tuple(s//2 for s in shape)

    K_hat =  R**2 / (piabsJ**2 - R**2) * (
        1 - np.exp(1j*R) * (np.cos(piabsJ) - 1j*R * np.sin(piabsJ) / piabsJ)
    )
    K_hat[Jzero] = -(1 - np.exp(1j*R) * (1 - 1j*R))
    K_hat[piabsJ == R] = -1j/4 * (2*R)**(-1/2) * (1 - np.exp(1j*R) * np.sin(R) / R)
    return (2*R)**(3/2) * fftshift(K_hat)


class MediumScatteringFixed(MediumScatteringBase):
    """Acoustic medium scattering with fixed measurement directions.

    Parameters
    ----------
    farfield_directions : array-like
        Array of measurement directions of the farfield, shape `(n, 2)` or `(n, 3)` depending on
        the problem dimension. All directions must be normalized.
    **kwargs
        All other (keyword-only) arguments are passed to the base class, which
        see.
    """

    def __init__(self, *, farfield_directions, **kwargs):
        super().__init__(**kwargs)

        farfield_directions = np.asarray(farfield_directions)
        assert farfield_directions.ndim == 2
        assert farfield_directions.shape[1] == self.domain.ndim
        assert np.allclose(np.linalg.norm(farfield_directions, axis=-1), 1)
        self.farfield_directions = farfield_directions
        """The farfield directions."""
        self.farfield_matrix = self.normalization_factor * np.exp(
            -1j * self.wave_number * (farfield_directions @ self.domain.coords[:, self.support])
        )
        """The farfield matrix."""

        self.codomain = vecsps.UniformGridFcts(
            axisdata=(self.farfield_directions, self.inc_directions),
            dtype=complex
        )

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
    **kwargs
        All other (keyword-only) arguments are passed to the base class, which
        see.
    """

    def __init__(self, *, farfield_directions, **kwargs):
        super().__init__(**kwargs)
        assert self.domain.ndim == 2

        farfield_directions = np.asarray(farfield_directions)
        assert farfield_directions.ndim == 3
        assert farfield_directions.shape[0] == self.inc_directions.shape[0]
        assert farfield_directions.shape[2] == self.domain.ndim
        assert np.allclose(np.linalg.norm(farfield_directions, axis=-1), 1)
        self.farfield_directions = farfield_directions
        """The farfield directions."""
        self.farfield_matrix = self.normalization_factor * np.exp(
            -1j * self.wave_number * (farfield_directions @ self.domain.coords[:, self.support])
        )
        """The farfield matrix."""

        ninc, nfarfield = farfield_directions.shape[:2]
        self.codomain = vecsps.VectorSpace(
            shape=(nfarfield, ninc),
            dtype=complex
        )

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

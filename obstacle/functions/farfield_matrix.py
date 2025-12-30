import numpy as np
from scipy.special import hankel1

def farfield_matrix(bd, dire, kappa, weight_sl, weight_dl):
     """  returns matrix A such that 

        farfield = A @ phi

     Parameters
     ----------
     bd : Curve 
        bd.z        : (2, Nbd)
        bd.normal   : (2, Nbd)
        bd.zpabs    : (Nbd,)
     dire : ndarray, shape (2, N_eval)
        directions for which the far field is evaluated
     kappa : float
        Wave number

     weightSL, weightDL : complex
        Single- and double-layer weights

      Returns
    -------
     A : ndarray, shape (N_eval, Nbd), complex
        farfield matrix
     """     
     FFmat=np.zeros((len(dire), np.size(bd.z,1)), dtype=complex)
     for l, meas in enumerate(dire):
         FFmat[l,:] = np.pi/(np.size(bd.z,1)*np.sqrt(8*np.pi*kappa))*np.exp(-complex(0,1)*np.pi/4)\
         *(weight_dl*kappa*meas.dot(bd.normal)+complex(0,1)*weight_sl*bd.zpabs)*np.exp(-complex(0,1)*kappa*(meas.dot(bd.z)))

     return FFmat

def nearfield_matrix(bd, zz, kappa, weightSL, weightDL):
     """  returns matrix A such that 

          field = A @ phi

     Parameters
     ----------
     bd : Curve 
          bd.z        : (2, Nbd)
          bd.normal   : (2, Nbd)
          bd.zpabs    : (Nbd,)
     zz : ndarray, shape (N_eval,2)
          Evaluation points
     kappa : float
          Wave number

     weightSL, weightDL : complex
          Single- and double-layer weights

     Returns
     -------
     A : ndarray, shape (N_eval, Nbd), complex
          Linear operator matrix
     """

     z_bd = bd.z                      # (2, Nbd)
     normals = bd.normal              # (2, Nbd)
     zpabs = bd.zpabs                 # (Nbd,)

     Nbd = z_bd.shape[1]
     N_eval = zz.shape[1]

     # ------------------------------------------------------------
     # Pairwise distances |zz_j - z_m|
     # ------------------------------------------------------------
     # diff: (2, N_eval, Nbd)
     diff = zz[:, :,None] - z_bd[None,:, :]
     r = np.linalg.norm(diff, axis=1)             # (N_eval, Nbd)
     kdist = kappa * r

     # ------------------------------------------------------------
     # Single-layer kernel
     # ------------------------------------------------------------
     G = hankel1(0, kdist)                         # (N_eval, Nbd)

     cSL = 0.25j * 2 * np.pi / Nbd
     SL = cSL * G * zpabs[None, :]                 # broadcasting

     # ------------------------------------------------------------
     # Double-layer kernel
     # ------------------------------------------------------------
     # zz^T * normal
     zn = zz @ normals                           # (N_eval, Nbd)

     # sum(normal .* z, axis=0)
     nz = np.sum(normals * z_bd, axis=0)           # (Nbd,)

     geom = zn - nz[None, :]                       # (N_eval, Nbd)

     H1 = hankel1(1, kdist)
     DL_kernel = np.zeros_like(H1)
     mask = r > 0
     DL_kernel[mask] = H1[mask] / kdist[mask]

     cDL = 0.25j * (kappa ** 2) * 2 * np.pi / Nbd
     DL = cDL * geom * DL_kernel

     return weightSL * SL + weightDL * DL


def farfield_matrix_trans(bd, dire, kappa_ex, weight_sl_ex, weight_dl_ex):
    FFmat=np.zeros((len(dire), 2*np.size(bd.z,1)), dtype=complex)
    FFmat_a=np.zeros((len(dire), np.size(bd.z,1)), dtype=complex)
    FFmat_b=np.zeros((len(dire), np.size(bd.z,1)), dtype=complex)

    for l, meas in enumerate(dire):
         FFmat_a[l,:] = 2*np.pi/(np.size(bd.z,1)*np.sqrt(8*np.pi*kappa_ex))*np.exp(complex(0,1)*np.pi/4)\
         *(-complex(0,1)*weight_dl_ex*kappa_ex*meas.dot(bd.normal))*np.exp(-complex(0,1)*kappa_ex*(meas.dot(bd.z)))
    for l, meas in enumerate(dire):
         FFmat_b[l,:] = 2*np.pi/(np.size(bd.z,1)*np.sqrt(8*np.pi*kappa_ex))*np.exp(complex(0,1)*np.pi/4)\
         *(weight_sl_ex*bd.zpabs)*np.exp(-complex(0,1)*kappa_ex*(meas.dot(bd.z)))
    
    FFmat = np.hstack((FFmat_a, FFmat_b))
    return FFmat


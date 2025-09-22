import numpy as np
from regpy.operators import FourierTransform

#defines the cut-off function, e.g., the bump function
def cutoff_function_2d(x, y):
    xx,yy = np.meshgrid(x,y)
    mask=((1-xx**2-yy**2)>0.01)
    func=np.zeros(xx.shape)
    func[mask]=(np.exp(1)*np.exp(-1/(1-xx[mask]**2-yy[mask]**2)))
    return func

# A C^1 nonnegative cutoff function that is 1 on [-\infty,1] and 0 on [2,\infty]
def cutoff_1d(x):
    x = np.asarray(x)
    y = np.empty_like(x, dtype=float)
    # left / right masks
    left = x <= 1
    right = x >= 2
    mid = (~left) & (~right)
    y[left] = 1.0
    y[right] = 0.0
    t = x[mid] - 1.0           # t in (0,1)
    y[mid] = 2*t**3 - 3*t**2 + 1
    return y

def cutoff_2d(X,Y,s=4):
    r = np.min([np.max(X),np.max(Y)])
    return cutoff_1d((s/r)*np.sqrt(X**2+Y**2)+2-s)

def _create_Vcov(N, N_b, create_type='fourier_random', sigma=0.5, xsample=None, ysample=None, grid=None, Nsample=20,s=4):    
    # create random vector V in spatial and frequency domain
    if xsample is None and grid is not None:
        xsample = grid.coords[0]
    if ysample is None and grid is not None:
        ysample = grid.coords[1]
    assert N_b<=Nsample
    rng = np.random.default_rng(seed=42)
    if create_type=='spatial':
        # create random vector V ( we suppose that K=VV* and V has small rank N_b)
        vec=np.zeros((N_b, N, N), dtype=complex)
        for i in range(0, N_b):
            vec[i, :, :]=1/np.sqrt(2)*(rng.normal(size=(N, N))+complex(0,1)*rng.normal(size=(N, N)))
        #Multiply with a rapidly decaying function
        vec=vec*np.exp(-(xsample)**2/(2*sigma**2)).reshape(N, 1)*np.exp(-(ysample)**2/(2*sigma**2)).reshape(1, N)
        #perform singular value decomposition ( a process to find PCA)
        U, S, V=np.linalg.svd(vec.reshape(N_b, N**2), full_matrices=False)
        # Vcov defines the orthogonal eigenvector (i.e., the principal components)))
        Vcov=V.T.conj()*S        
    elif create_type=='fourier_random':
        
        fourier=FourierTransform(grid, centered=True)
        #Multiply with the Gaussian function in Fourier domain,
        func_freq=np.exp(-np.linalg.norm(fourier.codomain.coords, axis=0)**2/sigma**2)
    
        # perform the cut-off function  
        vec_cut=cutoff_2d(xsample,ysample,s=s)
        # defines the vector V in the spatial domain after performing inverse Fourier transform
        vec_cutted=np.zeros((Nsample, N, N), dtype=complex)
        for i in range(0, Nsample):
            vec_cutted[i, :, :]=fourier.adjoint(rng.normal(size=fourier.domain.shape)*func_freq)*vec_cut
        # perform SVD procedure    
        U, S, V=np.linalg.svd(vec_cutted.reshape(Nsample, N**2), full_matrices=False)
        Vcov=V.T.conj()[:,:N_b]*S[:N_b]
        
    else:
        raise ValueError('No method specified')
        
    return Vcov.reshape(N,N,N_b), S[:N_b]
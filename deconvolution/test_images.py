from regpy.vecsps import UniformGridFcts
import numpy as np

def mixed(M=256,N=256,fac = 1.,c_ring = 1.,c_cross=2., c_smallbox=1.5,c_ramp=1., c_bubbles=-1.):
    grid = UniformGridFcts((-1, 1, 256), (-1.5, 1, 256),dtype = float, periodic = True)
    """Space of real-valued functions on a uniform grid with rectangular pixels"""
    X,Y = grid.coords
    """x and y coordinates."""
    cross = 1.0*np.logical_or((abs(X)<0.01) * (abs(Y)<0.3),(abs(X)<0.3) * (abs(Y)<0.01)) 
    rad = np.sqrt(X**2 + Y**2)
    ring = 1.0*np.logical_and(rad>=0.9, rad<=0.95)
    smallbox = (abs(X+0.55)<=0.05) * (abs(Y-0.55)<=0.05)
    bubbles = (1.001+np.sin(50/(X+1.3)))*np.exp(-((Y+1.25)/0.1)**2)*(X>-0.8)*(X<0.8)
    ramp = (Y<=-1)

    return grid, \
        fac * (c_ring*ring + c_cross*cross + c_smallbox*smallbox + c_ramp*ramp +c_bubbles*bubbles)
    

def fatcross_ring(M=256,N=256,fac = 1.,c_ring = 2.,c_cross=2):
    grid = UniformGridFcts((-1, 1, 256), (-1, 1, 256),dtype = float, periodic = True)
    """Space of real-valued functions on a uniform grid with rectangular pixels"""
    X,Y = grid.coords
    """x and y coordinates."""
    cross = 1.0*np.logical_or((abs(X)<0.1) * (abs(Y)<0.4),(abs(X)<0.4) * (abs(Y)<0.1)) 
    rad = np.sqrt(X**2 + Y**2)
    ring = 1.0*np.logical_and(rad>=0.7, rad<=0.85)

    return grid, fac * (c_ring*ring + c_cross*cross)
    
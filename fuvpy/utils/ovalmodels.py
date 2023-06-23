import os
import numpy as np
import pandas as pd
import xarray as xr


def oval_starkov1994(mlt,al,boundary='pb'):
    '''
    Location of the auroral boundaries from the Starkov1994 model.

    Starkov, G.V., Mathematical model of the auroral boundaries, Geomag. Aeron., 34 (3), 331–336, 1994a.
    See: https://www.swsc-journal.org/articles/swsc/pdf/2011/01/swsc110021.pdf

    Parameters
    ----------
    mlt : array_like
        Local times to return boundary locations
    al : array_like
        Values of the AL index to use
    boundary : str, optional
        'pb' for poleward boundary and 'eb' for equatorward boundary. Default is 'pb'

    Returns
    -------
    lat : array
        Magnetic latitude of the boundaries with shape len(mlt) x len(al)

    '''

    mlt = np.array([mlt]) if isinstance(mlt,(int,float)) else np.array(mlt)
    al = np.array([al]) if isinstance(al,(int,float)) else np.array(al) 

    basepath = os.path.dirname(__file__)
    if boundary =='pb':
        b = pd.read_csv(basepath+'/../data/ovalcoeffs_starkov1994_pb',sep=' ',header=0,index_col=0).values
        A = b[[0],:] + b[[1],:]*np.log10(al[:,None]) + b[[2],:]*np.log10(al[:,None])**2 + b[[3],:]*np.log10(al[:,None])**3
        clat = A[:,[0]] + A[:,[1]]*np.cos(np.deg2rad(15*(mlt[None,:]+ A[:,[4]]))) + A[:,[2]]*np.cos(np.deg2rad(15*(2*mlt[None,:]+A[:,[5]]))) + A[:,[3]]*np.cos(np.deg2rad(15*(3*mlt[None,:]+A[:,[6]])))
    elif boundary == 'eb':
        b = pd.read_csv(basepath+'/../data/ovalcoeffs_starkov1994_eb',sep=' ',header=0,index_col=0).values
        A = b[[0],:] + b[[1],:]*np.log10(al[:,None]) + b[[2],:]*np.log10(al[:,None])**2 + b[[3],:]*np.log10(al[:,None])**3
        clat = A[:,[0]] + A[:,[1]]*np.cos(np.deg2rad(15*(mlt[None,:]+ A[:,[4]]))) + A[:,[2]]*np.cos(np.deg2rad(15*(2*mlt[None,:]+A[:,[5]]))) + A[:,[3]]*np.cos(np.deg2rad(15*(3*mlt[None,:]+A[:,[6]])))
    return 90-clat

def oval_hu2017(Bx,By,Bz,Vp,Np,AE,boundary='pb'):
    '''
    Location of the auroral boundaries from the Hu2017 model

    Hu, Z.-J., Yang, Q.-J., Liang, J.-M., Hu, H.-Q., Zhang, B.-C., and Yang, H.-G. (2017),
    Variation and modeling of ultraviolet auroral oval boundaries associated with interplanetary and geomagnetic parameters,
    Space Weather, 15, 606– 622, doi:10.1002/2016SW001530. 

    Parameters
    Bx : array_like
        IMF Bx component (nT)
    By : array_like
        IMF By component (nT)
    Bz : array_like
        IMF Bz component (nT)
    Vp : array_like
        Solar wind speed (100 km/s)
    Np : array_like
        Solar wind proton density (cm^-3)
    AE : array_like
        AE index (nT)

    boundary : str, optional
        'pb' for poleward boundary and 'eb' for equatorward boundary. Default is 'pb'

    Returns
    -------
    lat : array
        Magnetic latitude of the boundaries from 0 to 23 magnetic local time [shape 24 x len(Bx)]

    '''

    Bx = np.array([Bx]) if isinstance(Bx,(int,float)) else np.array(Bx)
    By = np.array([By]) if isinstance(By,(int,float)) else np.array(By)
    Bz = np.array([Bz]) if isinstance(Bz,(int,float)) else np.array(Bz)
    Vp = np.array([Vp]) if isinstance(Vp,(int,float)) else np.array(Vp)
    Np = np.array([Np]) if isinstance(Np,(int,float)) else np.array(Np)
    AE = np.array([AE]) if isinstance(AE,(int,float)) else np.array(AE)
    if not Bx.shape==By.shape==Bz.shape==Vp.shape==Np.shape==AE.shape:
        raise ValueError('Input must have the same size')

    basepath = os.path.dirname(__file__)
    if boundary =='pb':
        A = pd.read_csv(basepath+'/../data/ovalcoeffs_hu2017_pb',sep='\t',header=0,index_col=0).values.T

        A =np.char.strip(A.astype(str))
        A = np.char.replace(A,'−', '-')
        A = A.astype(float)

        lat = A[0,:][None,:] + A[[1],:]*Bx[:,None] + A[[2],:]*By[:,None] + A[[3],:]*Bz[:,None] + A[[4],:]*Vp[:,None] + A[[5],:]*Np[:,None] + A[[6],:]*AE[:,None]

    elif boundary =='eb':
        A = pd.read_csv(basepath+'/../data/ovalcoeffs_hu2017_eb',sep='\t',header=0,index_col=0).values.T

        A =np.char.strip(A.astype(str))
        A = np.char.replace(A,'−', '-')
        A = A.astype(float)

        lat = A[0,:][None,:] + A[[1],:]*Bx[:,None] + A[[2],:]*By[:,None] + A[[3],:]*Bz[:,None] + A[[4],:]*Vp[:,None] + A[[5],:]*Np[:,None] + A[[6],:]*AE[:,None]

    return lat

def oval_gjerloev2008():
    basepath = os.path.dirname(__file__)
    path =  basepath+'/../data/ovalcoeffs_gjerloev2007'
    mlt = np.arange(0,24,0.1)
    epoch = np.arange(-4,7)

    A = pd.read_csv(path+'_pb',sep=',',header=None,index_col=0).values
    A =np.char.strip(A.astype(str))
    A = np.char.replace(A,'−', '-')
    A = A.astype(float)

    pb = A[:,[0]] + A[:,[1]]*np.cos(1*mlt*np.pi/12)+ A[:,[2]]*np.sin(1*mlt*np.pi/12)+ A[:,[3]]*np.cos(2*mlt*np.pi/12)+ A[:,[4]]*np.sin(2*mlt*np.pi/12)+ A[:,[5]]*np.cos(3*mlt*np.pi/12)+ A[:,[6]]*np.sin(3*mlt*np.pi/12)+ A[:,[7]]*np.cos(4*mlt*np.pi/12)+ A[:,[8]]*np.sin(4*mlt*np.pi/12)

    A = pd.read_csv(path+'_eb',sep=',',header=None,index_col=0).values
    A =np.char.strip(A.astype(str))
    A = np.char.replace(A,'−', '-')
    A = A.astype(float)

    eb = A[:,[0]] + A[:,[1]]*np.cos(1*mlt*np.pi/12)+ A[:,[2]]*np.sin(1*mlt*np.pi/12)+ A[:,[3]]*np.cos(2*mlt*np.pi/12)+ A[:,[4]]*np.sin(2*mlt*np.pi/12)+ A[:,[5]]*np.cos(3*mlt*np.pi/12)+ A[:,[6]]*np.sin(3*mlt*np.pi/12)+ A[:,[7]]*np.cos(4*mlt*np.pi/12)+ A[:,[8]]*np.sin(4*mlt*np.pi/12)

    ds = xr.Dataset(
        data_vars=dict(
            pb=(['epoch','mlt'], pb),
            eb=(['epoch','mlt'], eb),
            ),
        coords=dict(
            epoch = epoch,
            mlt = mlt
        ),
        )

    mu0 = 4e-7*np.pi # Vacuum magnetic permeability
    M_E = 8.05e22 # Earth's magnetic dipole moment
    R_E = 6371e3 # Earth radii
    height = 130e3
    R_I = R_E + height # Radius of ionosphere
    ds['dP'] = (mu0*M_E)/(4*np.pi*R_I) * (np.sin(np.deg2rad(90-ds['pb']))**2)

    ds['dA'] = (mu0*M_E)/(4*np.pi*R_I) * (np.sin(np.deg2rad(90-ds['eb']))**2) - ds['dP']

    #ind = np.r_[range(60),range(180,240)]
    ind = np.r_[range(240)]

    ds['P'] = 1e-6*np.deg2rad(15*0.1)*(ds['dP']).isel(mlt=ind).sum(dim='mlt')
    ds['A'] = 1e-6*np.deg2rad(15*0.1)*(ds['dA']).isel(mlt=ind).sum(dim='mlt')
    return ds

def q2kp(q):
    '''
    Convert q-index to Kp index 

    The algorithm is from USAF Research Laboratory (then Phillips Laboratory) report PL-TR-93-2267 by B. S. Dandekar (AD-A282 764). 
    https://apps.dtic.mil/sti/pdfs/ADA282764.pdf

    Parameters
    ----------
    q : array_like
        values to convert

    Returns
    -------
    kp : array_like
        converted values
    '''

    kp_lim = np.mean([7/3,8/3])
    qlim = np.mean([0.96*kp_lim-0.3,2.04*kp_lim-2.7])

    kp1 = (q+0.3)/0.96
    kp2 = (q+2.7)/2.04
    kp = np.where(q<qlim,kp1,kp2)
    return kp

def kp2al(kp):
    '''
    Convert Kp-index to AL index 

    The algorithm is from Starkov, G.V., Statistical dependences between the magnetic activity indices, Geomag. Aeron., 34 (1), 101–103, 1994b. 
    See https://www.swsc-journal.org/articles/swsc/pdf/2011/01/swsc110021.pdf

    Parameters
    ----------
    kp : array_like
        values to convert

    Returns
    -------
    al : array_like
        converted values
    '''

    c0 = 18
    c1 =-12.3
    c2 = 27.2
    c3 =-2
    return c0 + c1*kp + c2*kp**2 + c3*kp**3
import os
import numpy as np
import pandas as pd


def oval_starkov1994(mlt,al,boundary='pb'):
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

        lat = A[0,:][None,:] + A[[1],:]*Bx[:,None] + A[[2],:]*By[:,None] + A[[3],:]*Bz[:,None] + A[[4],:]*Vp[:,None] + A[[5],:]*Np[:,None] + A[[5],:]*AE[:,None]

    elif boundary =='eb':
        A = pd.read_csv(basepath+'/../data/ovalcoeffs_hu2017_eb',sep='\t',header=0,index_col=0).values.T

        A =np.char.strip(A.astype(str))
        A = np.char.replace(A,'−', '-')
        A = A.astype(float)

        lat = A[0,:][None,:] + A[[1],:]*Bx[:,None] + A[[2],:]*By[:,None] + A[[3],:]*Bz[:,None] + A[[4],:]*Vp[:,None] + A[[5],:]*Np[:,None] + A[[6],:]*AE[:,None]

    return lat

def q2kp(q):
    kp_lim = np.mean([7/3,8/3])
    qlim = np.mean([0.96*kp_lim-0.3,2.04*kp_lim-2.7])

    kp1 = (q+0.3)/0.96
    kp2 = (q+2.7)/2.04
    kp = np.where(q<qlim,kp1,kp2)
    return kp

def kp2al(kp):
    c0 = 18
    c1 =-12.3
    c2 = 27.2
    c3 =-2
    return c0 + c1*kp + c2*kp**2 + c3*kp**3
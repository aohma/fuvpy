import numpy as np
import pandas as pd


def oval_starkov1994(mlt,al,boundary='pb'):

    if boundary =='pb':
        b = pd.read_csv('../data/ovalcoeffs_starkov1994_pb',sep=' ',header=0,index_col=0).values
        A = b[0,:][None,:] + b[[1],:]*np.log10(al[:,None]) + b[[2],:]*np.log10(al[:,None])**2 + b[[3],:]*np.log10(al[:,None])**3
        clat = A[:,[0]] + A[:,[1]]*np.cos(np.deg2rad(15*(mlt[None,:]+ A[:,[4]]))) + A[:,[2]]*np.cos(np.deg2rad(15*(2*mlt[None,:]+A[:,[5]]))) + A[:,[3]]*np.cos(np.deg2rad(15*(3*mlt[None,:]+A[:,[6]])))
    elif boundary == 'eb':
        b = pd.read_csv('../data/ovalcoeffs_starkov_eb',sep=' ').values
        A = b[0,:] + b[1,:]*np.log10(al) + b[2,:]*np.log10(al)**2 + b[3,:]*np.log10(al)**3
        clat = A[0] + A[1]*np.cos(np.deg2rad(15*(mlt+ A[4]))) + A[2]*np.cos(np.deg2rad(15*(2*mlt+A[5]))) + A[3]*np.cos(np.deg2rad(15*(3*mlt+A[6]))) 
    return 90-clat
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:28:25 2022

@author: aohma
"""

import numpy as np
import xarray as xr

from fuvpy.src.utils import sh
from fuvpy.src.utils.sunlight import subsol

    
def makeDGmodel(imgs,inImg='img',transform=None,sOrder=3,dampingVal=0,tukeyVal=5,stop=1e-3,minlat=0,dzalim=80,sKnots=None,tKnotSep=None,tOrder=2,dzacorr = 0):
    '''
    Function to model the FUV dayglow and subtract it from the input image

    Parameters
    ----------
    imgs : xarray.Dataset
        Dataset with the FUV images, imported by readFUVimage()
    inImg : str, optional
        Name of the input image to be used in the model. The default is 'img'.
    sOrder : int, optional
        Order of the spatial spline fit. The default is 3.
    dampingVal : float, optional
        Damping (Tikhonov regularization).
        The default is 0 (no damping).
    tukeyVal : float, optional
        Determines to what degree outliers are down-weighted.
        Iterative reweights is (1-(residuals/(tukeyVal*rmse))^2)^2
        Larger tukeyVal means less down-weight
        Default is 5
    stop : float, optional
        When to stop the iteration. The default is 0.001.
    minlat : float, optional
        Lower mlat boundary to include in the model. The default is 0.
    dzalim : float, optional
        Maximum viewing angle to include. The default is 80.
    sKnots : array like, optional
        Location of the spatial Bspline knots. The default is None (default is used).
    tKnotSep : int, optional
        Approximate separation of temporal knots in minutes. The default is None (only knots at endpoints)
    tOrder : int, optional
        Order of the temporal spline fit. The default is 2.

    Returns
    -------
    imgs : xarray.Dataset
        A copy(?) of the image Dataset with three new fields:
            - imgs['dgmodel'] is the dayglow model
            - imgs['dgimg'] is the dayglow-corrected image (dayglow subtracked from the input image)
            - imgs['dgweight'] is the weights after the final iteration
    '''

    # Add temporal dimension if missing
    if len(imgs.sizes)==2: imgs = imgs.expand_dims('date')

    # Reshape the data
    sza   = imgs['sza'].stack(z=('row','col')).values
    dza   = imgs['dza'].stack(z=('row','col')).values
    if transform=='log':
        d = np.log(imgs[inImg].stack(z=('row','col')).values)
    else:
        d = imgs[inImg].stack(z=('row','col')).values
    mlat  = imgs['mlat'].stack(z=('row','col')).values
    remove = imgs['bad'].stack(z=('row','col')).values

    # Spatial knots and viewing angle correction
    if imgs['id'] in ['WIC','SI12','SI13','UVI']:
        fraction = np.exp(dzacorr*(1. - 1/np.cos(np.deg2rad(dza))))/np.cos(np.deg2rad(dza))*np.cos(np.deg2rad(sza))
        if sKnots is None: sKnots = [-5,-3,-1,-0.2,-0.1,0,0.1,0.2,1,3,5]
    elif imgs['id'] == 'VIS':
        fraction = np.cos(np.deg2rad(sza))
        if sKnots is None: sKnots= [-1,-0.2,-0.1,0,0.333,0.667,1]
    sKnots = np.r_[np.repeat(sKnots[0],sOrder),sKnots, np.repeat(sKnots[-1],sOrder)]

    # Minuets since first image
    date = imgs['date'].values
    time=(date-date[0])/ np.timedelta64(1, 'm')

    # temporal and spatial size
    n_t = d.shape[0]
    n_s = d.shape[1]

    # Temporal knots
    if tKnotSep==None:
        tKnots = np.linspace(time[0], time[-1], 2)
    else:
        tKnots = np.linspace(time[0], time[-1], int(np.round(time[-1]/tKnotSep)+1))
    tKnots = np.r_[np.repeat(tKnots[0],tOrder),tKnots, np.repeat(tKnots[-1],tOrder)]

    # Number of control points
    n_tcp = len(tKnots)-tOrder-1
    n_scp = len(sKnots)-sOrder-1

    # Temporal design matix
    print('Building dayglow G matrix')
    M = BSpline(tKnots, np.eye(n_tcp), tOrder)(time)

    G_g=[]
    G_s=[]
    for i in range(n_t):
        G= BSpline(sKnots, np.eye(n_scp), sOrder)(fraction[i,:]) # Spatial design matirx
        G_t = np.zeros((n_s, n_scp*n_tcp))

        for j in range(n_tcp):
            G_t[:, np.arange(j, n_scp*n_tcp, n_tcp)] = G*M[i, j]

        G_g.append(G)
        G_s.append(G_t)
    G_s=np.vstack(G_s)

    # Data
    ind = ((sza >= 0) & (dza <= dzalim) & (mlat >= minlat) & (np.isfinite(d)) & remove[None,:])
    d_s = d.flatten()
    ind = ind.flatten()
    w = np.ones(d_s.shape)

    # Damping
    damping = dampingVal*np.ones(G_s.shape[1])
    R = np.diag(damping)

    # Iterativaly solve the inverse problem
    diff = 1e10*stop
    iteration = 0
    m = None
    while (diff>stop)&(iteration < 100):
        print('Iteration:',iteration)

        m_s = np.linalg.lstsq((G_s[ind,:]*w[ind,None]).T@(G_s[ind,:]*w[ind,None])+R,(G_s[ind,:]*w[ind,None]).T@(d_s[ind]*w[ind]),rcond=None)[0]

        mNew    = M@m_s.reshape((n_scp, n_tcp)).T
        dm=[]
        for i, tt in enumerate(time):
            dm.append(G_g[i]@mNew[i, :])

        dm=np.array(dm).squeeze()
        residuals = dm.flatten()[ind] - d.flatten()[ind]
        rmse = np.sqrt(np.average(residuals**2,weights=w[ind]))

        iw = ((residuals)/(tukeyVal*rmse))**2
        iw[iw>1] = 1
        w[ind] = (1-iw)**2

        if m is not None:
            diff = np.sqrt(np.mean((mNew-m)**2))/(1+np.sqrt(np.mean(mNew**2)))
            print('Relative change model norm',diff)
        m = mNew
        iteration += 1

    # Add dayglow model and corrected image to the Dataset
    if transform=='log':
        imgs['dgmodel'] = (['date','row','col'],(np.exp(dm)).reshape((n_t,len(imgs.row),len(imgs.col))))
    else:
        imgs['dgmodel'] = (['date','row','col'],(dm).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['dgimg'] = imgs[inImg]-imgs['dgmodel']
    imgs['dgweight'] = (['date','row','col'],(w).reshape((n_t,len(imgs.row),len(imgs.col))))

    # Remove pixels outside model scope
    ind = (imgs.sza>=0)& (imgs.dza <= dzalim) & (imgs.mlat >= minlat) & imgs.bad
    imgs['dgmodel'] = xr.where(~ind,np.nan,imgs['dgmodel'])
    imgs['dgimg'] = xr.where(~ind,np.nan,imgs['dgimg'])
    imgs['dgweight'] = xr.where(~ind,np.nan,imgs['dgweight'])

    # Add attributes
    imgs['dgmodel'].attrs = {'long_name': 'Dayglow model'}
    imgs['dgimg'].attrs = {'long_name': 'Dayglow corrected image'}
    imgs['dgweight'].attrs = {'long_name': 'Dayglow model weights'}

    return imgs


def makeSHmodel(imgs,Nsh,Msh,order=2,dampingVal=0,tukeyVal=5,stop=1e-3,knotSep=None):
    '''
    Function to model the FUV residual background and subtract it from the input image

    Parameters
    ----------
    imgs : xarray.Dataset
        Dataset with the FUV images
    Nsh : int
        Order of the SH
    Msh : int
        Degree of the SH
    order: int, optional
        Order of the temporal spline fit. The default is 2.
    dampingVal : TYPE, optional
        Damping to reduce the influence of the time-dependent part of the model.
        The default is 0 (no damping).
    tukeyVal : float, optional
        Determines to what degree outliers are down-weighted.
        Iterative reweights is (1-(residuals/(tukeyVal*rmse))^2)^2
        Larger tukeyVal means less down-weight
        Default is 5
    stop : float, optional
        When to stop the iteration. The default is 0.001.
    knotSep : int, optional
        Approximate separation of temporal knots in minutes. The default is None (only knots at endpoints)

    Returns
    -------
    imgs : xarray.Dataset
        A copy(?) of the image Dataset with two new fields:
            - imgs['shmodel'] is the dayglow model
            - imgs['shimg'] is the dayglow-corrected image (dayglow subtracked from the input image)
            - imgs['shweight'] is the weight if each pixel after the final iteration
    '''
    date = imgs['date'].values
    time=(date-date[0])/ np.timedelta64(1, 'm')

    glat = imgs['glat'].stack(z=('row','col')).values
    glon = imgs['glon'].stack(z=('row','col')).values
    d = imgs['dgimg'].stack(z=('row','col')).values
    dg = imgs['dgsigma'].stack(z=('row','col')).values
    wdg = imgs['dgweight'].stack(z=('row','col')).values

    # Treat dg as variance
    d = d/dg
    

    sslat, sslon = map(np.ravel, subsol(date))
    phi = np.deg2rad((glon - sslon[:,None] + 180) % 360 - 180)

    n_t = glat.shape[0]
    n_s = glat.shape[1]

    # Temporal knots
    if knotSep==None:
        knots = np.linspace(time[0], time[-1], 2)
    else:
        knots = np.linspace(time[0], time[-1], int(np.round(time[-1]/knotSep)+1))
    knots = np.r_[np.repeat(knots[0],order),knots, np.repeat(knots[-1],order)]

    # Number of control points
    n_cp = len(knots)-order-1

    # Temporal design matix
    M = BSpline(knots, np.eye(n_cp), order)(time)

    # Iterative (few iterations)
    skeys = sh.SHkeys(Nsh, Msh).Mge(1).MleN().setNmin(1)
    ckeys = sh.SHkeys(Nsh, Msh).MleN().setNmin(1)

    print('Building sh G matrix')
    G_g=[]
    G_s=[]
    for i in range(n_t):
        # calculate Legendre functions at glat:
        P, dP = sh.get_legendre(Nsh, Msh, 90 - glat[i,:])
        Pc = np.hstack([P[key] for key in ckeys])
        Ps = np.hstack([P[key] for key in skeys])

        Gcos = Pc * np.cos(phi[i,:].reshape((-1, 1)) * ckeys.m)
        Gsin = Ps * np.sin(phi[i,:].reshape((-1, 1)) * skeys.m)

        G = np.hstack((Gcos, Gsin))
        G = G/dg[i,:][:,None]
        n_G = np.shape(G)[1]

        G_t = np.zeros((n_s, n_G*n_cp))

        for j in range(n_cp):
            G_t[:, np.arange(j, n_G*n_cp, n_cp)] = G*M[i, j]

        G_g.append(G)
        G_s.append(G_t)

    G_s = np.array(G_s)
    G_s = G_s.reshape(-1,G_s.shape[2])

    # Data
    ind = (np.isfinite(d))&(glat>0)&(imgs['bad'].values.flatten())[None,:]
    d_s = d.flatten()
    ind = ind.flatten()
    w = wdg.flatten() # Weights from dayglow model

    # Damping
    damping = dampingVal*np.ones(G_s.shape[1])
    R = np.diag(damping)

    diff = 1e10*stop
    iteration = 0
    m = None
    while (diff>stop)&(iteration < 100):
        print('Iteration:',iteration)
        # Solve for spline amplitudes
        m_s = np.linalg.lstsq((G_s[ind,:]*w[ind,None]).T@(G_s[ind,:]*w[ind,None])+R,(G_s[ind,:]*w[ind,None]).T@(d_s[ind]*w[ind]),rcond=None)[0]

        # Retrieve B-spline smooth model paramters (coarse)
        mNew    = M@m_s.reshape((n_G, n_cp)).T
        dm=[]
        for i, tt in enumerate(time):
            dm.append(G_g[i]@mNew[i, :])

        dm=np.array(dm).squeeze()
        residuals = dm.flatten()[ind] - d.flatten()[ind]
        rmse = np.sqrt(np.average(residuals**2,weights=w[ind]))

        iw = ((residuals)/(tukeyVal*rmse))**2
        iw[iw>1] = 1
        w[ind] = (1-iw)**2

        if m is not None:
            diff = np.sqrt(np.mean( (mNew-m)**2))/(1+np.sqrt(np.mean(mNew**2)))
            print('Relative change model norm:',diff)
        m = mNew
        iteration += 1


    imgs['shmodel'] = (['date','row','col'],(dm*dg).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['shimg'] = imgs['dgimg']-imgs['shmodel']
    imgs['shweight'] = (['date','row','col'],(w).reshape((n_t,len(imgs.row),len(imgs.col))))

    # Add attributes
    imgs['shmodel'].attrs = {'long_name': 'Spherical harmonics model'}
    imgs['shimg'].attrs = {'long_name': 'Spherical harmonics corrected image'}
    imgs['shweight'].attrs = {'long_name': 'Spherical harmonics model weights'}

    return imgs

def weightedBinning(fraction,d,dm,w,sKnots):
    bins = np.r_[sKnots[0],np.linspace(0,sKnots[-1],51)]
    binnumber = np.digitize(fraction,bins)
    
    rmse=np.full_like(d,np.nan)
    for i in range(1,len(bins)):
        ind = binnumber==i
        if np.sum(w[ind])==0:
            rmse[ind]=np.nan
        else:
            rmse[ind]=np.sqrt(np.average((d[ind]-dm[ind])**2,weights=w[ind]))
    
    # Linear fit
    ind2 = np.isfinite(rmse)
    G = np.vstack((dm**2,np.ones_like(dm))).T
    m =np.linalg.lstsq(G[ind2].T@G[ind2],G[ind2].T@np.log(rmse[ind2]**2),rcond=None)[0]
    
    return np.sqrt(np.exp(G@m))

def makeDGmodelTest(imgs,inImg='img',transform=None,sOrder=3,dampingVal=0,tukeyVal=5,stop=1e-3,minlat=0,dzalim=80,sKnots=None,tKnotSep=None,tOrder=2,dzacorr = 0):
    '''
    Function to model the FUV dayglow and subtract it from the input image

    Parameters
    ----------
    imgs : xarray.Dataset
        Dataset with the FUV images, imported by readFUVimage()
    inImg : str, optional
        Name of the input image to be used in the model. The default is 'img'.
    sOrder : int, optional
        Order of the spatial spline fit. The default is 3.
    dampingVal : float, optional
        Damping (Tikhonov regularization).
        The default is 0 (no damping).
    tukeyVal : float, optional
        Determines to what degree outliers are down-weighted.
        Iterative reweights is (1-(residuals/(tukeyVal*rmse))^2)^2
        Larger tukeyVal means less down-weight
        Default is 5
    stop : float, optional
        When to stop the iteration. The default is 0.001.
    minlat : float, optional
        Lower mlat boundary to include in the model. The default is 0.
    dzalim : float, optional
        Maximum viewing angle to include. The default is 80.
    sKnots : array like, optional
        Location of the spatial Bspline knots. The default is None (default is used).
    tKnotSep : int, optional
        Approximate separation of temporal knots in minutes. The default is None (only knots at endpoints)
    tOrder : int, optional
        Order of the temporal spline fit. The default is 2.

    Returns
    -------
    imgs : xarray.Dataset
        A copy(?) of the image Dataset with three new fields:
            - imgs['dgmodel'] is the dayglow model
            - imgs['dgimg'] is the dayglow-corrected image (dayglow subtracked from the input image)
            - imgs['dgweight'] is the weights after the final iteration
    '''
    imgs = imgs.copy()
    
    # Add temporal dimension if missing
    if len(imgs.sizes)==2: imgs = imgs.expand_dims('date')

    # Reshape the data
    sza   = imgs['sza'].stack(z=('row','col')).values
    dza   = imgs['dza'].stack(z=('row','col')).values
    if transform=='log':
        d = np.log(imgs[inImg].stack(z=('row','col')).values+1)
    elif transform=='asinh':
        background_mean = np.nanmean(imgs[inImg].values[imgs['bad'].values&(imgs['sza'].values>100|np.isnan(imgs['sza'].values))])
        background_std = np.nanstd(imgs[inImg].values[imgs['bad'].values&(imgs['sza'].values>100|np.isnan(imgs['sza'].values))])
        d = np.arcsinh((imgs[inImg].stack(z=('row','col')).values-background_mean)/background_std)        
    else:
        d = imgs[inImg].stack(z=('row','col')).values
    glat  = imgs['glat'].stack(z=('row','col')).values
    remove = imgs['bad'].stack(z=('row','col')).values

    # Spatial knots and viewing angle correction
    if imgs['id'] in ['WIC','SI12','SI13','UVI']:
        fraction = np.exp(dzacorr*(1. - 1/np.cos(np.deg2rad(dza))))/np.cos(np.deg2rad(dza))*np.cos(np.deg2rad(sza))
        if sKnots is None: sKnots = [-5,-3,-1,-0.2,-0.1,0,0.1,0.2,1,3,5]
    elif imgs['id'] == 'VIS':
        fraction = np.cos(np.deg2rad(sza))
        if sKnots is None: sKnots= [-1,-0.2,-0.1,0,0.333,0.667,1]
    sKnots = np.r_[np.repeat(sKnots[0],sOrder),sKnots, np.repeat(sKnots[-1],sOrder)]

    # Minuets since first image
    date = imgs['date'].values
    time=(date-date[0])/ np.timedelta64(1, 'm')

    # temporal and spatial size
    n_t = d.shape[0]
    n_s = d.shape[1]

    # Temporal knots
    if tKnotSep==None:
        tKnots = np.linspace(time[0], time[-1], 2)
    else:
        tKnots = np.linspace(time[0], time[-1], int(np.round(time[-1]/tKnotSep)+1))
    tKnots = np.r_[np.repeat(tKnots[0],tOrder),tKnots, np.repeat(tKnots[-1],tOrder)]

    # Number of control points
    n_tcp = len(tKnots)-tOrder-1
    n_scp = len(sKnots)-sOrder-1

    # Temporal design matix
    print('Building dayglow G matrix')
    M = BSpline(tKnots, np.eye(n_tcp), tOrder)(time)

    G_g=[]
    G_s=[]
    
    for i in range(n_t):
        G= BSpline(sKnots, np.eye(n_scp), sOrder)(fraction[i,:]) # Spatial design matirx
        G_t = np.zeros((n_s, n_scp*n_tcp))
        for j in range(n_tcp):
            G_t[:, np.arange(j, n_scp*n_tcp, n_tcp)] = G*M[i, j]

        G_g.append(G)
        G_s.append(G_t)
    G_s=np.vstack(G_s)

    # Data
    ind = (sza >= 0) & (dza <= dzalim) & (glat >= minlat) & (np.isfinite(d)) & remove[None,:] & (fraction>sKnots[0]) & (fraction<sKnots[-1])
    
    d_s = d.flatten()
    ind = ind.flatten()
    ws = np.ones_like(d_s)
    w = np.ones(d_s.shape)
    
    # Damping
    damping = dampingVal*np.ones(G_s.shape[1])
    R = np.diag(damping)

    # Iterativaly solve the inverse problem
    diff = 1e10
    iteration = 0
    m = None
    while (diff>stop)&(iteration < 100):
        print('Iteration:',iteration)

        m_s = np.linalg.lstsq((G_s[ind,:]*w[ind,None]*ws[ind,None]).T@(G_s[ind,:]*w[ind,None]*ws[ind,None])+R,(G_s[ind,:]*w[ind,None]*ws[ind,None]).T@(d_s[ind]*w[ind]*ws[ind]),rcond=None)[0]

        mNew    = M@m_s.reshape((n_scp, n_tcp)).T
        dm=[]
        for i, tt in enumerate(time):
            dm.append(G_g[i]@mNew[i, :])

        dm=np.array(dm).squeeze()
        residuals = (d.flatten()[ind] - dm.flatten()[ind])
        sigma = np.full_like(d_s,np.nan)
        sigma[ind] = weightedBinning(fraction.flatten()[ind], d.flatten()[ind],dm.flatten()[ind], w[ind],sKnots)

        iw = ((residuals)/(tukeyVal*sigma[ind]))**2
        iw[iw>1] = 1
        w[ind] = (1-iw)**2
        
        if m is not None:
            diff = np.sqrt(np.mean((mNew-m)**2))/(1+np.sqrt(np.mean(mNew**2)))
            print('Relative change model norm',diff)
        m = mNew
        iteration += 1
    
    # Add dayglow model and corrected image to the Dataset
    if transform=='log':
        imgs['dgmodel'] = (['date','row','col'],(np.exp(dm)).reshape((n_t,len(imgs.row),len(imgs.col))))
    elif transform == 'asinh':
        imgs['dgmodel'] = (['date','row','col'],(np.sinh(dm)*background_std +background_mean).reshape((n_t,len(imgs.row),len(imgs.col))))
    else:
        imgs['dgmodel'] = (['date','row','col'],(dm).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['dgimg'] = imgs[inImg]-imgs['dgmodel']
    imgs['dgweight'] = (['date','row','col'],(w).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['dgsigma'] = (['date','row','col'],(sigma).reshape((n_t,len(imgs.row),len(imgs.col))))

    # Remove pixels outside model scope
    ind = (imgs.sza>=0)& (imgs.dza <= dzalim) & (imgs.glat >= minlat) & imgs.bad
    imgs['dgmodel'] = xr.where(~ind,np.nan,imgs['dgmodel'])
    imgs['dgimg'] = xr.where(~ind,np.nan,imgs['dgimg'])
    imgs['dgweight'] = xr.where(~ind,np.nan,imgs['dgweight'])
    imgs['dgsigma'] = xr.where(~ind,np.nan,imgs['dgsigma'])
    
   
    return  imgs
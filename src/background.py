#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:28:25 2022

@author: aohma
"""

import numpy as np
import xarray as xr

from scipy.interpolate import BSpline
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic

from fuvpy.src.utils import sh
from fuvpy.src.utils.sunlight import subsol
from polplot import sdarngrid,bin_number


def makeBSmodel(imgs,inImg='img',sOrder=3,dampingVal=0,tukeyVal=5,stop=1e-3,minlat=-90,dzalim=75,sKnots=None,n_tKnots=2,tOrder=2,returnNorms=False):
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
        Lower glat boundary to include in the model. The default is -90.
    dzalim : float, optional
        Maximum viewing angle to include. The default is 75.
    sKnots : array like, optional
        Location of the spatial Bspline knots. The default is None (default is used).
    n_tKnots : int, optional
        Number of temporal knots, equally spaced between the endpoints. The default is 2 (only knots at endpoints)
    tOrder : int, optional
        Order of the temporal spline fit. The default is 2.
    returnNorms : bool, optional
        If True, also return the residual and model norms

    Returns
    -------
    imgs : xarray.Dataset
        A copy of the image Dataset with four new fields:
            - imgs['dgmodel'] is the B-spline based dayglow model
            - imgs['dgimg'] is the dayglow-corrected image (dayglow subtracked from the input image)
            - imgs['dgweight'] is the weights after the final iteration
            - imgs['dgsigma'] is the modelled spread
    norms : tuple, optional
        A tuple containing the residual and model norms. Only if returnNorms is True
    '''

    # Add temporal dimension if missing
    if len(imgs.sizes)==2: imgs = imgs.expand_dims('date')

    # Spatial knots and viewing angle correction
    if imgs['id'] in ['WIC','SI12','SI13','UVI']:
        fraction = np.cos(np.deg2rad(imgs['sza'].stack(z=('row','col')).values))/np.cos(np.deg2rad(imgs['dza'].stack(z=('row','col')).values))
        if sKnots is None: sKnots = [-3.5,-0.25,0,0.25,1.5,3.5]
    elif imgs['id'] == 'VIS':
        fraction = np.cos(np.deg2rad(imgs['sza'].stack(z=('row','col')).values))
        if sKnots is None: sKnots= [-1,-0.1,0,0.1,0.4,1]
    sKnots = np.r_[np.repeat(sKnots[0],sOrder),sKnots, np.repeat(sKnots[-1],sOrder)]

    # Minuets since first image
    time=(imgs['date'].values-imgs['date'].values[0])/ np.timedelta64(1, 'm')

    # temporal and spatial size
    n_t = fraction.shape[0]
    n_s = fraction.shape[1]

    # Temporal knots
    tKnots = np.linspace(time[0], time[-1], n_tKnots)
    tKnots = np.r_[np.repeat(tKnots[0],tOrder),tKnots, np.repeat(tKnots[-1],tOrder)]

    # Number of control points
    n_tcp = len(tKnots)-tOrder-1
    n_scp = len(sKnots)-sOrder-1

    # Temporal design matix
    print('Building dayglow G matrix')
    M = BSpline(tKnots, np.eye(n_tcp), tOrder)(time)

    # Spatial design matrix
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

    # Index of data to use in model
    ind = (imgs['sza'].stack(z=('row','col')).values >= 0) & (imgs['dza'].stack(z=('row','col')).values <= dzalim) & (imgs['glat'].stack(z=('row','col')).values >= minlat) & (np.isfinite(imgs[inImg].stack(z=('row','col')).values)) & imgs['bad'].stack(z=('row','col')).values[None,:] & (fraction>sKnots[0]) & (fraction<sKnots[-1])

    # Spatial weights
    ws = np.full_like(fraction,np.nan)
    for i in range(len(fraction)):
        count,bin_edges,bin_number=binned_statistic(fraction[i,ind[i,:]],fraction[i,ind[i,:]],statistic=
    'count',bins=np.arange(sKnots[0],sKnots[-1]+0.1,0.1))
        ws[i,ind[i,:]]=1/np.maximum(1,count[bin_number-1])

    # Make everything flat
    d_s = imgs[inImg].stack(z=('row','col')).values.flatten()
    ind = ind.flatten()
    ws = ws.flatten()
    w = np.ones_like(d_s)

    # Damping (zeroth-order Tikhonov regularization)
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
        residuals = (d_s[ind] - dm.flatten()[ind])

        if iteration==0:
            sigma = np.full_like(d_s,np.nan)
            sigma[ind] = _noiseModel(fraction.flatten()[ind], d_s[ind],dm.flatten()[ind], w[ind],sKnots)
        w[ind] = (1 - np.minimum(1,(residuals/(tukeyVal*sigma[ind]))**2))**2

        if m is not None:
            diff = np.sqrt(np.mean((mNew-m)**2))/(1+np.sqrt(np.mean(mNew**2)))
            print('Relative change model norm',diff)
        m = mNew
        iteration += 1

    normM = np.sqrt(np.average(m_s**2))
    normR = np.sqrt(np.average(residuals**2,weights=w[ind]))

    # Add dayglow model and corrected image to the Dataset
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

    # Add attributes
    imgs['dgmodel'].attrs = {'long_name': 'BS model'}
    imgs['dgimg'].attrs = {'long_name': 'BS corrected image'}
    imgs['dgweight'].attrs = {'long_name': 'BS model weights'}
    imgs['dgsigma'].attrs = {'long_name': 'BS model spread'}

    if returnNorms:
        return imgs,(normR,normM)
    else:
        return imgs

def _noiseModel(fraction,d,dm,w,sKnots):

    # Binned RMSE
    bins = np.r_[sKnots[0],np.arange(0,sKnots[-1]+0.25,0.25)]
    binnumber = np.digitize(fraction,bins)

    rmse=np.full_like(d,np.nan)
    for i in range(1,len(bins)):
        ind = binnumber==i
        if np.sum(w[ind])==0:
            rmse[ind]=np.nan
        else:
            rmse[ind]=np.sqrt(np.average((d[ind]-dm[ind])**2,weights=w[ind]))

    # Linear fit with bounds
    ind2 = np.isfinite(rmse)
    def func(x,a,b):
        return a*x+b

    popt, pcov=curve_fit(func, dm[ind2]**2, rmse[ind2]**2, bounds=(0,np.inf))
    rmseFit = np.sqrt(func(dm**2,*popt))

    return rmseFit


def makeSHmodel(imgs,Nsh,Msh,dampingVal=0,tukeyVal=5,stop=1e-3,minlat=0,n_tKnots=2,tOrder=2,returnNorms=False):
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
    minlat : float, optional
        Minimum geographic latitude to include
    n_tKnots : int, optional
        Number of temporal knots, equally spaced between the endpoints. The default is 2 (only knots at endpoints)
    tOrder: int, optional
        Order of the temporal spline fit. The default is 2.
    returnNorms : bool, optional
        If True, also return the residual and model norms
    Returns
    -------
    imgs : xarray.Dataset
        A copy of the image Dataset with two new fields:
            - imgs['shmodel'] is the dayglow model
            - imgs['shimg'] is the dayglow-corrected image (dayglow subtracked from the input image)
            - imgs['shweight'] is the weight if each pixel after the final iteration
    norms _ tuple, optional
        A tuple containing the residual and model norms. Only is returnNorms is True
    '''

    # Add temporal dimension if missing
    if len(imgs.sizes)==2: imgs = imgs.expand_dims('date')

    time=(imgs['date'].values-imgs['date'].values[0])/ np.timedelta64(1, 'm')

    glat = imgs['glat'].stack(z=('row','col')).values
    d = imgs['dgimg'].stack(z=('row','col')).values

    # Normalize on dayglow noise
    d = d/imgs['dgsigma'].stack(z=('row','col')).values

    sslat, sslon = map(np.ravel, subsol(imgs['date'].values))
    phi = np.deg2rad((imgs['glon'].stack(z=('row','col')).values - sslon[:,None] + 180) % 360 - 180)

    n_t = glat.shape[0]
    n_s = glat.shape[1]

    # Temporal knots
    if tKnotSep==None:
        knots = np.linspace(time[0], time[-1], 2)
    else:
        knots = np.linspace(time[0], time[-1], int(np.round(time[-1]/tKnotSep)+1))
    knots = np.r_[np.repeat(knots[0],tOrder),knots, np.repeat(knots[-1],tOrder)]

    # Number of control points
    n_cp = len(knots)-tOrder-1

    # Temporal design matix
    M = BSpline(knots, np.eye(n_cp), tOrder)(time)

    # Iterative (few iterations)
    skeys = sh.SHkeys(Nsh, Msh).Mge(1).MleN().NminusMeven()
    ckeys = sh.SHkeys(Nsh, Msh).MleN().NminusMeven()

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
        G = G/imgs['dgsigma'].stack(z=('row','col')).values[i,:][:,None]
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

    # Spatial weights
    grid,mltres=sdarngrid(5,5,minlat//5*5) # Equal area grid
    ws = np.full(glat.shape,np.nan)
    for i in range(len(glat)):
        gbin = bin_number(grid,glat[i,ind[i]],glon[i,ind[i]]/15)
        count=np.full(grid.shape[1],0)
        un,count_un=np.unique(gbin,return_counts=True)
        count[un]=count_un
        ws[i,ind[i]]=1/count[gbin]

    # Make everything flat
    d_s = d.flatten()
    ind = ind.flatten()
    ws = ws.flatten()
    w = imgs['dgweight'].stack(z=('row','col')).values.flatten()

    # Damping
    damping = dampingVal*np.ones(G_s.shape[1])
    R = np.diag(damping)

    diff = 1e10*stop
    iteration = 0
    m = None
    while (diff>stop)&(iteration < 100):
        print('Iteration:',iteration)
        # Solve for spline amplitudes
        m_s = np.linalg.lstsq((G_s[ind,:]*w[ind,None]*ws[ind,None]).T@(G_s[ind,:]*w[ind,None]*ws[ind,None])+R,(G_s[ind,:]*w[ind,None]*ws[ind,None]).T@(d_s[ind]*w[ind]*ws[ind]),rcond=None)[0]

        # Retrieve B-spline smooth model paramters (coarse)
        mNew    = M@m_s.reshape((n_G, n_cp)).T
        dm=[]
        for i, tt in enumerate(time):
            dm.append(G_g[i]@mNew[i, :])

        dm=np.array(dm).squeeze()
        residuals = dm.flatten()[ind] - d.flatten()[ind]

        w[ind] = (1 - np.minimum(1,(residuals/(tukeyVal))**2))**2

        if m is not None:
            diff = np.sqrt(np.mean( (mNew-m)**2))/(1+np.sqrt(np.mean(mNew**2)))
            print('Relative change model norm:',diff)
        m = mNew
        iteration += 1

    normM = np.sqrt(np.average(m_s**2))
    normR = np.sqrt(np.average(residuals[ind]**2,weights=w[ind]))

    imgs['shmodel'] = (['date','row','col'],(dm*imgs['dgsigma'].stack(z=('row','col')).values).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['shimg'] = imgs['dgimg']-imgs['shmodel']
    imgs['shweight'] = (['date','row','col'],(w).reshape((n_t,len(imgs.row),len(imgs.col))))

    # Remove pixels outside model scope
    ind = (imgs.glat >= minlat) & imgs.bad & np.isfinite(imgs['dgsimga'])
    imgs['shmodel'] = xr.where(~ind,np.nan,imgs['shmodel'])
    imgs['shimg'] = xr.where(~ind,np.nan,imgs['shimg'])
    imgs['shweight'] = xr.where(~ind,np.nan,imgs['shweight'])

    # Add attributes
    imgs['shmodel'].attrs = {'long_name': 'SH model'}
    imgs['shimg'].attrs = {'long_name': 'SH corrected image'}
    imgs['shweight'].attrs = {'long_name': 'SH model weights'}

    if returnNorms:
        return imgs,(normR,normM)
    else:
        return imgs

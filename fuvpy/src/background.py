#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:28:25 2022

@author: aohma
"""

import numpy as np
import xarray as xr
import time as timing

from scipy.interpolate import BSpline
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from scipy.sparse import csc_array
from scipy.linalg import lstsq

from fuvpy.utils import sh
from fuvpy.utils.sunlight import subsol
from fuvpy.polplot import sdarngrid,bin_number


def makeBSmodel(imgs,inImg='img',sOrder=3,dampingVal=0,tukeyVal=5,stop=1e-3,sKnots=None,n_tKnots=2,tOrder=2,returnNorms=False):
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
    start = timing.process_time()
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
    if n_t==1:
        tKnots = np.linspace(time[0], time[-1]+1, n_tKnots)
        tOrder = 0
    else:
        tKnots = np.linspace(time[0], time[-1], n_tKnots)
        tKnots = np.r_[np.repeat(tKnots[0],tOrder),tKnots, np.repeat(tKnots[-1],tOrder)]

    # Number of control points
    n_tcp = len(tKnots)-tOrder-1
    n_scp = len(sKnots)-sOrder-1

    # Temporal design matix
    print('Building design matrix')
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
    ind = (imgs['sza'].stack(z=('row','col')).values >= 0) & (np.isfinite(imgs[inImg].stack(z=('row','col')).values)) & (fraction>sKnots[0]) & (fraction<sKnots[-1])

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

    print(timing.process_time() - start)
    # Iterativaly solve the inverse problem
    diff = 1e10
    iteration = 0
    m = None
    while (diff>stop)&(iteration < 100):
        print('Iteration:',iteration)

        # m_s = np.linalg.lstsq((G_s[ind,:]*w[ind,None]*ws[ind,None]).T@(G_s[ind,:]*w[ind,None]*ws[ind,None])+R,(G_s[ind,:]*w[ind,None]*ws[ind,None]).T@(d_s[ind]*w[ind]*ws[ind]),rcond=None)[0]
        m_s = lstsq((G_s[ind,:]*w[ind,None]*ws[ind,None]).T@(G_s[ind,:]*w[ind,None]*ws[ind,None])+R,(G_s[ind,:]*w[ind,None]*ws[ind,None]).T@(d_s[ind]*w[ind]*ws[ind]),lapack_driver='gelsy')[0]
        mNew    = M@m_s.reshape((n_scp, n_tcp)).T
        dm=[]
        for i, tt in enumerate(time):
            dm.append(G_g[i]@mNew[i, :])

        dm=np.array(dm).squeeze()
        residuals = (d_s[ind] - dm.flatten()[ind])

        if iteration<=1:
            sigma = np.full_like(d_s,np.nan)
            sigma[ind] = _noiseModel(fraction.flatten()[ind], d_s[ind],dm.flatten()[ind], w[ind]*ws[ind],sKnots)
        w[ind] = (1 - np.minimum(1,(residuals/(tukeyVal*sigma[ind]))**2))**2

        if m is not None:
            diff = np.sqrt(np.mean((mNew-m)**2))/(1+np.sqrt(np.mean(mNew**2)))
            print('Relative change model norm',diff)
        m = mNew
        iteration += 1
        print(timing.process_time() - start)

    normM = np.sqrt(np.average(m_s**2))
    normR = np.sqrt(np.average(residuals**2,weights=w[ind]))

    # Add dayglow model and corrected image to the Dataset
    imgs['dgmodel'] = (['date','row','col'],(dm).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['dgimg'] = imgs[inImg]-imgs['dgmodel']
    imgs['dgweight'] = (['date','row','col'],(w).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['dgsigma'] = (['date','row','col'],(sigma).reshape((n_t,len(imgs.row),len(imgs.col))))

    # Remove pixels outside model scope
    ind = (imgs.sza>=0)
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

def makeBSmodelTest(imgs,inImg='img',dampingVal=0,tukeyVal=5,stop=1e-3,sKnots=None,sOrder=3,n_tKnots=2,tOrder=2,returnNorms=False):
    '''
    Function to model the FUV dayglow and subtract it from the input image. Testing scipy.sparse

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
    start = timing.process_time()
    # Spatial knots and viewing angle correction
    if imgs['id'] in ['WIC','SI12','SI13','UVI']:
        fraction = np.cos(np.deg2rad(imgs['sza'].stack(z=('date','row','col')).values))/np.cos(np.deg2rad(imgs['dza'].stack(z=('date','row','col')).values))
        if sKnots is None: sKnots = [-3.5,-0.25,0,0.25,1.5,3.5]
    elif imgs['id'] == 'VIS':
        fraction = np.cos(np.deg2rad(imgs['sza'].stack(z=('date','row','col')).values))
        if sKnots is None: sKnots= [-1,-0.1,0,0.1,0.4,1]
    sKnots = np.r_[np.repeat(sKnots[0],sOrder),sKnots, np.repeat(sKnots[-1],sOrder)]

    # Minuets since first image
    time=(imgs['date'].values-imgs['date'].values[0])/ np.timedelta64(1, 'm')

    # temporal and spatial size
    n_t = len(imgs.date)
    n_r = len(imgs.row)
    n_c = len(imgs.col)

    # Index of data to use in model
    ind = (imgs['sza'].stack(z=('date','row','col')).values >= 0) & (np.isfinite(imgs[inImg].stack(z=('date','row','col')).values)) & (fraction>sKnots[0]) & (fraction<sKnots[-1])

    # Temporal knots
    if n_t==1:
        tKnots = np.linspace(time[0], time[-1]+1, n_tKnots)
        tOrder = 0
    else:
        tKnots = np.linspace(time[0], time[-1], n_tKnots)
        tKnots = np.r_[np.repeat(tKnots[0],tOrder),tKnots, np.repeat(tKnots[-1],tOrder)]

    # Number of control points
    n_tcp = len(tKnots)-tOrder-1
    n_scp = len(sKnots)-sOrder-1

    # Temporal design matrix
    Gtime = BSpline.design_matrix(time[np.arange(n_t).repeat(n_r*n_c)][ind], tKnots,tOrder)

    # Spatial design matrix
    Gfrac = BSpline.design_matrix(fraction[ind],sKnots,sOrder)

    G = Gfrac[:,np.repeat(np.arange(n_scp),n_tcp)]*Gtime[:,np.tile(np.arange(n_tcp),n_scp)]

    # Spatial weights
    ws = []
    for i in range(n_t):
        count,bin_edges,bin_number=binned_statistic(fraction[i*(n_r*n_c):(i+1)*(n_r*n_c)][ind[i*(n_r*n_c):(i+1)*(n_r*n_c)]],None,statistic=
    'count',bins=np.arange(sKnots[0],sKnots[-1]+0.1,0.1))
        ws.extend(1/np.maximum(1,count[bin_number-1]))
    ws = csc_array(ws).T

    # Make everything flat
    d = imgs[inImg].stack(z=('date','row','col')).values[ind]

    # Initiate iterative weights
    w = np.ones_like(d)
    w = csc_array(w).T


    # Damping (zeroth-order Tikhonov regularization)
    damping = dampingVal*np.ones(n_scp*n_tcp)
    R = np.diag(damping)

    print(timing.process_time() - start)
    # Iterativaly solve the inverse problem
    diff = 1e10
    iteration = 0
    dm = np.full((n_t*n_r*n_c),np.nan)
    m = None
    while (diff>stop)&(iteration < 100):
        print('Iteration:',iteration)
        mNew = lstsq((G*w*ws).T.dot(G*w*ws)+R,(G*w*ws).T.dot(d*w.toarray().squeeze()*ws.toarray().squeeze()),lapack_driver='gelsy')[0]

        dm[ind]=G.dot(mNew)
        residuals = (d - dm[ind])

        if iteration<=1:
            sigma = np.full_like(dm,np.nan)
            sigma[ind] = _noiseModel(fraction[ind], d,dm[ind], w.toarray().squeeze()*ws.toarray().squeeze(),sKnots)
        w[:] = (1 - np.minimum(1,(residuals/(tukeyVal*sigma[ind]))**2))**2

        if m is not None:
            diff = np.sqrt(np.mean((mNew-m)**2))/(1+np.sqrt(np.mean(mNew**2)))
            print('Relative change model norm',diff)
        m = mNew
        iteration += 1
        print(timing.process_time() - start)

    # normM = np.sqrt(np.average(m_s**2))
    # normR = np.sqrt(np.average(residuals**2,weights=w[ind]))


    # Add dayglow model and corrected image to the Dataset
    imgs['dgmodel'] = (['date','row','col'],dm.reshape(n_t,n_r,n_c))
    imgs['dgimg'] = imgs[inImg]-imgs['dgmodel']
    # imgs['dgweight'] = (['date','row','col'],(w).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['dgsigma'] = (['date','row','col'],(sigma).reshape((n_t,n_r,n_c)))

    # Remove pixels outside model scope
    ind = (imgs.sza>=0)
    imgs['dgmodel'] = xr.where(~ind,np.nan,imgs['dgmodel'])
    imgs['dgimg'] = xr.where(~ind,np.nan,imgs['dgimg'])
    # imgs['dgweight'] = xr.where(~ind,np.nan,imgs['dgweight'])
    # imgs['dgsigma'] = xr.where(~ind,np.nan,imgs['dgsigma'])

    # Add attributes
    imgs['dgmodel'].attrs = {'long_name': 'BS model'}
    imgs['dgimg'].attrs = {'long_name': 'BS corrected image'}
    # imgs['dgweight'].attrs = {'long_name': 'BS model weights'}
    # imgs['dgsigma'].attrs = {'long_name': 'BS model spread'}

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
    tKnots = np.linspace(time[0], time[-1], n_tKnots)
    tKnots = np.r_[np.repeat(tKnots[0],tOrder),tKnots, np.repeat(tKnots[-1],tOrder)]

    # Number of control points
    n_cp = len(tKnots)-tOrder-1

    # Temporal design matix
    M = BSpline(tKnots, np.eye(n_cp), tOrder)(time)

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
        gbin = bin_number(grid,glat[i,ind[i]],imgs['glon'].stack(z=('row','col')).values[i,ind[i]]/15)
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
    normR = np.sqrt(np.average(residuals**2,weights=w[ind]))

    imgs['shmodel'] = (['date','row','col'],(dm*imgs['dgsigma'].stack(z=('row','col')).values).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['shimg'] = imgs['dgimg']-imgs['shmodel']
    imgs['shweight'] = (['date','row','col'],(w).reshape((n_t,len(imgs.row),len(imgs.col))))

    # Remove pixels outside model scope
    ind = (imgs.glat >= minlat) & imgs.bad & np.isfinite(imgs['dgsigma'])
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


def bin_number2(grid, mlat, mlt):
    """
    Ultra-fast routine to determine the bin number of mlat and mlt
    NB HARD CODED VERSION TO USE DATA FROM BOTH HEMISPHERES

    Parameters
    ----------
    grid : 2 x N array
        Array containing the edges of the mlat-mlt bins.
        First row is mlat and second row mlt.
        grid can be constructed using either equal_area_grid or sdarngrid
    mlat : array
        Array with the mlats to bin
    mlt : array
        Array with the mlts to bin

    Returns
    -------
    bin_n : array
        Array with the bin number for each mlat-mlt pair.
        Locations outside the defined grid are set to -1 in the returned array.


    SMH 2021/04/15
    Modified by JPR 2021/10/20
    2022-05-02: JPR added nan handling
    2022-05-05: AO added better handling of values outside grid (including nans).
    """

    llat = np.unique(grid[0]) # latitude circles
    assert np.allclose(np.sort(llat) - llat, 0) # should be in sorted order automatically. If not, the algorithm will not work
    dlat = np.diff(llat)[0] # latitude step
    latbins = np.hstack(( llat, llat[-1] + dlat )) # make latitude bin edges

    bin_n = -np.ones_like(mlat).astype(int) # initiate binnumber
    ii = (np.isfinite(mlat))&(np.isfinite(mlt))&(mlat>=latbins[0])&(mlat<=latbins[-1]) # index of real values inside the grid
    mlat,mlt = mlat[ii],mlt[ii] # Reduce to only real values inside grid

    latbin_n = np.digitize(mlat, latbins) - 1 # find the latitude index for each data point

    # number of longitude bins in each latitude ring:
    nlons = np.array([len(np.unique(grid[1][grid[0] == lat])) for lat in llat])

    # normalize all longitude bins to the equatorward ring:
    _mlt = mlt * nlons[latbin_n] / nlons[17]

    # make longitude bin edges for the equatorward ring:
    llon = np.unique(grid[1][grid[0] == llat[17]])
    dlon = np.diff(llon)[0]
    lonbins = np.hstack((llon, llon[-1] + dlon)) # make longitude bin edges
    lonbin_n = np.digitize(_mlt, lonbins) - 1 # find the longitude bin

    # map from 2D bin numbers to 1D by adding the number of bins in each row equatorward:
    bin_n[ii] = lonbin_n + np.cumsum(np.hstack((0, nlons)))[latbin_n]

    return bin_n

def makeSBSmodel(imgs):
    '''
    Testing Spherical B-spline modelling

    Parameters
    ----------
    imgs : TYPE
        DESCRIPTION.

    Returns
    -------
    imgs : TYPE
        DESCRIPTION.

    '''

    sslat, sslon = map(np.ravel, subsol(imgs['date'].values))


    r = 6500
    theta = np.deg2rad(90-imgs.glat.values)
    phi = np.deg2rad((imgs['glon'].values - sslon[:,None,None] + 180) % 360 - 180)

    xGEO = r * np.sin(theta) * np.cos(phi)
    yGEO = r * np.sin(theta) * np.sin(phi)
    zGEO = r * np.cos(theta)

    slat = np.full_like(theta,np.nan)
    slon = np.full_like(theta,np.nan)
    for i in range(len(imgs.date)):
        conv = np.array([[-np.sin(np.deg2rad(sslat[i])),0,np.cos(np.deg2rad(sslat[i]))],
                         [0,-1,0],
                         [np.cos(np.deg2rad(sslat[i])),0,np.sin(np.deg2rad(sslat[i]))]])
        xyz = conv @ np.vstack((xGEO[i,:,:].flatten(),yGEO[i,:,:].flatten(),zGEO[i,:,:].flatten()))

        slat[i,:,:] = 90-np.rad2deg(np.arccos(xyz[2,:]/r)).reshape(xGEO[i,:,:].shape)
        np.seterr(invalid='ignore', divide='ignore')
        slon[i,:,:] = np.rad2deg(((np.arctan2(xyz[1,:], xyz[0,:])*180/np.pi) % 360)/180*np.pi).reshape(xGEO[i,:,:].shape)

    imgs['slat'] = (['date','row','col'],slat)
    imgs['slon'] = (['date','row','col'],slon)

    # Viewving angle correction
    background=np.nanmedian(imgs['img'].values[(imgs['bad'].values)&(imgs['sza'].values>100|np.isnan(imgs['sza'].values))])
    dzacorr=0.15
    imgs['cimg'] = (imgs['img']-background)*np.cos(np.deg2rad(imgs['dza']))/np.exp(dzacorr*(1. - 1/np.cos(np.deg2rad(imgs['dza']))))

    latOrder = 3
    lonOrder = 3
    tOrder = 2
    tKnotSep=None

    dzalim=75
    minlat=-90
    dampingVal =1e-2
    tukeyVal = 3
    stop = 1e-3

    # Coordinates and data
    time=(imgs['date'].to_series().values-imgs['date'].values[0])/ np.timedelta64(1, 'm')
    slat = imgs['slat'].stack(z=('row','col')).values
    slon = imgs['slon'].stack(z=('row','col')).values
    d = imgs['cimg'].stack(z=('row','col')).values

    ind = (imgs['sza'].stack(z=('row','col')).values >= 0) & (imgs['dza'].stack(z=('row','col')).values <= dzalim) & (imgs['glat'].stack(z=('row','col')).values >= minlat) & (np.isfinite(imgs['cimg'].stack(z=('row','col')).values)) & imgs['bad'].stack(z=('row','col')).values[None,:]

    # temporal and spatial size
    n_t = slat.shape[0]

    # colat knots
    latKnots = [-90,-10,0,10,50,90]
    latKnots = np.r_[np.repeat(latKnots[0],latOrder),latKnots, np.repeat(latKnots[-1],latOrder)]
    n_latcp = len(latKnots)-latOrder-1 # Number of control points

    # lon knots full
    lonKnots = np.array([0,90,180,270])
    lonKnots = np.r_[lonKnots-360,lonKnots,lonKnots+360]
    lonKnots = np.r_[np.repeat(lonKnots[0],lonOrder),lonKnots, np.repeat(lonKnots[-1],lonOrder)]
    n_loncp = len(lonKnots)-lonOrder-1 # Number of control points

    # Temporal knots
    if tKnotSep==None:
        tKnots = np.linspace(time[0], time[-1], 2)
    else:
        tKnots = np.linspace(time[0], time[-1], int(np.round(time[-1]/tKnotSep)+1))
    tKnots = np.r_[np.repeat(tKnots[0],tOrder),tKnots, np.repeat(tKnots[-1],tOrder)]
    n_tcp = len(tKnots)-tOrder-1 # Number of control points

    # Temporal design matix
    M = BSpline(tKnots, np.eye(n_tcp), tOrder)(time)

    print('Building design matrix')

    G_g=[]
    G_s=[]
    for t in range(n_t):
        # Colat design matix
        n_s = np.sum(ind[t,:]) # Spatial data points at time t
        Glat = BSpline(latKnots, np.eye(n_latcp), latOrder)(slat[t,ind[t,:]])
        Gtemp = BSpline(lonKnots, np.eye(n_loncp), lonOrder)(slon[t,ind[t,:]])
        Glon = Gtemp[:,4:8].copy()+Gtemp[:,8:12].copy()

        G = np.tile(Glon,(1,n_latcp))*np.repeat(Glat,4,axis=1)
        G_t = np.tile(G,(1,n_tcp))*np.repeat(M[t,:],n_latcp*4)[None,:]

        G = np.tile(Glon,(1,n_latcp))*np.repeat(Glat,4,axis=1)
        G_t = np.repeat(G,n_tcp,axis=1)*np.tile(M[t,:],n_latcp*4)[None,:]

        G_g.append(G)
        G_s.append(G_t)
    G_s=np.vstack(G_s)


    # Spatial weights
    grid,mltres=sdarngrid(5,5,-90) # Equal area grid
    ws = np.full(slat.shape,np.nan)
    for i in range(len(slat)):
        sbin = bin_number2(grid,slat[i,ind[i]],slon[i,ind[i]]/15)
        count=np.full(grid.shape[1],0)
        un,count_un=np.unique(sbin,return_counts=True)
        count[un]=count_un
        ws[i,ind[i]]=1/count[sbin]

    # Make everything flat
    d_s = d[ind]
    ws = ws[ind]
    w = np.ones_like(d_s)

    # Damping (zeroth-order Tikhonov regularization)
    damping = dampingVal**np.ones(G_s.shape[1])
    R = np.diag(damping)

    latMax = np.arange(-90,90+0.1,0.1)[np.argmax(BSpline(latKnots, np.eye(n_latcp), latOrder)(np.arange(-90,90+0.1,0.1)),axis=0)]
    latLambda = 1/np.cos(np.deg2rad(latMax))
    latLambda[[0,-1]]=1e12

    L = np.array([[-1,1,0,0],[0,-1,1,0],[0,0,-1,1],[1,0,0,-1]])
    LTL = np.zeros((4*n_tcp*n_latcp,4*n_tcp*n_latcp))
    for i in range(n_latcp):
        for t in range(n_tcp):
            LTL[t+(i*4*n_tcp):t+4*n_tcp+(i*4*n_tcp):n_tcp,t+(i*4*n_tcp):t+4*n_tcp+(i*4*n_tcp):n_tcp] = L.T@L
    R = dampingVal*LTL
    # R = dampingVal*np.repeat(latLambda,4*n_tcp)[:,None]*LTL

    # Iterativaly solve the inverse problem
    diff = 1e10
    iteration = 0
    m = None


    while (diff>stop)&(iteration < 100):
        print('Iteration:',iteration)

        m_s = np.linalg.lstsq((G_s*w[:,None]*ws[:,None]).T@(G_s*w[:,None]*ws[:,None])+R,(G_s*w[:,None]*ws[:,None]).T@(d_s*w*ws),rcond=None)[0]

        mNew    = M@m_s.reshape((n_latcp*4, n_tcp)).T
        dm=[]
        for i, tt in enumerate(time):
            dm.append((G_g[i]@mNew[i, :]).flatten())
        dm = np.hstack(dm)
        residuals = (d_s - dm)

        # if iteration==0:
        #     sigma = noise(slat[ind], d_s,dm,w)
        # w = (1 - np.minimum(1,(residuals/(tukeyVal*sigma))**2))**2

        rmse = np.sqrt(np.average(residuals**2)     )
        w = (1 - np.minimum(1,(residuals/(tukeyVal*rmse))**2))**2

        if m is not None:
            diff = np.sqrt(np.mean((mNew-m)**2))/(1+np.sqrt(np.mean(mNew**2)))
            print('Relative change model norm',diff)
        m = mNew
        iteration += 1

    normM = np.sqrt(np.average(m_s**2))
    normR = np.sqrt(np.average(residuals**2,weights=w))

    # Add dayglow model and corrected image to the Dataset
    model = np.full_like(d,np.nan)
    model[ind]=dm
    imgs['dgmodel'] = (['date','row','col'],model.reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['dgimg'] = imgs['cimg']-imgs['dgmodel']


    return imgs

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:28:36 2022

@author: aohma
"""

import numpy as np
import pandas as pd
import xarray as xr

from scipy.interpolate import BSpline
from scipy.linalg import lstsq
from scipy.sparse import csc_array


def detect_boundaries(imgs,**kwargs):
    '''
    A function to identify auroral boundaries along latitudinal meridians

    Parameters
    ----------
    imgs : xarray.Dataset
        Dataset with the FUV images.
    inImg : str, optional
        Name of the image to use in the determination. The default is 'shimg'.
    lims : array_like, optional
        Count limits to detect boundaries. Default is np.arange(50,201,5)
    sigma : float, optional
        Distance in km from points on the meridian a pixel can be to be included in the median intensity profile. Default is 300 km.
    height : float, optional
        Assumed height of the emissions in km. Default is 130 
    clat_ev : array_like, optional
        Evalutation points of the latitudinal intensity profiles. Default is np.arange(0.5,50.5,0.5)
    mlt_ev : array_like, optional
        Which mlts to evaluate the latitudinal profiles. Default is np.arange(0.5,24,1)

    Returns
    -------
    xarray.Dataset
        A Dataset containing the identified boundaries.
    '''

    # Add temporal dimension if missing
    if len(imgs.sizes)==2: imgs = imgs.expand_dims('date')

    # Set keyword arguments to input or default values    
    inImg = kwargs.pop('inImg') if 'inImg' in kwargs.keys() else 'shimg'
    lims = kwargs.pop('lims') if 'lims' in kwargs.keys() else np.arange(50,201,5)
    sigma = kwargs.pop('sigma') if 'sigma' in kwargs.keys() else 300
    height = kwargs.pop('height') if 'height' in kwargs.keys() else 130
    clat_ev = kwargs.pop('clat_ev') if 'clat_ev' in kwargs.keys() else np.arange(0.5,50.5,0.5)
    mlt_ev = kwargs.pop('mlt_ev') if 'mlt_ev' in kwargs.keys() else np.arange(0.5,24,1)

    # Constants    
    R_E = 6371 # Earth radius (km)
    R_I = R_E+height # Assumed emission radius (km)
    km_per_lat = np.pi*R_I/180
    
    # Evaluation coordinates in cartesian projection
    r_ev = km_per_lat*(np.abs(clat_ev))
    a_ev = (mlt_ev- 6.)/12.*np.pi
    x_ev =  r_ev[:,None]*np.cos(a_ev[None,:])
    y_ev =  r_ev[:,None]*np.sin(a_ev[None,:])
    
    dfs=[]
    for t in range(len(imgs.date)):
        img = imgs.isel(date=t)
    
        # Data coordinates in cartesian projection
        r = km_per_lat*(90. - np.abs(img['mlat'].values))
        a = (img['mlt'].values - 6.)/12.*np.pi
        x =  r*np.cos(a)
        y =  r*np.sin(a)
        d = img[inImg].values
    
        # Make latitudinal intensity profiles
        d_ev = np.full_like(x_ev,np.nan)
        for i in range(len(clat_ev)):
            for j in range(len(mlt_ev)):
                ind = np.sqrt((x_ev[i,j]-x)**2+(y_ev[i,j]-y)**2)<sigma
                if np.sum(ind)>0: # non-zero weights
                    if (r_ev[i]>np.min(r[ind]))&(r_ev[i]<np.max(r[ind])): # Only between of pixels with non-zero weights
                        d_ev[i,j]=np.median(d[ind])

        # Make dataset with meridian intensity profiles
        ds = xr.Dataset(coords={'clat':clat_ev,'mlt':mlt_ev,'lim':lims})
        ds['d'] = (['clat','mlt'],d_ev)
        
        # Set values outside outer ring to nan
        ds['d'] = xr.where(ds['clat']>40+10*np.cos(np.pi*ds['mlt']/12),np.nan,ds['d'])
        
        ds['above'] = (ds['d']>ds['lim']).astype(float) 
        ds['above'] = xr.where(np.isnan(ds['d']),np.nan,ds['above'])
        
        diff = ds['above'].diff(dim='clat')
        
        # Find first above
        mask = diff==1
        ds['firstAbove'] = xr.where(mask.any(dim='clat'), mask.argmax(dim='clat'), np.nan)+1
        
        # Find last above
        mask = diff==-1
        val = len(diff.clat) - mask.isel(clat=slice(None,None,-1)).argmax(dim='clat') - 1
        ds['lastAbove']= xr.where(mask.any(dim='clat'), val, np.nan)
    
        # Identify poleward boundaries
        ind = ds['firstAbove'].stack(z=('mlt','lim'))[np.isfinite(ds['firstAbove'].stack(z=('mlt','lim')))].astype(int)
        
        df = ind.to_dataframe().reset_index()
        df['clatBelow'] = ds.isel(clat=ind-1)['clat'].values
        df['clatAbove'] = ds.isel(clat=ind)['clat'].values
        
        df = pd.merge(df,ds['d'].to_dataframe().reset_index(),left_on=['clatBelow','mlt'],right_on=['clat','mlt'])
        df = df.drop(columns=('clat')).rename(columns={'d':'dBelow'})
        
        df = pd.merge(df,ds['d'].to_dataframe().reset_index(),left_on=['clatAbove','mlt'],right_on=['clat','mlt'])
        df = df.drop(columns=('clat')).rename(columns={'d':'dAbove'})
        df['pb'] = np.average(df[['clatBelow','clatAbove']],weights=abs(df[['dAbove','dBelow']]-df['lim'].values[:,None]),axis=1) 
        df['date']= img.date.values
        df_pb = df[['date','mlt','lim','pb']].set_index(['date','mlt','lim'])    
        
        # Identify equatorward boundaries
        ind = ds['lastAbove'].stack(z=('mlt','lim'))[np.isfinite(ds['lastAbove'].stack(z=('mlt','lim')))].astype(int)
        
        df = ind.to_dataframe().reset_index()
        df['clatAbove'] = ds.isel(clat=ind)['clat'].values
        df['clatBelow'] = ds.isel(clat=ind+1)['clat'].values
        
        df = pd.merge(df,ds['d'].to_dataframe().reset_index(),left_on=['clatAbove','mlt'],right_on=['clat','mlt'])
        df = df.drop(columns=('clat')).rename(columns={'d':'dAbove'})
        
        df = pd.merge(df,ds['d'].to_dataframe().reset_index(),left_on=['clatBelow','mlt'],right_on=['clat','mlt'])
        df = df.drop(columns=('clat')).rename(columns={'d':'dBelow'})
        df['eb'] = np.average(df[['clatAbove','clatBelow']],weights=abs(df[['dBelow','dAbove']]-df['lim'].values[:,None]),axis=1) 
        df['date']= img.date.values
        df_eb = df[['date','mlt','lim','eb']].set_index(['date','mlt','lim'])  
    
        dfs.append(pd.merge(df_pb,df_eb,left_index=True,right_index=True,how='outer'))
    
    df = pd.concat(dfs)
    df[['pb','eb']]=90-df[['pb','eb']]

    ds = df.to_xarray()

    # Add attributes
    ds['mlt'].attrs = {'long_name': 'Magnetic local time','unit':'hrs'}
    ds['pb'].attrs = {'long_name': 'Poleward boundary','unit':'deg'}
    ds['eb'].attrs = {'long_name': 'Equatorward boundary','unit':'deg'}
    return ds

def boundarymodel_F(ds,**kwargs):
    '''
    Function to make a spatiotemporal Fourier model of auroral boundaries.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with initial boundaries.
    stop : float, optional
        When to stop the iterations. Default is 0.001
    dampingValE : float, optional
        Damping value (Tikhonov regularization) for the equatorward boundary.
        Default is 2e0
    dampingValP : float, optional
        Damping value (Tikhonov regulatization) for the poleward boundary.
        Default is 2e1
    n_termsE : int, optional
        Number of radial Fourier terms equatorward boundary
    n_termsP : int, optional
        Number of radial Fourier terms poleward boundary
    order : int, optional
        Order of the temporal B-spline. Default is 3
    knotSep : int, optional
        Approximate knot separation in minutes. Default is 10.

    Returns
    -------
    xarray.Dataset
        Dataset with model boundaries.
    '''

    # Set keyword arguments to input or default values
    stop = kwargs.pop('stop') if 'stop' in kwargs.keys() else 1e-3
    Leb = kwargs.pop('Leb') if 'Leb' in kwargs.keys() else 0
    Lpb = kwargs.pop('Lpb') if 'Lpb' in kwargs.keys() else 0
    tLeb = kwargs.pop('tLeb') if 'tLeb' in kwargs.keys() else 0
    tLpb = kwargs.pop('tLpb') if 'tLpb' in kwargs.keys() else 0
    tOrder = kwargs.pop('tOrder') if 'tOrder' in kwargs.keys() else 3
    tKnotSep = kwargs.pop('tKnotSep') if 'tKnotSep' in kwargs.keys() else 10
    n_terms_eb = kwargs.pop('n_terms_eb') if 'n_terms_eb' in kwargs.keys() else 3
    n_terms_pb = kwargs.pop('n_terms_pb') if 'n_terms_pb' in kwargs.keys() else 6
    max_iter = kwargs.pop('max_iter') if 'max_iter' in kwargs.keys() else 50

    for key in kwargs: print(f'Warning: {key} is not a valid keyword argument')

    # Constants
    mu0 = 4e-7*np.pi # Vacuum magnetic permeability
    M_E = 8.05e22 # Earth's magnetic dipole moment
    R_E = 6371e3 # Earth radii
    height = 130e3
    R_I = R_E + height # Radius of ionosphere  

    duration = (ds.date[-1]-ds.date[0])/ np.timedelta64(1, 'm')
    time=(ds.date-ds.date[0]).values/ np.timedelta64(1, 'm')
    mlt = np.tile(ds.mlt.values,len(ds.lim))
    n_t = len(ds.date)
    n_mlt = len(mlt)

    ## Eq boundary model
    theta_eb = np.deg2rad(90-ds['eb'].stack(z=('lim','mlt')).values)

    # Temporal knots
    tKnots = np.arange(0,duration+tKnotSep,tKnotSep)
    tKnots = np.r_[np.repeat(tKnots[0],tOrder),tKnots, np.repeat(tKnots[-1],tOrder)]

    # Number of control points
    n_cp = len(tKnots)-tOrder-1

    # Temporal design matix
    M = BSpline(tKnots, np.eye(n_cp), tOrder)(time)

    # Data kernel
    G=[]
    for i in range(n_mlt):
        phi = np.deg2rad(15*mlt[i])
        terms=[1]
        for tt in range(1,n_terms_eb): terms.extend([np.cos(tt*phi),np.sin(tt*phi)])
        G.append(terms)
    G = np.array(G)
    n_G = np.shape(G)[1]
    for i in range(n_t):
        G_t = np.zeros((n_mlt, n_G*n_cp))
        for j in range(n_cp):
            G_t[:, np.arange(j, n_G*n_cp, n_cp)] = G*M[i, j]

        if i == 0:
            G_s = G_t
        else:
            G_s = np.vstack((G_s, G_t))

    gtg_mag = np.median((G_s.T.dot(G_s)).diagonal())

    # Data
    d_s = theta_eb.flatten()
    ind = np.isfinite(d_s)
 
    # L0 regularization higher order terms
    damping = np.ones(G_s.shape[1])
    damping[:3*n_cp]=0

    # L1 regularization
    L = np.hstack((-np.identity(n_cp-1),np.zeros((n_cp-1,1))))+np.hstack((np.zeros((n_cp-1,1)),np.identity(n_cp-1)))
    LTL = np.zeros((n_cp*n_G,n_cp*n_G))
    for i in range(n_G): LTL[i*n_cp:(i+1)*n_cp,i*n_cp:(i+1)*n_cp] = L.T@L

    R = tLeb*gtg_mag*LTL + Leb*gtg_mag*np.diag(damping)
    
    # Iteratively estimation of model parameters
    diff = 10000
    w = np.ones(d_s[ind].shape)
    m = None
    iteration = 0
    while (diff > stop)&(iteration<max_iter):
        ms = np.linalg.inv((G_s[ind,:]*w[:,None]).T@(G_s[ind,:]*w[:,None])+R)@(G_s[ind,:]*w[:,None]).T@(d_s[ind]*w)
        ms = ms.reshape((n_G, n_cp)).T

        mNew    = M@ms
        mtau=[]
        for i, tt in enumerate(time):
            mtau.append(G@mNew[i, :])
        mtau=np.array(mtau).squeeze()

        residuals = mtau.flatten()[ind] - d_s[ind]
        rmse = np.sqrt(np.average(residuals**2,weights=w))

        # Change to Tukey?
        weights = 1.5*rmse/np.abs(residuals)
        weights[weights > 1] = 1.
        w = weights
        if m is not None:
            diff = np.sqrt(np.mean((mNew - m)**2))/(1+np.sqrt(np.mean(mNew**2)))

        m = mNew
        iteration += 1

    # Data kernel evaluation
    mlt_eval      = np.linspace(0,24,24*10+1)
    G=[]
    for i in range(len(mlt_eval)):
        phi = np.deg2rad(15*mlt_eval[i])
        terms=[1]
        for tt in range(1,n_terms_eb): terms.extend([np.cos(tt*phi),np.sin(tt*phi)])
        G.append(terms)
    G = np.array(G)
    n_G = np.shape(G)[1]

    tau_eb=[]
    for i, tt in enumerate(time):
        tau_eb.append(G@m[i, :])
    tau_eb=np.array(tau_eb).squeeze()

    ## DERIVATIVE

    # df/dt 
    dt =(tKnots[tOrder+1:-1]-tKnots[1:-tOrder-1])
    dms_dt = (ms[1:,:] - ms[:-1,:]) * tOrder / dt[:,None]
    knots_dt=tKnots[1:-1]
    n_cp_dt=len(knots_dt)-(tOrder-1)-1

    M_dt = BSpline(knots_dt, np.eye(n_cp_dt), tOrder-1)(time)
    dm_dt      = M_dt@dms_dt

    dtau_dt_eb=[]
    for i, tt in enumerate(time):
        dtau_dt_eb.append(G@dm_dt[i, :])
    dtau_dt_eb=np.array(dtau_dt_eb).squeeze()

    # df/d(phi)
    G_dphi=[]
    for i in range(len(mlt_eval)):
        phi = np.deg2rad(15*mlt_eval[i])
        terms=[0]
        for tt in range(1,n_terms_eb): terms.extend([-tt*np.sin(tt*phi),tt*np.cos(tt*phi)])
        G_dphi.append(terms)
    G_dphi = np.array(G_dphi)
    dtau_dphi_eb = []
    for i, tt in enumerate(time):
        dtau_dphi_eb.append(G_dphi@m[i, :])
    dtau_dphi_eb=np.array(dtau_dphi_eb).squeeze()


    R_I = 6500e3
    dphi_dt = -( dtau_dt_eb*dtau_dphi_eb)/(np.sin(tau_eb)**2+(dtau_dphi_eb)**2)
    dtheta_dt = dtau_dt_eb*np.sin(tau_eb)**2/(np.sin(tau_eb)**2+(dtau_dphi_eb)**2)

    # Boundary velocity
    u_phi = R_I*np.sin(tau_eb)*dphi_dt/60
    u_theta = R_I*dtheta_dt/60
    
    # Total flux inside EB
    dT = (mu0*M_E)/(4*np.pi*R_I) * (np.sin(tau_eb)**2)
    dT_dt = (mu0*M_E)/(4*np.pi*R_I) * np.sin(2*tau_eb)*dtau_dt_eb/60

    ## Poleward boundary model
    theta_pb = np.deg2rad(90 - ds['pb'].stack(z=('lim','mlt')).values)

    theta_pb1 = theta_pb/mtau

    # Temporal design matix
    M = BSpline(tKnots, np.eye(n_cp), tOrder)(time)

    # Data kernel
    G=[]
    for i in range(n_mlt):
        phi = np.deg2rad(15*mlt[i])
        terms=[1]
        for tt in range(1,n_terms_pb): terms.extend([np.cos(tt*phi),np.sin(tt*phi)])
        G.append(terms)
    G = np.array(G)
    n_G = np.shape(G)[1]
    for i in range(n_t):
        G_t = np.zeros((n_mlt, n_G*n_cp))

        for j in range(n_cp):
            G_t[:, np.arange(j, n_G*n_cp, n_cp)] = G*M[i, j]

        if i == 0:
            G_s = G_t
        else:
            G_s = np.vstack((G_s, G_t))

    gtg_mag = np.median((G_s.T.dot(G_s)).diagonal()) 

    # Data
    d_s = (np.log(theta_pb1)-np.log(1-theta_pb1)).flatten()
    ind = np.isfinite(d_s)
    
    # L0 regularization higher order terms
    damping = np.ones(G_s.shape[1])
    damping[:3*n_cp]=0

    # L1 regularization
    L = np.hstack((-np.identity(n_cp-1),np.zeros((n_cp-1,1))))+np.hstack((np.zeros((n_cp-1,1)),np.identity(n_cp-1)))
    LTL = np.zeros((n_cp*n_G,n_cp*n_G))
    for i in range(n_G): LTL[i*n_cp:(i+1)*n_cp,i*n_cp:(i+1)*n_cp] = L.T@L

    R = tLpb*gtg_mag*LTL + Lpb*gtg_mag*np.diag(damping)

    # Iteratively solve the full inverse problem
    diff = 10000
    w = np.ones(d_s[ind].shape)
    m = None
    iteration = 0
    while (diff > stop)&(iteration<max_iter):
        ms = np.linalg.inv((G_s[ind,:]*w[:,None]).T@(G_s[ind,:]*w[:,None])+R)@(G_s[ind,:]*w[:,None]).T@(d_s[ind]*w)
        ms = ms.reshape((n_G, n_cp)).T

        mNew    = M@ms
        mtau=[]
        for i, tt in enumerate(time):
            mtau.append(G@mNew[i, :])
        mtau=np.array(mtau).squeeze()

        residuals = mtau.flatten()[ind] - d_s[ind]
        rmse = np.sqrt(np.average(residuals**2,weights=w))

        # Change to Tukey?
        weights = 1.5*rmse/np.abs(residuals)
        weights[weights > 1] = 1.
        w = weights
        if m is not None: diff = np.sqrt(np.mean((mNew - m)**2))/(1+np.sqrt(np.mean(mNew**2)))

        m = mNew
        iteration += 1

    # Data kernel evaluation
    G=[]
    for i in range(len(mlt_eval)):
        phi = np.deg2rad(15*mlt_eval[i])
        terms=[1]
        for tt in range(1,n_terms_pb): terms.extend([np.cos(tt*phi),np.sin(tt*phi)])
        G.append(terms)
    G = np.array(G)
    n_G = np.shape(G)[1]

    # Model boundary in double primed space
    tau2=[]
    for i, tt in enumerate(time):
        tau2.append(G@m[i, :])
    tau2=np.array(tau2).squeeze()

    # Transform to unprimed
    tau1 = 1/(1+np.exp(-1*tau2))
    tau_pb  = tau_eb*tau1


    # df/dt in double primed space
    dt =(tKnots[tOrder+1:-1]-tKnots[1:-tOrder-1])
    dms_dt = (ms[1:,:] - ms[:-1,:]) * tOrder / dt[:,None]
    knots_dt=tKnots[1:-1]
    n_cp_dt=len(knots_dt)-(tOrder-1)-1

    # Temporal design matix
    M = BSpline(knots_dt, np.eye(n_cp_dt), tOrder-1)(time)
    dm_dt = M_dt@dms_dt

    dtau2_dt=[]
    for i, tt in enumerate(time):
        dtau2_dt.append(G@dm_dt[i, :])
    dtau2_dt=np.array(dtau2_dt).squeeze()

    # df/d(phi) in double primed coordinates
    G_dt=[]
    for i in range(len(mlt_eval)):
        phi = np.deg2rad(15*mlt_eval[i])
        terms=[0]
        for tt in range(1,n_terms_pb): terms.extend([-tt*np.sin(tt*phi),tt*np.cos(tt*phi)])
        G_dt.append(terms)
    G_dt = np.array(G_dt)
    dtau2_dphi = []
    for i, tt in enumerate(time):
        dtau2_dphi.append(G_dt@m[i, :])
    dtau2_dphi=np.array(dtau2_dphi).squeeze()

    # Transform derivatives to primed
    dtau1_dt = (np.exp(-tau2)/(np.exp(-tau2)+1)**2)*(dtau2_dt)
    dtau1_dphi = (np.exp(-tau2)/(np.exp(-tau2)+1)**2)*(dtau2_dphi)

    # Transform derivatives to unprimed
    dtau_dt = tau_eb*dtau1_dt  + dtau_dt_eb*tau1
    dtau_dphi = tau_eb*dtau1_dphi + dtau_dphi_eb*tau1

    # Transform from "wrapped" ionosphere to spherical ionosphere
    dphi_dt = -( dtau_dt*dtau_dphi)/(np.sin(tau_pb)**2+(dtau_dphi)**2)
    dtheta_dt = dtau_dt*np.sin(tau_pb)**2/(np.sin(tau_pb)**2+(dtau_dphi)**2)

    # Boundary velocity
    v_phi = R_I*np.sin(tau_pb)*dphi_dt/60
    v_theta = R_I*dtheta_dt/60

    # FLUX
    dP = (mu0*M_E)/(4*np.pi*R_I) * (np.sin(tau_pb)**2)
    dP_dt = (mu0*M_E)/(4*np.pi*R_I) * np.sin(2*tau_pb)*dtau_dt/60

    # Make Dataset with modelled boundary locations and velocities
    ds2 = xr.Dataset(
    data_vars=dict(
        pb=(['date','mlt'], 90-np.rad2deg(tau_pb)),
        eb=(['date','mlt'], 90-np.rad2deg(tau_eb)),
        ve_pb=(['date','mlt'], v_phi),
        vn_pb=(['date','mlt'], -v_theta),
        ve_eb=(['date','mlt'], u_phi),
        vn_eb=(['date','mlt'], -u_theta),
        dP=(['date','mlt'], dP),
        dA=(['date','mlt'], dT-dP),
        dP_dt=(['date','mlt'], dP_dt),
        dA_dt=(['date','mlt'], dT_dt-dP_dt),
        ),
    coords=dict(
        date = ds.date,
        mlt = mlt_eval
    ),
    )

    # Add attributes
    ds2['mlt'].attrs = {'long_name': 'Magnetic local time','unit':'hrs'}
    ds2['pb'].attrs = {'long_name': 'Poleward boundary','unit':'deg'}
    ds2['eb'].attrs = {'long_name': 'Equatorward boundary','unit':'deg'}
    ds2['ve_pb'].attrs = {'long_name': '$V_E^{pb}$','unit':'m/s'}
    ds2['vn_pb'].attrs = {'long_name': '$V_N^{pb}$','unit':'m/s'}
    ds2['ve_eb'].attrs = {'long_name': '$V_E^{eb}$','unit':'m/s'}
    ds2['vn_eb'].attrs = {'long_name': '$V_N^{eb}$','unit':'m/s'}
    ds2['dP'].attrs = {'long_name': 'Polar cap flux','unit':'Wb/rad'}
    ds2['dP_dt'].attrs = {'long_name': 'Change polar cap flux','unit':'V/rad'}
    ds2['dA'].attrs = {'long_name': 'Auroral flux','unit':'Wb/rad'}
    ds2['dA_dt'].attrs = {'long_name': 'Change auroral flux','unit':'V/rad'}
    return ds2


    # Make Dataset with modelled boundary locations and velocities
    ds2 = xr.Dataset(
    data_vars=dict(
        ocb=(['date','mlt'], 90-np.rad2deg(tau_pb)),
        eqb=(['date','mlt'], 90-np.rad2deg(tau_eb)),
        v_phi=(['date','mlt'], v_phi),
        v_theta=(['date','mlt'], v_theta),
        u_phi=(['date','mlt'], u_phi),
        u_theta=(['date','mlt'], u_theta),
        ),
    coords=dict(
        date = ds.date,
        mlt = mlt_eval
    ),
    )

    # Add attributes
    ds2['mlt'].attrs = {'long_name': 'Magnetic local time','unit':'hrs'}
    ds2['ocb'].attrs = {'long_name': 'Open-closed boundary','unit':'deg'}
    ds2['eqb'].attrs = {'long_name': 'Equatorward boundary','unit':'deg'}
    ds2['v_phi'].attrs = {'long_name': '$V_\\phi$','unit':'m/s'}
    ds2['v_theta'].attrs = {'long_name': '$V_\\theta$','unit':'m/s'}
    ds2['u_phi'].attrs = {'long_name': '$U_\\phi$','unit':'m/s'}
    ds2['u_theta'].attrs = {'long_name': '$U_\\theta$','unit':'m/s'}
    return ds2


def boundarymodel_BS(ds,**kwargs):
    '''
    Function to make a spatiotemporal model of auroral boundaries using periodic B-splines.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with initial boundaries.
    stop : float, optional
        When to stop the iterations. Default is 0.001
    tLeb,sLeb,tLpb,sLeb : float, optional
        Magnitude of 1st order Tikhonov regularization in temporal and spatial direction for the equatorward boundary (eb) and poleward boundary (pb).
        Default is 0        
    tOrder : int, optinal
        Order of the temporal B-spline. Default is 3
    tKnotSep : int, optional
        Temporal knot separation in minutes. Default is 10.
    sOrder : int, optinal
        Order of the spatial B-spline. Default is 3
    sKnots_eb,sKnots_pb, optional
        Locations of the spatial knot's.
    max_iter : int, optional
        Maximum number of iterations for the model to converge. Default is 50
    resample : bool, optional
        If True, the input boundaries are randomly resampled with replacement before the model is made. Default is False

    Returns
    -------
    xarray.Dataset
        Dataset with model boundaries.
    '''

    # Set keyword arguments to input or default values
    stop = kwargs.pop('stop') if 'stop' in kwargs.keys() else 1e-3
    tLeb = kwargs.pop('tLeb') if 'tLeb' in kwargs.keys() else 0
    sLeb = kwargs.pop('sLeb') if 'sLeb' in kwargs.keys() else 0
    tLpb = kwargs.pop('tLpb') if 'tLpb' in kwargs.keys() else 0
    sLpb = kwargs.pop('sLpb') if 'sLpb' in kwargs.keys() else 0
    tOrder = kwargs.pop('tOrder') if 'tOrder' in kwargs.keys() else 3
    tKnotSep = kwargs.pop('tKnotSep') if 'tKnotSep' in kwargs.keys() else 10
    sOrder = kwargs.pop('sOrder') if 'sOrder' in kwargs.keys() else 3
    sKnots_eb = kwargs.pop('sKnots_eb') if 'sKnots_eb' in kwargs.keys() else np.arange(0,24,6)
    sKnots_pb = kwargs.pop('sKnots_pb') if 'sKnots_pb' in kwargs.keys() else np.array([0,2,4,6,12,18,20,22])
    max_iter = kwargs.pop('max_iter') if 'max_iter' in kwargs.keys() else 50
    resample = bool(kwargs.pop('resample')) if 'resample' in kwargs.keys() else False
    
    for key in kwargs: print(f'Warning: {key} is not a valid keyword argument')
    
    1# Constants
    mu0 = 4e-7*np.pi # Vacuum magnetic permeability
    M_E = 8.05e22 # Earth's magnetic dipole moment
    R_E = 6371e3 # Earth radii
    height = 130e3
    R_I = R_E + height # Radius of ionosphere  

    # Make pandas dataframe
    if resample:
        df = ds.to_dataframe().reset_index().sample(frac=1,replace=True)
    else:
        df = ds.to_dataframe().reset_index()

    # Start date and duration
    dateS = df['date'].min().floor(str(tKnotSep)+'min')
    dateE = df['date'].max().ceil(str(tKnotSep)+'min')
    duration = (dateE-dateS)/ np.timedelta64(1, 'm')    
    time=(df['date']-dateS)/ np.timedelta64(1, 'm')
    phi = np.deg2rad(15*df['mlt'].values)

    ## Eq boundary model
    
    # Equatorward boundary to radias then convert to primed coordinates
    theta_eb = np.deg2rad(90-df['eb'].values) 
    theta_eb1 = np.log(theta_eb)

    # Index of finite data
    ind = np.isfinite(theta_eb1)

    # Temporal knots
    tKnots = np.arange(0,duration+tKnotSep,tKnotSep)
    tKnots = np.r_[np.repeat(tKnots[0],tOrder),tKnots, np.repeat(tKnots[-1],tOrder)]

    # Number of temporal control points
    n_tcp = len(tKnots)-tOrder-1

    # Temporal design matix
    Gtime = BSpline.design_matrix(time[ind],tKnots,tOrder)

    # Spatial knots (extended)
    mltKnots = sKnots_eb
    sKnots = np.deg2rad(15*mltKnots)
    sKnots = np.r_[sKnots-2*np.pi,sKnots,sKnots+2*np.pi]
    sKnots = np.r_[np.repeat(sKnots[0],sOrder),sKnots, np.repeat(sKnots[-1],sOrder)]

    # Number of spatial control points (extended)
    n_scp = len(sKnots)-sOrder-1
    
    # Spatial design matrix (extended)
    Gphi = BSpline.design_matrix(phi[ind], sKnots, sOrder)
    
    # Spatial design matrix (periodic)
    n_pcp = len(mltKnots)
    Gphi = Gphi[:,n_pcp:2*n_pcp]+Gphi[:,2*n_pcp:3*n_pcp]
 
    # Combine to full design matrix
    G_s = Gphi[:,np.repeat(np.arange(n_pcp),n_tcp)]*Gtime[:,np.tile(np.arange(n_tcp),n_pcp)]
    
    gtg_mag = np.median((G_s.T.dot(G_s)).diagonal()) 
    
    # 1st order Tikhonov regularization in time
    tLtemp = np.hstack((-np.identity(n_tcp-1),np.zeros((n_tcp-1,1))))+np.hstack((np.zeros((n_tcp-1,1)),np.identity(n_tcp-1)))
    tL = np.zeros((n_pcp*(n_tcp-1),n_pcp*n_tcp))
    for i in range(n_pcp): tL[i*(n_tcp-1):(i+1)*(n_tcp-1),i*n_tcp:(i+1)*n_tcp] = tLtemp
    tLTL = tL.T@tL
    
    # 1st order Tikhonov regularization in mlt
    sL = []
    for i in range(n_pcp): sL.append(np.roll(np.r_[-1,1,np.repeat(0,n_pcp-2)],i))
    sL=np.array(sL)
    sLTL = np.zeros((n_pcp*n_tcp,n_pcp*n_tcp))
    for t in range(n_tcp): sLTL[t:t+n_pcp*n_tcp:n_tcp,t:t+n_pcp*n_tcp:n_tcp] = sL.T@sL
    
    # Combined regularization
    R = tLeb*gtg_mag*tLTL + sLeb*gtg_mag*sLTL
     
    # Initiate iterative weights
    w = np.ones(theta_eb1[ind].shape)
    w = csc_array(w).T
    
    # Iteratively estimation of model parameters
    diff = 10000

    m = None
    iteration = 0
    mtau = np.full_like(theta_eb1,np.nan)
    while (diff > stop)&(iteration<max_iter):
        ms = lstsq((G_s*w).T.dot(G_s*w)+R,(G_s*w).T.dot(theta_eb1[ind]*w.toarray().squeeze()),lapack_driver='gelsy')[0]
        mtau[ind]=G_s@ms

        residuals = mtau[ind] - theta_eb1[ind]
        if iteration == 0: rmse = np.sqrt(np.average(residuals**2))

        w[:] = np.minimum(1.5*rmse/np.abs(residuals),1)
        if m is not None:
            diff = np.sqrt(np.mean((ms - m)**2))/(1+np.sqrt(np.mean(ms**2)))

        m = ms
        iteration += 1

    # Temporal evaluation matrix
    time_ev=(df['date'].drop_duplicates()-dateS).values/ np.timedelta64(1, 'm')
    Gtime = BSpline.design_matrix(time_ev, tKnots,tOrder).toarray()
    
    # Spatial evaluation matrix
    phi_ev = np.arange(0,2*np.pi,2*np.pi/240)
    Gphi = BSpline.design_matrix(phi_ev, sKnots, sOrder)
    Gphi = Gphi[:,n_pcp:2*n_pcp]+Gphi[:,2*n_pcp:3*n_pcp].toarray()
   
    # Combined evaluation matrix
    G_ev = np.tile(np.repeat(Gphi,n_tcp,axis=1),(len(time_ev),1))*np.repeat(np.tile(Gtime,(1,n_pcp)),len(phi_ev),axis=0)
    
    tau1=G_ev.dot(m)

    # Transform to unprimed
    tau_eb  = np.exp(tau1)

    ## DERIVATIVE
    
    mm = m.reshape((n_pcp,n_tcp))

    # df/dt in primed
    dt  =(tKnots[tOrder+1:-1]-tKnots[1:-tOrder-1])
    dmdt = (mm[:,1:] - mm[:,:-1]) * tOrder / dt[None,:]
    dtKnots=tKnots[1:-1]
    n_dtcp=len(dtKnots)-(tOrder-1)-1
    dMdt = BSpline(dtKnots, np.eye(n_dtcp), tOrder-1)(time_ev)

    GdMdt = np.tile(np.repeat(Gphi,n_dtcp,axis=1),(len(time_ev),1))*np.repeat(np.tile(dMdt,(1,n_pcp)),len(phi_ev),axis=0)  
    dtau1dt = GdMdt @ dmdt.flatten()
    
    # df/d(phi) in primed
    mm = np.vstack((mm,mm[:1,:]))
    dp =(sKnots[sOrder+1:-1]-sKnots[1:-sOrder-1])
    dmdp = (mm[1:,:] - mm[:-1,:]) * sOrder / dp[n_pcp:2*n_pcp,None]
    dpKnots=sKnots[1:-1]
    n_dpcp=len(dpKnots)-(sOrder-1)-1

    dGdp_temp = BSpline(dpKnots, np.eye(n_dpcp), sOrder-1)(phi_ev)
    dGdp = dGdp_temp[:,n_pcp:2*n_pcp].copy()+dGdp_temp[:,2*n_pcp:3*n_pcp].copy()
    
    dGdtM = np.tile(np.repeat(dGdp,n_tcp,axis=1),(len(time_ev),1))*np.repeat(np.tile(Gtime,(1,n_pcp)),len(phi_ev),axis=0)  
    dtau1dp = dGdtM @ dmdp.flatten()

    # Transform to unprimed
    dtau_dt_eb = np.exp(tau1)*(dtau1dt)
    dtau_dp_eb = np.exp(tau1)*(dtau1dp)

    # Temporal change of phi and theta
    R_I = 6500e3
    dphi_dt = -( dtau_dt_eb*dtau_dp_eb)/(np.sin(tau_eb)**2+(dtau_dp_eb)**2)
    dtheta_dt = dtau_dt_eb*np.sin(tau_eb)**2/(np.sin(tau_eb)**2+(dtau_dp_eb)**2)

    # Boundary velocity
    u_phi = R_I*np.sin(tau_eb)*dphi_dt/60
    u_theta = R_I*dtheta_dt/60

    ## FLUX INSIDE EB
    
    # TOTAL FLUX
    dT = (mu0*M_E)/(4*np.pi*R_I) * (np.sin(tau_eb)**2)
    dT_dt = (mu0*M_E)/(4*np.pi*R_I) * np.sin(2*tau_eb)*dtau_dt_eb/60

    #%% Poleward boundary model
    theta_pb  = np.deg2rad(90-df['pb'].values)
    theta_pb1 = theta_pb/np.exp(mtau)
    theta_pb2 = np.log(theta_pb1)-np.log(1-theta_pb1)
    
    # Finite data points
    ind = np.isfinite(theta_pb2)

    # Temporal design matix
    Gtime = BSpline.design_matrix(time[ind],tKnots,tOrder)

    # Spatial knots (extended)
    mltKnots = sKnots_pb
    sKnots = np.deg2rad(15*mltKnots)
    sKnots = np.r_[sKnots-2*np.pi,sKnots,sKnots+2*np.pi]
    sKnots = np.r_[np.repeat(sKnots[0],sOrder),sKnots, np.repeat(sKnots[-1],sOrder)]

    # Number of spatial control points (extended)
    n_scp = len(sKnots)-sOrder-1
    
    # Spatial design matrix (extended)
    Gphi = BSpline.design_matrix(phi[ind],sKnots,sOrder)
    
    # Spatial design matrix (periodic)
    n_pcp = len(mltKnots)
    Gphi = Gphi[:,n_pcp:2*n_pcp]+Gphi[:,2*n_pcp:3*n_pcp]
 
    # Combine to full design matrix
    G_s = Gphi[:,np.repeat(np.arange(n_pcp),n_tcp)]*Gtime[:,np.tile(np.arange(n_tcp),n_pcp)]   
  
    gtg_mag = np.median((G_s.T.dot(G_s)).diagonal())   
  
    # 1st order Tikhonov regularization in time
    tLtemp = np.hstack((-np.identity(n_tcp-1),np.zeros((n_tcp-1,1))))+np.hstack((np.zeros((n_tcp-1,1)),np.identity(n_tcp-1)))
    tL = np.zeros((n_pcp*(n_tcp-1),n_pcp*n_tcp))
    for i in range(n_pcp): tL[i*(n_tcp-1):(i+1)*(n_tcp-1),i*n_tcp:(i+1)*n_tcp] = tLtemp
    tLTL = tL.T@tL
    
    # 1st order regularization in mlt
    sL = []
    for i in range(n_pcp): sL.append(np.roll(np.r_[-1,1,np.repeat(0,n_pcp-2)],i))
    sL=np.array(sL)
    sLTL = np.zeros((n_pcp*n_tcp,n_pcp*n_tcp))
    for t in range(n_tcp): sLTL[t:t+n_pcp*n_tcp:n_tcp,t:t+n_pcp*n_tcp:n_tcp] = sL.T@sL
    
    # Combined regularization
    R = tLpb*gtg_mag*tLTL + sLpb*gtg_mag*sLTL

    # Initiate iterative weights
    w = np.ones(theta_pb2[ind].shape)
    w = csc_array(w).T
    
    # Iteratively estimation of model parameters
    diff = 10000
    m = None
    mtau = np.full_like(theta_pb2,np.nan)
    iteration = 0
    while (diff > stop)&(iteration<max_iter):
        ms = lstsq((G_s*w).T.dot(G_s*w)+R,(G_s*w).T.dot(theta_pb2[ind]*w.toarray().squeeze()),lapack_driver='gelsy')[0]
        mtau[ind]=G_s@ms

        residuals = mtau[ind] - theta_pb2[ind]
        if iteration == 0: rmse = np.sqrt(np.average(residuals**2,weights=w.toarray().squeeze()))

        w[:] = np.minimum(1.5*rmse/np.abs(residuals),1)
        if m is not None:
            diff = np.sqrt(np.mean((ms - m)**2))/(1+np.sqrt(np.mean(ms**2)))

        m = ms
        iteration += 1

    # Temporal evaluation matrix
    M = BSpline(tKnots, np.eye(n_tcp), tOrder)(time_ev)
    
    # Spatial evaluation matrix
    G = BSpline(sKnots, np.eye(n_scp), sOrder)(phi_ev)
    G = G[:,n_pcp:2*n_pcp]+G[:,2*n_pcp:3*n_pcp]
   
    
    # Combined evaluation matrix
    G_ev = np.tile(np.repeat(G,n_tcp,axis=1),(len(time_ev),1))*np.repeat(np.tile(M,(1,n_pcp)),len(phi_ev),axis=0)
    
    tau2=G_ev@m

    # Transform to unprimed
    tau1 = 1/(1+np.exp(-1*tau2))
    tau_pb  = tau_eb*tau1

    ## Derivative
  
    mm = m.reshape((n_pcp,n_tcp))

    # df/dt in double primed
    dt  =(tKnots[tOrder+1:-1]-tKnots[1:-tOrder-1])
    dmdt = (mm[:,1:] - mm[:,:-1]) * tOrder / dt[None,:]
    dtKnots=tKnots[1:-1]
    n_dtcp=len(dtKnots)-(tOrder-1)-1
    dMdt = BSpline(dtKnots, np.eye(n_dtcp), tOrder-1)(time_ev)

    GdMdt = np.tile(np.repeat(G,n_dtcp,axis=1),(len(time_ev),1))*np.repeat(np.tile(dMdt,(1,n_pcp)),len(phi_ev),axis=0)  
    dtau2dt = GdMdt @ dmdt.flatten()
    
    # df/d(phi) in double primed
    mm = np.vstack((mm,mm[:1,:]))
    dp =(sKnots[sOrder+1:-1]-sKnots[1:-sOrder-1])
    dmdp = (mm[1:,:] - mm[:-1,:]) * sOrder / dp[n_pcp:2*n_pcp,None]
    dpKnots=sKnots[1:-1]
    n_dpcp=len(dpKnots)-(sOrder-1)-1

    dGdp_temp = BSpline(dpKnots, np.eye(n_dpcp), sOrder-1)(phi_ev)
    dGdp = dGdp_temp[:,n_pcp:2*n_pcp].copy()+dGdp_temp[:,2*n_pcp:3*n_pcp].copy()
    
    dGdtM = np.tile(np.repeat(dGdp,n_tcp,axis=1),(len(time_ev),1))*np.repeat(np.tile(M,(1,n_pcp)),len(phi_ev),axis=0)  
    dtau2dp = dGdtM @ dmdp.flatten()

    # # Transform derivatives to primed
    dtau1dt = (np.exp(-tau2)/(np.exp(-tau2)+1)**2)*(dtau2dt)
    dtau1dp = (np.exp(-tau2)/(np.exp(-tau2)+1)**2)*(dtau2dp)

    # Transform derivatives to unprimed
    dtaudt = tau_eb*dtau1dt + dtau_dt_eb*tau1
    dtaudp = tau_eb*dtau1dp + dtau_dp_eb*tau1

    # Transform from "wrapped" ionosphere to spherical ionosphere
    dphidt = -( dtaudt*dtaudp)/(np.sin(tau_pb)**2+(dtaudp)**2)
    dthetadt = dtaudt*np.sin(tau_pb)**2/(np.sin(tau_pb)**2+(dtaudp)**2)

    # Boundary velocity
    v_phi = R_I*np.sin(tau_pb)*dphidt/60
    v_theta = R_I*dthetadt/60
    

    # FLUX
    dP = (mu0*M_E)/(4*np.pi*R_I) * (np.sin(tau_pb)**2)
    dP_dt = (mu0*M_E)/(4*np.pi*R_I) * np.sin(2*tau_pb)*dtaudt/60
    
    # Reshape modelled values
    tau_eb =  tau_eb.reshape((len(time_ev),len(phi_ev)))
    tau_pb = tau_pb.reshape((len(time_ev),len(phi_ev)))
    
    u_phi =  u_phi.reshape((len(time_ev),len(phi_ev)))
    u_theta = u_theta.reshape((len(time_ev),len(phi_ev)))
    
    v_phi =  v_phi.reshape((len(time_ev),len(phi_ev)))
    v_theta = v_theta.reshape((len(time_ev),len(phi_ev)))
    
    dA = (dT-dP).reshape((len(time_ev),len(phi_ev)))
    dA_dt = (dT_dt-dP_dt).reshape((len(time_ev),len(phi_ev)))
    dP = (dP).reshape((len(time_ev),len(phi_ev)))
    dP_dt = (dP_dt).reshape((len(time_ev),len(phi_ev)))

    
    # Make Dataset with modelled boundary locations and velocities
    ds2 = xr.Dataset(
    data_vars=dict(
        pb=(['date','mlt'], 90-np.rad2deg(tau_pb)),
        eb=(['date','mlt'], 90-np.rad2deg(tau_eb)),
        ve_pb=(['date','mlt'], v_phi),
        vn_pb=(['date','mlt'], -v_theta),
        ve_eb=(['date','mlt'], u_phi),
        vn_eb=(['date','mlt'], -u_theta),
        dP=(['date','mlt'], dP),
        dA=(['date','mlt'], dA),
        dP_dt=(['date','mlt'], dP_dt),
        dA_dt=(['date','mlt'], dA_dt),
        ),
    coords=dict(
        date = df['date'].drop_duplicates().values,
        mlt = np.rad2deg(phi_ev)/15
    ),
    )

    # Add attributes
    ds2['mlt'].attrs = {'long_name': 'Magnetic local time','unit':'hrs'}
    ds2['pb'].attrs = {'long_name': 'Poleward boundary','unit':'deg'}
    ds2['eb'].attrs = {'long_name': 'Equatorward boundary','unit':'deg'}
    ds2['ve_pb'].attrs = {'long_name': '$V_E^{pb}$','unit':'m/s'}
    ds2['vn_pb'].attrs = {'long_name': '$V_N^{pb}$','unit':'m/s'}
    ds2['ve_eb'].attrs = {'long_name': '$V_E^{eb}$','unit':'m/s'}
    ds2['vn_eb'].attrs = {'long_name': '$V_N^{eb}$','unit':'m/s'}
    ds2['dP'].attrs = {'long_name': 'Polar cap flux','unit':'Wb/rad'}
    ds2['dP_dt'].attrs = {'long_name': 'Change polar cap flux','unit':'V/rad'}
    ds2['dA'].attrs = {'long_name': 'Auroral flux','unit':'Wb/rad'}
    ds2['dA_dt'].attrs = {'long_name': 'Change auroral flux','unit':'V/rad'}
    return ds2
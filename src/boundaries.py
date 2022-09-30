#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:28:36 2022

@author: aohma
"""

import numpy as np
import pandas as pd
import xarray as xr

from scipy.interpolate import BSpline,griddata
from scipy.linalg import lstsq

import matplotlib.path as path

# Optional dependencies
try:
    import ppigrf
    import apexpy
except:
    print('ppigrf and/or apexpy not found. fuvpy.calcFlux will not work.'+
          'To install ppigrf: git clone https://github.com/klaundal/ppigrf.git'+
          'To install apexpy: pip install apexpy')    

    
def findBoundaries(imgs,inImg='shimg',mltRes=24,limFactors=None,order=3,dampingVal=0):
    '''
    A simple function to identify the ocb of a FUV image.
    NEEDS CLEANING + COMMENTS

    Parameters
    ----------
    imgs : xarray.Dataset
        Dataset with the FUV images.
    countlim : float
        countlimit when identifying the ocbs. All values lower than countlim are set to zero.
    inImg : str, optional
        Name of the image to use in the determination. The default is 'shimage'.
    mltRes : int, optional
        Number of MLT bins. Default is 24 (1-wide sectors).
    limFactors : array_like, optional
        Array containing fractions to multipy with noise used as threshold.
        Default is None, (np.linspace(0.5,1.5,5) is used).
    order : int, optional
        Order of the B-spline fitting the intensity profile in each circular sector.
        Default is 3.
    damplingVal : float, optional
        Damping value (Tikhonov regularization) in the B-spline fit.
        Default is 0.

    Returns
    -------
    xarray.Dataset
        A Dataset containing the identified boundaries.
    '''

    # Expand date dimension if missing
    if len(imgs.sizes)==2: imgs = imgs.expand_dims('date')

    if limFactors is None: limFactors = np.linspace(0.5,1.5,5)
    edges = np.linspace(0,24,mltRes+1)

    # Circle in which all boundaries are assumed to be located
    colatMax = np.concatenate((np.linspace(40,30,mltRes+1)[1:-1:2],np.linspace(30,40,mltRes+1)[1:-1:2]))


    colatAll = 90-abs(imgs['mlat'].values.copy().flatten())
    dAll = imgs[inImg].values.copy().flatten()
    wdgAll = imgs['shweight'].values.copy().flatten()
    jjj = (np.isfinite(dAll))&(colatAll<40)
    av = np.average(dAll[jjj],weights=wdgAll[jjj])
    rmse = np.sqrt(np.average((dAll[jjj]-av)**2,weights=wdgAll[jjj]))

    blistTime  = []
    for t in range(len(imgs['date'])):
        print('Image:',t)
        colat = 90-abs(imgs.isel(date=t)['mlat'].values.copy().flatten())
        mlt = imgs.isel(date=t)['mlt'].values.copy().flatten()
        d = imgs.isel(date=t)[inImg].values.copy().flatten()
        wDG = imgs.isel(date=t)['shweight'].values.copy().flatten()

        blistSec=[]
        for s in range(len(edges)-1):
            ocb=[]
            eqb=[]

            colatSec = colat[(mlt>edges[s])&(mlt<edges[s+1])&(colat<40)]
            dSec = d[(mlt>edges[s])&(mlt<edges[s+1])&(colat<40)]
            wSec = wDG[(mlt>edges[s])&(mlt<edges[s+1])&(colat<40)]



            if np.nansum(wSec[colatSec<colatMax[s]])==0:
                avSec=av
            else:
                jjj = (colatSec<colatMax[s])&(np.isfinite(dSec))
                avSec = np.average(dSec[jjj],weights=wSec[jjj])

            iii = np.isfinite(colatSec)&np.isfinite(dSec)
            knots = np.linspace(0,40,21).tolist()
            knots = np.r_[np.repeat(knots[0],order),knots, np.repeat(knots[-1],order)]

            if colatSec[iii].size == 0:
                ocb.extend(len(limFactors)*[np.nan])
                eqb.extend(len(limFactors)*[np.nan])
            else:
                # Prepare the model

                # Number of control points
                n_cp = len(knots)-order-1

                # Temporal design matix
                G = BSpline(knots, np.eye(n_cp), order)(colatSec[iii])

                damping = dampingVal*np.ones(G.shape[1])
                damping = dampingVal*(np.arange(G.shape[1],0,-1)/G.shape[1])
                RR = np.diag(damping)

                # Iterative estimation of model parameters
                diff = 10000

                w = 1-wSec[iii]
                m = None
                stop=1000
                while diff > stop:
                    Gw = G*w[:, np.newaxis]
                    GTG = Gw.T.dot(Gw)
                    GTd = Gw.T.dot((w*dSec[iii])[:, np.newaxis])
                    mNew = lstsq(GTG+RR, GTd)[0]
                    residuals = G.dot(mNew).flatten() - dSec[iii]
                    rmse = np.sqrt(np.average(residuals**2))
                    weights = 1.*rmse/np.abs(residuals)
                    weights[weights > 1] = 1.
                    w = weights
                    if m is not None:
                        diff = np.sqrt(np.mean((m - mNew)**2))/(1+np.sqrt(np.mean(mNew**2)))

                    m = mNew

                ev = np.linspace(0,40,401)
                G = BSpline(knots, np.eye(n_cp), order)(ev)
                dmSec = G.dot(m).flatten()
                dmSec[ev>np.max(colatSec[iii])]=np.nan

                ## identify main peak
                ind = (ev<colatMax[s])&(ev<np.max(colatSec[iii]))


                for l in range(len(limFactors)):
                    isAbove = dmSec[ind]>avSec+limFactors[l]*rmse
                    if (isAbove==True).all()|(isAbove==False).all(): #All above or below
                        ocb.append(np.nan)
                        eqb.append(np.nan)
                    else:
                        isAbove2 = np.concatenate(([False],isAbove,[False]))

                        firstAbove = np.argwhere(np.diff(isAbove2.astype(int))== 1).flatten()
                        firstBelow = np.argwhere(np.diff(isAbove2.astype(int))==-1).flatten()

                        if firstAbove[0]==0:
                            ind1=1
                        else:
                            ind1=0

                        if firstBelow[-1]==len(isAbove):
                            ind2=-1
                        else:
                            ind2=len(firstBelow)
                        #     firstAbove=firstAbove[:-1]
                        #     firstBelow=firstBelow[:-1]
                        firstAbove=(firstAbove[ind1:ind2])
                        firstBelow=(firstBelow[ind1:ind2])

                        areaSec=[]
                        for k in range(len(firstAbove)):
                            areaSec.append(np.sum(dmSec[ind][firstAbove[k]:firstBelow[k]]))


                        if len(areaSec)==0:
                            ocb.append(np.nan)
                            eqb.append(np.nan)
                        elif ((s<6)|(s>=18))&(len(areaSec)>1):
                            k = np.argmax(areaSec)
                            if k>0:
                                ocb.append(ev[firstAbove[k-1]-1])
                                eqb.append(ev[firstBelow[k]])
                            else:
                                ocb.append(ev[firstAbove[k]-1])
                                eqb.append(ev[firstBelow[k+1]])
                        else:
                            k = np.argmax(areaSec)
                            ocb.append(ev[firstAbove[k]-1])
                            eqb.append(ev[firstBelow[k]])

            ds = xr.Dataset(
                data_vars=dict(
                    ocb=(['lim'], 90-np.array(ocb)),
                    eqb=(['lim'], 90-np.array(eqb)),
                    ),

                coords=dict(
                    lim = limFactors
                ),
                )
            ds = ds.expand_dims(date=[pd.to_datetime(imgs.date[t].values)])
            ds = ds.expand_dims(mlt=[s+0.5])
            blistSec.append(ds)

        blistTime.append(xr.concat(blistSec,dim='mlt'))

    ds = xr.concat(blistTime,dim='date')

    # Add attributes
    ds['mlt'].attrs = {'long_name': 'Magnetic local time','unit':'hrs'}
    ds['ocb'].attrs = {'long_name': 'Open-closed boundary','unit':'deg'}
    ds['eqb'].attrs = {'long_name': 'Equatorward boundary','unit':'deg'}
    return ds

def makeBoundaryModel(ds,stop=1e-3,dampingValE=2e0,dampingValP=2e1,n_termsE=3,n_termsP=6,order = 3,knotSep = 10):
    '''
    Function to make a spatiotemporal Fourier model of auroral boundaries.
    INCLUDE L-CURVE script

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
    tuple
        Tuple with model and residual norms

    '''

    time=(ds.date-ds.date[0]).values/ np.timedelta64(1, 'm')
    mlt = np.tile(ds.mlt.values,len(ds.lim))
    n_t = len(ds.date)
    n_mlt = len(mlt)

    #%% Eq boundary model
    theta_eb = np.deg2rad(90-ds['eqb'].stack(z=('lim','mlt')).values)

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

    # Data kernel
    G=[]
    for i in range(n_mlt):
        phi = np.deg2rad(15*mlt[i])
        terms=[1]
        for tt in range(1,n_termsE): terms.extend([np.cos(tt*phi),np.sin(tt*phi)])
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

    # Data
    d_s = theta_eb.flatten()
    ind = np.isfinite(d_s)

    # # Temporal Damping
    # damping = dampingValE*np.ones(G_s.shape[1])
    # damping[3*n_cp:]=10*damping[3*n_cp:]

    # R = np.diag(damping)
    
    # Temporal Damping
    damping = 100*np.ones(G_s.shape[1])
    damping[:3*n_cp]=0

    
    # TEST L1 regularization
    L = np.hstack((-np.identity(n_cp-1),np.zeros((n_cp-1,1))))+np.hstack((np.zeros((n_cp-1,1)),np.identity(n_cp-1)))
    LTL = np.zeros((n_cp*n_G,n_cp*n_G))
    for i in range(n_G): LTL[i*n_cp:(i+1)*n_cp,i*n_cp:(i+1)*n_cp] = L.T@L
    R = dampingValE*LTL + np.diag(damping)
    

    

    # Iteratively estimation of model parameters
    diff = 10000
    w = np.ones(d_s[ind].shape)
    m = None
    iteration = 0
    while (diff > stop)&(iteration<100):
        print('Iteration:',iteration)
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
            print('Relative change model norm', diff)

        m = mNew
        iteration += 1


    # Model and residual norm
    m_eb = m
    norm_eb_m = np.sqrt(np.average((L@ms).flatten()**2))
    norm_eb_r = rmse

    # Data kernel evaluation
    mlt_eval      = np.linspace(0,24,24*10+1)
    G=[]
    for i in range(len(mlt_eval)):
        phi = np.deg2rad(15*mlt_eval[i])
        terms=[1]
        for tt in range(1,n_termsE): terms.extend([np.cos(tt*phi),np.sin(tt*phi)])
        G.append(terms)
    G = np.array(G)
    n_G = np.shape(G)[1]

    tau_eb=[]
    for i, tt in enumerate(time):
        tau_eb.append(G@m[i, :])
    tau_eb=np.array(tau_eb).squeeze()

    ## DERIVATIVE

    # df/dt 
    dt =(knots[order+1:-1]-knots[1:-order-1])
    dms_dt = (ms[1:,:] - ms[:-1,:]) * order / dt[:,None]
    knots_dt=knots[1:-1]
    n_cp_dt=len(knots_dt)-(order-1)-1

    M_dt = BSpline(knots_dt, np.eye(n_cp_dt), order-1)(time)
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
        for tt in range(1,n_termsE): terms.extend([-tt*np.sin(tt*phi),tt*np.cos(tt*phi)])
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


    #%% Poleward boundary model
    theta_pb = np.deg2rad(90 - ds['ocb'].stack(z=('lim','mlt')).values)

    theta_pb1 = theta_pb/mtau

    # Temporal design matix
    M = BSpline(knots, np.eye(n_cp), order)(time)

    # Data kernel
    G=[]
    for i in range(n_mlt):
        phi = np.deg2rad(15*mlt[i])
        terms=[1]
        for tt in range(1,n_termsP): terms.extend([np.cos(tt*phi),np.sin(tt*phi)])
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

    # Data
    d_s = (np.log(theta_pb1)-np.log(1-theta_pb1)).flatten()
    ind = np.isfinite(d_s)

    # # Temporal Damping
    # damping = dampingValP*np.ones(G_s.shape[1])
    # damping[3*n_cp:]=2*damping[3*n_cp:]
    # R = np.diag(damping)
    
    # Temporal Damping # Tst with L0 and L1 regularization
    damping = 10*np.ones(G_s.shape[1])
    damping[:3*n_cp]=0

    # TEST L1 regularization
    L = np.hstack((-np.identity(n_cp-1),np.zeros((n_cp-1,1))))+np.hstack((np.zeros((n_cp-1,1)),np.identity(n_cp-1)))
    LTL = np.zeros((n_cp*n_G,n_cp*n_G))
    for i in range(n_G): LTL[i*n_cp:(i+1)*n_cp,i*n_cp:(i+1)*n_cp] = L.T@L
    R = dampingValP*LTL+np.diag(damping)

    # Iteratively estimation of model parameters
    diff = 10000
    w = np.ones(d_s[ind].shape)
    m = None
    iteration = 0
    while diff > stop:
        print('Iteration:',iteration)
        ms = np.linalg.inv((G_s[ind,:]*w[:,None]).T@(G_s[ind,:]*w[:,None])+R)@(G_s[ind,:]*w[:,None]).T@(d_s[ind]*w)
        ms = ms.reshape((n_G, n_cp)).T
        # Retrieve B-spline smooth model paramters (coarse)
        mNew    = M@ms

        mtau2=[]
        for i, tt in enumerate(time):
            mtau2.append(G@mNew[i, :])
        mtau2=np.array(mtau2).squeeze()

        residuals = mtau2.flatten()[ind] - d_s[ind]
        rmse = np.sqrt(np.average(residuals**2,weights=w))
        weights = 1.5*rmse/np.abs(residuals)
        weights[weights > 1] = 1.
        w = weights
        if m is not None:
            diff = np.sqrt(np.mean((mNew - m)**2))/(1+np.sqrt(np.mean(mNew**2)))
            print('Relative change model norm', diff)

        m = mNew
        iteration += 1

    # Model and residual norm
    m_pb = m
    norm_pb_m = np.sqrt(np.average((L@ms).flatten()**2))
    norm_pb_r = rmse

    # Data kernel evaluation
    G=[]
    for i in range(len(mlt_eval)):
        phi = np.deg2rad(15*mlt_eval[i])
        terms=[1]
        for tt in range(1,n_termsP): terms.extend([np.cos(tt*phi),np.sin(tt*phi)])
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
    dt =(knots[order+1:-1]-knots[1:-order-1])
    dms_dt = (ms[1:,:] - ms[:-1,:]) * order / dt[:,None]
    knots_dt=knots[1:-1]
    n_cp_dt=len(knots_dt)-(order-1)-1

    # Temporal design matix
    M = BSpline(knots_dt, np.eye(n_cp_dt), order-1)(time)
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
        for tt in range(1,n_termsP): terms.extend([-tt*np.sin(tt*phi),tt*np.cos(tt*phi)])
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
    return ds2,(norm_eb_m,norm_eb_r,norm_pb_m,norm_pb_r),(m_eb,m_pb)


def makeBoundaryModelBSpline(ds,stop=1e-3,eL1=0,eL2=0,pL1=0,pL2=0,tOrder = 3,tKnotSep = 10):
    '''
    Function to make a spatiotemporal model of auroral boundaries using periodic B-splines.
    Note: The periodic B-splines are presently hard coded. Update this when scipy 1.10 is released!


    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with initial boundaries.
    stop : float, optional
        When to stop the iterations. Default is 0.001
    eL1 : float, optional
        1st order Tikhonov regularization for the equatorward boundary.
        Default is 0
    eL2 : float, optional
        2nd order Tikhonov regularization for the equatorward boundary.
        Default is 0
    pL1 : float, optional
        1st order Tikhonov regularization for the poleward boundary.
        Default is 0
    pL2 : float, optional
        2nd order Tikhonov regularization for the epoleward boundary.
        Default is 0        
    tOrder : int, optinal
        Order of the temporal B-spline. Default is 3
    tKnotSep : int, optional
        Approximate temporal knot separation in minutes. Default is 10.

    Returns
    -------
    xarray.Dataset
        Dataset with model boundaries.
    tuple
        Tuple with model and residual norms

    '''

    time=(ds.date-ds.date[0]).values/ np.timedelta64(1, 'm')
    phi = np.deg2rad(15*ds.mlt.values)
    n_t = len(ds.date)
    n_mlt = len(phi)
    n_lim = len(ds.lim)

    #%% Eq boundary model
    
    # Equatorward boundary to radias then convert to primed coordinates
    theta_eb = np.deg2rad(90-ds['eqb'].stack(z=('lim','mlt')).values)
    theta_eb1 = np.log(theta_eb)

    # Temporal knots
    if tKnotSep==None:
        tKnots = np.linspace(time[0], time[-1], 2)
    else:
        tKnots = np.linspace(time[0], time[-1], int(np.round(time[-1]/tKnotSep)+1))
    tKnots = np.r_[np.repeat(tKnots[0],tOrder),tKnots, np.repeat(tKnots[-1],tOrder)]

    # Number of control points
    n_tcp = len(tKnots)-tOrder-1

    # Temporal design matix
    M = BSpline(tKnots, np.eye(n_tcp), tOrder)(time)

    # Spatial design matrix
    sOrder = 3
    sKnots = np.deg2rad(15*np.array([0,6,12,18]))
    sKnots = np.r_[sKnots-2*np.pi,sKnots,sKnots+2*np.pi]
    sKnots = np.r_[np.repeat(sKnots[0],sOrder),sKnots, np.repeat(sKnots[-1],sOrder)]

    n_scp = len(sKnots)-sOrder-1
    Gtemp = BSpline(sKnots, np.eye(n_scp), sOrder)(phi)
    G = Gtemp[:,4:8].copy()+Gtemp[:,8:12].copy()
    G = np.tile(G,(n_lim,1))
    n_G = np.shape(G)[1]
    for i in range(n_t):
        G_t = np.zeros((n_mlt*n_lim, n_G*n_tcp))
        for j in range(n_tcp):
            G_t[:, np.arange(j, n_G*n_tcp, n_tcp)] = G*M[i, j]

        if i == 0:
            G_s = G_t
        else:
            G_s = np.vstack((G_s, G_t))

    # Data
    d_s = theta_eb1.flatten()
    ind = np.isfinite(d_s)
    
    # 1 order regularization
    L = np.hstack((-np.identity(n_tcp-1),np.zeros((n_tcp-1,1))))+np.hstack((np.zeros((n_tcp-1,1)),np.identity(n_tcp-1)))
    LTL = np.zeros((n_tcp*n_G,n_tcp*n_G))
    for i in range(n_G): LTL[i*n_tcp:(i+1)*n_tcp,i*n_tcp:(i+1)*n_tcp] = L.T@L
    
    # 2nd order regularization
    L2 = np.hstack((np.identity(n_tcp-2),np.zeros((n_tcp-2,2))))-2*np.hstack((np.zeros((n_tcp-2,1)),np.identity(n_tcp-2),np.zeros((n_tcp-2,1))))+np.hstack((np.zeros((n_tcp-2,2)),np.identity(n_tcp-2)))
    L2TL2 = np.zeros((n_tcp*n_G,n_tcp*n_G))
    for i in range(n_G): L2TL2[i*n_tcp:(i+1)*n_tcp,i*n_tcp:(i+1)*n_tcp] = L2.T@L2
    R =eL1*LTL+eL2*L2TL2
    
    # Iteratively estimation of model parameters
    diff = 10000
    w = np.ones(d_s[ind].shape)
    m = None
    iteration = 0
    while (diff > stop)&(iteration<100):
        print('Iteration:',iteration)
        ms = np.linalg.inv((G_s[ind,:]*w[:,None]).T@(G_s[ind,:]*w[:,None])+R)@(G_s[ind,:]*w[:,None]).T@(d_s[ind]*w)
        ms = ms.reshape((n_G, n_tcp)).T

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
            print('Relative change model norm', diff)

        m = mNew
        iteration += 1


    # Model and residual norm
    m_eb = m
    norm_eb_m = np.sqrt(np.average((L@ms).flatten()**2))
    norm_eb_r = rmse

    # Evaluation design matrix
    mlt_eval = np.arange(0,2*np.pi,2*np.pi/240)
    Gtemp = BSpline(sKnots, np.eye(n_scp), sOrder)(mlt_eval)
    G = Gtemp[:,4:8].copy()+Gtemp[:,8:12].copy()

    tau1=[]
    for i, tt in enumerate(time):
        tau1.append(G@m[i, :])
    tau1=np.array(tau1).squeeze()

    # Transform to unprimed
    tau_eb  = np.exp(tau1)

    ## DERIVATIVE

    # df/dt in primed
    dt  =(tKnots[tOrder+1:-1]-tKnots[1:-tOrder-1])
    dms_dt = (ms[1:,:] - ms[:-1,:]) * tOrder / dt[:,None]
    knots_dt=tKnots[1:-1]
    n_cp_dt=len(knots_dt)-(tOrder-1)-1

    M_dt = BSpline(knots_dt, np.eye(n_cp_dt), tOrder-1)(time)
    dm_dt      = M_dt@dms_dt

    dtau1_dt=[]
    for i, tt in enumerate(time):
        dtau1_dt.append(G@dm_dt[i, :])
    dtau1_dt=np.array(dtau1_dt).squeeze()

    # df/d(phi) in primed
    ms_temp = np.hstack((ms,ms[:,:1]))
    dp =(sKnots[sOrder+1:-1]-sKnots[1:-sOrder-1])
    dms_dp = (ms_temp[:,1:] - ms_temp[:,:-1]) * sOrder / dp[None,4:8]
    knots_dp=sKnots[1:-1]
    n_cp_dp=len(knots_dp)-(sOrder-1)-1

    M_dp_temp = BSpline(knots_dp, np.eye(n_cp_dp), sOrder-1)(mlt_eval)
    M_dp = M_dp_temp[:,4:8].copy()+M_dp_temp[:,8:12].copy()
    dm_dp      = dms_dp@M_dp.T

    dtau1_dp = []
    for i, tt in enumerate(mlt_eval):
        dtau1_dp.append(M@dm_dp[:,i])
    dtau1_dp=np.array(dtau1_dp).squeeze().T
    
    # Transform to unprimed
    dtau_dt_eb = np.exp(tau1)*(dtau1_dt)
    dtau_dp_eb = np.exp(tau1)*(dtau1_dp)

    # Temporal change of phi and theta
    R_I = 6500e3
    dphi_dt = -( dtau_dt_eb*dtau_dp_eb)/(np.sin(tau_eb)**2+(dtau_dp_eb)**2)
    dtheta_dt = dtau_dt_eb*np.sin(tau_eb)**2/(np.sin(tau_eb)**2+(dtau_dp_eb)**2)

    # Boundary velocity
    u_phi = R_I*np.sin(tau_eb)*dphi_dt/60
    u_theta = R_I*dtheta_dt/60


    #%% Poleward boundary model
    
    
    theta_pb = np.deg2rad(90 - ds['ocb'].stack(z=('lim','mlt')).values)
    theta_pb1 = theta_pb/np.exp(mtau)

    # Temporal design matix
    M = BSpline(tKnots, np.eye(n_tcp), tOrder)(time)

    # Spatial design matrix
    sOrder = 3
    sKnots = np.deg2rad(15*np.array([0,2,4,6,12,18,20,22]))
    sKnots = np.r_[sKnots-2*np.pi,sKnots,sKnots+2*np.pi]
    sKnots = np.r_[np.repeat(sKnots[0],sOrder),sKnots, np.repeat(sKnots[-1],sOrder)]
    
    n_scp = len(sKnots)-sOrder-1
    Gtemp = BSpline(sKnots, np.eye(n_scp), sOrder)(phi)
    G = Gtemp[:,8:16].copy()+Gtemp[:,16:24].copy()
    G = np.tile(G,(n_lim,1))
    n_G = np.shape(G)[1]
    for i in range(n_t):
        G_t = np.zeros((n_mlt*n_lim, n_G*n_tcp))

        for j in range(n_tcp):
            G_t[:, np.arange(j, n_G*n_tcp, n_tcp)] = G*M[i, j]

        if i == 0:
            G_s = G_t
        else:
            G_s = np.vstack((G_s, G_t))

    # Data
    d_s = (np.log(theta_pb1)-np.log(1-theta_pb1)).flatten()
    ind = np.isfinite(d_s)

    # 1st order regularization
    L = np.hstack((-np.identity(n_tcp-1),np.zeros((n_tcp-1,1))))+np.hstack((np.zeros((n_tcp-1,1)),np.identity(n_tcp-1)))
    LTL = np.zeros((n_tcp*n_G,n_tcp*n_G))
    for i in range(n_G): LTL[i*n_tcp:(i+1)*n_tcp,i*n_tcp:(i+1)*n_tcp] = L.T@L

    # 2nd order regularization
    L2 = np.hstack((np.identity(n_tcp-2),np.zeros((n_tcp-2,2))))-2*np.hstack((np.zeros((n_tcp-2,1)),np.identity(n_tcp-2),np.zeros((n_tcp-2,1))))+np.hstack((np.zeros((n_tcp-2,2)),np.identity(n_tcp-2)))
    L2TL2 = np.zeros((n_tcp*n_G,n_tcp*n_G))
    for i in range(n_G): L2TL2[i*n_tcp:(i+1)*n_tcp,i*n_tcp:(i+1)*n_tcp] = L2.T@L2
    R = pL1*LTL+pL2*L2TL2
    
    # Iteratively estimation of model parameters
    diff = 10000
    w = np.ones(d_s[ind].shape)
    m = None
    iteration = 0
    while diff > stop:
        print('Iteration:',iteration)
        ms = np.linalg.inv((G_s[ind,:]*w[:,None]).T@(G_s[ind,:]*w[:,None])+R)@(G_s[ind,:]*w[:,None]).T@(d_s[ind]*w)
        ms = ms.reshape((n_G, n_tcp)).T
        # Retrieve B-spline smooth model paramters (coarse)
        mNew    = M@ms

        mtau2=[]
        for i, tt in enumerate(time):
            mtau2.append(G@mNew[i, :])
        mtau2=np.array(mtau2).squeeze()

        residuals = mtau2.flatten()[ind] - d_s[ind]
        rmse = np.sqrt(np.average(residuals**2,weights=w))
        weights = 1.5*rmse/np.abs(residuals)
        weights[weights > 1] = 1.
        w = weights
        if m is not None:
            diff = np.sqrt(np.mean((mNew - m)**2))/(1+np.sqrt(np.mean(mNew**2)))
            print('Relative change model norm', diff)

        m = mNew
        iteration += 1

    # Model and residual norm
    m_pb = m
    norm_pb_m = np.sqrt(np.average((L@ms).flatten()**2))
    norm_pb_r = rmse

    # Evaluation
    Gtemp = BSpline(sKnots, np.eye(n_scp), sOrder)(mlt_eval)
    G = Gtemp[:,8:16].copy()+Gtemp[:,16:24].copy()
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
    M_dt = BSpline(knots_dt, np.eye(n_cp_dt), tOrder-1)(time)
    dm_dt = M_dt@dms_dt

    dtau2_dt=[]
    for i, tt in enumerate(time):
        dtau2_dt.append(G@dm_dt[i, :])
    dtau2_dt=np.array(dtau2_dt).squeeze()

    # df/d(phi) in primed
    ms_temp = np.hstack((ms,ms[:,:1]))
    dp =(sKnots[sOrder+1:-1]-sKnots[1:-sOrder-1])
    dms_dp = (ms_temp[:,1:] - ms_temp[:,:-1]) * sOrder / dp[None,8:16]
    knots_dp=sKnots[1:-1]
    n_cp_dp=len(knots_dp)-(sOrder-1)-1

    M_dp_temp = BSpline(knots_dp, np.eye(n_cp_dp), sOrder-1)(mlt_eval)
    M_dp = M_dp_temp[:,8:16].copy()+M_dp_temp[:,16:24].copy()
    dm_dp      = dms_dp@M_dp.T

    dtau2_dp = []
    for i, tt in enumerate(mlt_eval):
        dtau2_dp.append(M@dm_dp[:,i])
    dtau2_dp=np.array(dtau2_dp).squeeze().T

    # Transform derivatives to primed
    dtau1_dt = (np.exp(-tau2)/(np.exp(-tau2)+1)**2)*(dtau2_dt)
    dtau1_dp = (np.exp(-tau2)/(np.exp(-tau2)+1)**2)*(dtau2_dp)

    # Transform derivatives to unprimed
    dtau_dt = tau_eb*dtau1_dt + dtau_dt_eb*tau1
    dtau_dp = tau_eb*dtau1_dp + dtau_dp_eb*tau1

    # Transform from "wrapped" ionosphere to spherical ionosphere
    dphi_dt = -( dtau_dt*dtau_dp)/(np.sin(tau_pb)**2+(dtau_dp)**2)
    dtheta_dt = dtau_dt*np.sin(tau_pb)**2/(np.sin(tau_pb)**2+(dtau_dp)**2)

    # Boundary velocity
    v_phi = R_I*np.sin(tau_pb)*dphi_dt/60
    v_theta = R_I*dtheta_dt/60

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
        mlt = np.rad2deg(mlt_eval)/15
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
    return ds2,(norm_eb_m,norm_eb_r,norm_pb_m,norm_pb_r),(m_eb,m_pb)    
    
def calcFlux(ds,height=130):
    '''
    Function to estimate the amount of open flux inside the given boundaries.
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with the boundaries.
        Coordinates must be 'date' and 'mlt'
        Data variable must be 'ocb' and 'eqb'
    height : float, optional
        Assumed height of the given boundary
    Returns
    -------
    ds : xarray.Dataset
        Copy(?) of the Dataset with calculated flux
    '''
    
    
    # Calc km/gdlat
    lat = np.arange(29.5,90,1)
    theta,R = ppigrf.ppigrf.geod2geoc(lat,height,0,0)[0:2]
    km_per_lat = np.sqrt(np.diff(R*np.cos(np.deg2rad(theta)))**2+np.diff(R*np.sin(np.deg2rad(theta)))**2) 
    
    # Grid in geodetic
    grid_lat,grid_lon = np.meshgrid(np.arange(30,90,1),np.arange(0,360,1))
        
    # Upward magnetic flux in grid
    dateIGRF = ds.date[len(ds.date)//2].values # Date for IGRF values
    Be, Bn, Bu = ppigrf.igrf(grid_lon,grid_lat,height,dateIGRF)
    Bu = abs(Bu.squeeze())
    
    # Transform grid to cartesian
    theta = np.cumsum(km_per_lat[::-1])[::-1]
    grid_x = theta[None,:]*np.cos(np.deg2rad(grid_lon))
    grid_y = theta[None,:]*np.sin(np.deg2rad(grid_lon))
    
    xi,yi = np.meshgrid(np.arange(-10000,10001,100),np.arange(-10000,10001,100))
    xi=xi.flatten()
    yi=yi.flatten()
    Bui = griddata((grid_x.flatten(),grid_y.flatten()),Bu.flatten(),(xi,yi))
    dflux = (1e-9*Bui)*(100e3)**2

    oFlux=[]
    aFlux=[]
    for t in range(len(ds.date)):
        date = pd.to_datetime(ds.date[t].values)
        mlt = ds.mlt.values

        # OCB coordinates
        ocb = ds.isel(date=t)['ocb'].values

        # Convert boundary to geo
        A = apexpy.Apex(date)
        gdlat,glon = A.convert(ocb, mlt, 'mlt', 'geo', height=height,datetime=date)

        # Create an OCB polygon
        x,y=np.interp(gdlat,np.arange(30,90,1),theta)*np.cos(np.deg2rad(glon)),np.interp(gdlat,np.arange(30,90,1),theta)*np.sin(np.deg2rad(glon))
        poly = path.Path(np.stack((x,y),axis=1))

        # Identify gridcell with center inside the OCB polygon
        inocb = poly.contains_points(np.stack((xi,yi),axis=1))

        # Summarize
        if np.isnan(x).any(): # Boundary exceeds the grid
            oFlux.append(np.nan)
        else:
            oFlux.append(np.sum(dflux.flatten()[inocb])*1e-6)

        # EQB coordinates
        eqb = ds.isel(date=t)['eqb'].values

        # Convert boundary to geo
        gdlat,glon = A.convert(eqb, mlt, 'mlt', 'geo', height=height,datetime=date)

        # Create an OCB polygon
        x,y=np.interp(gdlat,np.arange(30,90,1),theta)*np.cos(np.deg2rad(glon)),np.interp(gdlat,np.arange(30,90,1),theta)*np.sin(np.deg2rad(glon))
        poly = path.Path(np.stack((x,y),axis=1))

        # Identify gridcell with center inside the OCB polygon
        inocb = poly.contains_points(np.stack((xi,yi),axis=1))
        
        # Summarize
        if np.isnan(x).any(): # Boundary exceeds the grid
            aFlux.append(np.nan)
        else:
            aFlux.append(np.sum(dflux.flatten()[inocb])*1e-6)

    ds=ds.assign({'openFlux':('date',np.array(oFlux)),'auroralFlux':('date',np.array(aFlux)-np.array(oFlux))})
    ds['openFlux'].attrs = {'long_name':'Open flux','unit':'MWb'}
    ds['auroralFlux'].attrs = {'long_name':'Auroral flux','unit':'MWb'}
    return ds

def calcIntensity(wic,bm):
    wic['shimg'].attrs = {'long_name': 'Counts', 'units': ''}
  
    r = (90. - np.abs(bm['ocb']))
    a = (bm.mlt.values - 6.)/12.*np.pi
    bm['px'] =  r*np.cos(a)
    bm['py'] =  r*np.sin(a)
    
    r = (90. - np.abs(bm['eqb']))
    a = (bm.mlt.values - 6.)/12.*np.pi
    bm['ex'] =  r*np.cos(a)
    bm['ey'] =  r*np.sin(a)
    
    r = (90. - np.abs(wic['mlat']))
    a = (wic.mlt.values - 6.)/12.*np.pi
    wic['x'] =  r*np.cos(a)
    wic['y'] =  r*np.sin(a)
    
    mc=[]
    mc0=[]
    mc6=[]
    mc12=[]
    mc18=[]
    for t in range(len(bm.date)):
        # Create an PB polygon
        poly = path.Path(np.stack((bm.isel(date=t).px.values,bm.isel(date=t).py.values),axis=1))

        # Identify gridcell with center inside the PB polygon
        inpb = poly.contains_points(np.stack((wic.isel(date=t).x.values.flatten(),wic.isel(date=t).y.values.flatten()),axis=1))
    
        # Create an EB polygon
        poly = path.Path(np.stack((bm.isel(date=t).ex.values,bm.isel(date=t).ey.values),axis=1))

        # Identify gridcell with center inside the EB polygon
        ineb = poly.contains_points(np.stack((wic.isel(date=t).x.values.flatten(),wic.isel(date=t).y.values.flatten()),axis=1))
    
        mc.append(np.nanmedian(wic.isel(date=t).shimg.values.flatten()[ineb & ~inpb]))
        mc0.append(np.nanmedian(wic.isel(date=t).shimg.values.flatten()[ineb & ~inpb & ((wic.isel(date=t)['mlt'].values.flatten()<3)|(wic.isel(date=t)['mlt'].values.flatten()>21))]))
        mc6.append(np.nanmedian(wic.isel(date=t).shimg.values.flatten()[ineb & ~inpb & ((wic.isel(date=t)['mlt'].values.flatten()>3)&(wic.isel(date=t)['mlt'].values.flatten()<9))]))
        mc12.append(np.nanmedian(wic.isel(date=t).shimg.values.flatten()[ineb & ~inpb & ((wic.isel(date=t)['mlt'].values.flatten()>9)&(wic.isel(date=t)['mlt'].values.flatten()<15))]))
        mc18.append(np.nanmedian(wic.isel(date=t).shimg.values.flatten()[ineb & ~inpb & ((wic.isel(date=t)['mlt'].values.flatten()>15)&(wic.isel(date=t)['mlt'].values.flatten()<21))]))
    
    bm=bm.assign({'median':('date',np.array(mc)),
                  'median00':('date',np.array(mc0)),
                  'median06':('date',np.array(mc6)),
                  'median12':('date',np.array(mc12)),
                  'median18':('date',np.array(mc18))})
    return bm
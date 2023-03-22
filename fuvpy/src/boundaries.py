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
from scipy.sparse import csc_array

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

def solveInverseIteratively(G_s,d_s,ind,time,R,M,G,n_G,n_cp,stop):
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
    return ms,m,rmse,w

def makeBoundaryModelF(ds,stop=1e-3,dampingValE=2e0,dampingValP=2e1,n_termsE=3,n_termsP=6,order = 3,knotSep = 10):
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
    
    # Iteratively solve the full inverse problem
    ms,m,rmse,w=solveInverseIteratively(G_s,d_s,ind,time,R,M,G,n_G,n_cp,stop)


    # # Model and residual norm
    # m_eb = m
    # norm_eb_m = np.sqrt(np.average((L@ms).flatten()**2))
    # norm_eb_r = rmse

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

    theta_pb1 = theta_pb/tau_eb[::10]

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

    # Iteratively solve the full inverse problem
    ms,m,rmse,w=solveInverseIteratively(G_s,d_s,ind,time,R,M,G,n_G,n_cp,stop)

    # # Model and residual norm # ONLY VALID FOR 
    # m_pb = m
    # norm_pb_m = np.sqrt(np.average((L@ms).flatten()**2))
    # norm_pb_r = rmse

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
    return ds2


def makeBoundaryModelBS(ds,stop=1e-3,eL1=0,eL2=0,pL1=0,pL2=0,tOrder = 3,tKnotSep = 10,estimateError=False,return_norms=False):
    '''
    Function to make a spatiotemporal model of auroral boundaries using periodic B-splines.
    Note: The periodic B-splines are presently hard coded. Update when scipy 1.10 is released!


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
    
    # Iteratively solve the full inverse problem
    ms,m_eb,rmse,w=solveInverseIteratively(G_s,d_s,ind,time,R,M,G,n_G,n_tcp,stop)



    # EVALUATION PER MIN?

    # Evaluation design matrix
    mlt_eval = np.arange(0,2*np.pi,2*np.pi/240)
    Gtemp = BSpline(sKnots, np.eye(n_scp), sOrder)(mlt_eval)
    G = Gtemp[:,4:8].copy()+Gtemp[:,8:12].copy()

    tau1=[]
    for i, tt in enumerate(time):
        tau1.append(G@m_eb[i, :])
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

    # if return_norms:
    #     rmse_eb = rmse
    #     norm_eb_m = np.sqrt(np.average((L@ms).flatten()**2))
    #     norm_eb_r = np.sqrt(np.average(residuals**2,weights=w))

    if estimateError=='cov':
        # Cs =  1/(rmse**2)*np.linalg.inv((G_s[ind,:]*w[:,None]).T@(G_s[ind,:]*w[:,None]))   
        Cs  = rmse**2*(np.linalg.inv((G_s[ind,:]*w[:,None]).T@(G_s[ind,:]*w[:,None])+R)@(G_s[ind,:]*w[:,None]).T)@(np.linalg.inv((G_s[ind,:]*w[:,None]).T@(G_s[ind,:]*w[:,None])+R)@(G_s[ind,:]*w[:,None]).T).T
        # A = G @ Cs * G.t 
        n_r = 1000
        rms = np.random.multivariate_normal(ms.T.flatten(),Cs,n_r)
        rtau = np.full((tau_eb.shape[0],tau_eb.shape[1],n_r),np.nan)
        
        for r in range(n_r):
            rm = M@(rms[r,:].reshape((n_G, n_tcp)).T)
            tau1=[]
            for i, tt in enumerate(time):
                tau1.append(G@rm[i, :])
            tau1=np.array(tau1).squeeze()
    
            # Transform to unprimed
            rtau[:,:,r]  = np.exp(tau1)
        return tau_eb,rtau
    # elif estimateError=='bootstrap':
        
    #%% Poleward boundary model
    
    
    theta_pb = np.deg2rad(90 - ds['ocb'].stack(z=('lim','mlt')).values)
    theta_pb1 = theta_pb/np.exp(tau_eb[::10])

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
    
    # Iteratively solve the full inverse problem
    ms,m,rmse,w=solveInverseIteratively(G_s,d_s,ind,time,R,M,G,n_G,n_tcp,stop)
    
    # # Model and residual norm
    # m_pb = m
    # norm_pb_m = np.sqrt(np.average((L@ms).flatten()**2))
    # norm_pb_r = rmse

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
    return ds2
    
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

def makeBoundaryModelBStest(df,stop=1e-3,tLeb=0,sLeb=0,tLpb=0,sLpb=0,tOrder = 3,tKnotSep = 10,max_iter=50,resample=False,return_norms=False):
    '''
    Function to make a spatiotemporal model of auroral boundaries using periodic B-splines.
    Note: The periodic B-splines are presently hard coded. Update when scipy 1.10 is released!


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
    
    # Make pandas dataframe
    if resample:
        df = df.reset_index().sample(frac=1,replace=True)
    else:
        df = df.reset_index()

    # Start date and duration
    dateS = df['date'].min().floor(str(tKnotSep)+'min')
    dateE = df['date'].max().ceil(str(tKnotSep)+'min')
    duration = (dateE-dateS)/ np.timedelta64(1, 'm')    
    time=(df['date']-dateS)/ np.timedelta64(1, 'm')
    phi = np.deg2rad(15*df['mlt'].values)


    #%% Eq boundary model
    
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
    sOrder = 3
    mltKnots = np.arange(0,24,4)
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
    
   
    
    # 1st order Tikhonov regularization in time
    tL = np.hstack((-np.identity(n_tcp-1),np.zeros((n_tcp-1,1))))+np.hstack((np.zeros((n_tcp-1,1)),np.identity(n_tcp-1)))
    tLTL = np.zeros((n_tcp*n_pcp,n_tcp*n_pcp))
    for i in range(n_pcp): tLTL[i*n_tcp:(i+1)*n_tcp,i*n_tcp:(i+1)*n_tcp] = tL.T@tL
    
    # 1st order Tikhonov regularization in mlt
    sL = []
    for i in range(n_pcp): sL.append(np.roll(np.r_[-1,1,np.repeat(0,n_pcp-2)],i))
    sL=np.array(sL)
    sLTL = np.zeros((n_pcp*n_tcp,n_pcp*n_tcp))
    for t in range(n_tcp): sLTL[t:t+n_pcp*n_tcp:n_tcp,t:t+n_pcp*n_tcp:n_tcp] = sL.T@sL
    
    # Combined regularization
    R = tLeb*tLTL + sLeb*sLTL
     
    # Initiate iterative weights
    w = np.ones(theta_eb1[ind].shape)
    w = csc_array(w).T
    
    # Iteratively estimation of model parameters
    diff = 10000

    m = None
    iteration = 0
    mtau = np.full_like(theta_eb1,np.nan)
    while (diff > stop)&(iteration<max_iter):
        print('Iteration:',iteration)
        ms = lstsq((G_s*w).T.dot(G_s*w)+R,(G_s*w).T.dot(theta_eb1[ind]*w.toarray().squeeze()),lapack_driver='gelsy')[0]
        mtau[ind]=G_s@ms

        residuals = mtau[ind] - theta_eb1[ind]
        if iteration == 0: rmse = np.sqrt(np.average(residuals**2))

        w[:] = np.minimum(0.5*rmse/np.abs(residuals),1)
        if m is not None:
            diff = np.sqrt(np.mean((ms - m)**2))/(1+np.sqrt(np.mean(ms**2)))
            print('Relative change model norm', diff)

        m = ms
        iteration += 1

    
    # Temporal evaluation matrix
    time_ev=(df['date'].drop_duplicates()-dateS).values/ np.timedelta64(1, 'm')
    Gtime = BSpline.design_matrix(time_ev, tKnots,tOrder).toarray()
    
    # Spatial evaluation matrix
    phi_ev = np.arange(0,2*np.pi,2*np.pi/240)
    Gphi = BSpline.design_matrix(phi_ev, sKnots, sOrder)
    Gphi = Gphi[:,n_pcp:2*n_pcp]+Gphi[:,2*n_pcp:3*n_pcp].toarray()
   
    # return time_ev,Gtime,phi_ev,Gphi
    # Combined evaluation matrix
    # G_ev = Gphi[:,np.repeat(np.arange(n_pcp),n_tcp)][np.tile(np.arange(len(phi_ev)),len(time_ev)),:]*Gtime[:,np.tile(np.arange(n_tcp),n_pcp)][np.repeat(np.arange(len(time_ev)),len(phi_ev)),:]
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

        
    #%% Poleward boundary model
    theta_pb  = np.deg2rad(90-df['pb'].values)
    theta_pb1 = theta_pb/np.exp(mtau)
    theta_pb2 = np.log(theta_pb1)-np.log(1-theta_pb1)
    
    # Finite data points
    ind = np.isfinite(theta_pb2)

    # Temporal design matix
    Gtime = BSpline.design_matrix(time[ind],tKnots,tOrder)

    # Spatial knots (extended)
    mltKnots = np.array([0,2,4,8,12,16,20,22])
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
  
    # 1st order regularization in time
    tL = np.hstack((-np.identity(n_tcp-1),np.zeros((n_tcp-1,1))))+np.hstack((np.zeros((n_tcp-1,1)),np.identity(n_tcp-1)))
    tLTL = np.zeros((n_tcp*n_pcp,n_tcp*n_pcp))
    for i in range(n_pcp): tLTL[i*n_tcp:(i+1)*n_tcp,i*n_tcp:(i+1)*n_tcp] = tL.T@tL
    
    # 1st order regularization in mlt
    # sL = np.array([[-1,1,0,0,0,0,0,0],[0,-1,1,0,0,0,0,0],[0,0,-1,1,0,0,0,0],[0,0,0,-1,1,0,0,0],[0,0,0,0,-1,1,0,0],[0,0,0,0,0,-1,1,0],[0,0,0,0,0,0,-1,1],[1,0,0,0,0,0,0,-1]])
    sL = []
    for i in range(n_pcp): sL.append(np.roll(np.r_[-1,1,np.repeat(0,n_pcp-2)],i))
    sL=np.array(sL)
    sLTL = np.zeros((n_pcp*n_tcp,n_pcp*n_tcp))
    for t in range(n_tcp): sLTL[t:t+n_pcp*n_tcp:n_tcp,t:t+n_pcp*n_tcp:n_tcp] = sL.T@sL
    
    # Combined regularization
    R = tLpb*tLTL + sLpb*sLTL

    # Initiate iterative weights
    w = np.ones(theta_pb2[ind].shape)
    w = csc_array(w).T
    
    # Iteratively estimation of model parameters
    diff = 10000
    m = None
    mtau = np.full_like(theta_pb2,np.nan)
    iteration = 0
    while (diff > stop)&(iteration<max_iter):
        print('Iteration:',iteration)
        ms = lstsq((G_s*w).T.dot(G_s*w)+R,(G_s*w).T.dot(theta_pb2[ind]*w.toarray().squeeze()),lapack_driver='gelsy')[0]
        mtau[ind]=G_s@ms

        residuals = mtau[ind] - theta_pb2[ind]
        rmse = np.sqrt(np.average(residuals**2,weights=w.toarray().squeeze()))

        w[:] = np.minimum(1.5*rmse/np.abs(residuals),1)
        if m is not None:
            diff = np.sqrt(np.mean((ms - m)**2))/(1+np.sqrt(np.mean(ms**2)))
            print('Relative change model norm', diff)

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

    # Reshape modelled values
    tau_eb =  tau_eb.reshape((len(time_ev),len(phi_ev)))
    tau_pb = tau_pb.reshape((len(time_ev),len(phi_ev)))
    
    u_phi =  u_phi.reshape((len(time_ev),len(phi_ev)))
    u_theta = u_theta.reshape((len(time_ev),len(phi_ev)))
    
    v_phi =  v_phi.reshape((len(time_ev),len(phi_ev)))
    v_theta = v_theta.reshape((len(time_ev),len(phi_ev)))

    
    # Make Dataset with modelled boundary locations and velocities
    ds2 = xr.Dataset(
    data_vars=dict(
        pb=(['date','mlt'], 90-np.rad2deg(tau_pb)),
        eb=(['date','mlt'], 90-np.rad2deg(tau_eb)),
        v_phi=(['date','mlt'], v_phi),
        v_theta=(['date','mlt'], v_theta),
        u_phi=(['date','mlt'], u_phi),
        u_theta=(['date','mlt'], u_theta),
        dtaudphi = (['date','mlt'], dtaudp.reshape((len(time_ev),len(phi_ev)))),
        ),
    coords=dict(
        date = df['date'].drop_duplicates().values,
        mlt = np.rad2deg(phi_ev)/15
    ),
    )

    # Add attributes
    ds2['mlt'].attrs = {'long_name': 'Magnetic local time','unit':'hrs'}
    ds2['pb'].attrs = {'long_name': 'Open-closed boundary','unit':'deg'}
    ds2['eb'].attrs = {'long_name': 'Equatorward boundary','unit':'deg'}
    ds2['v_phi'].attrs = {'long_name': '$V_\\phi$','unit':'m/s'}
    ds2['v_theta'].attrs = {'long_name': '$V_\\theta$','unit':'m/s'}
    ds2['u_phi'].attrs = {'long_name': '$U_\\phi$','unit':'m/s'}
    ds2['u_theta'].attrs = {'long_name': '$U_\\theta$','unit':'m/s'}
    return ds2
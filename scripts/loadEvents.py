#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:18:55 2022

@author: aohma
"""
import glob
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import BSpline
from scipy.linalg import lstsq
from scipy.stats import binned_statistic_2d
# import statsmodels.api as sm
import fuvpy as fuv
from polplot import pp,sdarngrid,bin_number
import matplotlib.path as path
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit


def makeData():
    wicfiles = glob.glob('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/projects/aurora_dayglow/fuv_2000-08-28-1st/idl/wic/*')
    s12files = glob.glob('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/projects/aurora_dayglow/fuv_2000-08-28-1st/idl/s12/*')
    s13files = glob.glob('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/projects/aurora_dayglow/fuv_2000-08-28-1st/idl/s13/*')

    wic = fuv.readImg(wicfiles)
    s12 = fuv.readImg(s12files)
    s13 = fuv.readImg(s13files)
    
    wic = fuv.makeDGmodel(wic,transform='log',stop=1e-2,tKnotSep=240)
    s12 = fuv.makeDGmodel(s12,stop=1e-2,tKnotSep=240)
    s13 = fuv.makeDGmodel(s13,stop=1e-2,tKnotSep=240)
    
    wic = fuv.makeSHmodel(wic,4,4,stop=1e-2,knotSep=240)
    s13 = fuv.makeSHmodel(s13,4,4,stop=1e-2,knotSep=240)
    
    wic.to_netcdf('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/projects/aurora_dayglow/fuv_2000-08-28-1st/wic.nc')
    s12.to_netcdf('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/projects/aurora_dayglow/fuv_2000-08-28-1st/s12.nc')
    s13.to_netcdf('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/projects/aurora_dayglow/fuv_2000-08-28-1st/s13.nc')    
 
def weightedBinning(fraction,d,dm,w,sKnots):
    bins = np.r_[sKnots[0],np.arange(0,sKnots[-1]+0.25,0.25)]
    # bins = np.quantile(fraction,np.linspace(0,1,101))
    # bins = np.r_[-1e10,bins,1e10]
    binnumber = np.digitize(fraction,bins)
    
    rmse=np.full_like(d,np.nan)
    for i in range(1,len(bins)):
        ind = binnumber==i
        if np.sum(w[ind])==0:
            rmse[ind]=np.nan
        else:
            rmse[ind]=np.sqrt(np.average((d[ind]-dm[ind])**2,weights=w[ind]))
    
    ind2 = np.isfinite(rmse)
    # Fit using Bspline
    
    # test with smoother knots
    
    # G= BSpline(sKnots, np.eye(n_scp), sOrder)(fraction)
    # m =np.linalg.lstsq(G[ind2].T@G[ind2],G[ind2].T@np.log(rmse[ind2]**2),rcond=None)[0]
    
    # # test with linear 
    # G = np.vstack((dm,np.ones_like(dm))).T
    # m =np.linalg.lstsq(G[ind2].T@G[ind2],G[ind2].T@rmse[ind2],rcond=None)[0]
    
    # # test with linear 
    # G = np.vstack((dm**2,np.ones_like(dm))).T
    # m =np.linalg.lstsq(G[ind2].T@G[ind2],G[ind2].T@rmse[ind2]**2,rcond=None)[0]
    
    # Test, non-linear fit
    def func(x,a,b):
        return a*x+b
    
    popt, pcov=curve_fit(func, dm[ind2]**2, rmse[ind2]**2, bounds=(0,np.inf))
    # rmseFit=np.full_like(d,np.nan)
    # rmseFit[ind2]=np.sqrt(popt)
    rmseFit = np.sqrt(func(dm**2,*popt))
    
    return rmseFit,rmse
    
    
def makeFig1c(wic,s12,s13,idate,outpath):
    ''' 
    Plot a few WIC images
    wic : Dataset with the images to be plotted
    outpath : path to where the image is saved
    
    '''  

    fig,axs = plt.subplots(1,3,figsize=(9,3.5),constrained_layout = True)
    
    pc=[]
    pc.append(axs[0].pcolormesh(wic['img'].isel(date=idate).values,vmin=0, vmax=12000,cmap='Oranges_r'))
    pc.append(axs[1].pcolormesh(s12['img'].isel(date=idate).values,vmin=0, vmax=50,cmap='Blues_r'))
    pc.append(axs[2].pcolormesh(s13['img'].isel(date=idate).values,vmin=0, vmax=100,cmap='Greens_r'))
    names = ['WIC','SI12','SI13']
    abc = 'abc'
    for i in range(3):
        axs[i].axes.xaxis.set_visible(False)
        axs[i].axes.yaxis.set_visible(False)
        axs[i].set_aspect('equal')
        plt.colorbar(pc[i],ax=axs[i],fraction=0.05,pad=0.01,location='bottom',extend='max',label=names[i]+' intensity [counts]')
        axs[i].text(0.05, 0.95, abc[i], fontsize=12, horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes,color='w')
        
    
    
    # wic['img'].isel(date=idate).plot(vmin=0,vmax=12000,xticks=[],yticks=[],subplot_kws={'xlabel':''},cbar_kwargs={'label':'WIC intensity [counts]'})
    plt.savefig(outpath + 'wicExample.png',bbox_inches='tight',dpi = 300)
    plt.clf()
    plt.close()    
    
    # cmap
    
def makeFig1b(inpath,outpath,idate):
    wicfiles = glob.glob(inpath)
    wicfiles.sort()
    wic = fuv.readImg(wicfiles[idate])
    wic0 = fuv.readImg(wicfiles[idate],reflat=False)
    print(wic.date)
    ff = xr.concat([wic0, wic], pd.Index(["original", "corrected"], name="flat-field"))
    ff['img'].plot(x='col',y='row',col='flat-field',vmin=550,vmax=950,xticks=[],yticks=[],subplot_kws={'xlabel':''},cbar_kwargs={'label':'WIC intensity [counts]'},cmap='Oranges_r')
    plt.gcf().text(0.06, 0.83, 'a', fontsize=12,color='k')
    plt.gcf().text(0.443, 0.83, 'b', fontsize=12,color='w')
    
    plt.savefig(outpath + 'wicFF.png',bbox_inches='tight',dpi = 300)
    plt.clf()
    plt.close()


def makeFig2c(imgs,outpath,idate,inImg='img',transform=None,sOrder=3,dampingVal=0,tukeyVal=5,stop=1e-3,minlat=0,dzalim=80,sKnots=None,tKnotSep=None,tOrder=2,dzacorr = 0):
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
    
    
    
    
    
    # imgs['fraction'] = np.cos(np.deg2rad(imgs['sza']))/np.cos(np.deg2rad(imgs['dza']))
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
        # fraction = np.exp(dzacorr*(1. - 1/np.cos(np.deg2rad(dza))))/np.cos(np.deg2rad(dza))*np.cos(np.deg2rad(sza))
        fraction = np.cos(np.deg2rad(sza))/np.cos(np.deg2rad(dza))
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
    
    # G_s2=[]
    for i in range(n_t):
        G= BSpline(sKnots, np.eye(n_scp), sOrder)(fraction[i,:]) # Spatial design matirx
        G_t = np.zeros((n_s, n_scp*n_tcp))
        # G_t2 = np.zeros((n_s, n_scp*n_tcp))
        for j in range(n_tcp):
            G_t[:, np.arange(j, n_scp*n_tcp, n_tcp)] = G*M[i, j]
        # for j in range(n_scp):
        #     G_t2[:,j*n_tcp:(j+1)*n_tcp] = np.outer(G[:,j],M[i,:])

        G_g.append(G)
        G_s.append(G_t)
        # G_s2.append(G_t2)
    G_s=np.vstack(G_s)
    # G_s2=np.vstack(G_s2)

    # Data
    ind = (sza >= 0) & (dza <= dzalim) & (glat >= minlat) & (np.isfinite(d)) & remove[None,:] & (fraction>sKnots[0]) & (fraction<sKnots[-1])
    
    # return sza,fraction,ind,sKnots
    # Spatial weights
    ws = np.full_like(fraction,np.nan)
    for i in range(len(sza)):
        count,bin_edges,bin_number=binned_statistic(fraction[i,ind[i,:]],fraction[i,ind[i,:]],statistic=
    'count',bins=np.arange(sKnots[0],sKnots[-1]+0.1,0.1))
        ws[i,ind[i,:]]=1/np.maximum(1,count[bin_number-1])

    d_s = d.flatten()
    ind = ind.flatten()
    ws = ws.flatten()
    # ws = np.ones_like(d_s)
    w = np.ones(d_s.shape)
    # sigma = np.zeros(d_s.shape)
    # Damping
    damping = dampingVal*np.ones(G_s.shape[1])
    R = np.diag(damping)

    # Iterativaly solve the inverse problem
    diff = 1e10
    iteration = 0
    m = None
    dm=None
    
    # return d_s,ind,ws,fraction
    
    while (diff>stop)&(iteration < 100):
        print('Iteration:',iteration)

        m_s = np.linalg.lstsq((G_s[ind,:]*w[ind,None]*ws[ind,None]).T@(G_s[ind,:]*w[ind,None]*ws[ind,None])+R,(G_s[ind,:]*w[ind,None]*ws[ind,None]).T@(d_s[ind]*w[ind]*ws[ind]),rcond=None)[0]

        mNew    = M@m_s.reshape((n_scp, n_tcp)).T
        
        if dm is not None:
            dmOld=dm
        
        dm=[]
        for i, tt in enumerate(time):
            dm.append(G_g[i]@mNew[i, :])

        dm=np.array(dm).squeeze()
        residuals = (d.flatten()[ind] - dm.flatten()[ind])#/dm.flatten()[ind]
        # rmse = np.sqrt(np.average(residuals**2,weights=w[ind]))
        if iteration==0:
            sigma = np.full_like(d_s,np.nan)
            sigmaBinned = np.full_like(d_s,np.nan)
            sigma[ind],sigmaBinned[ind] = weightedBinning(fraction.flatten()[ind], d.flatten()[ind],dm.flatten()[ind], w[ind]*ws[ind],sKnots)

        # # # Heteroskedasitic consistent covariance
        # # V = ((G_s[ind,:]*w[ind,None]*residuals[:,None]).T@(G_s[ind,:]*w[ind,None]*residuals[:,None]))
        # GTG = (G_s[ind,:]*w[ind,None]*ws[ind,None]).T@(G_s[ind,:]*w[ind,None]*ws[ind,None])
        # C = np.linalg.inv(GTG) @ ((G_s[ind,:]*w[ind,None]*ws[ind,None]*residuals[:,None]).T@(G_s[ind,:]*w[ind,None]*ws[ind,None]*residuals[:,None]))@np.linalg.inv(GTG)
        # out=np.std(np.random.multivariate_normal(m_s,C,100),axis=0)

        # # # # var_m = M@ (abs(np.diag(C))).reshape((n_scp, n_tcp)).T
        # # # var_m=[]
        # # # for i in range(n_scp*n_tcp):
        # # #     var_m.append(M@out.reshape((n_scp, n_tcp)).T)
        # var_m    = M@out.reshape((n_scp, n_tcp)).T
        # sigma = []
        # for i, tt in enumerate(time):
        #     sigma.append(abs(G_g[i]@var_m[i, :]))
        # sigma=np.array(sigma).squeeze().flatten()
        # sigma = np.where(np.isnan(sigma),0,sigma)

        # m_r = np.random.multivariate_normal(m_s,C,100)
        # # return m_s,m_r
        # dm_r = []
        # for j in range(len(m_r)):
        #     dm_r.append([])
        #     mTemp = M@m_r[j,:].reshape((n_scp, n_tcp)).T
        #     for i, tt in enumerate(time):
        #         dm_r[j].append(G_g[i]@mTemp[i, :])
        #     dm_r[j]=np.array(dm_r[j]).squeeze()  
        # return M,m_s,dm,dm_r,C
        # list_m.append(m_s)
        # list_G.append(G_s[ind,:]*w[ind,None]*ws[ind,None])
        # list_C.append(C)
        
        # TEST CO
        
        
        # TEST covar with no time dep, maybe?
        # G0 = np.vstack(G_g)
        # G0TG0 = (G0[ind,:]*w[ind,None]*ws[ind,None]).T@(G0[ind,:]*w[ind,None]*ws[ind,None])
        # C0 = np.linalg.inv(G0TG0) @ ((G0[ind,:]*w[ind,None]*ws[ind,None]*residuals[:,None]).T@(G0[ind,:]*w[ind,None]*ws[ind,None]*residuals[:,None]))@np.linalg.inv(G0TG0)
        # S0 = G0TG0*C0
        # Gk = BSpline(sKnots, np.eye(n_scp), sOrder)(sKnots)[sOrder-1:-sOrder+1]
        # Sd = np.array([np.sqrt(np.sum(Gk[i,:][None,:]*S0)) for i in range(n_scp)])
        # return Sd
        # GTG = np.vstack(G_g)[ind,:].T@np.vstack(G_g)[ind,:]
        
        # Covar with no time dependence:
        # G0 = np.vstack(G_g)
        # C = np.linalg.inv((G0[ind,:]*w[ind,None]).T@(G0[ind,:]*w[ind,None])) @ ((G0[ind,:]*w[ind,None]*residuals[:,None]).T@(G0[ind,:]*w[ind,None]*residuals[:,None]))@np.linalg.inv((G0[ind,:]*w[ind,None]).T@(G0[ind,:]*w[ind,None]))
        # return C
        
        # C = np.linalg.inv((G_s[ind,:]*w[ind,None]).T@(G_s[ind,:]*w[ind,None])) @ ((G_s[ind,:]*w[ind,None]*residuals[:,None]).T@(G_s[ind,:]*w[ind,None]*residuals[:,None]))#@np.linalg.inv((G_s[ind,:]*w[ind,None]).T@(G_s[ind,:]*w[ind,None]))
        # sigma[ind] = np.sqrt(((G_s[ind,:]*w[ind,None])@C*(G_s[ind,:]*w[ind,None])).sum(-1))
        # sigma = np.where(np.isnan(sigma),0,sigma)
        # ## Sigma from binning
        # binned_statistic(x, values, statistic='mean'
        # # return C,var_m,sigma.flatten()[ind],rmse,residuals
        # # sqrt(diag(X.T X)^(-1)X.T diag(e_i^(2)) X(X.T X)^(-1)
        # # return ((G_s[ind,:]*w[ind,None]*residuals[:,None]).T@(G_s[ind,:]*w[ind,None]*residuals[:,None]))
        # print(np.nanmean(dm.flatten()[ind][fraction.flatten()[ind]<0]))
        # print(np.average(residuals[fraction.flatten()[ind]<0]**2,weights=w[ind][fraction.flatten()[ind]<0]))
        # print(np.nanmean(dm.flatten()[ind][fraction.flatten()[ind]>=0]))
        # print(np.average(residuals[fraction.flatten()[ind]>=0]**2,weights=w[ind][fraction.flatten()[ind]>=0]))
        
        # mean1=(np.nanmean(dm.flatten()[ind][fraction.flatten()[ind]<0]))
        # mse1=(np.average(residuals[fraction.flatten()[ind]<0]**2,weights=w[ind][fraction.flatten()[ind]<0]))
        # mean2=(np.nanmean(dm.flatten()[ind][fraction.flatten()[ind]>=0]))
        # mse2=(np.average(residuals[fraction.flatten()[ind]>=0]**2,weights=w[ind][fraction.flatten()[ind]>=0]))
        
        # if mse1>=mse2:
        #     sigma = np.sqrt(mse1) + 0*dm.flatten()
        # else:
        #     sigma = np.sqrt(mse1 + (mse2-mse1)*(dm.flatten()-mean1)/(mean2-mean1))
        # # sigma = rmse/np.nanmean(dm.flatten()[ind])*dm.flatten()

        # iw = ((residuals)/(tukeyVal*rmse))**2
        w[ind] = (1 - np.minimum(1,(residuals/(tukeyVal*sigma[ind]))**2))**2
        # iw = ((residuals)/(tukeyVal*sigma[ind]))**2
        # iw[iw>1] = 1
        # w[ind] = (1-iw)**2
        
        if diff == 1e10:
            dm0 =dm
            sigma0 = sigma
            sigma0B = sigmaBinned
            diff = 1e9

        if m is not None:
            diff = np.sqrt(np.mean((mNew-m)**2))/(1+np.sqrt(np.mean(mNew**2)))
            diff2 = np.sqrt(np.nanmean((dm.flatten()[ind]-dmOld.flatten()[ind])**2))/(1+np.sqrt(np.nanmean(dm.flatten()[ind]**2)))
            print('Relative change model norm',diff)
            print('Relative change model',diff2)
        m = mNew
        iteration += 1

    print(np.sqrt(np.average(m_s**2)))
    print(np.sqrt(np.average((residuals/sigma[ind])**2,weights=w[ind])))

    # Add dayglow model and corrected image to the Dataset
    if transform=='log':
        imgs['dgmodel'] = (['date','row','col'],(np.exp(dm)).reshape((n_t,len(imgs.row),len(imgs.col))))
    elif transform == 'asinh':
        imgs['dgmodel'] = (['date','row','col'],(np.sinh(dm)*background_std +background_mean).reshape((n_t,len(imgs.row),len(imgs.col))))
    else:
        imgs['dgmodel'] = (['date','row','col'],(dm).reshape((n_t,len(imgs.row),len(imgs.col))))
        imgs['dgmodel0'] = (['date','row','col'],(dm0).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['dgimg'] = imgs[inImg]-imgs['dgmodel']
    imgs['dgweight'] = (['date','row','col'],(w).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['dgsigma'] = (['date','row','col'],(sigma).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['dgsigma0'] = (['date','row','col'],(sigma0).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['dgsigmaB'] = (['date','row','col'],(sigmaBinned).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['dgsigma0B'] = (['date','row','col'],(sigma0B).reshape((n_t,len(imgs.row),len(imgs.col))))

    # Remove pixels outside model scope
    ind = (imgs.sza>=0)& (imgs.dza <= dzalim) & (imgs.glat >= minlat) & imgs.bad
    # ind = (sza >= 0) & (dza <= dzalim) & (glat >= minlat) & (np.isfinite(d)) & remove[None,:] & (fraction>sKnots[0]) & (fraction<sKnots[-1])
    imgs['img'] = xr.where(~ind,np.nan,imgs['img'])
    imgs['dgmodel'] = xr.where(~ind,np.nan,imgs['dgmodel'])
    imgs['dgmodel0'] = xr.where(~ind,np.nan,imgs['dgmodel0'])
    imgs['dgimg'] = xr.where(~ind,np.nan,imgs['dgimg'])
    imgs['dgweight'] = xr.where(~ind,np.nan,imgs['dgweight'])
    imgs['dgsigma'] = xr.where(~ind,np.nan,imgs['dgsigma'])
    imgs['dgsigma0'] = xr.where(~ind,np.nan,imgs['dgsigma0'])
    imgs['dgsigmaB'] = xr.where(~ind,np.nan,imgs['dgsigmaB'])
    imgs['dgsigma0B'] = xr.where(~ind,np.nan,imgs['dgsigma0B'])
    
    # change time to hours
    time = time/60
    tEdge = np.concatenate((time-np.mean(np.diff(time))/2,time[-1:]+np.mean(np.diff(time))/2))
    
    fig,axs = plt.subplots(4,1,figsize=(9,8.5),constrained_layout = True)    
    
    x = np.repeat(time, n_s)
    y = np.cos(np.deg2rad(imgs['sza'].values.flatten()))/np.cos(np.deg2rad(imgs['dza'].values.flatten()))
    z = imgs['img'].values.flatten()
    zMean,xEdge,yEdge,binNumber = binned_statistic_2d(y, x, z,statistic=np.nanmean,bins=(np.linspace(-5,5,51),tEdge))
    pc=axs[0].pcolormesh(yEdge,xEdge,zMean,cmap='magma',vmin=0,vmax=16000)
    axs[0].set_xticklabels([])
    axs[0].set_xlim([0,time[-1]])
    axs[0].set_ylim([-3.5,3.5])
    axs[0].set_ylabel('$\\cos (\\alpha_s) / \cos(\\alpha_d)$')
    # cbaxes = inset_axes(axs[1], width="1%", height="30%", loc=2) 
    cbaxes = axs[0].inset_axes([.85,.3,.1,.03]) 
    cb = plt.colorbar(pc,cax=cbaxes,ticks=[0,8000,16000],orientation='horizontal')
    cb.set_label('Mean intensity [counts]',color='white')
    cb.outline.set_edgecolor('white')
    cbaxes.tick_params(axis='both', colors='white')

    
    cmap = plt.get_cmap('copper',n_tcp)
    for i in range(n_tcp):
        axs[1].plot(time,M[:,i],c=cmap(i))
    axs[1].vlines(tKnots[tOrder+1:-tOrder-1]/60,0,1,color='r',linestyle='--',linewidth=0.6)
    axs[1].set_xticklabels([])
    axs[1].set_xlim([0,time[-1]])
    axs[1].set_ylim([0,1])
    axs[1].set_ylabel('Temporal B-splines')   
    
    
    x = np.repeat(time, n_s)
    y = np.cos(np.deg2rad(imgs['sza'].values.flatten()))/np.cos(np.deg2rad(imgs['dza'].values.flatten()))
    z = imgs['dgmodel'].values.flatten()
    zMean,xEdge,yEdge,binNumber = binned_statistic_2d(y, x, z,statistic=np.nanmean,bins=(np.linspace(-5,5,51),tEdge))
    pc=axs[2].pcolormesh(yEdge,xEdge,zMean,cmap='magma',vmin=0,vmax=16000)
    axs[2].set_xticklabels([])
    axs[2].set_xlim([0,time[-1]])
    axs[2].set_ylim([-3.5,3.5])
    axs[2].set_ylabel('$\\cos (\\alpha_s) / \cos(\\alpha_d)$')
    # cbaxes = inset_axes(axs[1], width="1%", height="30%", loc=2) 
    cbaxes = axs[2].inset_axes([.85,.3,.1,.03]) 
    cb = plt.colorbar(pc,cax=cbaxes,ticks=[0,8000,16000],orientation='horizontal')
    cb.set_label('Model intensity [counts]',color='white')
    cb.outline.set_edgecolor('white')
    cbaxes.tick_params(axis='both', colors='white')
    
    
    
    cmap = plt.get_cmap('cool',n_scp)
    for i in range(n_scp):
        axs[3].plot(time,m[:,i],c=cmap(i))
    
    axs[3].set_xlim([0,time[-1]])
    axs[3].set_xlabel('Time since first image [hrs]')
    axs[3].set_ylabel('Model coefficients [counts]')
    
    axs[0].text(0.03, 0.94, 'a', fontsize=12, horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
    axs[1].text(0.03, 0.94, 'b', fontsize=12, horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
    axs[2].text(0.03, 0.94, 'c', fontsize=12, horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes)
    axs[3].text(0.03, 0.94, 'd', fontsize=12, horizontalalignment='center', verticalalignment='center', transform=axs[3].transAxes)
    
    
    plt.savefig(outpath + 'dayglowFit2.png',bbox_inches='tight',dpi = 300)
    plt.clf()
    plt.close()
    
    wic = imgs.isel(date=idate).copy()
    
    fig,axs = plt.subplots(4,1,figsize=(9,8.5),constrained_layout = True)
    
    G= BSpline(sKnots, np.eye(n_scp), sOrder)(np.linspace(-5,5,1001))
    G = np.where(G==0,np.nan,G)
    cmap = plt.get_cmap('cool',n_scp)
    for i in range(n_scp):
        axs[1].plot(np.linspace(-5,5,1001),G[:,i],c=cmap(i))
    axs[1].vlines(sKnots[sOrder+1:-sOrder-1],0,1,color='r',linestyle='--',linewidth=0.6)
    axs[1].set_xticklabels([])
    axs[1].set_xlim([-3.5,3.5])
    axs[1].set_ylim([0,1])
    axs[1].set_ylabel('Spatial B-splines')
    
    
    # Dayglow
    x = np.cos(np.deg2rad(wic['sza'].values.flatten()))/np.cos(np.deg2rad(wic['dza'].values.flatten()))
    y = wic['img'].values.flatten()
    w = wic['dgweight'].values.flatten()
    mid = int(len(wic.date)/2) # Only model of middle image
    xm = np.cos(np.deg2rad(wic.isel(date=mid)['sza'].values.flatten()))/np.cos(np.deg2rad(wic.isel(date=mid)['dza'].values.flatten()))
    m = wic.isel(date=mid)['dgmodel'].values.flatten()
    m0 = wic.isel(date=mid)['dgmodel0'].values.flatten()
    e = wic.isel(date=mid)['dgsigma'].values.flatten()
    e0 = wic.isel(date=mid)['dgsigma0'].values.flatten()
    b = wic.isel(date=mid)['dgsigmaB'].values.flatten()
    b0 = wic.isel(date=mid)['dgsigma0B'].values.flatten()
    

    
    axs[0].scatter(x,y,c='C0',s=0.5,alpha=0.2)
    axs[0].set_xlim([-3.5,3.5])
    # axs[0].set_ylim([0,16000]) 
    axs[0].set_xticklabels([])
    axs[0].set_ylabel('Intensity [counts]')

    # Sort to interp
    m = m[xm.argsort()]
    m0 = m0[xm.argsort()]
    e = e[xm.argsort()]
    e0 = e0[xm.argsort()]
    b = b[xm.argsort()]
    b0 = b0[xm.argsort()]
    xm = xm[xm.argsort()]
    
    # interp for plotting
    xi = np.linspace(-3.5,3.5,1001)
    mi = np.interp(xi,xm[np.isfinite(m)],m[np.isfinite(m)],left=np.nan,right=np.nan)
    mi0 = np.interp(xi,xm[np.isfinite(m0)],m0[np.isfinite(m0)],left=np.nan,right=np.nan)
    ei = np.interp(xi,xm[np.isfinite(e)],e[np.isfinite(e)],left=np.nan,right=np.nan)
    ei0 = np.interp(xi,xm[np.isfinite(e0)],e0[np.isfinite(e0)],left=np.nan,right=np.nan)

    axs[3].plot(xi,mi0,c='C0',linestyle='-',label='Initial $I_{bs}$')
    # axs[3].plot(xi,ei0,c='C0',linestyle=':',label='$\\sigma_{bs}$ first iteration')
    
    axs[3].plot(xi,mi,c='C1',linestyle='-',label='Final $I_{bs}$')
    axs[3].plot(xi,ei,c='C3',linestyle=':',label='$\\sigma_{bs}$')
    # axs[3].plot(x[np.isfinite(b)],b[np.isfinite(b)],c='C1',linewidth=0.6)
    # axs[3].fill_between(x[np.isfinite(m)],m[np.isfinite(m)]-e[np.isfinite(m)],m[np.isfinite(m)]+e[np.isfinite(m)],facecolor='C1',edgecolor=None,alpha=0.4)
    
    # axs[3].plot(x[np.isfinite(b0)],b0[np.isfinite(b0)],c='C0',linewidth=0.6)
    # axs[3].fill_between(x[np.isfinite(m0)],m0[np.isfinite(m0)]-e0[np.isfinite(m0)],m0[np.isfinite(m0)]+e0[np.isfinite(m0)],facecolor='C0',edgecolor=None,alpha=0.4)
    axs[3].set_xlim([-3.5,3.5])
    # axs[2].set_ylim([0,15499]) 
    axs[3].set_xlabel('$\\cos (\\alpha_s) / \cos(\\alpha_d)$')
    axs[3].set_ylabel('$I_{bs}$ and $\\sigma_{bs}$ [counts]')
    axs[3].set_yscale('log')
    axs[3].legend(loc=(0.1,0.5),frameon=False)

    sc= axs[2].scatter(x,y,c=w,s=0.5,vmin=0, vmax=1)
    axs[2].plot(xi,mi,c='C1')
    # axs[2].fill_between(x[np.isfinite(m)],m[np.isfinite(m)]-e[np.isfinite(m)],m[np.isfinite(m)]+e[np.isfinite(m)],color='r',alpha=0.2)
    # axs[2].plot(x[np.isfinite(m)],e[np.isfinite(m)],c='C1')
    axs[2].set_xlim([-3.5,3.5])
    axs[2].set_xticklabels([])
    axs[2].set_ylabel('Intensity [counts]')
    # cbaxes = inset_axes(axs[1], width="1%", height="30%", loc=2) 
    cbaxes = axs[2].inset_axes([.85,.3,.1,.03])    
    cb = plt.colorbar(sc,cax=cbaxes, ticks=[0,0.5,1.], orientation='horizontal')
    cb.set_label('Weight')
    
    axs[0].text(0.03, 0.94, 'a', fontsize=12, horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
    axs[1].text(0.03, 0.94, 'b', fontsize=12, horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
    axs[2].text(0.03, 0.94, 'c', fontsize=12, horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes)
    axs[3].text(0.03, 0.94, 'd', fontsize=12, horizontalalignment='center', verticalalignment='center', transform=axs[3].transAxes)
    
    plt.savefig(outpath + 'dayglowFit.png',bbox_inches='tight',dpi = 300)
    plt.clf()
    plt.close()
    
    return  imgs


def makeFig3(wic,s12,s13,outpath):
    ''' 
    Plot dayglow model overview for all cameras
    wic (xarray.Dataset): Wic image to be plotted. 
    '''
    
    
    fig = plt.figure(figsize=(11,9))

    gs = gridspec.GridSpec(nrows=3,ncols=4,hspace=0.3,wspace=0.01)
    
    ## WIC ##
    pax = pp(plt.subplot(gs[0,0]),minlat=50)
    fuv.plotimg(wic,'img',pax=pax,crange=(0,5000),cmap='magma')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='max')
    cb.set_label('Projected image [counts]')
    pax.ax.set_title('WIC',rotation='vertical',x=-0.03,y=0.45,verticalalignment='center',horizontalalignment='center')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    # Dayglow
    pax = pp(plt.subplot(gs[0,1]),minlat=50)
    fuv.plotimg(wic,'dgmodel',pax=pax,crange=(0,5000),cmap='magma')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='max')
    cb.set_label('BS model [counts]')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    # Corr
    pax = pp(plt.subplot(gs[0,2]),minlat=50)
    fuv.plotimg(wic,'dgimg',pax=pax,crange=(-1000,1000),cmap='coolwarm')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='both')
    cb.set_label('BS corrected image [counts]')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    #Weight
    pax = pp(plt.subplot(gs[0,3]),minlat=50)
    fuv.plotimg(wic,'dgweight',pax=pax,crange=(0,1))
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal')
    cb.set_label('Weights')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    ## S13 ##
    pax = pp(plt.subplot(gs[2,0]),minlat=50)
    fuv.plotimg(s13,'img',pax=pax,crange=(0,40),cmap='magma')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='max')
    cb.set_label('Projected image [counts]')
    pax.ax.set_title('SI13',rotation='vertical',x=-0.03,y=0.45,verticalalignment='center',horizontalalignment='center')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    # Dayglow
    pax = pp(plt.subplot(gs[2,1]),minlat=50)
    fuv.plotimg(s13,'dgmodel',pax=pax,crange=(0,40),cmap='magma')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='max')
    cb.set_label('BS model [counts]')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    # Corr
    pax = pp(plt.subplot(gs[2,2]),minlat=50)
    fuv.plotimg(s13,'dgimg',pax=pax,crange=(-15,15),cmap='coolwarm')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='both')
    cb.set_label('BS corrected image [counts]')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    #Weight
    pax = pp(plt.subplot(gs[2,3]),minlat=50)
    fuv.plotimg(s13,'dgweight',pax=pax,crange=(0,1))
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal')
    cb.set_label('Weights')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    
    
    ## S12 ##
    pax = pp(plt.subplot(gs[1,0]),minlat=50)
    fuv.plotimg(s12,'img',pax=pax,crange=(0,20),cmap='magma')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='max')
    cb.set_label('Projected image [counts]')
    pax.ax.set_title('SI12',rotation='vertical',x=-0.03,y=0.45,verticalalignment='center',horizontalalignment='center')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    # Dayglow
    pax = pp(plt.subplot(gs[1,1]),minlat=50)
    fuv.plotimg(s12,'dgmodel',pax=pax,crange=(0,20),cmap='magma')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='max')
    cb.set_label('BS model [counts]')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    # Corr
    pax = pp(plt.subplot(gs[1,2]),minlat=50)
    fuv.plotimg(s12,'dgimg',pax=pax,crange=(-5,5),cmap='coolwarm')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='both')
    cb.set_label('BS corrected image [counts]')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    #Weight
    pax = pp(plt.subplot(gs[1,3]),minlat=50)
    fuv.plotimg(s12,'dgweight',pax=pax,crange=(0,1))
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal')
    cb.set_label('Weights')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    plt.savefig(outpath + 'dayglowFUV.png',bbox_inches='tight',dpi = 300)
    plt.clf()
    plt.close()

def makeFig4(wic,s12,s13,outpath):

    
    fig = plt.figure(figsize=(11,9))

    gs = gridspec.GridSpec(nrows=3,ncols=4,hspace=0.3,wspace=0.01)
    
    ## WIC ##
    pax = pp(plt.subplot(gs[0,0]),minlat=50)
    fuv.plotimg(wic,'dgimg',pax=pax,crange=(-1000,1000),cmap='coolwarm')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='both')
    cb.set_label('BS corrected image [counts]')
    pax.ax.set_title('WIC',rotation='vertical',x=-0.03,y=0.45,verticalalignment='center',horizontalalignment='center')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    # Dayglow
    pax = pp(plt.subplot(gs[0,1]),minlat=50)
    fuv.plotimg(wic,'shmodel',pax=pax,crange=(-1000,1000),cmap='coolwarm')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='both')
    cb.set_label('SH model [counts]')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    # Corr
    pax = pp(plt.subplot(gs[0,2]),minlat=50)
    fuv.plotimg(wic,'shimg',pax=pax,crange=(-1000,1000),cmap='coolwarm')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='both')
    cb.set_label('SH corrected image [counts]')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    #Weight
    pax = pp(plt.subplot(gs[0,3]),minlat=50)
    fuv.plotimg(wic,'shweight',pax=pax,crange=(0,1))
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal')
    cb.set_label('Weights')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
        

    ## S13 ##
    pax = pp(plt.subplot(gs[2,0]),minlat=50)
    fuv.plotimg(s13,'dgimg',pax=pax,crange=(-15,15),cmap='coolwarm')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='both')
    cb.set_label('BS corrected image [counts]')
    pax.ax.set_title('SI13',rotation='vertical',x=-0.03,y=0.45,verticalalignment='center',horizontalalignment='center')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    # Dayglow
    pax = pp(plt.subplot(gs[2,1]),minlat=50)
    fuv.plotimg(s13,'shmodel',pax=pax,crange=(-15,15),cmap='coolwarm')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='both')
    cb.set_label('SH model [counts]')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    # Corr
    pax = pp(plt.subplot(gs[2,2]),minlat=50)
    fuv.plotimg(s13,'shimg',pax=pax,crange=(-15,15),cmap='coolwarm')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='both')
    cb.set_label('SH corrected image [counts]')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    #Weight
    pax = pp(plt.subplot(gs[2,3]),minlat=50)
    fuv.plotimg(s13,'shweight',pax=pax,crange=(0,1))
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal')
    cb.set_label('Weights')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    ## S12 ##
    pax = pp(plt.subplot(gs[1,0]),minlat=50)
    fuv.plotimg(s12,'dgimg',pax=pax,crange=(-5,5),cmap='coolwarm')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='both')
    cb.set_label('BS corrected image [counts]')
    pax.ax.set_title('SI12',rotation='vertical',x=-0.03,y=0.45,verticalalignment='center',horizontalalignment='center')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    # Dayglow
    pax = pp(plt.subplot(gs[1,1]),minlat=50)
    fuv.plotimg(s12,'shmodel',pax=pax,crange=(-5,5),cmap='coolwarm')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='both')
    cb.set_label('SH model [counts]')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    # Corr
    pax = pp(plt.subplot(gs[1,2]),minlat=50)
    fuv.plotimg(s12,'shimg',pax=pax,crange=(-5,5),cmap='coolwarm')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='both')
    cb.set_label('SH corrected image [counts]')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    #Weight
    pax = pp(plt.subplot(gs[1,3]),minlat=50)
    fuv.plotimg(s12,'shweight',pax=pax,crange=(0,1))
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal')
    cb.set_label('Weights')
    pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='left',fontsize=8)
    pax.write(50, 12, '12',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
    pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='right',fontsize=8)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
    
    
    
    plt.savefig(outpath + 'shFUV.png',bbox_inches='tight',dpi = 300)
    plt.clf()
    plt.close()


def makeSHmodelTest(imgs,Nsh,Msh,order=2,dampingVal=0,tukeyVal=5,stop=1e-3,minlat=0,knotSep=None):
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

    from fuvpy.src.utils import sh
    from fuvpy.src.utils.sunlight import subsol
    
    date = imgs['date'].values
    time=(date-date[0])/ np.timedelta64(1, 'm')

    glat = imgs['glat'].stack(z=('row','col')).values
    glon = imgs['glon'].stack(z=('row','col')).values
    d = imgs['dgimg'].stack(z=('row','col')).values
    # dg = imgs['dgmodel'].stack(z=('row','col')).values
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
    # skeys = sh.SHkeys(Nsh, Msh).Mge(1).MleN().setNmin(1)
    # ckeys = sh.SHkeys(Nsh, Msh).MleN().setNmin(1)
    # skeys = sh.SHkeys(Nsh, Msh).Mge(1).MleN().NminusModd()
    # ckeys = sh.SHkeys(Nsh, Msh).MleN().NminusModd()
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
        G = G/dg[i,:][:,None]
        n_G = np.shape(G)[1]

        G_t = np.zeros((n_s, n_G*n_cp))

        for j in range(n_cp):
            G_t[:, np.arange(j, n_G*n_cp, n_cp)] = G*M[i, j]

        G_g.append(G)
        G_s.append(G_t)

    G_s = np.array(G_s)
    G_s = G_s.reshape(-1,G_s.shape[2])

    # Pixels to include in model
    ind = (np.isfinite(d))&(glat>minlat)&(imgs['bad'].values.flatten())[None,:] &(np.isfinite(dg))  

    # Spatial weights
    grid,mltres=sdarngrid(5,5,minlat//5*5) # Equal area grid
    ws = np.full(glat.shape,np.nan)
    for i in range(len(glat)):
        gbin = bin_number(grid,glat[i,ind[i]],glon[i,ind[i]]/15)
        count=np.full(grid.shape[1],0)
        un,count_un=np.unique(gbin,return_counts=True)
        count[un]=count_un
        ws[i,ind[i]]=1/count[gbin]

    # Data
    d_s = d.flatten()
    ind = ind.flatten()
    w = wdg.flatten() # Weights from dayglow model
    ws = ws.flatten() 
    sigma = dg.flatten()
    # Damping
    damping = dampingVal*np.ones(G_s.shape[1])
    R = np.diag(damping)
    
    # # 1st order regularization
    # L = np.hstack((-np.identity(n_cp-1),np.zeros((n_cp-1,1))))+np.hstack((np.zeros((n_cp-1,1)),np.identity(n_cp-1)))
    # LTL = np.zeros((n_cp*n_G,n_cp*n_G))
    # for i in range(n_G): LTL[i*n_cp:(i+1)*n_cp,i*n_cp:(i+1)*n_cp] = L.T@L
    # R =damping*LTL

    diff = 1e10
    iteration = 0
    m = None
    dm = None
    while (diff>stop)&(iteration < 100):
        print('Iteration:',iteration)
        # Solve for spline amplitudes
        m_s = np.linalg.lstsq((G_s[ind,:]*w[ind,None]*ws[ind,None]).T@(G_s[ind,:]*w[ind,None]*ws[ind,None])+R,(G_s[ind,:]*w[ind,None]*ws[ind,None]).T@(d_s[ind]*w[ind]*ws[ind]),rcond=None)[0]

        if dm is not None:
            dmOld=dm

        # Retrieve B-spline smooth model paramters (coarse)
        mNew    = M@m_s.reshape((n_G, n_cp)).T
        plt.plot(mNew)
        dm=[]
        for i, tt in enumerate(time):
            dm.append(G_g[i]@mNew[i, :])

        dm=np.array(dm).squeeze()
        residuals = dm.flatten()[ind] - d.flatten()[ind]
        # if diff == 1e10: rmse = np.sqrt(np.average(residuals**2,weights=w[ind]*ws[ind]))

        w[ind] = (1-np.minimum(1,((residuals)/(tukeyVal))**2))**2

        if diff==1e10: diff=1e9

        if m is not None:
            diff = np.sqrt(np.mean( (mNew-m)**2))/(1+np.sqrt(np.mean(mNew**2)))
            diff2 = np.sqrt(np.nanmean((dm.flatten()[ind]-dmOld.flatten()[ind])**2))/(1+np.sqrt(np.nanmean(dm.flatten()[ind]**2)))
            print('Relative change model norm:',diff)
            print('Relative change model norm:',diff2)
        m = mNew
        iteration += 1

    
    imgs['shmodel'] = (['date','row','col'],(dm*dg).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['shimg'] = imgs['dgimg']-imgs['shmodel']
    imgs['shweight'] = (['date','row','col'],(w).reshape((n_t,len(imgs.row),len(imgs.col))))

    # Remove pixels outside model scope
    ind = (imgs.glat >= minlat) & imgs.bad & np.isfinite(imgs.dgsigma)
    imgs['shmodel'] = xr.where(~ind,np.nan,imgs['shmodel'])
    imgs['shimg'] = xr.where(~ind,np.nan,imgs['shimg'])
    imgs['shweight'] = xr.where(~ind,np.nan,imgs['shweight'])

    # Add attributes
    imgs['shmodel'].attrs = {'long_name': 'Spherical harmonics model'}
    imgs['shimg'].attrs = {'long_name': 'Spherical harmonics corrected image'}
    imgs['shweight'].attrs = {'long_name': 'Spherical harmonics model weights'}

    return imgs

def runEvent(inpath,outpath):
    # NO ws SH: 0.01,0.3,0.3
    wic = xr.load_dataset(inpath + 'wic.nc')
    wic = makeFig2c(wic,outpath,[66,67,68],sKnots=[-3.5,-0.25,0,0.25,1.5,3.5],stop=0.01,tKnotSep=240,minlat=-90,tukeyVal=5,dzalim=75,dampingVal=0.03)
    wic = makeSHmodelTest(wic,4,4,knotSep=240,stop=0.01,tukeyVal=5,dampingVal=1e-4)
    
    s12 = xr.load_dataset(inpath + 's12.nc')
    s12 = makeFig2c(s12,outpath,[66,67,68],sKnots=[-3.5,-0.25,0,0.25,1.5,3.5],stop=0.01,tKnotSep=240,minlat=-90,tukeyVal=5,dzalim=75,dampingVal=1)
    s12 = makeSHmodelTest(s12,4,4,knotSep=240,stop=0.01,tukeyVal=5,dampingVal=0.3)
    
    s13 = xr.load_dataset(inpath + 's13.nc')
    s13 = makeFig2c(s13,outpath,[66,67,68],sKnots=[-3.5,-0.25,0,0.25,1.5,3.5],stop=0.01,tKnotSep=240,minlat=-90,tukeyVal=5,dzalim=75,dampingVal=1)
    s13 = makeSHmodelTest(s13,4,4,knotSep=240,stop=0.01,tukeyVal=5,dampingVal=1e1)
    
    return wic,s12,s13



def lcurve(imgs,model='BS'):
    L0s = np.r_[0,np.geomspace(1e-5,1e2,7+1)]
    
    norms = []
    for i in range(len(L0s)):
        if model == 'BS':
            _,temp=findNormsBS(imgs,sKnots=[-3.5,-0.25,0,0.25,1.5,3.5],stop=0.01,tKnotSep=150,minlat=-90,tukeyVal=5,dzalim=75,dampingVal=L0s[i])
        elif model= 'SH':
            _,temp=findNormsBS(imgs,sKnots=[-3.5,-0.25,0,0.25,1.5,3.5],stop=0.01,tKnotSep=150,minlat=-90,tukeyVal=5,dzalim=75,dampingVal=L0s[i])
        norms.append(temp)
    
    plt.loglog(norm_r,norm_m,'.-')
    return norm_m,norm_r


    





    

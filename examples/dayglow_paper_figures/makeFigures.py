#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:18:55 2022

@author: aohma
"""
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.interpolate import BSpline
from scipy.stats import binned_statistic,binned_statistic_2d,spearmanr
from scipy.optimize import curve_fit

import fuvpy as fuv
from polplot import pp

def runEvent(inpath):
    '''
    Load premade image files

    Parameters
    ----------
    inpath : str
        Location of netcdf files with the images. These files can be made with fuvpy.readImg()

    Returns
    -------
    wic : xr.Dataset
        Dataset with wic images
    s12 : xr.Dataset
        Dataset with the SI12 images
    s13 : xr.Dataset
        Dataset with the SI13 images

    '''
    wic = xr.load_dataset(inpath + 'wic.nc')
    wic = fuv.makeBSmodel(wic,sKnots=[-3.5,-0.25,0,0.25,1.5,3.5],stop=0.01,n_tKnots=5,minlat=-90,tukeyVal=5,dzalim=75,dampingVal=1e-2)
    wic = fuv.makeSHmodel(wic,4,4,n_tKnots=5,stop=0.01,tukeyVal=5,dampingVal=1e-4)
    
    s12 = xr.load_dataset(inpath + 's12.nc')
    s12 = fuv.makeBSmodel(s12,sKnots=[-3.5,-0.25,0,0.25,1.5,3.5],stop=0.01,n_tKnots=5,minlat=-90,tukeyVal=5,dzalim=75,dampingVal=1e-1)
    s12 = fuv.makeSHmodel(s12,4,4,n_tKnots=5,stop=0.01,tukeyVal=5,dampingVal=1e1)
    
    s13 = xr.load_dataset(inpath + 's13.nc')
    s13 = fuv.makeBSmodel(s13,sKnots=[-3.5,-0.25,0,0.25,1.5,3.5],stop=0.01,n_tKnots=5,minlat=-90,tukeyVal=5,dzalim=75,dampingVal=1e-1)
    s13 = fuv.makeSHmodel(s13,4,4,n_tKnots=5,stop=0.01,tukeyVal=5,dampingVal=1e1)
    
    return wic,s12,s13

def lcurve(imgs,model='BS'):
    '''
    L-curve analysis on the images.

    Parameters
    ----------
    imgs : xr.Dataset
        Dataset with the images
    model : str, optional
        Which model to do the analysis on ('BS' or 'SH'). The default is 'BS'.
        Note that 'SH' only works on datasets where the BS model have already been applied.

    Returns
    -------
    norms : array
        2D array containing the norm of the data misfit and model norm.

    '''
    
    L0s = np.r_[0,np.geomspace(1e-5,1e2,7+1)]
    
    norms = []
    for i in range(len(L0s)):
        if model == 'BS':
            _,temp=fuv.makeBSmodel(imgs,sKnots=[-3.5,-0.25,0,0.25,1.5,3.5],stop=0.01,n_tKnots=5,minlat=-90,tukeyVal=5,dzalim=75,dampingVal=L0s[i],returnNorms=True)
        elif model == 'SH':
            _,temp=fuv.makeSHmodel(imgs,4,4,n_tKnots=5,stop=0.01,tukeyVal=5,dampingVal=L0s[i],returnNorms=True)
        norms.append(temp)
    
    return np.array(norms)


def _noiseModel(fraction,d,dm,w,sKnots):
    # Noise model to be used by makeFig3and4()
    
    bins = np.r_[sKnots[0],np.arange(0,sKnots[-1]+0.25,0.25)]
    binnumber = np.digitize(fraction,bins)
    
    # Binned RMSE
    rmse=np.full_like(d,np.nan)
    for i in range(1,len(bins)):
        ind = binnumber==i
        if np.sum(w[ind])==0:
            rmse[ind]=np.nan
        else:
            rmse[ind]=np.sqrt(np.average((d[ind]-dm[ind])**2,weights=w[ind]))
    
    ind2 = np.isfinite(rmse)
    
    # Linear fit with bounds
    def func(x,a,b):
        return a*x+b
    
    popt, pcov=curve_fit(func, dm[ind2]**2, rmse[ind2]**2, bounds=(0,np.inf))
    rmseFit = np.sqrt(func(dm**2,*popt))
    
    return rmseFit
    
    
def makeFig1(wic,s12,s13,idate,outpath):
    ''' 
    Figure displaying three FUV images
    wic : xr.Dataset with wic images
    s12 : xr.Dataset with s12 images
    s13 : xr.Dataset with s13 images
    idate : index of date to be plotted
    outpath : path to where the figure is saved
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
        
    plt.savefig(outpath + 'fig1.png',bbox_inches='tight',dpi = 300)
    plt.clf()
    plt.close()
    
def makeFig2(inpath,idate,outpath):
    '''
    Figure showing WIC flat-field correction

    Parameters
    ----------
    inpath : str
        Path to wic .idl file.
    outpath : str
        Path to where the figure is saved.
    idate : int
        index of date to be used.

    Returns
    -------
    None.

    '''
    
    wicfiles = glob.glob(inpath)
    wicfiles.sort()
    wic = fuv.readImg(wicfiles[idate])
    wic0 = fuv.readImg(wicfiles[idate],reflat=False)
    print(wic.date)
    ff = xr.concat([wic0, wic], pd.Index(["original", "corrected"], name="flat-field"))
    ff['img'].plot(x='col',y='row',col='flat-field',vmin=550,vmax=950,xticks=[],yticks=[],subplot_kws={'xlabel':''},cbar_kwargs={'label':'WIC intensity [counts]'},cmap='Oranges_r')
    plt.gcf().text(0.06, 0.83, 'a', fontsize=12,color='k')
    plt.gcf().text(0.443, 0.83, 'b', fontsize=12,color='w')
    
    plt.savefig(outpath + 'fig2.png',bbox_inches='tight',dpi = 300)
    plt.clf()
    plt.close()


def makeFig3and4(imgs,outpath,idate,inImg='img',sOrder=3,dampingVal=0,tukeyVal=5,stop=1e-3,minlat=0,dzalim=75,sKnots=None,n_tKnots=2,tOrder=2):
    '''
    Figures showing the performance of the B-spline model

    Parameters
    ----------
    imgs : xarray.Dataset
        Dataset with the FUV images, imported by readFUVimage()
    outpath : str
        Path to where the figure is saved.
    idate : int or list of int
        Indices to be used when displaying the spatial B-splines
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
    n_tKnots : int, optional
        Number of temporal knots, equally spaced between the endpoints. The default is 2 (only knots at endpoints)
    tOrder : int, optional
        Order of the temporal spline fit. The default is 2.
    '''
    
    imgs = imgs.copy()
    
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
        
        if iteration == 0: dm0 = dm

        if iteration<=1:
            sigma = np.full_like(d_s,np.nan)
            sigma[ind] = _noiseModel(fraction.flatten()[ind], d_s[ind],dm.flatten()[ind], w[ind]*ws[ind],sKnots)
            
        w[ind] = (1 - np.minimum(1,(residuals/(tukeyVal*sigma[ind]))**2))**2

        if m is not None:
            diff = np.sqrt(np.mean((mNew-m)**2))/(1+np.sqrt(np.mean(mNew**2)))
            print('Relative change model norm',diff)
        m = mNew
        iteration += 1


    # Add dayglow model and corrected image to the Dataset
    imgs['dgmodel'] = (['date','row','col'],(dm).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['dgmodel0'] = (['date','row','col'],(dm0).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['dgimg'] = imgs[inImg]-imgs['dgmodel']
    imgs['dgweight'] = (['date','row','col'],(w).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['dgsigma'] = (['date','row','col'],(sigma).reshape((n_t,len(imgs.row),len(imgs.col))))

    # Remove pixels outside model scope
    ind = (imgs.sza>=0)& (imgs.dza <= dzalim) & (imgs.glat >= minlat) & imgs.bad
    # ind = (sza >= 0) & (dza <= dzalim) & (glat >= minlat) & (np.isfinite(d)) & remove[None,:] & (fraction>sKnots[0]) & (fraction<sKnots[-1])
    imgs['img'] = xr.where(~ind,np.nan,imgs['img'])
    imgs['dgmodel'] = xr.where(~ind,np.nan,imgs['dgmodel'])
    imgs['dgmodel0'] = xr.where(~ind,np.nan,imgs['dgmodel0'])
    imgs['dgimg'] = xr.where(~ind,np.nan,imgs['dgimg'])
    imgs['dgweight'] = xr.where(~ind,np.nan,imgs['dgweight'])
    imgs['dgsigma'] = xr.where(~ind,np.nan,imgs['dgsigma'])
    
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
    
    
    plt.savefig(outpath + 'fig4.png',bbox_inches='tight',dpi = 300)
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

    

    
    axs[0].scatter(x,y,c='C0',s=0.5,alpha=0.2)
    axs[0].set_xlim([-3.5,3.5])
    # axs[0].set_ylim([0,16000]) 
    axs[0].set_xticklabels([])
    axs[0].set_ylabel('Intensity [counts]')

    # Sort to interp
    m = m[xm.argsort()]
    m0 = m0[xm.argsort()]
    e = e[xm.argsort()]
    xm = xm[xm.argsort()]
    
    # interp for plotting
    xi = np.linspace(-3.5,3.5,1001)
    mi = np.interp(xi,xm[np.isfinite(m)],m[np.isfinite(m)],left=np.nan,right=np.nan)
    mi0 = np.interp(xi,xm[np.isfinite(m0)],m0[np.isfinite(m0)],left=np.nan,right=np.nan)
    ei = np.interp(xi,xm[np.isfinite(e)],e[np.isfinite(e)],left=np.nan,right=np.nan)

    axs[3].plot(xi,mi0,c='C0',linestyle='-',label='Initial $I_{bs}$')
    
    axs[3].plot(xi,mi,c='C1',linestyle='-',label='Final $I_{bs}$')
    axs[3].plot(xi,ei,c='C3',linestyle=':',label='$\\sigma$')
    axs[3].set_xlim([-3.5,3.5]) 
    axs[3].set_xlabel('$\\cos (\\alpha_s) / \cos(\\alpha_d)$')
    axs[3].set_ylabel('$I_{bs}$ and $\\sigma$ [counts]')
    axs[3].set_yscale('log')
    axs[3].legend(loc=(0.1,0.5),frameon=False)

    sc= axs[2].scatter(x,y,c=w,s=0.5,vmin=0, vmax=1)
    axs[2].plot(xi,mi,c='C1')
    axs[2].set_xlim([-3.5,3.5])
    axs[2].set_xticklabels([])
    axs[2].set_ylabel('Intensity [counts]')
    cbaxes = axs[2].inset_axes([.85,.3,.1,.03])    
    cb = plt.colorbar(sc,cax=cbaxes, ticks=[0,0.5,1.], orientation='horizontal')
    cb.set_label('Weight')
    
    axs[0].text(0.03, 0.94, 'a', fontsize=12, horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
    axs[1].text(0.03, 0.94, 'b', fontsize=12, horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
    axs[2].text(0.03, 0.94, 'c', fontsize=12, horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes)
    axs[3].text(0.03, 0.94, 'd', fontsize=12, horizontalalignment='center', verticalalignment='center', transform=axs[3].transAxes)
    
    plt.savefig(outpath + 'fig3.png',bbox_inches='tight',dpi = 300)
    plt.clf()
    plt.close()
    

def makeFig5(wic,s12,s13,idate,outpath):
    ''' 
    Plot BS model overview for all cameras
    
    wic,s12,s13 : xarray.Dataset
        Datasets with the FUV images
    idate : int
        Index tof date to be displayed
    outpath : str
        Path to where the figure is saved.
    
    '''
    
    wic = wic.isel(date=idate).copy()
    s12 = s12.isel(date=idate).copy()
    s13 = s13.isel(date=idate).copy()
    
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
    
    plt.savefig(outpath + 'fig5.png',bbox_inches='tight',dpi = 300)
    plt.clf()
    plt.close()

def makeFig6(wic,s12,s13,idate,outpath):
    ''' 
    Plot SH model overview for all cameras
    
    wic,s12,s13 : xarray.Dataset
        Datasets with the FUV images
    idate : int
        Index tof date to be displayed
    outpath : str
        Path to where the figure is saved.
    
    '''
    wic = wic.isel(date=idate).copy()
    s12 = s12.isel(date=idate).copy()
    s13 = s13.isel(date=idate).copy()
    
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
    
    
    
    plt.savefig(outpath + 'fig6.png',bbox_inches='tight',dpi = 300)
    plt.clf()
    plt.close()






    





    

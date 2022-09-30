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
import statsmodels.api as sm
import fuvpy as fuv
from polplot import pp
import matplotlib.path as path
from scipy.stats import binned_statistic
# events = glob.glob('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/data/fuv200012/north/*')
# events.sort()

# for e in events[2:3]:
#     wicfiles = glob.glob(e+'/*.idl')   
#     wic = fuv.readFUVimage(wicfiles)
#     wic = fuv.makeFUVdayglowModel(wic,transform='log')
#     wic = fuv.makeFUVshModel(wic,4,4)
    
 
    # wic.to_netcdf('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/data/fuv200012/wic_'+e[63:]+'.nc')
# events = glob.glob('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/data/fuv200012/north/*')
# events.sort()

# for e in events:
#     wicfiles = glob.glob(e+'/*.idl')   
#     wic = readFUVimage(wicfiles)
#     wic = makeFUVdayglowModelC(wic,transform='log')
#     wic = makeFUVshModelNew(wic,4,4)
    
#     wic.to_netcdf('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/data/fuv200012/wic_'+e[63:]+'.nc')

# wicfiles = glob.glob('/Users/aohma/BCSS-DAG Dropbox/TMP/lompe_tutorial_2022-02-20/simon_event/wic_data/*.idl')   
# wic = readFUVimage(wicfiles)
# wic = makeFUVdayglowModelC(wic,transform='log')
# wic = makeFUVshModelNew(wic,4,4)
# wic.to_netcdf('/Users/aohma/BCSS-DAG Dropbox/TMP/lompe_tutorial_2022-02-20/simon_event/wic_20020519.nc')


# Field to include:
# ind,date,row,col,mlat,mlt,hemisphere,img,dgimg,shimg,onset,rind 

# HDD events

# find time between images

# events = sorted(glob.glob('/Volumes/Seagate Backup Plus Drive/fuv/wic/idl/north/*'))

# for e in events:
#     wicfiles = glob.glob(e+'/*.idl')   
#     wic = fuv.readImg(wicfiles)
     
#     wic = fuv.makeDGmodelTest(wic,stop=1e-3,tKnotSep=240,minlat=40)
#     wic = fuv.makeSHmodel(wic,4,4,stop=1e-3,knotSep=240)
#     wic.to_netcdf('/Volumes/Seagate Backup Plus Drive/fuv/wic/nc/wic_'+e[53:]+'.nc')


events = glob.glob('/Volumes/Seagate Backup Plus Drive/fuv/wic/idl/north/*')
events.sort()

ddate = []
ddates = []
for e in events:
    wicfiles = glob.glob(e+'/*.idl')   
#     wic = fuv.readImg(wicfiles)
#     ddate.append(wic.date.diff(dim='date').mean().values/ np.timedelta64(1, 's'))
#     ddates.append(wic.date.diff(dim='date').values/ np.timedelta64(1, 's'))
    
#     wic = makeFUVdayglowModelC(wic,transform='log')
#     wic = makeFUVshModelNew(wic,4,4)
    
#     wic.to_netcdf('/Volumes/Seagate Backup Plus Drive/fuv/wic/nc/wic_'+e[53:]+'.nc')

# HDD events s12

# events = glob.glob('/Volumes/Seagate Backup Plus Drive/fuv/s12/idl/north/*')
# events.sort()

# for e in events:
#     wicfiles = glob.glob(e+'/*.idl')   
#     wic = readFUVimage(wicfiles)
#     wic = makeFUVdayglowModelC(wic)
#     wic['shmodel'] = wic['dgmodel']
#     wic['shimg'] = wic['dgimg']
#     wic['shweight'] = wic['dgweight']
    
#     wic.to_netcdf('/Volumes/Seagate Backup Plus Drive/fuv/s12/nc/s12_'+e[53:]+'.nc')

# wicfiles = glob.glob('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/data/wic/2001-06-01-2nd/*.idl')   
# wic = fuv.readImg(wicfiles)
# wic = fuv.makeDGmodel(wic,transform='log',stop=1e-2,tKnotSep=240)
# wic = fuv.makeSHmodel(wic,4,4,stop=1e-2,knotSep=240)

# wicfiles = glob.glob('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/python/git/fuvpy/data/wicFiles/*.idl')   
# wic = fuv.readImg(wicfiles)
# wic = fuv.makeDGmodel(wic,transform='log')
# wic = fuv.makeSHmodel(wic,4,4)

# events = sorted(glob.glob('/Volumes/Seagate Backup Plus Drive/fuv/wic/nc/wic*'))
# for e in events:
#     wic = xr.load_dataset(e)
#     bi = fuv.findBoundaries(wic,dampingVal=1e1)
#     bm = fuv.makeBoundaryModel(bi,knotSep=10,dampingValE=1e1,dampingValP=1e1)[0]
#     bb = fuv.makeBoundaryModelBSpline(bi,tKnotSep=10,dampingValE=1e1,dampingValP=1e1)[0]
#     # bm = fuv.calcFlux(bm)
#     # bm = fuv.calcIntensity(wic,bm)
    
#     bi.to_netcdf('/Volumes/Seagate Backup Plus Drive/fuv/wic/nc/bi_'+e[-17:])
#     bm.to_netcdf('/Volumes/Seagate Backup Plus Drive/fuv/wic/nc/bm_'+e[-17:])
#     bb.to_netcdf('/Volumes/Seagate Backup Plus Drive/fuv/wic/nc/bb_'+e[-17:])

# Boundaries
# events = sorted(glob.glob('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/data/fuv200012/nc/wic*'))
# events = sorted(glob.glob('/Volumes/Seagate Backup Plus Drive/fuv/wic/nc/wic*'))

# for e in events[:1]:
#     wic = xr.load_dataset(e)
    # bi = fuv.findBoundaries(wic,dampingVal=1e1)
    # bb = fuv.makeBoundaryModelBSpline(bi,tKnotSep=10,eL1=1e1,eL2=1e1,pL1=1e0,pL2=1e1)[0]

    # bb = fuv.calcFlux(bb)
    # bb = fuv.calcIntensity(wic,bb)
    
    # bi.to_netcdf('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/data/fuv200012/nc/bi_'+e[-17:])
    # bb.to_netcdf('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/data/fuv200012/nc/bb_'+e[-17:]) 

# bi = fuv.findBoundaries(wic)
# bm = fuv.makeBoundaryModel(bi,dampingValE=0.001,dampingValP=0.001)

# fi1 = fuv.calcFlux(bi.sel(lim=1))
# fm1 = fuv.calcFlux(bm)
# fi2 = fuv.calcFlux2(bi.sel(lim=1))
# fm2 = fuv.calcFlux2(bm)

# plt.figure()
# # fi1['openFlux'].plot()
# # fm1['openFlux'].plot()
# fi2['openFlux'].plot()
# fm2['openFlux'].plot()

# plt.figure()
# # fi1['auroralFlux'].plot()
# # fm1['auroralFlux'].plot()
# fi2['auroralFlux'].plot()
# fm2['auroralFlux'].plot()

## L-curve test and "discrepancy" test
# events = sorted(glob.glob('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/data/fuv200012/nc/wic*'))
# # lmbds = np.r_[0,np.geomspace(1e-4,1e5,10)]
# lmbds1 = np.array([1e-4,1e0,1e1,1e2,1e1,1e1,1e1,1e1])
# lmbds2 = np.array([0,0  ,0  ,0  ,1e0,1e1,1e2,1e3])
# n = len(lmbds1)

# bbs = []
# b_norm_r = []
# b_norm_m = []

# for e in events:
#     wic = xr.load_dataset(e)
#     bi = fuv.findBoundaries(wic,dampingVal=1e1)
#     # Gtemp,G = fuv.makeBoundaryModelBSpline(bi,tKnotSep=10,dampingValE=0,dampingValP=0)
#     for i in range(n):
#         # bm = fuv.makeBoundaryModel(bi,knotSep=10,dampingValE=l,dampingValP=1)[2]
#         blist = fuv.makeBoundaryModelBSpline(bi,tKnotSep=10,eL1=lmbds1[i],eL2=lmbds2[i],pL1=1e10)
#         # f_norm_r.append(bm[1])
#         # f_norm_m.append(bm[0])
#         bb = blist[0]
#         r = (90. - np.abs(bb['eqb']))
#         a = (bb.mlt.values - 6.)/12.*np.pi
#         bb['ex'] =  113.5*r*np.cos(a)
#         bb['ey'] =  113.5*r*np.sin(a)
#         r = (90. - np.abs(bb['ocb']))
#         a = (bb.mlt.values - 6.)/12.*np.pi
#         bb['px'] =  113.5*r*np.cos(a)
#         bb['py'] =  113.5*r*np.sin(a)
#         bbs.append(bb)
#         b_norm_r.append(blist[1][1])
#         b_norm_m.append(blist[1][0])

# bbc = []
# for i in range(n):
#     bbc.append(xr.concat([bbs[j] for j in range(i,len(bbs),n)], dim='date'))


# ## "Discrepancy" test
# events = sorted(glob.glob('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/data/fuv200012/nc/wic*'))
# #lmbds = np.geomspace(1e-5,1e6,12)
# lmbds = np.r_[0,np.geomspace(1e-4,1e5,10)]
# bbs = []

# for e in events[2:3]:
#     wic = xr.load_dataset(e)
#     bi = fuv.findBoundaries(wic,dampingVal=1e1)
#     # Gtemp,G = fuv.makeBoundaryModelBSpline(bi,tKnotSep=10,dampingValE=0,dampingValP=0)
#     for l in lmbds:
#         # bm = fuv.makeBoundaryModel(bi,knotSep=10,dampingValE=l,dampingValP=1)[2]
#         bb = fuv.makeBoundaryModelBSpline(bi,tKnotSep=10,eL1=1e1,eL2=1e3,pL1=1e0,pL2=l)[0]
#         # f_norm_r.append(bm[1])
#         # f_norm_m.append(bm[0])
#         # b_norm_r.append(bb[1])
#         # b_norm_m.append(bb[0])
        
#         r = (90. - np.abs(bb['ocb']))
#         a = (bb.mlt.values - 6.)/12.*np.pi
#         bb['ex'] = 113.5*r*np.cos(a)
#         bb['ey'] = 113.5*r*np.sin(a)
#         bbs.append(bb)
        


# pplot(wic.isel(date=slice(2,6)),'img',robust=True,cbar_kwargs={'extend':'both'})
# pplot(wic.isel(date=slice(2,6)),'img',row=True)
# pplot(wic.isel(date=slice(2,6)),'img',col_wrap=2)

# pplot(wic.isel(date=slice(2,6)),'img',cbar_kwargs={'orientation':'horizontal'})
# pplot(wic.isel(date=slice(2,6)),'img',row=True,cbar_kwargs={'orientation':'horizontal'})
# pplot(wic.isel(date=slice(2,6)),'img',col_wrap=2,cbar_kwargs={'orientation':'horizontal','extend':'max'})

# pplot(wic.isel(date=slice(2,6)),'img',add_cbar = False)
# pplot(wic.isel(date=slice(2,6)),'img',row=True,add_cbar = False)
# pplot(wic.isel(date=slice(2,6)),'img',col_wrap=2,add_cbar = False)

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
 
def weightedBinning(fraction,d,dm,w,sKnots,n_scp,sOrder):
    bins = np.r_[sKnots[0],np.linspace(0,sKnots[-1],51)]
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
    
    ind2 = np.isfinite(rmse) & (dm>0)
    # Fit using Bspline
    
    # test with smoother knots
    
    # G= BSpline(sKnots, np.eye(n_scp), sOrder)(fraction)
    # m =np.linalg.lstsq(G[ind2].T@G[ind2],G[ind2].T@rmse[ind2],rcond=None)[0]
    
    # test with linear 
    G = np.vstack((dm,np.ones_like(dm))).T
    m =np.linalg.lstsq(G[ind2].T@G[ind2],G[ind2].T@rmse[ind2],rcond=None)[0]
    
    # test with linear 
    G = np.vstack((dm**2,np.ones_like(dm))).T
    m =np.linalg.lstsq(G[ind2].T@G[ind2],G[ind2].T@rmse[ind2]**2,rcond=None)[0]
    
    return np.sqrt(G@m),rmse
    
def makeFig1(wic,outpath):
    ''' 
    Plot a few WIC images
    wic : Dataset with the images to be plotted
    outpath : path to where the image is saved
    
    '''  
    
    wic['img'].plot(x='col',y='row',col='date',vmin=0,vmax=12000,xticks=[],yticks=[],subplot_kws={'xlabel':''},cbar_kwargs={'label':'WIC intensity [counts]'})
    plt.savefig(outpath + 'wicExample.png',bbox_inches='tight',dpi = 300)
    plt.clf()
    plt.close()
    
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
def makeFig2(wic,outpath):
    ''' 
    Plot B-splines and dayglow fit for a wic image
    wic : Dataset with the image to be plotted. Must be a single date.
    outpath (str) : Path to where the image is saved
    '''
    wic = wic.copy()
    if len(wic.sizes)!=2: 
        raise Exception('multiple dates given.')
    
    
    fig,axs = plt.subplots(3,1,figsize=(9,6.5),constrained_layout = True)
    

    
    # B-spline
    sOrder = 3
    sKnots = [-5,-3,-1,-0.2,-0.1,0,0.1,0.2,1,3,5]
    sKnots = np.r_[np.repeat(sKnots[0],sOrder),sKnots, np.repeat(sKnots[-1],sOrder)]
    n_scp = len(sKnots)-sOrder-1
    G= BSpline(sKnots, np.eye(n_scp), sOrder)(np.linspace(-5,5,1001))
    G = np.where(G==0,np.nan,G)
    
    cmap = plt.get_cmap('cool',n_scp)
    for i in range(n_scp):
        axs[1].plot(np.linspace(-5,5,1001),G[:,i],c=cmap(i))
    axs[1].set_xticklabels([])
    axs[1].set_xlim([-5,5])
    axs[1].set_ylim([0,1])
    axs[1].set_ylabel('Spatial B-splines')
    
    
    # Dayglow
    x = np.cos(np.deg2rad(wic['sza'].values.flatten()))/np.cos(np.deg2rad(wic['dza'].values.flatten()))
    y = wic['img'].values.flatten()
    w = wic['dgweight'].values.flatten()
    m = wic['dgmodel'].values.flatten()
    e = wic['dgsigma'].values.flatten()
    
    y = y[x.argsort()]
    w = w[x.argsort()]
    m = m[x.argsort()]
    e = e[x.argsort()]
    x = x[x.argsort()]
    
    axs[0].scatter(x,10**y,c='C0',s=0.5,alpha=0.2)
    axs[0].set_xlim([-5,5])
    axs[0].set_ylim([0,16000]) 
    axs[0].set_xticklabels([])
    axs[0].set_ylabel('WIC intensity [counts]')
    
    sc= axs[2].scatter(x,y,c=w,s=0.5,vmin=0, vmax=1)
    axs[2].plot(x[np.isfinite(m)],m[np.isfinite(m)],c='r')
    axs[2].plot(x[np.isfinite(m)],m[np.isfinite(m)]-e[np.isfinite(m)],c='r')
    axs[2].plot(x[np.isfinite(m)],m[np.isfinite(m)]+e[np.isfinite(m)],c='r')
    axs[2].set_xlim([-5,5])
    # axs[2].set_ylim([6.1,10.2]) 
    axs[2].set_xlabel('$\\cos (\\alpha_s) / \cos(\\alpha_d)$')
    axs[2].set_ylabel('WIC intensity [counts]')
    
    # cbaxes = inset_axes(axs[1], width="1%", height="30%", loc=2) 
    cbaxes = axs[2].inset_axes([.85,.3,.1,.03]) 
    
    cb = plt.colorbar(sc,cax=cbaxes, ticks=[0,0.5,1.], orientation='horizontal')
    cb.set_label('Weight')
    
    axs[0].text(0.03, 0.94, 'a', fontsize=12, horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
    axs[1].text(0.03, 0.94, 'b', fontsize=12, horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
    axs[2].text(0.03, 0.94, 'c', fontsize=12, horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes)
    
    plt.savefig(outpath + 'dayglowFit.png',bbox_inches='tight',dpi = 300)
    plt.clf()
    plt.close()
    
def makeFig2b(wic,outpath):
    ''' 
    Plot B-splines and dayglow fit for a wic image
    wic : Dataset with the image to be plotted. Must be a single date.
    outpath (str) : Path to where the image is saved
    '''
    
    
    
    fig,axs = plt.subplots(3,1,figsize=(10,9),constrained_layout = True)
    
    # B-spline
    sOrder = 3
    sKnots = [-5,-3,-1,-0.2,-0.1,0,0.1,0.2,1,3,5]
    sKnots = np.r_[np.repeat(sKnots[0],sOrder),sKnots, np.repeat(sKnots[-1],sOrder)]
    n_scp = len(sKnots)-sOrder-1
    G= BSpline(sKnots, np.eye(n_scp), sOrder)(np.linspace(-5,5,1001))
    
    axs[0].plot(np.linspace(-5,5,1001),G)
    axs[0].set_xticklabels([])
    axs[0].set_xlim([-5,5])
    axs[0].set_ylim([0,1])
    axs[0].set_ylabel('B-splines')
    
    
    # Dayglow
    x = np.cos(np.deg2rad(wic.isel(date=67)['sza'].values.flatten()))/np.cos(np.deg2rad(wic.isel(date=67)['dza'].values.flatten()))
    y = np.log(wic.isel(date=67)['img'].values.flatten())
    w = wic.isel(date=67)['dgweight'].values.flatten()
    m = np.log(wic.isel(date=67)['dgmodel'].values.flatten())
    
    y = y[x.argsort()]
    w = w[x.argsort()]
    m = m[x.argsort()]
    x = x[x.argsort()]
    
    sc= axs[1].scatter(x,y,c=w,s=0.5,vmin=0, vmax=1)
    axs[1].plot(x[np.isfinite(m)],m[np.isfinite(m)],c='r')
    axs[1].set_xlim([-5,5]) 
    axs[1].set_xticklabels([])
    axs[1].set_ylabel('Original image [log(counts)]')
    
    # cbaxes = inset_axes(axs[1], width="1%", height="30%", loc=2) 
    cbaxes = axs[1].inset_axes([.85,.15,.1,.03]) 
    
    cb = plt.colorbar(sc,cax=cbaxes, ticks=[0,1.], orientation='horizontal')
    cb.set_label('Weight', labelpad=-8)
    
    
    # Temporal Dayglow
    idates = [27,67,107,147,187,227,267]
    cmap = plt.get_cmap('plasma',6)
    for i, idate in enumerate(idates):
        x = np.cos(np.deg2rad(wic.isel(date=idate)['sza'].values.flatten()))/np.cos(np.deg2rad(wic.isel(date=idate)['dza'].values.flatten()))
        m = np.log(wic.isel(date=idate)['dgmodel'].values.flatten())
        
        m = m[x.argsort()]
        x = x[x.argsort()]
        axs[2].plot(x[np.isfinite(m)],m[np.isfinite(m)],c=cmap(i))
        
    axs[2].set_xlim([-5,5]) 
    axs[2].set_xlabel('$x = \\cos (\\theta_s) / \cos(\\theta_d)$')
    axs[2].set_ylabel('Dayglow [log(counts)]')
    
    plt.savefig(outpath + 'dayglowFit.png',bbox_inches='tight',dpi = 300)
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
    
    # # Spatial weights
    # ws = np.full(sza.shape,np.nan)
    # for i in range(len(sza)):
    #     count,bin_edges,bin_number=binned_statistic(fraction[i,ind[i,:]],fraction[i,ind[i,:]],statistic=
    # 'count',bins=np.linspace(sKnots[0],sKnots[-1],21))
    #     ws[i,ind[i,:]]=1/np.maximum(10,count[bin_number-1])

    d_s = d.flatten()
    ind = ind.flatten()
    ws = np.ones_like(d_s)
    w = np.ones(d_s.shape)
    # sigma = np.zeros(d_s.shape)
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
        residuals = (d.flatten()[ind] - dm.flatten()[ind])#/dm.flatten()[ind]
        rmse = np.sqrt(np.average(residuals**2,weights=w[ind]))
        sigma = np.full_like(d_s,np.nan)
        sigmaBinned = np.full_like(d_s,np.nan)
        sigma[ind],sigmaBinned[ind] = weightedBinning(fraction.flatten()[ind], d.flatten()[ind],dm.flatten()[ind], w[ind],sKnots,n_scp,sOrder)
        # return sigma[ind]
        # # # Heteroskedasitic consistent covariance
        # # V = ((G_s[ind,:]*w[ind,None]*residuals[:,None]).T@(G_s[ind,:]*w[ind,None]*residuals[:,None]))
        # # GTG = (G_s[ind,:]*w[ind,None]).T@(G_s[ind,:]*w[ind,None])
        # C = np.linalg.inv((G_s[ind,:]*w[ind,None]).T@(G_s[ind,:]*w[ind,None])) @ ((G_s[ind,:]*w[ind,None]*residuals[:,None]).T@(G_s[ind,:]*w[ind,None]*residuals[:,None]))#@np.linalg.inv((G_s[ind,:]*w[ind,None]).T@(G_s[ind,:]*w[ind,None]))
        # # return C
        # # var_m = M@ (abs(np.diag(C))).reshape((n_scp, n_tcp)).T
        # var_m=[]
        # for i in range(n_scp*n_tcp):
        #     var_m.append(M@C[:,i].reshape((n_scp, n_tcp)).T)
        # return C,var_m,G_g
        # sigma = []
        # for i, tt in enumerate(time):
        #     sigma.append(np.sqrt(abs(G_g[i]@var_m[i, :])))
        # sigma=np.array(sigma).squeeze()
        # sigma = np.where(np.isnan(sigma),0,sigma)
        
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
        iw = ((residuals)/(tukeyVal*sigma[ind]))**2
        iw[iw>1] = 1
        w[ind] = (1-iw)**2
        
        if diff == 1e10:
            dm0 =dm
            sigma0 = sigma
            sigma0B = sigmaBinned

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
        imgs['dgmodel0'] = (['date','row','col'],(dm0).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['dgimg'] = imgs[inImg]-imgs['dgmodel']
    imgs['dgweight'] = (['date','row','col'],(w).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['dgsigma'] = (['date','row','col'],(sigma).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['dgsigma0'] = (['date','row','col'],(sigma0).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['dgsigmaB'] = (['date','row','col'],(sigmaBinned).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['dgsigma0B'] = (['date','row','col'],(sigma0B).reshape((n_t,len(imgs.row),len(imgs.col))))

    # Remove pixels outside model scope
    ind = (imgs.sza>=0)& (imgs.dza <= dzalim) & (imgs.glat >= minlat) & imgs.bad
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
    axs[0].set_ylim([-5,5])
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
    axs[2].set_ylim([-5,5])
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
    axs[1].set_xticklabels([])
    axs[1].set_xlim([-5,5])
    axs[1].set_ylim([0,1])
    axs[1].set_ylabel('Spatial B-splines')
    
    
    # Dayglow
    x = np.cos(np.deg2rad(wic['sza'].values.flatten()))/np.cos(np.deg2rad(wic['dza'].values.flatten()))
    y = wic['img'].values.flatten()
    w = wic['dgweight'].values.flatten()
    m = wic['dgmodel'].values.flatten()
    m0 = wic['dgmodel0'].values.flatten()
    e = wic['dgsigma'].values.flatten()
    e0 = wic['dgsigma0'].values.flatten()
    b = wic['dgsigmaB'].values.flatten()
    b0 = wic['dgsigma0B'].values.flatten()
    
    y = y[x.argsort()]
    w = w[x.argsort()]
    m = m[x.argsort()]
    m0 = m0[x.argsort()]
    e = e[x.argsort()]
    e0 = e0[x.argsort()]
    b = b[x.argsort()]
    b0 = b0[x.argsort()]
    x = x[x.argsort()]
    
    axs[0].scatter(x,y,c='C0',s=0.5,alpha=0.2)
    axs[0].set_xlim([-5,5])
    # axs[0].set_ylim([0,16000]) 
    axs[0].set_xticklabels([])
    axs[0].set_ylabel('Intensity [counts]')

    axs[3].plot(x[np.isfinite(m)],m[np.isfinite(m)],c='C1',linewidth=1)
    
    xi = np.linspace(-5,5,1001)
    mi = np.interp(xi,x[np.isfinite(m)],m[np.isfinite(m)])
    mi0 = np.interp(xi,x[np.isfinite(m0)],m0[np.isfinite(m0)])
    ei = np.interp(xi,x[np.isfinite(e)],e[np.isfinite(e)])
    ei0 = np.interp(xi,x[np.isfinite(e0)],e0[np.isfinite(e0)])

    axs[3].plot(xi,mi0,c='C0',linestyle='-',label='$I_{bs}$ first iteration')
    axs[3].plot(xi,ei0,c='C0',linestyle=':',label='$\\sigma_{bs}$ first iteration')
    
    axs[3].plot(xi,mi,c='C1',linestyle='-',label='$I_{bs}$ final iteration')
    axs[3].plot(xi,ei,c='C1',linestyle=':',label='$\\sigma_{bs}$ final iteration')
    # axs[3].plot(x[np.isfinite(b)],b[np.isfinite(b)],c='C1',linewidth=0.6)
    # axs[3].fill_between(x[np.isfinite(m)],m[np.isfinite(m)]-e[np.isfinite(m)],m[np.isfinite(m)]+e[np.isfinite(m)],facecolor='C1',edgecolor=None,alpha=0.4)
    
    # axs[3].plot(x[np.isfinite(b0)],b0[np.isfinite(b0)],c='C0',linewidth=0.6)
    # axs[3].fill_between(x[np.isfinite(m0)],m0[np.isfinite(m0)]-e0[np.isfinite(m0)],m0[np.isfinite(m0)]+e0[np.isfinite(m0)],facecolor='C0',edgecolor=None,alpha=0.4)
    axs[3].set_xlim([-5,5])
    # axs[2].set_ylim([0,15499]) 
    axs[3].set_xlabel('$\\cos (\\alpha_s) / \cos(\\alpha_d)$')
    axs[3].set_ylabel('$I_{bs}$ and $\\sigma_{bs}$ [counts]')
    axs[3].set_yscale('log')
    axs[3].legend(loc=(0.1,0.5),frameon=False)

    sc= axs[2].scatter(x,y,c=w,s=0.5,vmin=0, vmax=1)
    axs[2].plot(x[np.isfinite(m)],m[np.isfinite(m)],c='C1')
    # axs[2].fill_between(x[np.isfinite(m)],m[np.isfinite(m)]-e[np.isfinite(m)],m[np.isfinite(m)]+e[np.isfinite(m)],color='r',alpha=0.2)
    # axs[2].plot(x[np.isfinite(m)],e[np.isfinite(m)],c='C1')
    axs[2].set_xlim([-5,5])
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

def makeFig5(wic,outpath,sectors,inImg='shimg',limFactor=1,order=3,dampingVal=0):
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
    
    fig = plt.figure(figsize=(11,9))

    gs = gridspec.GridSpec(nrows=3,ncols=4,hspace=0.25,wspace=0.3)
    
    # Corr
    pax = pp(plt.subplot(gs[:,:3]),minlat=50)
    fuv.plotimg(wic,'shimg',pax=pax,crange=(0,200),cmap='Greens')
    cbaxes = pax.ax.inset_axes([.2,.0,.6,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='both')
    cb.set_label('Corrected image [Counts]')
    pax.ax.set_title(wic['id'].values.tolist() + ': ' + 
             wic['date'].dt.strftime('%Y-%m-%d %H:%M:%S').values.tolist(),pad=-550)
    pax.writeMLTlabels(mlat=49.5)
    pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center')
    
    mltRes=24
    edges = np.linspace(0,24,mltRes+1)

    # Circle in which all boundaries are assumed to be located
    colatMax = np.concatenate((np.linspace(40,30,mltRes+1)[1:-1:2],np.linspace(30,40,mltRes+1)[1:-1:2]))


    colatAll = 90-abs(wic['mlat'].values.copy().flatten())
    dAll = wic[inImg].values.copy().flatten()
    wdgAll = wic['shweight'].values.copy().flatten()
    jjj = (np.isfinite(dAll))&(colatAll<40)
    av = np.average(dAll[jjj],weights=wdgAll[jjj])
    mae = np.average(abs(dAll[jjj]-av),weights=wdgAll[jjj])
    
    print(av)
    print(mae)
    print(np.sqrt(np.average(dAll[jjj]**2,weights=wdgAll[jjj])))
    print(np.sqrt(np.average((dAll[jjj]-av)**2,weights=wdgAll[jjj])))
    
    colat = 90-abs(wic['mlat'].values.copy().flatten())
    mlt = wic['mlt'].values.copy().flatten()
    d = wic[inImg].values.copy().flatten()
    wDG = wic['shweight'].values.copy().flatten()

    ocb=[]
    eqb=[]
    for s in range(len(edges)-1):
        

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

        ev = np.linspace(0,40,401)
        if colatSec[iii].size == 0:
            ocb.append(np.nan)
            eqb.append(np.nan)
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

            
            G = BSpline(knots, np.eye(n_cp), order)(ev)
            dmSec = G.dot(m).flatten()
            dmSec[ev>np.max(colatSec[iii])]=np.nan

            ## identify main peak
            ind = (ev<colatMax[s])&(ev<np.max(colatSec[iii]))



            # isAbove = dmSec[ind]>avSec+limFactor*mae
            isAbove = dmSec[ind]>limFactor*mae
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

        ## Plot
        if s in sectors:
            ax = plt.subplot(gs[(sectors==s).nonzero()[0][0],3])
            ax.scatter(90-colatSec,dSec,c=1-wSec,s=.5)
            ax.plot(90-ev,dmSec,c='C1',linewidth = 1)
            ax.set_xlim([90,50])
            ax.set_ylim([-500,3000])
            ax.set_title(str(s) + '-' + str(s+1) + ' MLT' )
            ax.axhspan(-500,limFactor*mae,facecolor='k',edgecolor=None,alpha=0.2)
            
            ax.axvline(90-ocb[s],c='r',linewidth=1)
            ax.axvline(90-eqb[s],c='r',linewidth=1)
            
            ax.set_ylabel('Intensity [counts]')
            if (sectors==s).nonzero()[0][0] == 2:
                ax.set_xlabel('Magnetic latitude [deg]')
            else:
                ax.set_xticklabels([])


    ds = xr.Dataset(
        data_vars=dict(
            ocb=(['mlt'], 90-np.array(ocb)),
            eqb=(['mlt'], 90-np.array(eqb)),
            ),

        coords=dict(
            mlt = edges[:-1]+0.5
        ),
        )
    
    pax.scatter(ds.ocb.values, ds.mlt.values,c='r')
    pax.scatter(ds.eqb.values, ds.mlt.values,c='r')
    
    pax.plot([50,90],[sectors[0],sectors[0]],c='C0')
    pax.plot([50,90],[sectors[0]+1,sectors[0]+1],c='C0')
    pax.plot([50,90],[sectors[1],sectors[1]],c='C0')
    pax.plot([50,90],[sectors[1]+1,sectors[1]+1],c='C0')
    pax.plot([50,90],[sectors[2],sectors[2]],c='C0')
    pax.plot([50,90],[sectors[2]+1,sectors[2]+1],c='C0')
    
    plt.savefig(outpath + 'boundaryDetection.png',bbox_inches='tight',dpi = 300)
    plt.clf()
    plt.close()
  


def makeFig9(wic,bi,bf,outpath):
    if len(wic.date)!=20: 
        raise Exception('Must be 20 images.')
    
    n_lims = len(bi.lim)
    fig = plt.figure(figsize=(11,15))

    gs = gridspec.GridSpec(nrows=5,ncols=4,hspace=0.05,wspace=0)
    
    for i in range(5):
        for j in range(4):
            ## WIC ##
            t= 4*i+j
            pax = pp(plt.subplot(gs[i,j]),minlat=50)
            fuv.plotimg(wic.isel(date=t),'shimg',pax=pax,crange=(0,1000),cmap='Greens')
            pax.scatter(bi.isel(date=t)['ocb'].values,np.tile(bi.mlt.values,(5,1)).T,s=2,color='k')
            pax.scatter(bi.isel(date=t)['eqb'].values,np.tile(bi.mlt.values,(5,1)).T,s=2,color='k')
            pax.plot(bf.isel(date=t)['ocb'].values,bf.mlt.values,color='r')
            pax.plot(bf.isel(date=t)['eqb'].values,bf.mlt.values,color='r')
            
            pax.write(50, 12,wic.isel(date=t)['date'].dt.strftime('%H:%M:%S').values.tolist(),verticalalignment='bottom',horizontalalignment='center',fontsize=12)
            pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='right',fontsize=8)
            pax.write(50, 12, '12',verticalalignment='top',horizontalalignment='center',fontsize=8)
            pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='left',fontsize=8)
            pax.write(50, 24, '24',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
            pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
            
    cbaxes = pax.ax.inset_axes([-1.22,.05,.4,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes,ticks=[0,1000.],orientation='horizontal',extend='both')
    cb.set_label('Counts', labelpad=-8)
    
    
    plt.savefig(outpath + 'boundaryModel.png',bbox_inches='tight',dpi = 300)
    plt.clf()
    plt.close()
    
    
def lmbdaE(bi):
    dampingVals=np.geomspace(1e-4,1e3,22)
    
    norms_m_eb = []
    norms_r_eb = []
    for ii,lmbda in enumerate(dampingVals):
        bm,norms = fuv.makeBoundaryModel(bi,dampingValE=lmbda)
        norms_m_eb.append(norms[0])
        norms_r_eb.append(norms[1])
    
    return np.array(norms_m_eb),np.array(norms_r_eb)

def lmbdaP(bi,dampingValE):
    dampingVals=np.geomspace(1e-5,1e2,22)
    
    norms_m_pb = []
    norms_r_pb = []
    for ii,lmbda in enumerate(dampingVals):
        bm,norms = fuv.makeBoundaryModel(bi,dampingValE=dampingValE,dampingValP=lmbda)
        norms_m_pb.append(norms[2])
        norms_r_pb.append(norms[3])
    
    return np.array(norms_m_pb),np.array(norms_r_pb)    

def makeFig10(wic,bf,outpath):
    if len(wic.date)!=20: 
        raise Exception('Must be 20 images.')
    
    fig = plt.figure(figsize=(11,15))

    gs = gridspec.GridSpec(nrows=5,ncols=4,hspace=0.05,wspace=0)
    
    for i in range(5):
        for j in range(4):
            ## WIC ##
            t= 4*i+j
            pax = pp(plt.subplot(gs[i,j]),minlat=50)
            pax.plot(bf.isel(date=t)['ocb'].values,bf.mlt.values,color='r')
            # pax.plot(bf.isel(date=t)['eqb'].values,bf.mlt.values,color='r')
            pax.plotpins(bf.isel(date=t)['ocb'].values[::2],bf.mlt.values[::2],-bf.isel(date=t)['v_theta'].values[::2],bf.isel(date=t)['v_phi'].values[::2],SCALE=300,unit='m/s',markersize=0)
            # pax.plotpins(bf.isel(date=t)['eqb'].values[::2],bf.mlt.values[::2],-bf.isel(date=t)['u_theta'].values[::2],bf.isel(date=t)['u_phi'].values[::2],SCALE=500,markersize=0)
            
            pax.write(50, 12,wic.isel(date=t)['date'].dt.strftime('%H:%M:%S').values.tolist(),verticalalignment='bottom',horizontalalignment='center',fontsize=12)
            pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='right',fontsize=8)
            pax.write(50, 12, '12',verticalalignment='top',horizontalalignment='center',fontsize=8)
            pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='left',fontsize=8)
            pax.write(50, 24, '24',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
            pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
            
    
    
    plt.savefig(outpath + 'boundaryVel.png',bbox_inches='tight',dpi = 300)
    plt.clf()
    plt.close()   

def makeFigA1(path):
    events = glob.glob('/Volumes/Seagate Backup Plus Drive/fuv/wic/nc/wic*')
    events.sort()
    
    # wics = []
    # for e in events:
    #     wics.append(xr.load_dataset(e))
    # wic = xr.concat(wics, dim='date')
    wic = xr.load_dataset(events[3])
    
    fig,axs = plt.subplots(1,2,figsize=(10,5))
    
    wic.plot.scatter(x='sza',y='shimg',hue='shweight',ax=axs[0],s=0.01,alpha=0.4)
    wic['Model'] = wic['dgmodel'] + wic['shmodel']
    wic.plot.scatter(x='Model',y='img',hue='shweight',ax=axs[1],s=0.01,alpha=0.4)
    
    plt.savefig(path + 'dayglowPerformance.png',bbox_inches='tight',dpi=150)
    plt.clf()
    plt.close()


def makeFigX(path):
    ''' Make radial Fourier example '''
    
    mlt = np.linspace(0,24,241)
    phi = np.deg2rad(15*mlt)
    
    c = np.array([25,7,2,-3,1,-1,2])
    f = np.array([np.ones_like(phi),np.cos(phi),np.sin(phi),np.cos(2*phi),np.sin(2*phi),np.cos(3*phi),np.sin(3*phi)])
    
    fig,axs = plt.subplots(1,4,figsize=(10,3),constrained_layout = True)
    
    for i in range(4):
        pax = pp(axs[i])
        pax.plot(90-c[:1+2*i]@f[:1+2*i,:],mlt)
        if i == 0:
            pax.ax.set_title('$n = '+str(i) + '$\n $a_'+str(i)+' = '+str(c[0])+'$',pad=-10)
        else:
            pax.ax.set_title('$n = '+str(i) + '$\n $a_'+str(i)+' = '+str(c[2*i-1])+'$ and $b_'+str(i)+' = '+str(c[2*i])+'$',pad=-10)
        
    plt.savefig(path + 'shExample.pdf',bbox_inches='tight')
    plt.clf()
    plt.close()
    
    
def makeGIF(wic,bi,bm,outpath,minlat=50):

    ospath  = outpath.replace(r' ',r'\ ')

    wic['shimg'].attrs = {'long_name': 'Counts', 'units': ''}
  
    r = (90. - np.abs(bm['ocb']))/(90. - minlat)
    a = (bm.mlt.values - 6.)/12.*np.pi
    bm['px'] =  r*np.cos(a)
    bm['py'] =  r*np.sin(a)
    
    r = (90. - np.abs(bm['eqb']))/(90. - minlat)
    a = (bm.mlt.values - 6.)/12.*np.pi
    bm['ex'] =  r*np.cos(a)
    bm['ey'] =  r*np.sin(a)
    
    for t in range(len(bm.date)):
        fig,ax = plt.subplots(figsize=(7,6))
        pax = pp(ax,plotgrid=False)
        ax.fill(bm.isel(date=t)['ex'],bm.isel(date=t)['ey'],color='k')
        ax.fill(bm.isel(date=t)['px'],bm.isel(date=t)['py'],color='w')
        ax.set_title(bm.date[t].dt.strftime('%H:%M:%S').values.tolist(),y=1.0, pad=-30)
        
        plt.savefig(outpath + 'temp/binary'+str(t).zfill(4)+'.png',bbox_inches='tight',dpi=150)
        plt.clf()
        plt.close()
        
        fig,ax = plt.subplots(figsize=(7,6))
        pax = pp(ax)
        fuv.plotimg(wic.isel(date=t),'shimg',pax=pax,crange=(0,300),cmap='Greens')
        pax.scatter(bi.isel(date=t)['ocb'].values,np.tile(bi.mlt.values,(5,1)).T,s=1,color='k')
        pax.scatter(bi.isel(date=t)['eqb'].values,np.tile(bi.mlt.values,(5,1)).T,s=1,color='k')
        pax.plot(bm.isel(date=t)['ocb'].values,bm.mlt.values,color='r')
        pax.plot(bm.isel(date=t)['eqb'].values,bm.mlt.values,color='r')
        cbar = plt.colorbar(pax.ax.collections[0],ax=ax,extend='both')
        cbar.set_label(wic['shimg'].attrs['long_name'])
        
        ax.set_title(bm.date[t].dt.strftime('%H:%M:%S').values.tolist())
        plt.savefig(outpath + 'temp/wic'+str(t).zfill(4)+'.png',bbox_inches='tight',dpi=150)
        plt.clf()
        plt.close()

    os.system('convert '+ospath+'temp/wic*.png '+ospath+'imgs.gif')
    os.system('convert '+ospath+'temp/binary*.png '+ospath+'oval.gif')
    os.system('rm '+ospath+'temp/*.png')
    
def makeFigAutocorr(bi,outpath):
    bm,norms,ms = fuv.makeBoundaryModel(bi,knotSep=5,dampingValE=0,dampingValP=0)
    bm0,norms0,ms0 = fuv.makeBoundaryModel(bi,knotSep=15,dampingValE=10,dampingValP=2e2)
    
    x = 123/60*np.arange(len(sm.tsa.acf(ms[0][:,0])))
    
    fig,axs = plt.subplots(1,2,figsize=(6,3))
    # axs[0].plot(x,sm.tsa.acf(ms[0][:,0]))
    axs[0].plot(x,sm.tsa.acf(ms[0][:,1]))
    axs[0].plot(x,sm.tsa.acf(ms[0][:,2]))
    
    axs[0].legend(['$a_1$','$b_1$'],loc=1,frameon=False)
    axs[0].set_xlim([0,50])
    axs[0].set_xlabel('Minutes')
    axs[0].set_ylim([-0.5,1])
    
    
    # axs[1].plot(x,sm.tsa.acf(ms[0][:,0]-ms0[0][:,0]))
    axs[1].plot(x,sm.tsa.acf(ms[0][:,1]-ms0[0][:,1]))
    axs[1].plot(x,sm.tsa.acf(ms[0][:,2]-ms0[0][:,2]))
    
    axs[1].legend(['$a_1$','$b_1$'],loc=1,frameon=False)
    axs[1].set_xlim([0,50])
    axs[1].set_xlabel('Minutes')
    axs[1].set_ylim([-0.5,1])
    axs[1].set_yticklabels([])
    
    plt.savefig(outpath + 'boundary_autocorr.pdf',bbox_inches='tight')
    plt.clf()
    plt.close()
    
def calcIntensity(wic,bm,outpath):
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

def makeSHmodelTest(imgs,Nsh,Msh,order=2,dampingVal=0,tukeyVal=5,stop=1e-3,knotSep=None):
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
    from fuvpy.utils import sh
    from fuvpy.utils.sunlight import subsol
    
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
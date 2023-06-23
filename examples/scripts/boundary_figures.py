#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:29:45 2023

@author: aohma
"""

import numpy as np
import pandas as pd
import xarray as xr
from xhistogram.xarray import histogram

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.colors as colors

from scipy.stats import pearsonr, binned_statistic
from scipy.optimize import curve_fit
# from sklearn.linear_model import LinearRegression,HuberRegressor,RANSACRegressor,Lasso

import fuvpy as fuv
from polplot import pp

import cdflib # pip install cdflib
import os

def fig1(img,**kwargs):
    '''
    Plot a figure to illustrate the boundary detection method

    dataset with example image    
    '''


    # Set keyword arguments to input or default values    
    inImg = kwargs.pop('inImg') if 'inImg' in kwargs.keys() else 'shimg'
    lims = kwargs.pop('lims') if 'lims' in kwargs.keys() else np.arange(50,201,5)
    sigma = kwargs.pop('sigma') if 'sigma' in kwargs.keys() else 300
    height = kwargs.pop('height') if 'height' in kwargs.keys() else 130
    clat_ev = kwargs.pop('clat_ev') if 'clat_ev' in kwargs.keys() else np.arange(0.5,50.5,0.5)
    mlt_ev = kwargs.pop('mlt_ev') if 'mlt_ev' in kwargs.keys() else np.arange(0.5,24,1)
    mlt_profile = kwargs.pop('mlt_profile') if 'mlt_profile' in kwargs.keys() else [3,10,20]
    minlat = kwargs.pop('minlat') if 'minlat' in kwargs.keys() else 40 
    outpath = kwargs.pop('outpath') if 'outpath' in kwargs.keys() else None 
    
    # Constants    
    R_E = 6371 # Earth radius (km)
    R_I = R_E+height # Assumed emission radius (km)
    km_per_lat = np.pi*R_I/180
    
    # Evaluation coordinates in cartesian projection
    r_ev = km_per_lat*(np.abs(clat_ev))
    a_ev = (mlt_ev- 6.)/12.*np.pi
    x_ev =  r_ev[:,None]*np.cos(a_ev[None,:])
    y_ev =  r_ev[:,None]*np.sin(a_ev[None,:])

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

    df = pd.merge(df_pb,df_eb,left_index=True,right_index=True,how='outer')
    
    df[['pb','eb']]=90-df[['pb','eb']]

    ds = df.to_xarray()
    ds = ds.isel(date=0)

    # Add attributes
    ds['mlt'].attrs = {'long_name': 'Magnetic local time','unit':'hrs'}
    ds['pb'].attrs = {'long_name': 'Poleward boundary','unit':'deg'}
    ds['eb'].attrs = {'long_name': 'Equatorward boundary','unit':'deg'}

    lims_profile = [50,125,200]
    cmap = plt.cm.get_cmap('Reds',1+len(lims_profile))
    
    # Outer ring
    mlt_out = np.arange(0,24.1,0.1)
    mlat_out = 90-(40+10*np.cos(np.pi*mlt_out/12))
    mlat_min = minlat*np.ones_like(mlat_out)

    fig = plt.figure(figsize=(9,6))
    gs = gridspec.GridSpec(nrows=3,ncols=4,hspace=0,wspace=0.01)
    
    # Corr
    pax = pp(plt.subplot(gs[:,1:]),minlat=minlat)
    fuv.plotimg(img,'shimg',pax=pax,crange=(0,500),cmap='Greens')
    cbaxes = pax.ax.inset_axes([.3,.0,.4,.017]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='both')
    cb.set_label('Intensity [counts]')
    pax.ax.set_title(img['id'].values.tolist() + ': ' + 
             img['date'].dt.strftime('%Y-%m-%d %H:%M:%S').values.tolist(),pad=-550)
    pax.writeLTlabels(lat=minlat-0.1)

    # Lat ticks
    pax.write(80, 2, str(80),verticalalignment='center',horizontalalignment='center')
    pax.write(50, 2, str(50),verticalalignment='center',horizontalalignment='center')

    # Lower latitude
    pax.fill(np.concatenate((mlat_out,mlat_min)),np.concatenate((mlt_out,mlt_out[::-1])),color='C7',alpha=0.3,edgecolor=None)

    pax.ax.text(0.2,0.9,'d',horizontalalignment='center', verticalalignment='center', transform=pax.ax.transAxes,fontsize=12)

    bcd = 'abc'

    sc=[]
    for i,l in enumerate(lims_profile):
        sctemp = pax.scatter(ds.sel(lim=l).pb.values, ds.mlt.values,color=cmap(i+1),s=10,zorder=20)
        sc.append(sctemp)
        pax.scatter(ds.sel(lim=l).eb.values, ds.mlt.values,color=cmap(i+1),s=10,zorder=20)
    

    for i,p in enumerate(mlt_profile):
        pax.plot([minlat,90],[mlt_ev[p],mlt_ev[p]],c='C7',zorder=2,linewidth=1)
        ax = plt.subplot(gs[i,0])
        ax.plot(90-clat_ev,d_ev[:,p],c='g')
        ax.axvspan(90-(40+10*np.cos(np.pi*mlt_ev[p]/12)),minlat,facecolor='C7',edgecolor=None,alpha=0.3)

        ax.set_xlim([90,minlat])
        ax.set_ylim([-499,1999])
        #ax.set_title(str(mlt_ev[p]) + ' MLT' )
        ax.text(0.1, 0.9, bcd[i], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontsize=12)
        ax.text(0.8, 0.9, str(mlt_ev[p]) + ' MLT', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontsize=8)

        ax.set_ylabel('Intensity [counts]')
        if i == 2:
            ax.set_xticks([90,65,40])
            ax.set_xlabel('Magnetic latitude [$^\circ$]')
        else:
            ax.set_xticks([])

        for j,l in enumerate(lims_profile):
            ax.axvline(ds.sel(lim=l,mlt=mlt_ev[p])['pb'],color=cmap(j+1),linewidth=0.7)
            ax.axvline(ds.sel(lim=l,mlt=mlt_ev[p])['eb'],color=cmap(j+1),linewidth=0.7)
    
    # Add median ring
    r0 = 300/km_per_lat/(90-minlat)
    x0,y0=pax._latlt2xy(77,mlt_ev[p])
    pax.ax.scatter(x0,y0,s=15,zorder=25,c='k')
    a0 = np.linspace(0,2*np.pi,361)
    pax.ax.plot(x0+r0*np.cos(a0),y0+r0*np.sin(a0),zorder=22,c='k')

    ax.scatter(77,d_ev[np.argwhere(clat_ev==13),p],s=15,zorder=22,c='k')
    pax.ax.legend(sc,[' 50 counts','125 counts','200 counts'],frameon=False)

    if outpath: plt.savefig(outpath + 'fig01.png',bbox_inches='tight',dpi = 600)

def fig2(wic,bi,bf,outpath):
    if len(wic.date)!=16: 
        raise Exception('Must be 16 images.')
    
    fig = plt.figure(figsize=(11,12))

    gs = gridspec.GridSpec(nrows=4,ncols=4,hspace=0.05,wspace=0)
    
    for i in range(4):
        for j in range(4):
            ## WIC ##
            t= 4*i+j
            pax = pp(plt.subplot(gs[i,j]),minlat=50)
            fuv.plotimg(wic.isel(date=t),'shimg',pax=pax,crange=(0,1000),cmap='Greens')
            pax.scatter(bi.isel(date=t)['pb'].values,np.tile(bi.mlt.values,(len(bi.lim),1)).T,s=2,color='C6')
            pax.scatter(bi.isel(date=t)['eb'].values,np.tile(bi.mlt.values,(len(bi.lim),1)).T,s=2,color='C9')
            pax.plot(bf.isel(date=t)['pb'].values,bf.mlt.values,color='C3')
            pax.plot(bf.isel(date=t)['eb'].values,bf.mlt.values,color='C0')
            
            # ERRORS
            df = bf.isel(date=t)[['pb','pb_err','eb','eb_err']].to_dataframe().reset_index().set_index('date')
            mlat_err = np.concatenate((df['pb'].values+df['pb_err'].values,df['pb'].values[[0]]+df['pb_err'].values[[0]],df['pb'].values[[0]]-df['pb_err'].values[[0]],df['pb'].values[::-1]-df['pb_err'].values[::-1]))
            mlt_err = np.concatenate((df['mlt'].values,df['mlt'].values[[0,0]],df['mlt'].values[::-1]))
            pax.fill(mlat_err,mlt_err,color='C3',alpha=0.3,edgecolor=None)

            mlat_err = np.concatenate((df['eb'].values+df['eb_err'].values,df['eb'].values[[0]]+df['eb_err'].values[[0]],df['eb'].values[[0]]-df['eb_err'].values[[0]],df['eb'].values[::-1]-df['eb_err'].values[::-1]))
            pax.fill(mlat_err,mlt_err,color='C0',alpha=0.3,edgecolor=None)        
                    


            pax.write(50, 12,wic.isel(date=t)['date'].dt.strftime('%H:%M:%S').values.tolist(),verticalalignment='bottom',horizontalalignment='center',fontsize=12)
            pax.write(50,  6, '06',verticalalignment='center',horizontalalignment='right',fontsize=8)
            pax.write(50, 12, '12',verticalalignment='top',horizontalalignment='center',fontsize=8)
            pax.write(50, 18, '18',verticalalignment='center',horizontalalignment='left',fontsize=8)
            pax.write(50, 24, '24',verticalalignment='bottom',horizontalalignment='center',fontsize=8)
            pax.write(50, 9, '50',verticalalignment='center',horizontalalignment='center',fontsize=8)
            
    cbaxes = pax.ax.inset_axes([-1.22,.05,.4,.03]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes,ticks=[0,1000.],orientation='horizontal',extend='both')
    cb.set_label('Counts', labelpad=-8)
    
    
    plt.savefig(outpath + 'fig02.png',bbox_inches='tight',dpi = 600)
    plt.clf()
    plt.close()

def fig3(bi,bm,outpath):
    

    bm['v_pb'] = xr.where(bm['vn_pb']>0,np.sqrt(bm['ve_pb']**2+bm['vn_pb']**2),-np.sqrt(bm['ve_pb']**2+bm['vn_pb']**2))

    fig,axs = plt.subplots(1,4,figsize=(10,8))

    bi['pb_median'] = bi['pb'].median(dim='lim')
    #bi['pb_median'] = bi['pb'].sel(lim=200)
    bi['pb_median'].attrs = {'long_name':'Median boundary','unit':'$^\circ$'}
    bm['pb'].attrs = {'long_name':'Model boundary','unit':'$^\circ$'}
    bm['pb_err'].attrs = {'long_name':'Model uncertainty','unit':'$^\circ$'}
    bm['v_pb'].attrs = {'long_name':'Boundary velocity','unit':'m/s'}

    bi['pb_median'].plot(ax=axs[0],vmin=65,vmax=80,cbar_kwargs={'orientation':'horizontal','pad':0.09})
    bm['pb'].plot(ax=axs[1],vmin=65,vmax=80,cbar_kwargs={'orientation':'horizontal','pad':0.09})
    bm['pb_err'].plot(ax=axs[2],vmin=0,vmax=3,cbar_kwargs={'orientation':'horizontal','pad':0.09})
    bm['v_pb'].plot(ax=axs[3],vmin=-600,vmax=600,cbar_kwargs={'orientation':'horizontal','pad':0.09},cmap='coolwarm')

    abc = 'abcd'
    cc = 'wwww'

    for i in range(4):
        bm['isglobal'].plot.contour(ax=axs[i],colors='r')
        
        axs[i].set_xlabel('MLT [h]')
        axs[i].set_xticks([0,6,12,18])
        axs[i].text(0.05,0.94,abc[i],horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes,fontsize=12,color=cc[i])


    axs[0].yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axs[0].set_ylabel('Universsal time')

    for i in range(1,4):
        axs[i].set_ylabel('')
        axs[i].set_yticks([])
    plt.subplots_adjust(wspace=0.05)

    plt.savefig(outpath + 'fig03.png',bbox_inches='tight',dpi = 600)

def fig4(ds,outpath):
    fig = plt.figure(figsize=(12,5))
    gs = gridspec.GridSpec(nrows=2,ncols=6,hspace=0,wspace=0.05,width_ratios=[1,1,0.4,1,1,1])

    ebind = np.isfinite(ds['eb'])&np.isfinite(ds['eb_bas'])
    pbind = np.isfinite(ds['pb'])&np.isfinite(ds['pb_bas']) 

    ds['eb'] = xr.where(ebind,ds['eb'],np.nan)
    ds['eb_bas'] = xr.where(ebind,ds['eb_bas'],np.nan)
    ds['pb'] = xr.where(pbind,ds['pb'],np.nan)
    ds['pb_bas'] = xr.where(pbind,ds['pb_bas'],np.nan)

    pax = pp(plt.subplot(gs[:,:2]),minlat=50)
    pax.plot(np.r_[ds['eb'].median(dim='date'),ds['eb'].median(dim='date')[0]],np.r_[ds.mlt,0.5],c='C0')
    pax.plot(np.r_[ds['pb'].median(dim='date'),ds['pb'].median(dim='date')[0]],np.r_[ds.mlt,0.5],c='C0')
    pax.plot(np.r_[ds['eb_bas'].median(dim='date'),ds['eb_bas'].median(dim='date')[0]],np.r_[ds.mlt,0.5],c='C1')
    pax.plot(np.r_[ds['pb_bas'].median(dim='date'),ds['pb_bas'].median(dim='date')[0]],np.r_[ds.mlt,0.5],c='C1')
    pax.plot(np.r_[(ds['eb_bas']-ds['Leb']).median(dim='date'),(ds['eb_bas']-ds['Leb']).median(dim='date')[0]],np.r_[ds.mlt,0.5],c='C2')
    pax.plot(np.r_[(ds['pb_bas']-ds['Lpb']).median(dim='date'),(ds['pb_bas']-ds['Lpb']).median(dim='date')[0]],np.r_[ds.mlt,0.5],c='C2')

    pax.ax.set_title('Median boundary locations')


    pax.fill(np.r_[90,np.linspace(50,50,101),90],np.r_[10,np.linspace(10,16,101),16],facecolor='C7',edgecolor=None,alpha=0.3)
    pax.fill(np.r_[90,np.linspace(50,50,101),90],np.r_[22,np.linspace(22,24+6,101)%24,6],facecolor='C7',edgecolor=None,alpha=0.3)

    pax.writeLTlabels(49.5)
    pax.text(50,9,'50',verticalalignment='center',horizontalalignment='center')
    pax.ax.text(0.25,0,'This study',horizontalalignment='center',verticalalignment='center', transform=pax.ax.transAxes,color='C0')
    pax.ax.text(0.5,0,'BAS',horizontalalignment='center',verticalalignment='center', transform=pax.ax.transAxes,color='C1')
    pax.ax.text(0.75,0,'Corrected BAS',horizontalalignment='center',verticalalignment='center', transform=pax.ax.transAxes,color='C2')
    pax.ax.text(0.2,0.9,'a',horizontalalignment='center', verticalalignment='center', transform=pax.ax.transAxes,fontsize=12)

    ax = plt.subplot(gs[0,3])
    ds['eb'].plot.hist(ax=ax,histtype='step',bins=np.arange(45,91))
    ds['eb_bas'].plot.hist(ax=ax,histtype='step',bins=np.arange(45,91))
    (ds['eb_bas']-ds['Leb']).plot.hist(ax=ax,histtype='step',bins=np.arange(45,91))
    ax.tick_params(direction="in")
    ax.set_xticklabels([])
    ax.set_ylim([0,55000])
    ax.set_ylabel('Equatorward boundary [counts]')
    ax.set_title('Locations')
    ax.text(0.1,0.9,'b',horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontsize=12)

    ax = plt.subplot(gs[0,4])
    (ds['eb_bas']-ds['eb']).plot.hist(ax=ax,bins=np.arange(-10,10.1,0.5),histtype='step',color='C1')
    (ds['eb_bas']-ds['Leb']-ds['eb']).plot.hist(ax=ax,bins=np.arange(-10,10.1,0.5),histtype='step',color='C2')
    ax.tick_params(direction="in")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_ylim([0,55000])
    ax.set_title('Relavite locations')
    ax.text(0.1,0.9,'d',horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontsize=12)

    ax = plt.subplot(gs[1,3])
    ds['pb'].plot.hist(ax=ax,histtype='step',bins=np.arange(45,91))
    ds['pb_bas'].plot.hist(ax=ax,histtype='step',bins=np.arange(45,91))
    (ds['pb_bas']-ds['Lpb']).plot.hist(ax=ax,histtype='step',bins=np.arange(45,91))
    ax.tick_params(direction="in")
    ax.set_ylim([0,55000])
    ax.set_xlabel('Latitude [$^\circ$]')
    ax.set_ylabel('Poleward boundary [counts]')
    ax.text(0.1,0.9,'c',horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontsize=12)

    ax = plt.subplot(gs[1,4])
    (ds['pb_bas']-ds['pb']).plot.hist(ax=ax,bins=np.arange(-10,10.1,0.5),histtype='step',color='C1')
    (ds['pb_bas']-ds['Lpb']-ds['pb']).plot.hist(ax=ax,bins=np.arange(-10,10.1,0.5),histtype='step',color='C2')
    ax.tick_params(direction="in")
    ax.set_yticklabels([])
    ax.set_ylim([0,55000])
    ax.set_xlabel('Relative lat [$^\circ$]')
    ax.text(0.1,0.9,'e',horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontsize=12)

    ind = [6,7,8,9,16,17,18,19,20,21]
    ds2 = ds.isel(mlt=ind)

    ax = plt.subplot(gs[0,5])
    (ds2['eb_bas']-ds2['eb']).plot.hist(ax=ax,bins=np.arange(-10,10.1,0.5),histtype='step',color='C1')
    (ds2['eb_bas']-ds2['Leb']-ds2['eb']).plot.hist(ax=ax,bins=np.arange(-10,10.1,0.5),histtype='step',color='C2')
    ax.tick_params(direction="in")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_ylim([0,55000])
    ax.set_title('Relavite locations')
    ax.text(0.1,0.9,'f',horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontsize=12)
    #ax.axvline((ds2['eb_bas']-ds2['eb']).mean(),color='C1')
    #ax.axvline((ds2['eb_bas']-ds2['Leb']-ds2['eb']).mean(),color='C2')


    ax = plt.subplot(gs[1,5])
    (ds2['pb_bas']-ds2['pb']).plot.hist(ax=ax,bins=np.arange(-10,10.1,0.5),histtype='step',color='C1')
    (ds2['pb_bas']-ds2['Lpb']-ds2['pb']).plot.hist(ax=ax,bins=np.arange(-10,10.1,0.5),histtype='step',color='C2')
    ax.tick_params(direction="in")
    ax.set_yticklabels([])
    ax.set_ylim([0,55000])
    ax.set_xlabel('Relative lat [$^\circ$]')
    ax.text(0.1,0.9,'g',horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontsize=12)

    plt.savefig(outpath + 'fig04.png',bbox_inches='tight',dpi = 600)

def fig5(bb,outpath):
    bbins = np.arange(40,91,1)

    fig,axs = plt.subplots(1,3,figsize=(10,4))

    h = histogram(bb['pb'], bins=[bbins], dim=['date'],density=True)
    bb['pb'].quantile([0.25,0.75],dim='date').plot.line(ax=axs[0],x='mlt',c='w',add_legend=False)
    h['pb_bin'].attrs={'long_name':'Latitude','unit':'$^\circ$'}
    h.plot(x='mlt',ax=axs[0],add_colorbar=False,norm=colors.LogNorm(vmin=5e-3, vmax=2e-1))
    

    h = histogram(bb['eb'], bins=[bbins], dim=['date'],density=True)
    pc = h.plot(x='mlt',ax=axs[1],add_colorbar=False,norm=colors.LogNorm(vmin=5e-3, vmax=2e-1))
    bb['eb'].quantile([0.25,0.75],dim='date').plot.line(ax=axs[1],x='mlt',c='w',add_legend=False)

    bb['pb'].plot.hist(ax=axs[2], bins=bbins, density=True,histtype='step',edgecolor='C3',orientation='horizontal')
    bb['eb'].plot.hist(ax=axs[2], bins=bbins, density=True,histtype='step',edgecolor='C0',orientation='horizontal')

    # Colorbar
    cbaxes = axs[0].inset_axes([.1,.14,.4,.03]) 
    cb = plt.colorbar(pc,cax=cbaxes,ticks=[1e-2,1e-2,1e-1],orientation='horizontal')
    cb.set_label('counts',color='white')
    cb.outline.set_edgecolor('white')
    cbaxes.tick_params(axis='both', colors='white')

    # Labels and ticks
    axs[1].set_yticklabels([])
    axs[2].set_yticklabels([])
    axs[1].set_ylabel('')
    axs[2].set_ylim([bbins[0],bbins[-1]])

    axs[0].set_xticks([0,6,12,18])
    axs[1].set_xticks([0,6,12,18])
    axs[0].set_xlabel('MLT [h]')
    axs[1].set_xlabel('MLT [h]')
    axs[2].set_xlabel('Normalized counts')

    # Titles
    axs[0].set_title('Poleward boundary location')
    axs[1].set_title('Equatorward boundary location')
    axs[2].legend(['pb','eb'],frameon=False)

    # Background
    background = plt.cm.get_cmap('viridis')(0)
    axs[0].set_facecolor(background)
    axs[1].set_facecolor(background)

    # Letters
    axs[0].text(0.05,0.95,'a',horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes,fontsize=12,color='w')
    axs[1].text(0.05,0.95,'b',horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes,fontsize=12,color='w')
    axs[2].text(0.05,0.95,'c',horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes,fontsize=12,color='k')

    plt.subplots_adjust(wspace=0.05)

    plt.savefig(outpath + 'fig05.png',bbox_inches='tight',dpi = 600)

    print(bb['eb'].quantile([0.25,0.75],dim='date').diff(dim='qunatile'))

def fig6(bb,outpath):
    vbins = np.arange(-500,501,20)

    bb['v_eb'] = xr.where(bb['vn_eb']>0,np.sqrt(bb['ve_eb']**2+bb['vn_eb']**2),-np.sqrt(bb['ve_eb']**2+bb['vn_eb']**2))
    bb['v_pb'] = xr.where(bb['vn_pb']>0,np.sqrt(bb['ve_pb']**2+bb['vn_pb']**2),-np.sqrt(bb['ve_pb']**2+bb['vn_pb']**2))


    fig,axs = plt.subplots(1,3,figsize=(10,4))

    h = histogram(bb['v_pb'], bins=[vbins], dim=['date'],density=True)
    bb['v_pb'].quantile([0.25,0.75],dim='date').plot.line(ax=axs[0],x='mlt',c='w',add_legend=False)
    h['v_pb_bin'].attrs={'long_name':'Velocity','unit':'m/s'}
    h.plot(x='mlt',ax=axs[0],add_colorbar=False,norm=colors.LogNorm(vmin=5e-5, vmax=6e-3))
    


    h = histogram(bb['v_eb'], bins=[vbins], dim=['date'],density=True)
    pc = h.plot(x='mlt',ax=axs[1],add_colorbar=False,norm=colors.LogNorm(vmin=5e-5, vmax=6e-3))
    bb['v_eb'].quantile([0.25,0.75],dim='date').plot.line(ax=axs[1],x='mlt',c='w',add_legend=False)

    bb['v_pb'].plot.hist(ax=axs[2], bins=vbins, density=True,histtype='step',edgecolor='C3',orientation='horizontal')
    bb['v_eb'].plot.hist(ax=axs[2], bins=vbins, density=True,histtype='step',edgecolor='C0',orientation='horizontal')

    # Colorbar
    cbaxes = axs[1].inset_axes([.05,.14,.3,.03]) 
    cb = plt.colorbar(pc,cax=cbaxes,ticks=[1e-5,1e-4,1e-3],orientation='horizontal')
    cb.set_label('counts',color='white')
    cb.outline.set_edgecolor('white')
    cbaxes.tick_params(axis='both', colors='white')

    # Labels and ticks
    axs[1].set_yticklabels([])
    axs[2].set_yticklabels([])
    axs[1].set_ylabel('')
    axs[2].set_ylim([vbins[0],vbins[-1]])

    axs[0].set_xticks([0,6,12,18])
    axs[1].set_xticks([0,6,12,18])
    axs[0].set_xlabel('MLT [h]')
    axs[1].set_xlabel('MLT [h]')
    axs[2].set_xlabel('Normalized counts')

    # Titles
    axs[0].set_title('Poleward boundary velocity')
    axs[1].set_title('Equatorward boundary velocity')
    axs[2].legend(['pb','eb'],frameon=False)

    # BAckground
    background = plt.cm.get_cmap('viridis')(0)
    axs[0].set_facecolor(background)
    axs[1].set_facecolor(background)

    # Letters
    axs[0].text(0.05,0.95,'a',horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes,fontsize=12,color='w')
    axs[1].text(0.05,0.95,'b',horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes,fontsize=12,color='w')
    axs[2].text(0.05,0.95,'c',horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes,fontsize=12,color='k')

    plt.subplots_adjust(wspace=0.05)

    #(bb['ve_pb']-bb['ve_eb']).plot.hist(ax=axs[0,0], bins=ebins, density=True,histtype='step',edgecolor='C5')
    #axs[0,0].set_xlim(ebins[[0,-1]])
    #axs[0,0].tick_params(axis="y",direction="in", pad=-24)
    #axs[0,0].set_yticks([0.02,0.04])
    #axs[0,0].set_ylabel('Normalized counts')

    plt.savefig(outpath + 'fig06.png',bbox_inches='tight',dpi = 600)


def fig7_8_10(bb,outpath):
    bb['P'] = 1e-6*np.deg2rad(15*0.1)*(bb['dP']).sum(dim='mlt')
    bb['A'] = 1e-6*np.deg2rad(15*0.1)*(bb['dA']).sum(dim='mlt')
    bb['P_dt'] = 1e-3*np.deg2rad(15*0.1)*(bb['dP_dt']).sum(dim='mlt')
    bb['A_dt'] = 1e-3*np.deg2rad(15*0.1)*(bb['dA_dt']).sum(dim='mlt')
    bb['PhiN'] = bb['PhiD'] - bb['P_dt']
    bb['L'] = 1.63*bb['PhiN'] - bb['A_dt']
    # bb['LM'] = 1e3*(1/(20*60)*np.maximum(bb['A']-700,0)+1/(10*60*60)*bb['A'])
    bb['LM'] = 1.7*bb['PhiN'] - 1e3/(3.66*60*60)*np.maximum(bb['A'],0)
    
    #bb['LM'] = 1.63*bb['PhiN'] - bb['A']**4/6e9
    
    

    # Linear fit with bounds
    ind = np.isfinite(bb['L']) & (bb['A']>0) & (bb['A']<2000)
    
    count,bin_edges,bin_number=binned_statistic(bb['A'][ind],bb['A'][ind],statistic='count',bins=np.arange(0,2001,250))
    w=1/count[bin_number-1]
    
    def func(x,a,b):
        return a*x**b

    popt, pcov=curve_fit(func, bb['A'][ind], bb['L'][ind], bounds=(0,np.inf))
    testFit = func(bb['A'],*popt)
    #bb['LM'] = 1.63*bb['PhiN'] - testFit  
    
    def func2(x,a):
        return a*x

    print(popt)
    popt2, pcov2=curve_fit(func2, bb['A'][ind], bb['L'][ind], bounds=(0,np.inf))
    # testFit = func(bb['A'],*popt)
    testFit = func2(bb['A'],*popt2)
    # bb['LM'] = 1.4*bb['PhiN'] - testFit 
    print(popt2)
    bb['LM'] = 1.7*bb['PhiN'] - popt2[0]*np.maximum(bb['A'],0)
    ## MAGNETC FLUX AND dF/dt FIGURE

    fig,axs = plt.subplots(2,2,figsize=(6,6))
    
    bb['P'].plot.hist(axs[0,0],bins=np.arange(0,2001,25),histtype='step',color='C1')
    bb['A'].plot.hist(axs[0,0],bins=np.arange(0,2001,25),histtype='step',color='C2')
    # (1e-6*np.deg2rad(15*0.1)*(bb['dA']+bb['dP']).sum(dim='mlt')).plot.hist(axs[0],bins=np.arange(0,2701,25),histtype='step')
    
    axs[0,0].set_ylabel('Counts')
    axs[0,0].set_xlabel('Magnetic flux [MWb]')
    axs[0,0].legend(['P', 'A'],frameon=False)
    axs[0,0].set_xlim([0,2000])

    bb['P_dt'].plot.hist(axs[0,1],bins=np.arange(-500,501,10),histtype='step',color='C1')
    bb['A_dt'].plot.hist(axs[0,1],bins=np.arange(-500,501,10),histtype='step',color='C2')
    # (1e-3*np.deg2rad(15*0.1)*(bb['dA_dt']+bb['dP_dt']).sum(dim='mlt')).plot.hist(axs[1],bins=np.arange(-1000,1001,25),histtype='step')
    
    axs[0,1].set_ylabel('Counts')
    axs[0,1].set_xlabel('Change flux [kV]')
    axs[0,1].legend(['dP/dt', 'dA/dt'],frameon=False)
    axs[0,1].set_xlim([-500,500])

    axs[1,0].scatter(bb['P'],bb['A'],s=0.1,alpha=0.1)
    r = np.round(pearsonr(bb['P'],bb['A'])[0],3)
    axs[1,0].text(0.8,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[1,0].transAxes)

    axs[1,0].set_xlim([0,2000])
    axs[1,0].set_ylim([0,2000])
    axs[1,0].set_ylabel('A [MWb]')
    axs[1,0].set_xlabel('P [MWb]')

    x = bb['P_dt'].values
    y = bb['A_dt'].values
    ind = (x>-1000)&(x<1000)&(y>-1000)&(y<1000)
    x=x[ind]
    y=y[ind]
    
    m = np.linalg.eig(np.cov(x,y))[1][0][0]/np.linalg.eig(np.cov(x,y))[1][0][1]
    print(m)

    axs[1,1].scatter(bb['P_dt'],bb['A_dt'],s=0.1,alpha=0.1)
    r = np.round(pearsonr(bb['P_dt'],bb['A_dt'])[0],3)
    axs[1,1].text(0.8,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[1,1].transAxes)

    axs[1,1].set_xlim([-500,500])
    axs[1,1].set_ylim([-500,500])
    axs[1,1].set_ylabel('dA/dt [kV]')
    axs[1,1].set_xlabel('dP/dt [kV]')

    # Letters
    axs[0,0].text(0.05,0.95,'a',horizontalalignment='center',verticalalignment='center', transform=axs[0,0].transAxes)
    axs[1,0].text(0.05,0.95,'b',horizontalalignment='center',verticalalignment='center', transform=axs[1,0].transAxes)
    axs[0,1].text(0.05,0.95,'c',horizontalalignment='center',verticalalignment='center', transform=axs[0,1].transAxes)
    axs[1,1].text(0.05,0.95,'d',horizontalalignment='center',verticalalignment='center', transform=axs[1,1].transAxes)
    
    fig.tight_layout()
    plt.savefig(outpath + 'fig07.png',bbox_inches='tight',dpi = 600)
    plt.close()

    ## GEOMAGNETIC FORCING AND INDICES
    fig,axs = plt.subplots(2,3,figsize=(9,6))
    
    axs[0,0].scatter(bb['PhiD'],bb['A'],s=0.1,alpha=0.1)
    ind = np.isfinite(bb['PhiD'])
    r = np.round(pearsonr(bb['PhiD'][ind],bb['A'][ind])[0],3)
    axs[0,0].text(0.8,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[0,0].transAxes)

    axs[0,0].set_xlim([0,249.99])
    axs[0,0].set_ylim([0,2000])
    axs[0,0].set_ylabel('A [MWb]')
    axs[0,0].set_xlabel('$\\Phi_D$ [kV]')
    
    axs[0,1].scatter(bb['AE_INDEX'],bb['A'],s=0.1,alpha=0.1)
    ind = np.isfinite(bb['AE_INDEX'])
    r = np.round(pearsonr(bb['AE_INDEX'][ind],bb['A'][ind])[0],3)
    axs[0,1].text(0.8,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[0,1].transAxes)

    axs[0,1].set_xlim([0,1499.9])
    axs[0,1].set_ylim([0,2000])
    axs[0,1].set_yticklabels('')
    axs[0,1].set_xlabel('AE index [nT]')
    
    axs[0,2].scatter(bb['SYM_H'],bb['A'],s=0.1,alpha=0.1)
    ind = np.isfinite(bb['SYM_H'])
    r = np.round(pearsonr(bb['SYM_H'][ind],bb['A'][ind])[0],3)
    axs[0,2].text(0.8,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[0,2].transAxes)

    axs[0,2].set_xlim([-300,100])
    axs[0,2].set_ylim([0,2000])
    axs[0,2].set_yticklabels('')
    axs[0,2].set_xlabel('SYM-H index [nT]')

    axs[1,0].scatter(bb['PhiD'],bb['A_dt'],s=0.1,alpha=0.1)
    ind = np.isfinite(bb['PhiD'])
    r = np.round(pearsonr(bb['PhiD'][ind],bb['A_dt'][ind])[0],3)
    axs[1,0].text(0.8,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[1,0].transAxes)

    axs[1,0].set_xlim([0,249.9])
    axs[1,0].set_ylim([-500,500])
    axs[1,0].set_ylabel('dA/dt [kV]')
    axs[1,0].set_xlabel('$\\Phi_D$ [kV]')
    
    axs[1,1].scatter(bb['dAE_dt'],bb['A_dt'],s=0.1,alpha=0.1)
    ind = np.isfinite(bb['dAE_dt'])
    r = np.round(pearsonr(bb['dAE_dt'][ind],bb['A_dt'][ind])[0],3)
    axs[1,1].text(0.8,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[1,1].transAxes)

    axs[1,1].set_xlim([-50,50])
    axs[1,1].set_ylim([-500,500])
    axs[1,1].set_yticklabels('')
    axs[1,1].set_xlabel('d(AE)/dt [nT/min]')
    
    axs[1,2].scatter(bb['dSYMH_dt'],bb['A_dt'],s=0.1,alpha=0.1)
    ind = np.isfinite(bb['dSYMH_dt'])
    r = np.round(pearsonr(bb['dSYMH_dt'][ind],bb['A_dt'][ind])[0],3)
    axs[1,2].text(0.8,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[1,2].transAxes)

    axs[1,2].set_xlim([-5,5])
    axs[1,2].set_ylim([-500,500])
    axs[1,2].set_yticklabels('')
    axs[1,2].set_xlabel('d(SYM-H)/dt [nT/min]')
    
    # Letters
    axs[0,0].text(0.05,0.93,'a',horizontalalignment='center', verticalalignment='center', transform=axs[0,0].transAxes,fontsize=12)
    axs[0,1].text(0.05,0.93,'c',horizontalalignment='center', verticalalignment='center', transform=axs[0,1].transAxes,fontsize=12)
    axs[0,2].text(0.05,0.93,'e',horizontalalignment='center', verticalalignment='center', transform=axs[0,2].transAxes,fontsize=12)
    axs[1,0].text(0.05,0.93,'b',horizontalalignment='center', verticalalignment='center', transform=axs[1,0].transAxes,fontsize=12)
    axs[1,1].text(0.05,0.93,'d',horizontalalignment='center', verticalalignment='center', transform=axs[1,1].transAxes,fontsize=12)
    axs[1,2].text(0.05,0.93,'f',horizontalalignment='center', verticalalignment='center', transform=axs[1,2].transAxes,fontsize=12)

    plt.subplots_adjust(hspace=0.25,wspace=0.07)
    
    plt.savefig(outpath + 'fig08.png',bbox_inches='tight',dpi = 600)
    plt.close()

    ## AURORAL MODEL
    fig,axs = plt.subplots(1,3,figsize=(10,3))
    
    axs[0].scatter(bb['PhiD']-bb['P_dt'],bb['A_dt'],s=0.1,alpha=0.1)
    ind = np.isfinite(bb['PhiD'])
    r = np.round(pearsonr(bb['PhiD'][ind]-bb['P_dt'][ind],bb['A_dt'][ind])[0],3)
    axs[0].text(0.25,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[0].transAxes)

    axs[0].set_xlim([-500,500])
    axs[0].set_ylim([-500,500])
    axs[0].set_ylabel('dA/dt [kV]')
    axs[0].set_xlabel('$\\Phi_N$ [kV]')
    
    x = bb['PhiD'].values-bb['P_dt'].values
    y = bb['A_dt'].values
    ind = (x>-1000)&(x<1000)&(y>-1000)&(y<1000)
    x=x[ind]
    y=y[ind]
    
    m = np.linalg.eig(np.cov(x,y))[1][0][0]/np.linalg.eig(np.cov(x,y))[1][0][1]
    b = np.mean(y) - m * np.mean(x)
    axs[0].axline(xy1=(0, b), slope=m,color='C6',linestyle='--')
    axs[0].text(0.7,0.1,'dA/dt $\\propto$ '+str(np.round(m,1))+' $\\Phi_N$',horizontalalignment='center',verticalalignment='center', transform=axs[0].transAxes)

    axs[1].scatter(bb['PhiD'],1e-3*np.deg2rad(15*0.1)*bb['dA_dt'].sel(mlt=12),s=0.1,alpha=0.1)
    ind = np.isfinite(bb['PhiD'])
    r = np.round(pearsonr(bb['PhiD'][ind],1e-3*np.deg2rad(15*0.1)*bb['dA_dt'].sel(mlt=12)[ind])[0],3)
    axs[1].text(0.75,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[1].transAxes)
    
    axs[1].set_xlim([0,200])
    axs[1].set_ylim([-2,2])
    axs[1].set_ylabel('dA$_{12}$/dt [kV]')
    axs[1].set_xlabel('$\\Phi_D$ [kV]')

    axs[2].scatter(bb['A'],bb['L'],s=0.1,alpha=0.1)
    ind = np.isfinite(bb['PhiD'])
    r = np.round(pearsonr(bb['A'][ind],bb['L'][ind])[0],3)
    axs[2].text(0.75,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[2].transAxes)

    axs[2].set_xlim([0,2000])
    axs[2].set_ylim([-399,399])
    axs[2].set_ylabel('L [kV]')
    axs[2].set_xlabel('$A$ [MWb]')
    
    x = bb['A'].values[ind]
    y = bb['L'].values[ind]

    
    tau = 1/(popt2[0]*1e-3)/60/60
    axs[2].text(0.3,0.1,'L $\\propto$ A/'+str(np.round(tau,1))+' hrs',horizontalalignment='center',verticalalignment='center', transform=axs[2].transAxes)

    a = np.linspace(0,2000,101)
    axs[2].plot(a,func2(a,*popt2),color='C6',linestyle='--')
    #axs[2].plot(a,func(a,*popt),color='C8',linestyle=':')

    ind = np.isfinite(bb['PhiD'])
    r = np.round(pearsonr(bb['LM'][ind],bb['A_dt'][ind])[0],3)

    # Letters
    axs[0].text(0.05,0.93,'a',horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes,fontsize=12)
    axs[1].text(0.05,0.93,'b',horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes,fontsize=12)
    axs[2].text(0.05,0.93,'c',horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes,fontsize=12)
    
    print(r)

    plt.subplots_adjust(wspace=0.33)

    plt.savefig(outpath + 'fig10.png',bbox_inches='tight',dpi = 600)
    plt.close()

    ## GEOMAGNETIC FORCING AND BOUNDARIES
    fig,axs = plt.subplots(2,3,figsize=(9,6))
    
    axs[0,0].scatter(bb['PhiD'],bb['eb'].median(dim='mlt'),s=0.1,alpha=0.1)
    ind = np.isfinite(bb['PhiD'])
    r = np.round(pearsonr(bb['PhiD'][ind],bb['eb'].median(dim='mlt')[ind])[0],3)
    axs[0,0].text(0.8,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[0,0].transAxes)

    axs[0,0].set_xlim([0,249.99])
    axs[0,0].set_ylim([50,80])
    axs[0,0].set_ylabel('Equatorward boundary [$^\circ$]')
    axs[0,0].set_xticklabels('')
    
    axs[0,1].scatter(bb['AE_INDEX'],bb['eb'].median(dim='mlt'),s=0.1,alpha=0.1)
    ind = np.isfinite(bb['AE_INDEX'])
    r = np.round(pearsonr(bb['AE_INDEX'][ind],bb['eb'].median(dim='mlt')[ind])[0],3)
    axs[0,1].text(0.8,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[0,1].transAxes)

    axs[0,1].set_xlim([0,1499.9])
    axs[0,1].set_ylim([50,80])
    axs[0,1].set_yticklabels('')
    axs[0,1].set_xticklabels('')
    
    axs[0,2].scatter(bb['SYM_H'],bb['eb'].median(dim='mlt'),s=0.1,alpha=0.1)
    ind = np.isfinite(bb['SYM_H'])
    r = np.round(pearsonr(bb['SYM_H'][ind],bb['eb'].median(dim='mlt')[ind])[0],3)
    axs[0,2].text(0.8,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[0,2].transAxes)

    axs[0,2].set_xlim([-300,100])
    axs[0,2].set_ylim([50,80])
    axs[0,2].set_yticklabels('')
    axs[0,2].set_xticklabels('')

    axs[1,0].scatter(bb['PhiD'],bb['pb'].median(dim='mlt'),s=0.1,alpha=0.1)
    ind = np.isfinite(bb['PhiD'])
    r = np.round(pearsonr(bb['PhiD'][ind],bb['pb'].median(dim='mlt')[ind])[0],3)
    axs[1,0].text(0.8,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[1,0].transAxes)

    axs[1,0].set_xlim([0,249.99])
    axs[1,0].set_ylim([60,90])
    axs[1,0].set_ylabel('Poleward boundary [$^\circ$]')
    axs[1,0].set_xlabel('$\\Phi_D$ [kV]')
    
    axs[1,1].scatter(bb['AE_INDEX'],bb['pb'].median(dim='mlt'),s=0.1,alpha=0.1)
    ind = np.isfinite(bb['AE_INDEX'])
    r = np.round(pearsonr(bb['AE_INDEX'][ind],bb['pb'].median(dim='mlt')[ind])[0],3)
    axs[1,1].text(0.8,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[1,1].transAxes)

    axs[1,1].set_xlim([0,1499.9])
    axs[1,1].set_ylim([60,90])
    axs[1,1].set_yticklabels('')
    axs[1,1].set_xlabel('AE index [nT]')
    
    axs[1,2].scatter(bb['SYM_H'],bb['pb'].median(dim='mlt'),s=0.1,alpha=0.1)
    ind = np.isfinite(bb['SYM_H'])
    r = np.round(pearsonr(bb['SYM_H'][ind],bb['pb'].median(dim='mlt')[ind])[0],3)
    axs[1,2].text(0.8,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[1,2].transAxes)

    axs[1,2].set_xlim([-300,100])
    axs[1,2].set_ylim([60,90])
    axs[1,2].set_yticklabels('')
    axs[1,2].set_xlabel('SYM-H index [nT]')
    
    # Letters
    axs[0,0].text(0.05,0.93,'a',horizontalalignment='center', verticalalignment='center', transform=axs[0,0].transAxes,fontsize=12)
    axs[0,1].text(0.05,0.93,'c',horizontalalignment='center', verticalalignment='center', transform=axs[0,1].transAxes,fontsize=12)
    axs[0,2].text(0.05,0.93,'e',horizontalalignment='center', verticalalignment='center', transform=axs[0,2].transAxes,fontsize=12)
    axs[1,0].text(0.05,0.93,'b',horizontalalignment='center', verticalalignment='center', transform=axs[1,0].transAxes,fontsize=12)
    axs[1,1].text(0.05,0.93,'d',horizontalalignment='center', verticalalignment='center', transform=axs[1,1].transAxes,fontsize=12)
    axs[1,2].text(0.05,0.93,'f',horizontalalignment='center', verticalalignment='center', transform=axs[1,2].transAxes,fontsize=12)

    plt.subplots_adjust(hspace=0.07,wspace=0.07)
    
    plt.savefig(outpath + 'figXX.png',bbox_inches='tight',dpi = 600)
    plt.close()

def fig11(orbits,path,outpath):
    ''' Plot boundary evolution
    orbits (list) : list of orbit numbers to plot
    path (str) : path to boundary h5 file
    outpath (str) : path to save the image
    '''
    n=len(orbits)
    
    fig,axs = plt.subplots(n,2,figsize=(8,3))
    
    for i in range(n):
        
        bm = pd.read_hdf(path,where='orbit=="{}"'.format(orbits[i])).to_xarray()
        print(bm.date[[0,-1]])

        df = processOMNI(bm.date.values)
        df.index.name = 'date'
        bm = xr.merge((bm,df.to_xarray()))
        
        bm['A'] = 1e-3*np.deg2rad(15*0.1)*(bm['dA']).sum(dim='mlt')
        bm['A_dt'] = 1e-3*np.deg2rad(15*0.1)*(bm['dA_dt']).sum(dim='mlt')
        bm['P_dt'] = 1e-3*np.deg2rad(15*0.1)*(bm['dP_dt']).sum(dim='mlt')
        bm['PhiN'] = bm['PhiD']-bm['P_dt']
        bm['L'] = 1.63*bm['PhiN'] - bm['A_dt']
        bm['t2'] = 1/(3.66*60*60)*bm['A']

        time=(bm.date-bm.date[0]).values/ np.timedelta64(1, 'h')
    
        
        axs[i,0].plot(time,bm['A_dt'].values,color='C2')
        axs[i,0].plot(time,bm['PhiN'].values,color='C1')
        
        # axs[0].legend(['dA/dt [kV]','$\\Phi_N$ [kV]'],frameon=False,ncol=2)
        axs[0,0].text(0.25,0.88,'dA/dt [kV]',color='C2',horizontalalignment='center',verticalalignment='center', transform=axs[0,0].transAxes)
        axs[0,0].text(0.75,0.88,'$\\Phi_N$ [kV]',color='C1',horizontalalignment='center',verticalalignment='center', transform=axs[0,0].transAxes)

        axs[i,0].set_xlim([1,7.999])
        axs[i,0].set_ylim([-399,599])
        axs[i,0].set_ylabel('Orbit '+str(orbits[i]))
        if i != n-1:
            axs[i,0].xaxis.set_tick_params(labelbottom=False)
        else:
            axs[i,0].set_xlabel('time [h]')

        axs[i,1].plot(time,bm['L'].values,color='C2')
        axs[i,1].plot(time,bm['t2'].values,color='C4')
        
        axs[0,1].text(0.25,0.88,'L [kV]',color='C2',horizontalalignment='center',verticalalignment='center', transform=axs[0,1].transAxes)
        axs[0,1].text(0.75,0.88,'A/3.6 [kV]',color='C4',horizontalalignment='center',verticalalignment='center', transform=axs[0,1].transAxes)

        axs[i,1].set_xlim([1.001,8])
        axs[i,1].set_ylim([-399,599])
        axs[i,1].set_yticklabels('')
        if i != n-1:
            axs[i,1].xaxis.set_tick_params(labelbottom=False)
        else:
            axs[i,1].set_xlabel('time [h]')
    
    plt.subplots_adjust(wspace=0.0,hspace=0.0)
    plt.savefig(outpath + 'fig12.png',bbox_inches='tight',dpi = 300)
    plt.close()      

def load_bb(path,omnipath):
    '''Load the reduced boundary model dataset
    path (str) : path to the boundary model h5 file
    omnipath (str) : path to the omni dataset h5 file'''

    bm = pd.read_hdf(path).to_xarray()
    print(len(bm.date))

    ind0 = bm.isel(mlt=0)['isglobal'].values
    ind1 = bm.isel(mlt=0)['A_mean'].values > bm.isel(mlt=0)['P_mean'].values + 2*bm.isel(mlt=0)['P_std'].values
    ind2 = bm.isel(mlt=0)['A_mean'].values > bm.isel(mlt=0)['S_mean'].values + 2*bm.isel(mlt=0)['S_std'].values
    ind3 = bm.isel(mlt=0)['count'].values > 12

    ind4 = bm['pb_err'].quantile(0.75,dim='mlt').values<1.5
    ind5 = bm['eb_err'].quantile(0.75,dim='mlt').values<1.5

    dates = bm.date[ind0&ind1&ind2&ind3&ind4&ind5]
    
    bb = bm[['pb','eb','pb_err','eb_err','dP','dA','dP_dt','dA_dt','ve_pb','vn_pb','ve_eb','vn_eb']].sel(date=dates)
    
    df = processOMNI(bb.date.values,omnipath)
    df.index.name = 'date'
    bb = xr.merge((bb,df.to_xarray()))
    return bb

def load_bas(path):
    ''' Load BAS boundaries into xarray.Dataset
    path (str) : path to the BAS WIC boundary file'''
    # BAS WIC boundaries
    bas = pd.read_csv(path,index_col=0)

    mlt = np.arange(0.5,24)

    bas = bas.rename(columns={'Date_UTC_s':'date'})
    bas['date'] = pd.to_datetime(bas['date'])
    bas = bas.set_index('date')

    bas_eb = bas.iloc[:,1:25]
    bas_pb = bas.iloc[:,26:-1]

    bas_eb.columns = mlt
    bas_eb=bas_eb.stack()
    bas_eb.index = bas_eb.index.set_names('mlt',level=1)
    bas_eb.name='eb_bas'

    bas_pb.columns = mlt
    bas_pb=bas_pb.stack()
    bas_pb.index = bas_pb.index.set_names('mlt',level=1)
    bas_pb.name='pb_bas'

    bas = pd.merge(bas_eb,bas_pb,left_index=True,right_index=True,how='outer')
    bas = bas.to_xarray()
    #bas['date'] = bas.date.dt.round('1min')
    return bas

def combine_datasets(bb,bas):
    bf = bb[['eb','pb']].isel(mlt=slice(5,None,10))
    mlt = np.arange(0.5,24)
    bf['mlt'] = mlt
    ds = xr.merge((bf[['eb','pb']],bas))

    Cpb = [1.0298, -1.1249, -0.7380, 0.1838, -0.6171]
    Lpb = Cpb[0] + Cpb[1]*np.cos(np.deg2rad(15*mlt))+Cpb[2]*np.sin(np.deg2rad(15*mlt))+Cpb[3]*np.cos(2*np.deg2rad(15*mlt))+Cpb[4]*np.sin(2*np.deg2rad(15*mlt))

    ds['Lpb'] = (['mlt'],Lpb)

    Cpb = [-0.4935,	-2.1186, 0.3188,0.5749, -0.3118]
    Lpb = Cpb[0] + Cpb[1]*np.cos(np.deg2rad(15*mlt))+Cpb[2]*np.sin(np.deg2rad(15*mlt))+Cpb[3]*np.cos(2*np.deg2rad(15*mlt))+Cpb[4]*np.sin(2*np.deg2rad(15*mlt))

    ds['Leb'] = (['mlt'],Lpb)

    return ds

def loadOMNI(inputfile,fromDate='1981-01-01',toDate='2020-01-01'):
    # Load and process omni data

    # Columns to include from the unaltered omni data stored in omni_1min.h5.
    # This file is generated by the omni_download_1min_data() function.
    columns = ['BX_GSE','BY_GSM', 'BZ_GSM','flow_speed','Vx','Vy','Vz',
   'proton_density','Pressure','AL_INDEX','AE_INDEX', 'SYM_H']

    # Read the file
    omni = pd.read_hdf(inputfile,where='index>="{}"&index<"{}"'.format(fromDate,toDate),columns=columns)
    omni = omni.rename(columns={"BX_GSE":"Bx","BY_GSM":"By","BZ_GSM":"Bz","flow_speed":"V"})
    
  
    return omni

def processOMNI(dates,omnipath):
    fromDate = '2000-05-01 00:00'
    toDate = '2003-01-01 00:00'
    
    omni = loadOMNI(omnipath,fromDate,toDate)
    omni = omni.rolling(10,min_periods=0,center=True).mean()
    
    omni['dAE_dt'] = np.gradient(omni['AE_INDEX'])
    omni['dSYMH_dt'] = np.gradient(omni['SYM_H'])
    
    allDates=np.concatenate((omni.index.values,dates))
    omni = omni.reindex(allDates).sort_index()
    omni = omni[~omni.index.duplicated(keep='first')]
    omni = omni.interpolate(method='cubic',limit=1,limit_direction='both')
    
    omni['ca'] = np.arctan2(omni['By'],omni['Bz'])
    
    B_y,B_z,V_x = omni['By']*1e-9, omni['Bz']*1e-9, abs(omni['V'])*1e3
    B_t = np.sqrt(B_y**2+B_z**2)
    omni['PhiD'] = 3.3e2*V_x**(4./3)*B_t*np.sin(abs(omni['ca'])/2)**(9./2)
    
    
    return omni.loc[dates]


def validinput(inputstr, positive_answer, negative_answer):
    answer= input(inputstr+'\n').lower()
    if answer==positive_answer:
        return True
    elif answer== negative_answer:
        return False
    else:
        print('Invalid response should be either '+ str(positive_answer)+ ' or ' +str(negative_answer))
        return validinput(inputstr, positive_answer, negative_answer)
def download_omni_1min(fromYear,toYear,monthFirstYear=1,monthLastYear=12, path='./', file='omni_1min.h5'):
    '''
    The function downloads omni 1min data and stores it in a hdf file. 
    
    Parameters
    ==========
    fromYear : int
        Download from and including fromYear
    toYear : int,
        Download to and including toYear
    monthFirstYear : int, default 1
        First month to include from the first year.
    monthLastYear : int, default 12
        Last month to include from the last year.
    '''
    
    if fromYear < 1981:
        raise ValueError('fromYear must be >=1981')
    if os.path.isfile(path+file):
        if not validinput('file already exists and more omni will be added which can lead to duplication of data continue? (y/n)', 'y', 'n'):
            raise ValueError('User Cancelled Download, Alter file name or path or remove or move the existing file and retry')
    years = np.arange(fromYear,toYear+1,1)
    months= []
    for i in np.arange(1,13,1): months.append('%02i' % i)
        
    for y in years:
        for m in months:
            if not ((y==years[0])& (int(m)<monthFirstYear)) | ((y==years[-1]) & (int(m)>monthLastYear)):
                command = 'wget https://cdaweb.gsfc.nasa.gov/sp_phys/data/omni/hro_1min/' + str(y) + \
                    '/omni_hro_1min_' + str(y) + str(m) + '01_v01.cdf'
                os.system(command)
                
                omni = pd.DataFrame()
                cdf_file = cdflib.CDF('omni_hro_1min_' + str(y) + str(m) + '01_v01.cdf')
                varlist = cdf_file.cdf_info()['zVariables']
                for v in varlist:
                    omni[v] = cdf_file.varget(v)
                    fillval = cdf_file.varattsget(v)['FILLVAL']
                    omni[v] = omni[v].replace(fillval,np.nan)
                omni.index = pd.to_datetime(cdflib.cdfepoch.unixtime(cdf_file.varget('Epoch')),unit='s')
                omni[['AE_INDEX','AL_INDEX','AU_INDEX', 'PC_N_INDEX']] = omni[['AE_INDEX','AL_INDEX','AU_INDEX', 'PC_N_INDEX']].astype('float64')
                omni.to_hdf(path+file,'omni',mode='a',append=True,format='t', data_columns=True)
                cdf_file.close()
                os.remove('omni_hro_1min_' + str(y) + str(m) + '01_v01.cdf')
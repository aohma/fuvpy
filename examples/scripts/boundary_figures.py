#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:29:45 2023

@author: aohma
"""

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.stats import pearsonr, binned_statistic
from scipy.optimize import curve_fit
# from sklearn.linear_model import LinearRegression,HuberRegressor,RANSACRegressor,Lasso

import fuvpy as fuv
from polplot import pp

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
    gs = gridspec.GridSpec(nrows=3,ncols=4,hspace=0,wspace=0.3)
    
    # Corr
    pax = pp(plt.subplot(gs[:,:3]),minlat=minlat)
    fuv.plotimg(img,'shimg',pax=pax,crange=(0,500),cmap='Greens')
    cbaxes = pax.ax.inset_axes([.3,.0,.4,.02]) 
    cb = plt.colorbar(pax.ax.collections[0],cax=cbaxes, orientation='horizontal',extend='both')
    cb.set_label('Intensity [counts]')
    pax.ax.set_title(img['id'].values.tolist() + ': ' + 
             img['date'].dt.strftime('%Y-%m-%d %H:%M:%S').values.tolist(),pad=-550)
    pax.writeLTlabels(lat=minlat+0.5)

    # Lat ticks
    pax.write(80, 2, str(80),verticalalignment='center',horizontalalignment='center')
    pax.write(50, 2, str(50),verticalalignment='center',horizontalalignment='center')

    # Lower latitude
    pax.fill(np.concatenate((mlat_out,mlat_min)),np.concatenate((mlt_out,mlt_out[::-1])),color='C7',alpha=0.3,edgecolor=None)

    #pax.plot(mlat_out,mlt_out,c='C7',linewidth=1)



    for i,l in enumerate(lims_profile):
        pax.scatter(ds.sel(lim=l).pb.values, ds.mlt.values,color=cmap(i+1),s=10,zorder=20)
        pax.scatter(ds.sel(lim=l).eb.values, ds.mlt.values,color=cmap(i+1),s=10,zorder=20)
    

    for i,p in enumerate(mlt_profile):
        pax.plot([minlat,90],[mlt_ev[p],mlt_ev[p]],c='C7',zorder=2,linewidth=1)
        ax = plt.subplot(gs[i,3])
        ax.plot(90-clat_ev,d_ev[:,p],c='g')
        ax.axvspan(90-(40+10*np.cos(np.pi*mlt_ev[p]/12)),minlat,facecolor='C7',edgecolor=None,alpha=0.3)

        ax.set_xlim([90,minlat])
        ax.set_ylim([-499,1999])
        #ax.set_title(str(mlt_ev[p]) + ' MLT' )
        ax.text(0.25, 0.9, str(mlt_ev[p]) + ' MLT', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        ax.set_ylabel('Intensity [counts]')
        if i == 2:
            ax.set_xticks([90,65,40])
            ax.set_xlabel('Magnetic latitude [deg]')
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

    print(np.argwhere(clat_ev==13))

    ax.scatter(77,d_ev[np.argwhere(clat_ev==13),p],s=15,zorder=22,c='k')

    if outpath: plt.savefig(outpath + 'fig01.png',bbox_inches='tight',dpi = 600)




def load_bb():
    path = '/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/data/fuvAuroralFlux/'
    bm = pd.read_hdf(path+'hdf/final_boundaries.h5').to_xarray()
    
    ind = bm.isel(mlt=0)['isglobal'].values & ((bm.isel(mlt=0)['rmse_in'].values/bm.isel(mlt=0)['rmse_out'].values)>3) & (bm['pb_err'].quantile(0.75,dim='mlt').values<1.5) & (bm['eb_err'].quantile(0.75,dim='mlt').values<1.5)
    dates = bm.date[ind]
    
    bb = bm[['pb','eb','pb_err','eb_err','dP','dA','dP_dt','dA_dt','ve_pb','vn_pb','ve_eb','vn_eb']].sel(date=dates)
    
    df = processOMNI(bb.date.values)
    df.index.name = 'date'
    bb = xr.merge((bb,df.to_xarray()))
    return bb



def loadOMNI(fromDate='1981-01-01',toDate='2020-01-01'):
    # Load and process omni data

    # Columns to include from the unaltered omni data stored in omni_1min.h5.
    # This file is generated by the omni_download_1min_data() function.
    columns = ['BX_GSE','BY_GSM', 'BZ_GSM','flow_speed','Vx','Vy','Vz',
   'proton_density','Pressure','AL_INDEX','AE_INDEX', 'SYM_H']

    # Read the file
    inputfile = '/Users/aohma/BCSS-DAG dropbox/Anders Ohma/data/omni_1min.h5'
    omni = pd.read_hdf(inputfile,where='index>="{}"&index<"{}"'.format(fromDate,toDate),columns=columns)
    omni = omni.rename(columns={"BX_GSE":"Bx","BY_GSM":"By","BZ_GSM":"Bz","flow_speed":"V"})
    
    # # Window when estimating mean and variance of the IMF clock angle
    # caStart = 60 # Minutes before each measurement
    # caEnd = 60 # Minutes after each measurement

    # ca = pd.DataFrame()
    # ca['ca'] = np.arctan2(omni['By'],omni['Bz'])
    # ca['sin'] = np.sin(ca['ca'])
    # ca['cos'] = np.cos(ca['ca'])
    # ca[['sinMean','cosMean']] = ca[['sin','cos']].rolling(window=caStart+caEnd,min_periods=60).mean()
    # ca['caMean'] = np.rad2deg(np.arctan2(ca['sinMean'],ca['cosMean']))
    # ca['caVar'] = 1-np.sqrt(ca['sinMean']**2+ca['cosMean']**2) # Circular variance
    # ca['caCount'] = ca['ca'].rolling(window=caStart+caEnd,min_periods=60).count()
    # ca['BtMean'] = np.sqrt(omni['By']**2+omni['Bz']**2).rolling(window=caStart+caEnd,min_periods=60).mean()
    
    # B_y,B_z,V_x = omni['By']*1e-9, omni['Bz']*1e-9, abs(omni['V'])*1e3
    # B_t = np.sqrt(B_y**2+B_z**2)
    
    # ca['PhiD'] = 3.3e2*V_x**(4./3)*B_t*np.sin(abs(ca['ca'])/2)**(9./2)
    # ca['PhiDmean'] = ca['PhiD'].rolling(window=caStart+caEnd,min_periods=60).mean()
    
    # ca.index = ca.index.shift(-caEnd,freq='min')
    # omni = omni.join(ca[['ca','caMean','caVar','caCount','PhiDmean']])
    
    return omni

def processOMNI(dates):
    fromDate = '2000-05-01 00:00'
    toDate = '2003-01-01 00:00'
    
    omni = loadOMNI(fromDate,toDate)
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

def plotplot(bb,outpath):
    bb['P'] = 1e-6*np.deg2rad(15*0.1)*(bb['dP']).sum(dim='mlt')
    bb['A'] = 1e-6*np.deg2rad(15*0.1)*(bb['dA']).sum(dim='mlt')
    bb['P_dt'] = 1e-3*np.deg2rad(15*0.1)*(bb['dP_dt']).sum(dim='mlt')
    bb['A_dt'] = 1e-3*np.deg2rad(15*0.1)*(bb['dA_dt']).sum(dim='mlt')
    bb['PhiN'] = bb['PhiD'] - bb['P_dt']
    bb['L'] = 1.4*bb['PhiN'] - bb['A_dt']
    # bb['LM'] = 1e3*(1/(20*60)*np.maximum(bb['A']-700,0)+1/(10*60*60)*bb['A'])
    bb['LM'] = 1.4*bb['PhiN'] - 1e3/(1.8*60*60)*np.maximum(bb['A'],0)
    
    bb['LM'] = 1.4*bb['PhiN'] - bb['A']**4/6e9
    
    

    # Linear fit with bounds
    ind = np.isfinite(bb['L'])
    
    count,bin_edges,bin_number=binned_statistic(bb['A'][ind],bb['A'][ind],statistic='count',bins=np.arange(0,2001,100))
    # w=1/count[bin_number-1]
    
    def func(x,a,b):
        return a*x**b

    popt, pcov=curve_fit(func, bb['A'][ind], bb['L'][ind], bounds=(0,np.inf))
    testFit = func(bb['A'],*popt)
    bb['LM'] = 1.4*bb['PhiN'] - testFit  
    
    def func2(x,a):
        return a*x

    print(popt)
    popt2, pcov2=curve_fit(func2, bb['A'][ind], bb['L'][ind], bounds=(0,np.inf))
    # testFit = func(bb['A'],*popt)
    testFit = func2(bb['A'],*popt2)
    # bb['LM'] = 1.4*bb['PhiN'] - testFit 
    print(popt2)
    fig,axs = plt.subplots(1,3,figsize=(9,3))
    
    (1e-6*np.deg2rad(15*0.1)*(bb['dP']).sum(dim='mlt')).plot.hist(axs[0],bins=np.arange(0,2001,25),histtype='step',color='C1')
    (1e-6*np.deg2rad(15*0.1)*(bb['dA']).sum(dim='mlt')).plot.hist(axs[0],bins=np.arange(0,2001,25),histtype='step',color='C2')
    # (1e-6*np.deg2rad(15*0.1)*(bb['dA']+bb['dP']).sum(dim='mlt')).plot.hist(axs[0],bins=np.arange(0,2701,25),histtype='step')
    
    # axs[0].set_ylabel('Count')
    axs[0].set_xlabel('Magnetic flux [MWb]')
    axs[0].legend(['P', 'A'],frameon=False)
    
    bb['P_dt'].plot.hist(axs[1],bins=np.arange(-500,501,10),histtype='step',color='C1')
    bb['A_dt'].plot.hist(axs[1],bins=np.arange(-500,501,10),histtype='step',color='C2')
    # (1e-3*np.deg2rad(15*0.1)*(bb['dA_dt']+bb['dP_dt']).sum(dim='mlt')).plot.hist(axs[1],bins=np.arange(-1000,1001,25),histtype='step')
    
    axs[1].set_xlabel('Change flux [kV]')
    axs[1].legend(['dP/dt', 'dA/dt'],frameon=False)
    plt.tight_layout()
    
    axs[2].scatter(bb['P'],bb['A'],s=0.1,alpha=0.1)
    r = np.round(pearsonr(bb['P'],bb['A'])[0],3)
    axs[2].text(0.75,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[2].transAxes)

    axs[2].set_xlim([0,2000])
    axs[2].set_ylim([0,2000])
    axs[2].set_ylabel('A [MWb]')
    axs[2].set_xlabel('P [MWb]')
    
    fig.tight_layout()
    plt.savefig(outpath + 'fig01.png',bbox_inches='tight',dpi = 300)
    plt.close()
    
    fig,axs = plt.subplots(1,3,figsize=(9,3))
    
    bb['ve_pb'].plot.hist(axs[0],bins=np.arange(-150,151,2),histtype='step',color='C3')
    bb['ve_eb'].plot.hist(axs[0],bins=np.arange(-150,151,2),histtype='step',color='C0')
    axs[0].set_xlim([-150,150])
    axs[0].set_ylim([0,2e6])
    
    axs[0].set_xlabel('Eastward velocity [m/s]')
    axs[0].legend(['pb','eb'],frameon=False)
    
    bb['vn_pb'].plot.hist(axs[1],bins=np.arange(-750,751,10),histtype='step',color='C3')
    bb['vn_eb'].plot.hist(axs[1],bins=np.arange(-750,751,10),histtype='step',color='C0')
    axs[1].set_xlim([-750,750])
    axs[1].set_ylim([0,1.0e6])
    
    axs[1].set_xlabel('Northward velocity [m/s]')
    axs[1].legend(['pb','eb'],frameon=False)
    
    np.sqrt(bb['ve_pb']**2+bb['vn_pb']**2).plot.hist(axs[2],bins=np.arange(0,751,10),histtype='step',color='C3')
    np.sqrt(bb['ve_eb']**2+bb['vn_eb']**2).plot.hist(axs[2],bins=np.arange(0,751,10),histtype='step',color='C0')
    axs[2].set_xlim([0,750])
    axs[2].set_ylim([0,1.5e6])
    axs[2].set_xlabel('Speed [m/s]')
    axs[2].legend(['pb','eb'],frameon=False)
    plt.tight_layout()
    # (1e-3*np.deg2rad(15*0.1)*(bb['dA_dt']).sum(dim='mlt')).plot.hist(axs[1],bins=np.arange(-1000,1001,25),histtype='step')
    # (1e-3*np.deg2rad(15*0.1)*(bb['dA_dt']+bb['dP_dt']).sum(dim='mlt')).plot.hist(axs[1],bins=np.arange(-1000,1001,25),histtype='step')
    
    plt.savefig(outpath + 'fig02.png',bbox_inches='tight',dpi = 300)
    plt.close()
    
    fig,axs = plt.subplots(1,2,figsize=(6,3))
    
    axs[0].scatter(1e-3*np.deg2rad(15*0.1)*bb['dP_dt'].sum(dim='mlt'),1e-3*np.deg2rad(15*0.1)*bb['dA_dt'].sum(dim='mlt'),s=0.1,alpha=0.1)
    r = np.round(pearsonr(bb['dP_dt'].sum(dim='mlt'),bb['dA_dt'].sum(dim='mlt'))[0],3)
    axs[0].text(0.75,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[0].transAxes)

    axs[0].set_xlim([-500,500])
    axs[0].set_ylim([-500,500])
    axs[0].set_ylabel('dA/dt [kV]')
    axs[0].set_xlabel('dP/dt [kV]')
    
    x = 1e-3*np.deg2rad(15*0.1)*bb['dP_dt'].sum(dim='mlt').values
    y = 1e-3*np.deg2rad(15*0.1)*bb['dA_dt'].sum(dim='mlt').values
    
    # ind = (x>-1000)&(x<1000)&(y>-1000)&(y<1000)
    # x=x[ind]
    # y=y[ind]
    # count,bin_edges,bin_number=binned_statistic(y,y,statistic='count',bins=np.arange(-1000,1001,100))
    # w=1/count[bin_number-1]
    
    # count,bin_edges,bin_number=binned_statistic(x,x,statistic='count',bins=np.arange(-1000,1001,100))
    # w=w*1/count[bin_number-1]

    # reg = LinearRegression().fit(x[:, None],y,w)
    # b = reg.intercept_
    # m = reg.coef_[0]
    # axs[0].axline(xy1=(0, b), slope=m, color='C1')
    
    # reg = HuberRegressor().fit(x[:, None],y,w)
    # b = reg.intercept_
    # m = reg.coef_[0]
    # axs[0].axline(xy1=(0, b), slope=m, color='C2')
    
    # reg = RANSACRegressor().fit(x[:, None],y,w)
    # b = reg.estimator_.intercept_
    # m = reg.estimator_.coef_[0]
    # axs[0].axline(xy1=(0, b), slope=m, color='C3')
    # label=f'$y = {m:.1f}x {b:+.1f}$'

    m = np.linalg.eig(np.cov(x,y))[1][0][0]/np.linalg.eig(np.cov(x,y))[1][0][1]
    b = np.mean(y) - m * np.mean(x)
    axs[0].axline(xy1=(0, b), slope=m,color='C6',linestyle='--')
    axs[0].text(0.3,0.1,'dA/dt $\\propto$ '+str(np.round(m,1))+' dP/dt',horizontalalignment='center',verticalalignment='center', transform=axs[0].transAxes)

    

    axs[1].scatter(bb['PhiD']-1e-3*np.deg2rad(15*0.1)*bb['dP_dt'].sum(dim='mlt'),1e-3*np.deg2rad(15*0.1)*bb['dA_dt'].sum(dim='mlt'),s=0.1,alpha=0.1)
    ind = np.isfinite(bb['PhiD'])
    r = np.round(pearsonr(bb['PhiD'][ind]-1e-3*np.deg2rad(15*0.1)*bb['dP_dt'].sum(dim='mlt')[ind],1e-3*np.deg2rad(15*0.1)*bb['dA_dt'].sum(dim='mlt')[ind])[0],3)
    axs[1].text(0.25,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[1].transAxes)

    axs[1].set_xlim([-500,500])
    axs[1].set_ylim([-500,500])
    axs[1].set_ylabel('dA/dt [kV]')
    axs[1].set_xlabel('$\\Phi_N$ [kV]')
    
    x = bb['PhiD'].values-1e-3*np.deg2rad(15*0.1)*bb['dP_dt'].sum(dim='mlt').values
    y = 1e-3*np.deg2rad(15*0.1)*bb['dA_dt'].sum(dim='mlt').values
    ind = (x>-1000)&(x<1000)&(y>-1000)&(y<1000)
    x=x[ind]
    y=y[ind]
    
    m = np.linalg.eig(np.cov(x,y))[1][0][0]/np.linalg.eig(np.cov(x,y))[1][0][1]
    b = np.mean(y) - m * np.mean(x)
    axs[1].axline(xy1=(0, b), slope=m,color='C6',linestyle='--')
    axs[1].text(0.7,0.1,'dA/dt $\\propto$ '+str(np.round(m,1))+' $\\Phi_N$',horizontalalignment='center',verticalalignment='center', transform=axs[1].transAxes)

    
    # axs[2].scatter(bb['PhiD'],1e-3*np.deg2rad(15*0.1)*bb['dA_dt'].sel(mlt=12),s=0.1,alpha=0.1)
    # ind = np.isfinite(bb['PhiD'])
    # r = np.round(pearsonr(bb['PhiD'][ind],1e-3*np.deg2rad(15*0.1)*bb['dA_dt'].sel(mlt=12)[ind])[0],3)
    # axs[2].text(0.75,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[2].transAxes)

    # axs[2].set_xlim([0,250])
    # axs[2].set_ylim([-5,5])
    # axs[2].set_ylabel('dA$_{12}$/dt [kV]')
    # axs[2].set_xlabel('$\\Phi_D$ [kV]')
    
    plt.tight_layout()
    plt.savefig(outpath + 'fig03.png',bbox_inches='tight',dpi = 300)
    plt.close()
    # dA/dt vs A
    
    fig,axs = plt.subplots(1,3,figsize=(9,3))
    
    ind = np.isfinite(bb['dP_dt'].sum(dim='mlt'))
    # ind = bb['dP_dt'].sum(dim='mlt')>0
    
    axs[0].scatter(1e-6*np.deg2rad(15*0.1)*bb['dP'].sum(dim='mlt')[ind],abs(1e-3*np.deg2rad(15*0.1)*bb['dA_dt'].sum(dim='mlt')[ind]),s=0.1,alpha=0.1)
    r = np.round(pearsonr(bb['dP'].sum(dim='mlt')[ind],abs(bb['dA_dt'].sum(dim='mlt')[ind]))[0],3)
    axs[0].text(0.75,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[0].transAxes)

    axs[0].set_xlim([0,2000])
    axs[0].set_ylim([0,1000])
    axs[0].set_ylabel('dA/dt [kV]')
    axs[0].set_xlabel('P [MWb]')
    
    axs[1].scatter(1e-6*np.deg2rad(15*0.1)*bb['dA'].sum(dim='mlt')[ind],abs(1e-3*np.deg2rad(15*0.1)*bb['dA_dt'].sum(dim='mlt')[ind]),s=0.1,alpha=0.1)
    r = np.round(pearsonr(bb['dA'].sum(dim='mlt')[ind],abs(bb['dA_dt'].sum(dim='mlt')[ind]))[0],3)
    axs[1].text(0.75,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[1].transAxes)

    axs[1].set_xlim([0,2000])
    axs[1].set_ylim([0,1000])
    axs[1].set_ylabel('dA/dt [kV]')
    axs[1].set_xlabel('A [MWb]')
    
    axs[2].scatter(1e-6*np.deg2rad(15*0.1)*(bb['dP']+bb['dA']).sum(dim='mlt')[ind],abs(1e-3*np.deg2rad(15*0.1)*bb['dA_dt'].sum(dim='mlt')[ind]),s=0.1,alpha=0.1)
    r = np.round(pearsonr(1e-6*np.deg2rad(15*0.1)*(bb['dP']+bb['dA']).sum(dim='mlt')[ind],abs(1e-3*np.deg2rad(15*0.1)*bb['dA_dt'].sum(dim='mlt')[ind]))[0],3)
    axs[2].text(0.75,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[2].transAxes)

    axs[2].set_xlim([700,2700])
    axs[2].set_ylim([0,1000])
    axs[2].set_ylabel('dA/dt (kV)')
    axs[2].set_xlabel('T [MWb]')
    
    plt.tight_layout()
    
    plt.savefig(outpath + 'fig04.png',bbox_inches='tight',dpi = 300)
    plt.close()
    
    fig,axs = plt.subplots(1,3,figsize=(9,3))
    
    axs[0].scatter(bb['PhiD'],1e-6*np.deg2rad(15*0.1)*bb['dA'].sum(dim='mlt'),s=0.1,alpha=0.1)
    ind = np.isfinite(bb['PhiD'])
    r = np.round(pearsonr(bb['PhiD'][ind],bb['dA'].sum(dim='mlt')[ind])[0],3)
    axs[0].text(0.75,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[0].transAxes)

    axs[0].set_xlim([0,250])
    axs[0].set_ylim([0,2000])
    axs[0].set_ylabel('A [MWb]')
    axs[0].set_xlabel('$\\Phi_D$ [kV]')
    
    axs[1].scatter(bb['AE_INDEX'],1e-6*np.deg2rad(15*0.1)*bb['dA'].sum(dim='mlt'),s=0.1,alpha=0.1)
    ind = np.isfinite(bb['AE_INDEX'])
    r = np.round(pearsonr(bb['AE_INDEX'][ind],1e-6*np.deg2rad(15*0.1)*bb['dA'].sum(dim='mlt')[ind])[0],3)
    axs[1].text(0.75,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[1].transAxes)

    axs[1].set_xlim([0,1500])
    axs[1].set_ylim([0,2000])
    axs[1].set_ylabel('A [MWb]')
    axs[1].set_xlabel('AE index [nT]')
    
    axs[2].scatter(bb['SYM_H'],1e-6*np.deg2rad(15*0.1)*bb['dA'].sum(dim='mlt'),s=0.1,alpha=0.1)
    ind = np.isfinite(bb['SYM_H'])
    r = np.round(pearsonr(bb['SYM_H'][ind],1e-6*np.deg2rad(15*0.1)*bb['dA'].sum(dim='mlt')[ind])[0],3)
    axs[2].text(0.75,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[2].transAxes)

    axs[2].set_xlim([-300,100])
    axs[2].set_ylim([0,2000])
    axs[2].set_ylabel('A [MWb]')
    axs[2].set_xlabel('SYM_H index [nT]')
    
    plt.tight_layout()
    
    plt.savefig(outpath + 'fig05.png',bbox_inches='tight',dpi = 300)
    plt.close()
    
    fig,axs = plt.subplots(1,2,figsize=(6,3))
    
    # axs[0].scatter(bb['PhiD'],bb['L'],s=0.1,alpha=0.1)
    # ind = np.isfinite(bb['PhiD'])
    # r = np.round(pearsonr(bb['PhiD'][ind],bb['L'][ind])[0],3)
    # axs[0].text(0.75,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[0].transAxes)

    # axs[0].set_xlim([0,200])
    # axs[0].set_ylim([-399,399])
    # axs[0].set_ylabel('L [kV]')
    # axs[0].set_xlabel('$\\Phi_D$ [kV]')
    
    axs[1].scatter(bb['PhiD'],1e-3*np.deg2rad(15*0.1)*bb['dA_dt'].sel(mlt=12),s=0.1,alpha=0.1)
    ind = np.isfinite(bb['PhiD'])
    r = np.round(pearsonr(bb['PhiD'][ind],1e-3*np.deg2rad(15*0.1)*bb['dA_dt'].sel(mlt=12)[ind])[0],3)
    axs[1].text(0.75,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[1].transAxes)
    
    axs[1].set_xlim([0,200])
    axs[1].set_ylim([-2,2])
    axs[1].set_ylabel('dA$_{12}$/dt [kV]')
    axs[1].set_xlabel('$\\Phi_D$ [kV]')
    

    
    # axs[1].scatter(bb['AE_INDEX'],1e-6*np.deg2rad(15*0.1)*(bb['dP']+bb['dA']).sum(dim='mlt'),s=0.1,alpha=0.1)
    # ind = np.isfinite(bb['AE_INDEX'])
    # r = np.round(pearsonr(bb['AE_INDEX'][ind],1e-6*np.deg2rad(15*0.1)*(bb['dP']+bb['dA']).sum(dim='mlt')[ind])[0],3)
    # axs[1].text(0.75,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[1].transAxes)

    # axs[1].set_xlim([0,1000])
    # axs[1].set_ylim([700,2700])
    # axs[1].set_ylabel('T [MWb]')
    # axs[1].set_xlabel('AE index [nT]')
    
    # axs[2].scatter(bb['SYM_H'],1e-6*np.deg2rad(15*0.1)*(bb['dP']+bb['dA']).sum(dim='mlt'),s=0.1,alpha=0.1)
    # ind = np.isfinite(bb['SYM_H'])
    # r = np.round(pearsonr(bb['SYM_H'][ind],1e-6*np.deg2rad(15*0.1)*(bb['dP']+bb['dA']).sum(dim='mlt')[ind])[0],3)
    # axs[2].text(0.75,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[2].transAxes)

    # axs[2].set_xlim([-300,100])
    # axs[2].set_ylim([700,2700])
    # axs[2].set_ylabel('T [MWb]')
    # axs[2].set_xlabel('SYM_H index [nT]')
    
    plt.tight_layout()
    
    plt.savefig(outpath + 'fig06.png',bbox_inches='tight',dpi = 300)
    plt.close()
    
    fig,axs = plt.subplots(1,2,figsize=(6,3))
    
    axs[0].scatter(bb['A'],bb['L'],s=0.1,alpha=0.1)
    ind = np.isfinite(bb['PhiD'])
    r = np.round(pearsonr(bb['A'][ind],bb['L'][ind])[0],3)
    axs[0].text(0.75,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[0].transAxes)

    axs[0].set_xlim([0,2000])
    axs[0].set_ylim([-399,399])
    axs[0].set_ylabel('L [kV]')
    axs[0].set_xlabel('$A$ [MWb]')
    
    x = bb['A'].values[ind]
    y = bb['L'].values[ind]
    # return x,y
    m = np.linalg.eig(np.cov(x,y))[1][1][0]/np.linalg.eig(np.cov(x,y))[1][1][1]
    b = np.mean(y) - m * np.mean(x)
    axs[0].axline(xy1=(0, b), slope=m,color='C6',linestyle='--')
    
    tau = 1/(m*1e-3)/60/60
    axs[0].text(0.3,0.1,'L $\\propto$ A/'+str(np.round(tau,1))+' hrs',horizontalalignment='center',verticalalignment='center', transform=axs[0].transAxes)

    a = np.linspace(0,1500,101)
    axs[0].plot(a,a**4/6e9,color='C7',linestyle=':')
    
    axs[0].plot(a,func(a,*popt),color='C8',linestyle=':')
    axs[0].plot(a,func2(a,*popt2),color='C9',linestyle=':')

    
    axs[1].scatter(bb['LM'],bb['A_dt'],s=0.1,alpha=0.1)
    ind = np.isfinite(bb['PhiD'])
    r = np.round(pearsonr(bb['LM'][ind],bb['A_dt'][ind])[0],3)
    axs[1].text(0.75,0.9,'r = '+str(r),horizontalalignment='center',verticalalignment='center', transform=axs[1].transAxes)
    
    axs[1].set_xlim([-500,500])
    axs[1].set_ylim([-500,500])
    axs[1].set_ylabel('dA/dt [kV]')
    axs[1].set_xlabel('1.4$\\Phi_N$ - A/1.8 [kV]')
    
    # x = bb['LM'].values[ind]
    # y = bb['A_dt'].values[ind]
    # # return x,y
    # m = np.linalg.eig(np.cov(x,y))[1][1][0]/np.linalg.eig(np.cov(x,y))[1][1][1]
    # b = np.mean(y) - m * np.mean(x)
    # axs[1].axline(xy1=(0, b), slope=m,color='C1')
    
    # axs[1].text(0.3,0.1,'dA/dt $\\propto$ '+str(np.round(m,1)),horizontalalignment='center',verticalalignment='center', transform=axs[1].transAxes)

    
    plt.tight_layout()
    
    plt.savefig(outpath + 'fig06b.png',bbox_inches='tight',dpi = 300)
    plt.close()
    
    fig,axs = plt.subplots(1,2,figsize=(6,3))
    pax = fuv.pp(axs[0])
    cmap = plt.get_cmap('plasma',5).colors
    
    dates = bb.date[(bb['PhiD']>=0)&(bb['PhiD']<5)]
    pax.plot(bb['pb'].sel(date=dates).mean(dim='date'),bb.mlt,color=cmap[0],linewidth=0.8)
    pax.plot(bb['eb'].sel(date=dates).mean(dim='date'),bb.mlt,color=cmap[0],linewidth=0.8)
    dates = bb.date[(bb['PhiD']>=5)&(bb['PhiD']<15)]
    pax.plot(bb['pb'].sel(date=dates).mean(dim='date'),bb.mlt,color=cmap[1],linewidth=0.8)
    pax.plot(bb['eb'].sel(date=dates).mean(dim='date'),bb.mlt,color=cmap[1],linewidth=0.8)
    dates = bb.date[(bb['PhiD']>=15)&(bb['PhiD']<30)]
    pax.plot(bb['pb'].sel(date=dates).mean(dim='date'),bb.mlt,color=cmap[2],linewidth=0.8)
    pax.plot(bb['eb'].sel(date=dates).mean(dim='date'),bb.mlt,color=cmap[2],linewidth=0.8)
    dates = bb.date[(bb['PhiD']>=30)&(bb['PhiD']<50)]
    pax.plot(bb['pb'].sel(date=dates).mean(dim='date'),bb.mlt,color=cmap[3],linewidth=0.8)
    pax.plot(bb['eb'].sel(date=dates).mean(dim='date'),bb.mlt,color=cmap[3],linewidth=0.8)
    axs[0].set_title('Mean boundary vs $\\Phi_D$')
    pax = fuv.pp(axs[1])
    dates = bb.date[(bb['By']<-3)]
    pax.plot(bb['pb'].sel(date=dates).mean(dim='date'),bb.mlt,color='b',linewidth=0.8)
    pax.plot(bb['eb'].sel(date=dates).mean(dim='date'),bb.mlt,color='b',linewidth=0.8)
    dates = bb.date[(bb['By']>=-3)&(bb['By']<=3)]
    pax.plot(bb['pb'].sel(date=dates).mean(dim='date'),bb.mlt,color='k',linewidth=0.8)
    pax.plot(bb['eb'].sel(date=dates).mean(dim='date'),bb.mlt,color='k',linewidth=0.8)
    dates = bb.date[(bb['By']>3)]
    pax.plot(bb['pb'].sel(date=dates).mean(dim='date'),bb.mlt,color='r',linewidth=0.8)
    pax.plot(bb['eb'].sel(date=dates).mean(dim='date'),bb.mlt,color='r',linewidth=0.8)
    axs[1].set_title('Mean boundary vs IMF $B_y$')
    
    plt.savefig(outpath + 'fig07.png',bbox_inches='tight',dpi = 300)
    plt.close()
    
    fig,axs = plt.subplots(1,2,figsize=(6,3))
    pax = fuv.pp(axs[0],minlat=85)
    cmap = plt.get_cmap('plasma',5).colors
    
    dates = bb.date[(bb['PhiD']>=0)&(bb['PhiD']<5)]
    pax.plot(90-bb['pb_err'].sel(date=dates).mean(dim='date'),bb.mlt,color=cmap[0],linewidth=0.8)
    dates = bb.date[(bb['PhiD']>=5)&(bb['PhiD']<15)]
    pax.plot(90-bb['pb_err'].sel(date=dates).mean(dim='date'),bb.mlt,color=cmap[1],linewidth=0.8)
    dates = bb.date[(bb['PhiD']>=15)&(bb['PhiD']<30)]
    pax.plot(90-bb['pb_err'].sel(date=dates).mean(dim='date'),bb.mlt,color=cmap[2],linewidth=0.8)
    dates = bb.date[(bb['PhiD']>=30)&(bb['PhiD']<50)]
    pax.plot(90-bb['pb_err'].sel(date=dates).mean(dim='date'),bb.mlt,color=cmap[3],linewidth=0.8)
    axs[0].set_title('Mean error pb')
    
    pax = fuv.pp(axs[1],minlat=85)
    
    dates = bb.date[(bb['PhiD']>=0)&(bb['PhiD']<5)]
    pax.plot(90-bb['eb_err'].sel(date=dates).mean(dim='date'),bb.mlt,color=cmap[0],linewidth=0.8)
    dates = bb.date[(bb['PhiD']>=5)&(bb['PhiD']<15)]
    pax.plot(90-bb['eb_err'].sel(date=dates).mean(dim='date'),bb.mlt,color=cmap[1],linewidth=0.8)
    dates = bb.date[(bb['PhiD']>=15)&(bb['PhiD']<30)]
    pax.plot(90-bb['eb_err'].sel(date=dates).mean(dim='date'),bb.mlt,color=cmap[2],linewidth=0.8)
    dates = bb.date[(bb['PhiD']>=30)&(bb['PhiD']<50)]
    pax.plot(90-bb['eb_err'].sel(date=dates).mean(dim='date'),bb.mlt,color=cmap[3],linewidth=0.8)
    axs[1].set_title('Mean error eb')
    
    plt.savefig(outpath + 'fig08.png',bbox_inches='tight',dpi = 300)
    plt.close()
    
    fig,axs = plt.subplots(1,2,figsize=(6,3))
    pax = fuv.pp(axs[0],minlat=85)
    cmap = plt.get_cmap('plasma',5).colors
    
    dates = bb.date[(bb['PhiD']>=0)&(bb['PhiD']<5)]
    pax.plot(90-bb['pb'].sel(date=dates).std(dim='date'),bb.mlt,color=cmap[0],linewidth=0.8)
    pax.plot(90-bb['pb'].sel(date=dates).std(dim='date')/np.sqrt(len(dates)),bb.mlt,color=cmap[0],linewidth=0.8)
    dates = bb.date[(bb['PhiD']>=5)&(bb['PhiD']<15)]
    pax.plot(90-bb['pb'].sel(date=dates).std(dim='date'),bb.mlt,color=cmap[1],linewidth=0.8)
    pax.plot(90-bb['pb'].sel(date=dates).std(dim='date')/np.sqrt(len(dates)),bb.mlt,color=cmap[1],linewidth=0.8)
    dates = bb.date[(bb['PhiD']>=15)&(bb['PhiD']<30)]
    pax.plot(90-bb['pb'].sel(date=dates).std(dim='date'),bb.mlt,color=cmap[2],linewidth=0.8)
    pax.plot(90-bb['pb'].sel(date=dates).std(dim='date')/np.sqrt(len(dates)),bb.mlt,color=cmap[2],linewidth=0.8)
    dates = bb.date[(bb['PhiD']>=30)&(bb['PhiD']<50)]
    pax.plot(90-bb['pb'].sel(date=dates).std(dim='date'),bb.mlt,color=cmap[3],linewidth=0.8)
    pax.plot(90-bb['pb'].sel(date=dates).std(dim='date')/np.sqrt(len(dates)),bb.mlt,color=cmap[3],linewidth=0.8)
    axs[0].set_title('std and sem for pb')
    
    pax = fuv.pp(axs[1],minlat=85)
    
    dates = bb.date[(bb['PhiD']>=0)&(bb['PhiD']<5)]
    pax.plot(90-bb['eb'].sel(date=dates).std(dim='date'),bb.mlt,color=cmap[0],linewidth=0.8)
    pax.plot(90-bb['eb'].sel(date=dates).std(dim='date')/np.sqrt(len(dates)),bb.mlt,color=cmap[0],linewidth=0.8)
    dates = bb.date[(bb['PhiD']>=5)&(bb['PhiD']<15)]
    pax.plot(90-bb['eb'].sel(date=dates).std(dim='date'),bb.mlt,color=cmap[1],linewidth=0.8)
    pax.plot(90-bb['eb'].sel(date=dates).std(dim='date')/np.sqrt(len(dates)),bb.mlt,color=cmap[1],linewidth=0.8)
    dates = bb.date[(bb['PhiD']>=15)&(bb['PhiD']<30)]
    pax.plot(90-bb['eb'].sel(date=dates).std(dim='date'),bb.mlt,color=cmap[2],linewidth=0.8)
    pax.plot(90-bb['eb'].sel(date=dates).std(dim='date')/np.sqrt(len(dates)),bb.mlt,color=cmap[2],linewidth=0.8)
    dates = bb.date[(bb['PhiD']>=30)&(bb['PhiD']<50)]
    pax.plot(90-bb['eb'].sel(date=dates).std(dim='date'),bb.mlt,color=cmap[3],linewidth=0.8)
    pax.plot(90-bb['eb'].sel(date=dates).std(dim='date')/np.sqrt(len(dates)),bb.mlt,color=cmap[3],linewidth=0.8)
    axs[1].set_title('std and sem for eb')
    
    plt.savefig(outpath + 'fig09.png',bbox_inches='tight',dpi = 300)
    plt.close()
    
    fig,ax = plt.subplots(1,1,figsize=(3,3))
    pax = fuv.pp(ax)

    pax.plot(bb['pb'].mean(dim='date'),bb.mlt,linewidth=0.8,color='C0')
    pax.plot(bb['eb'].mean(dim='date'),bb.mlt,linewidth=0.8,color='C0')
    
    # BAS WIC boundaries
    bas = pd.read_csv('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/data/Wic_boundaries_V2.csv',index_col=0)

    mlt = np.arange(0.5,24)
    eb = bas.iloc[:,2:26].mean(axis=0).values
    pb = bas.iloc[:,27:-1].mean(axis=0).values
    
    mlt = np.append(mlt,mlt[0])
    eb = np.append(eb,eb[0])
    pb = np.append(pb,pb[0])
    
    pax.plot(pb,mlt,linewidth=0.8,color='C1')
    pax.plot(eb,mlt,linewidth=0.8,color='C1')
    
    ax.set_title('BCSS vs BAS all data')

    plt.savefig(outpath + 'fig10.png',bbox_inches='tight',dpi = 300)
    plt.close()    
    
def plot_ex(orbits,outpath):
    n=len(orbits)
    path = '/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/data/fuvAuroralFlux/'
    
    fig,axs = plt.subplots(n,figsize=(4,3))
    
    for i in range(n):
        bm = pd.read_hdf(path+'hdf/final_boundaries.h5',where='orbit=="{}"'.format(orbits[i])).to_xarray()
    
        df = processOMNI(bm.date.values)
        df.index.name = 'date'
        bm = xr.merge((bm,df.to_xarray()))
        
        bm['A_dt'] = 1e-3*np.deg2rad(15*0.1)*(bm['dA_dt']).sum(dim='mlt')
        bm['P_dt'] = 1e-3*np.deg2rad(15*0.1)*(bm['dP_dt']).sum(dim='mlt')
        bm['PhiN'] = bm['PhiD']-bm['P_dt']
        bm['L'] = 1.4*bm['PhiN'] - bm['A_dt']
        bm['A12'] = 1e-3*np.deg2rad(15*0.1)*(bm['dP_dt']).sel(mlt=12)
        
        time=(bm.date-bm.date[0]).values/ np.timedelta64(1, 'h')
    
        axs[i].plot(time,100*bm['A12'].values,color='C2')
        axs[i].plot(time,bm['PhiD'].values,color='C1')
        
        # axs[0].legend(['100 dA$_{12}$/dt [kV]','$\\Phi_D$ [kV]'],frameon=False,ncol=2)
        axs[0].text(0.25,0.88,'100 dA$_{12}$/dt [kV]',color='C2',horizontalalignment='center',verticalalignment='center', transform=axs[0].transAxes)
        axs[0].text(0.75,0.88,'$\\Phi_D$ [kV]',color='C1',horizontalalignment='center',verticalalignment='center', transform=axs[0].transAxes)


        axs[i].set_xlim([1,8])
        axs[i].set_ylim([-99,199])
        axs[i].set_ylabel('Orbit '+str(orbits[i]))
        if i != n-1:
            axs[i].xaxis.set_tick_params(labelbottom=False)
        else:
            axs[i].set_xlabel('time [hrs]')
    
    plt.subplots_adjust(wspace=0.0,hspace=0.0)
    plt.savefig(outpath + 'fig11.png',bbox_inches='tight',dpi = 300)
    plt.close() 
    
    fig,axs = plt.subplots(n,figsize=(4,3))
    
    for i in range(n):
        bm = pd.read_hdf(path+'hdf/final_boundaries.h5',where='orbit=="{}"'.format(orbits[i])).to_xarray()
    
        df = processOMNI(bm.date.values)
        df.index.name = 'date'
        bm = xr.merge((bm,df.to_xarray()))
        
        bm['A_dt'] = 1e-3*np.deg2rad(15*0.1)*(bm['dA_dt']).sum(dim='mlt')
        bm['P_dt'] = 1e-3*np.deg2rad(15*0.1)*(bm['dP_dt']).sum(dim='mlt')
        bm['PhiN'] = bm['PhiD']-bm['P_dt']
        bm['L'] = 1.4*bm['PhiN'] - bm['A_dt']
        
        time=(bm.date-bm.date[0]).values/ np.timedelta64(1, 'h')
    
        
        axs[i].plot(time,bm['A_dt'].values,color='C2')
        axs[i].plot(time,bm['PhiN'].values,color='C1')
        
        # axs[0].legend(['dA/dt [kV]','$\\Phi_N$ [kV]'],frameon=False,ncol=2)
        axs[0].text(0.25,0.88,'dA/dt [kV]',color='C2',horizontalalignment='center',verticalalignment='center', transform=axs[0].transAxes)
        axs[0].text(0.75,0.88,'$\\Phi_N$ [kV]',color='C1',horizontalalignment='center',verticalalignment='center', transform=axs[0].transAxes)

        axs[i].set_xlim([1,8])
        axs[i].set_ylim([-299,599])
        axs[i].set_ylabel('Orbit '+str(orbits[i]))
        if i != n-1:
            axs[i].xaxis.set_tick_params(labelbottom=False)
        else:
            axs[i].set_xlabel('time [hrs]')
    
    plt.subplots_adjust(wspace=0.0,hspace=0.0)
    plt.savefig(outpath + 'fig12.png',bbox_inches='tight',dpi = 300)
    plt.close() 
    
    
    fig,axs = plt.subplots(n,figsize=(4,3))
    
    for i in range(n):
        bm = pd.read_hdf(path+'hdf/final_boundaries.h5',where='orbit=="{}"'.format(orbits[i])).to_xarray()
    
        df = processOMNI(bm.date.values)
        df.index.name = 'date'
        bm = xr.merge((bm,df.to_xarray()))
        
        bm['A_dt'] = 1e-3*np.deg2rad(15*0.1)*(bm['dA_dt']).sum(dim='mlt')
        bm['P_dt'] = 1e-3*np.deg2rad(15*0.1)*(bm['dP_dt']).sum(dim='mlt')
        bm['PhiN'] = bm['PhiD']-bm['P_dt']
        bm['L'] = 1.4*bm['PhiN'] - bm['A_dt']
        bm['A'] = 1e-3*np.deg2rad(15*0.1)*(bm['dA']).sum(dim='mlt')
        bm['t1'] = 1/(1.8*60*60)*np.maximum(bm['A']-200e3,0)#+1/(5*60*60)*bm['A']
        bm['t1'] = bm['A']**4/6e21
        bm['t1'] = 2.54589386e-06 * (1e-3*bm['A'])**(2.51008498e+00)
        bm['t2'] = 1/(1.8*60*60)*bm['A']
        
        time=(bm.date-bm.date[0]).values/ np.timedelta64(1, 'h')
    
        axs[i].plot(time,bm['L'].values,color='C2')
        axs[i].plot(time,bm['t1'].values,color='C4',linestyle=':')
        axs[i].plot(time,bm['t2'].values,color='C4')
        
        axs[0].text(0.25,0.88,'L [kV]',color='C2',horizontalalignment='center',verticalalignment='center', transform=axs[0].transAxes)
        axs[0].text(0.75,0.88,'A/1.8 hrs [kV]',color='C4',horizontalalignment='center',verticalalignment='center', transform=axs[0].transAxes)

        axs[i].set_xlim([1,8])
        axs[i].set_ylim([-199,299])
        axs[i].set_ylabel('Orbit '+str(orbits[i]))
        if i != n-1:
            axs[i].xaxis.set_tick_params(labelbottom=False)
        else:
            axs[i].set_xlabel('time [hrs]')

    plt.subplots_adjust(wspace=0.0,hspace=0.0)
    plt.savefig(outpath + 'fig13.png',bbox_inches='tight',dpi = 300)
    plt.close() 

        
        
        
        
        
        
        

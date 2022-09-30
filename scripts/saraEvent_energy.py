#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:02:40 2022

@author: aohma
"""

import glob
import fuvpy as fuv
import numpy as np
import pandas as pd
import xarray as xr
from polplot.grids import equal_area_grid,sdarngrid,bin_number
from polplot import pp

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata,interp1d

def loadEvent(inpath):
    wicfiles = glob.glob(inpath+'/idl_files/*.idl')
    s12files = glob.glob(inpath+'/idl_files_SI12/*.idl')  
    s13files = glob.glob(inpath+'/idl_files_SI13/*.idl')  
    
    wic = fuv.readImg(wicfiles)
    s12 = fuv.readImg(s12files)
    s13 = fuv.readImg(s13files)
    
    wic = fuv.makeDGmodelTest(wic,stop=1e-3,tKnotSep=240)
    s12 = fuv.makeDGmodelTest(s12,stop=1e-3,tKnotSep=240)
    s13 = fuv.makeDGmodelTest(s13,stop=1e-3,tKnotSep=240)
    
    wic = fuv.makeSHmodel(wic,4,4,stop=1e-3,knotSep=240)
    s12 = fuv.makeSHmodel(s12,4,4,stop=1e-3,knotSep=240)
    s13 = fuv.makeSHmodel(s13,4,4,stop=1e-3,knotSep=240)
    
    return wic,s12,s13

def regridEvent(wic,s12,s13,dlat=2,dlon=2,latmin=50):
    
    # Identify conugate images
    tolerance='20s' # largest allowed difference between images
    wicDate = pd.DataFrame({'dateWIC':wic['date'],'indWIC':range(len(wic['date']))})
    s12Date = pd.DataFrame({'dateS12':s12['date'],'indS12':range(len(s12['date']))})
    s13Date = pd.DataFrame({'dateS13':s13['date'],'indS13':range(len(s13['date']))})

    con = pd.merge_asof(wicDate, s12Date,direction='nearest',left_on='dateWIC',right_on='dateS12',tolerance=pd.Timedelta(tolerance))
    con = pd.merge_asof(con, s13Date,direction='nearest',left_on='dateWIC',right_on='dateS13',tolerance=pd.Timedelta(tolerance))
        # It is possible to get duplicates in sh if VIS for some reason have a lower sampling. Should be removed here

    con.index=con['dateWIC']
    con = con.sort_index()
    con = con.dropna()
    
    # Set up the almost equal area grid
    grid,mltres=sdarngrid(dlat = dlat, dlon = dlat, latmin = latmin)
    
    detectors = ['WIC','S12','S13']
       
    wic['simage']=wic['shimg']
    s12['simage']=s12['shimg']
    s13['simage']=s13['shimg']
    
    imgs = [wic,s12,s13]
    
    ximgs = []
    for t in range(len(con)):
        timgs=[]
        for i,c in enumerate(detectors):
            img = imgs[i].isel(date=t)[['mlat','mlt','simage']].to_dataframe()
            ind = (np.isfinite(img['simage']))&(img.mlat>latmin)
            bin_num = np.zeros_like(img.mlt.values)
            bin_num[:] = np.nan
            
            bin_num[ind] = bin_number(grid,img[ind]['mlat'].values,img[ind]['mlt'].values)
            
            img['bin_num']=bin_num.astype(int)
            
            dfgrid=pd.DataFrame()
            dfgrid['mlat']=grid[0,:]
            dfgrid['mlt']=grid[1,:]
            dfgrid['mltres']=mltres
            dfgrid['mlatres']=dlat
            dfgrid = dfgrid.join(img.groupby('bin_num').median()['simage'])
            
            # fig,ax=plt.subplots()
            # pax = Polarsubplot(ax)
            # pc=pax.filled_cells(dfgrid.mlat.values, dfgrid.mlt.values, 2, dfgrid.mltres.values, 
            #                     dfgrid.shimage.values,crange=(0,1000))
            # plt.colorbar(pc)
            
            dfgrid = dfgrid.rename(columns={'simage':c})
            timg=dfgrid[[c]].to_xarray()
            timg = timg.expand_dims(date=[con.index[t]])
            timgs.append(timg)
        ximgs.append(xr.merge(timgs))
    ximgs = xr.concat(ximgs, dim='date')
    ximgs['mlat'] = (['ind'],grid[0,:])
    ximgs['mlatres'] = dlat
    ximgs['mlt'] = (['ind'],grid[1,:])
    ximgs['mltres'] = (['ind'],mltres)
    ximgs['mlatmin'] = latmin
    
    return ximgs

def plotRegrid(ximg):
    fig = plt.figure(facecolor = 'white',figsize=(12,10))
    gs = gridspec.GridSpec(2,3)
    
    gs2 = gridspec.GridSpecFromSubplotSpec(2,1,height_ratios=[90,10],subplot_spec=gs[0,0])
    ax2 = plt.subplot(gs2[0])
    ax2c = plt.subplot(gs2[1])
    pax2 = pp(ax2)
    pax2.filled_cells(ximg.mlat.values, ximg.mlt.values, ximg.mlatres.values, ximg.mltres.values, 
                     ximg.S12.values,crange=(1,90),cmap='gist_earth')
    ax2c.axis('off')
    plt.colorbar(pax2.ax.collections[0],orientation='horizontal',ax=ax2c,fraction=1,
                 extend='max')

    # Polar plot of the dayglow model
    gs3 = gridspec.GridSpecFromSubplotSpec(2,1,height_ratios=[90,10],subplot_spec=gs[0,1])
    ax3 = plt.subplot(gs3[0])
    ax3c = plt.subplot(gs3[1])
    pax3 = pp(ax3)
    pax3.filled_cells(ximg.mlat.values, ximg.mlt.values, ximg.mlatres.values, ximg.mltres.values, 
                     ximg.WIC.values,crange=(0,2500),cmap='gist_earth')
    ax3c.axis('off')
    plt.colorbar(pax3.ax.collections[0],orientation='horizontal',ax=ax3c,fraction=1,
                 extend='max')

    # Polar plot of the dayglow-corrected image
    gs4 = gridspec.GridSpecFromSubplotSpec(2,1,height_ratios=[90,10],subplot_spec=gs[0,2])
    ax4 = plt.subplot(gs4[0])
    ax4c = plt.subplot(gs4[1])
    pax4 = pp(ax4)
    pax4.filled_cells(ximg.mlat.values, ximg.mlt.values, ximg.mlatres.values, ximg.mltres.values, 
                     ximg.S13.values,crange=(1,50),cmap='gist_earth')
    ax4c.axis('off')
    plt.colorbar(pax4.ax.collections[0],orientation='horizontal',ax=ax4c,fraction=1,
                 extend='max')
    
    # Polar plot of input image
    gs2 = gridspec.GridSpecFromSubplotSpec(2,1,height_ratios=[90,10],subplot_spec=gs[1,0])
    ax2 = plt.subplot(gs2[0])
    ax2c = plt.subplot(gs2[1])
    pax2 = pp(ax2)
    pax2.filled_cells(ximg.mlat.values, ximg.mlt.values, ximg.mlatres.values, ximg.mltres.values, 
                     ximg.aEnergy.values,crange=(0.1,10),cmap='gist_earth')
    ax2c.axis('off')
    plt.colorbar(pax2.ax.collections[0],orientation='horizontal',ax=ax2c,fraction=1,
                 extend='max')

    # Polar plot of the dayglow model
    gs3 = gridspec.GridSpecFromSubplotSpec(2,1,height_ratios=[90,10],subplot_spec=gs[1,1])
    ax3 = plt.subplot(gs3[0])
    ax3c = plt.subplot(gs3[1])
    pax3 = pp(ax3)
    pax3.filled_cells(ximg.mlat.values, ximg.mlt.values, ximg.mlatres.values, ximg.mltres.values, 
                      ximg.eEnergy.values,crange=(0.1,10),cmap='plasma')
    ax3c.axis('off')
    plt.colorbar(pax3.ax.collections[0],orientation='horizontal',ax=ax3c,fraction=1,
                 extend='max')

    # Polar plot of the dayglow-corrected image
    gs4 = gridspec.GridSpecFromSubplotSpec(2,1,height_ratios=[90,10],subplot_spec=gs[1,2])
    ax4 = plt.subplot(gs4[0])
    ax4c = plt.subplot(gs4[1])
    pax4 = pp(ax4)
    pax4.filled_cells(ximg.mlat.values, ximg.mlt.values, ximg.mlatres.values, ximg.mltres.values, 
                     ximg.eFlux.values,crange=(0.1,10),cmap='plasma')
    ax4c.axis('off')
    plt.colorbar(pax4.ax.collections[0],orientation='horizontal',ax=ax4c,fraction=1,
                 extend='max')
    
def quantifyEvent(ximgs,pEnergy=1):
    
    PE = np.array([0.47,2.00,8.00,25.0,46.7])
    P1 = np.array([145,319,554,601,562])
    P2 = np.array([23.7,35.6,30.2,17.0,11.7])
    P3 = np.array([2.14,4.39,7.09,7.20,6.51])
    
    fP1 = interp1d(PE,P1,kind='linear')
    fP2 = interp1d(PE,P2,kind='linear')
    fP3 = interp1d(PE,P3,kind='linear')
    
    
    EE = np.array([0.2,0.5,1.0,5.0,10.0,25.0])
    E1 = np.array([446,470,511,377,223,101])
    E3 = np.array([12.8,11.3,8.75,4.26,2.11,0.74])
    
    fE1 = interp1d(EE,E1,fill_value=(446,101),bounds_error=False)
    fE3 = interp1d(EE,E3,fill_value=(12.8,0.74),bounds_error=False)
    fE1divE3 = interp1d(E1/E3,EE,fill_value=(np.nan,np.nan),bounds_error=False)
    
    pFs=[]
    eEs=[]
    eFs=[]
    aEs=[]
    for t in range(len(ximgs.date)):
        gwic = ximgs.isel(date=t)['WIC'].copy().values
        gs12 = ximgs.isel(date=t)['S12'].copy().values
        gs13 = ximgs.isel(date=t)['S13'].copy().values
        
        gwic[gwic<0]=0
        gs12[gs12<0]=0
        gs13[gs13<0]=0
        
        pFlux_s12 = gs12/fP2(pEnergy)
        pwic = 1*fP1(pEnergy)*pFlux_s12
        ps13 = 1*fP3(pEnergy)*pFlux_s12
        
        ewic = gwic.copy()-pwic
        es13 = gs13.copy()-ps13
        
        # ewic[ewic<0]=0
        # es13[es13<0]=0
        # plt.figure()
        # plt.plot(gwic)
        # plt.plot(pwic)
        # plt.plot(ewic)
        # plt.figure()
        # plt.plot(gs13)
        # plt.plot(ps13)
        # plt.plot(es13)
        # plt.figure()
        # plt.plot(gwic/gs13)
        # plt.plot(ewic/es13)
        # ewic[ewic<1]=np.nan
        
        es13[(es13<1)&(ewic<50)]=np.nan
        ewic[(es13<1)&(ewic<50)]=np.nan
        
        es13[es13<0.01]=0.01
        ewic[ewic<0]=0
        eEnergy = fE1divE3(ewic/es13)
        # eEnergy[np.isnan(eEnergy)]=0
        
        eFluxwic = ewic/fE1(eEnergy)
        eFluxs13 = es13/fE3(eEnergy)
        
        gs13[gs13<1]=np.nan
        aEnergy = fE1divE3(gwic/gs13)
        # aEnergy[np.isnan(aEnergy)]=0
        
        pFs.append(pwic)
        eEs.append(eEnergy)
        eFs.append(eFluxwic)
        aEs.append(aEnergy)
    
    ximgs['pEnergy']= pEnergy
    ximgs['pFlux'] = (['date','ind'],np.array(pFs))
    ximgs['eEnergy'] = (['date','ind'],np.array(eEs))
    ximgs['eFlux'] = (['date','ind'],np.array(eFs))
    ximgs['aEnergy'] = (['date','ind'],np.array(aEs))
    return ximgs
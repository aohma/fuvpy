#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:02:51 2023

@author: aohma
"""

import glob
import numpy as np
import pandas as pd
import xarray as xr
import fuvpy as fuv

def make_wicfiles(orbit,path):
    '''
    Make a dataframe with date, wic filename and orbit number

    Parameters
    ----------
    orbit : pandas.DataFrame
        Data frame with IMAGE orbit information.
    path : str
        Path to the wicfiles.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with the date, filename and orbit number.

    '''
    
    path = path + '*'
    wicfiles = sorted(glob.glob(path))
    
    df = pd.DataFrame()
    df['date']=[pd.to_datetime(file[-15:-4],format='%Y%j%H%M') for file in wicfiles]
    df['wicfile']=[file[-18:] for file in wicfiles]
    df = pd.merge_asof(df,orbit[['orbit_number']],left_on='date',right_index=True,direction='nearest')
    df = df.set_index(['date','orbit_number'])
    return df

def background_removal(orbits):
    orbitpath = '/Home/siv24/aoh013/python/image_analysis/'
    wicpath = '/mnt/0b3b8cce-3469-42cb-b694-60a7ca36e03a/IMAGE_FUV/wic/'
    outpath = '/mnt/5fa6bccc-fa9d-4efc-9ddc-756f65699a0a/aohma/fuv/wic/'
    
    
    for orbit in orbits:
        files = pd.read_hdf(orbitpath+'wicfiles.h5',where='orbit_number=="{}"'.format(orbit))
        files['path']=wicpath
        
        try:
            wic = fuv.readImg((files['path']+files['wicfile']).tolist(),dzalim=75) # Load
            wic = wic.sel(date=wic.hemisphere.date[wic.hemisphere=='north']) # Remove SH
            wic = fuv.makeBSmodel(wic,sKnots=[-3.5,-0.25,0,0.25,1.5,3.5],stop=0.01,n_tKnots=5,minlat=-90,tukeyVal=5,dzalim=75,dampingVal=1e-2)
            wic = fuv.makeSHmodel(wic,4,4,n_tKnots=5,stop=0.01,tukeyVal=5,dampingVal=1e-4)
            wic.to_netcdf(outpath+'wic_or'+str(orbit).zfill(4)+'.nc') 
            # df = wic[['img','dgimg','shimg','mlat','mlt']].to_dataframe().dropna(subset='dgimg')
            # df.to_hdf(outpath+'wic_or'+str(orbit).zfill(4)+'.h5','wic',format='table',append=True,data_columns=True)     
        except Exception as e: print(e)
        
def boundary_detection(imgs):


    thresholds = [100,150,200] # Peak threshold in counts
    sigma = 300
    
    R_E = 6371 # Earth radius (km)
    R_I = R_E+130 # Assumed emission radius (km)
    km_per_lat = np.pi*R_I/180
    
    
    # Evaluation grid
    clat_ev = np.arange(0.5,41,0.5)
    mlt_ev = np.arange(0.5,24,1)
    
    r_ev = km_per_lat*(np.abs(clat_ev))
    a_ev = (mlt_ev- 6.)/12.*np.pi
    x_ev =  r_ev[:,None]*np.cos(a_ev[None,:])
    y_ev =  r_ev[:,None]*np.sin(a_ev[None,:])

    #%% Model test
    dfs=[]
    for t in range(len(imgs.date)):
        print('■', end='', flush=True)
        img = imgs.isel(date=t)
    
        
        # cartesian projection
        r = km_per_lat*(90. - np.abs(img['mlat'].values))
        a = (img['mlt'].values - 6.)/12.*np.pi
        x =  r*np.cos(a)
        y =  r*np.sin(a)
        d = img['shimg'].values
        
        
        # Make latitudinal intensity profiles 
        d_ev = np.full_like(x_ev,np.nan)
        for i in range(len(clat_ev)):
            for j in range(len(mlt_ev)):
                ind = np.sqrt((x_ev[i,j]-x)**2+(y_ev[i,j]-y)**2)<sigma
                if np.sum(ind)>0: # non-zero weights
                    if (r_ev[i]>np.min(r[ind]))&(r_ev[i]<np.max(r[ind])): # Only between of pixels with non-zero weights
                        d_ev[i,j]=np.median(d[ind])
    
        for i,lt in enumerate(mlt_ev): # Find peaks in each intensity profile
            max_colat = 35+10*np.cos(np.pi*lt/12)
            dp = d_ev[:,i]
            
            for j in range(len(thresholds)):
                threshold = thresholds[j]
                pb = []
                eb = []
                for k in range(1,len(dp)):
                    if (dp[k-1]<threshold)&(dp[k]>threshold)&np.isfinite(dp[[k-1,k]]).all()&(clat_ev[[k-1,1]]<max_colat).all():
                        pb.append(np.average(clat_ev[[k-1,k]],weights=abs(dp[[k-1,k]]-threshold)))
                        
                    if (dp[k-1]>threshold)&(dp[k]<threshold)&np.isfinite(dp[[k-1,k]]).all()&(clat_ev[[k-1,1]]<max_colat).all():
                        eb.append(np.average(clat_ev[[k-1,k]],weights=abs(dp[[k-1,k]]-threshold)))
            
                df = pd.DataFrame(np.nan,index=[0],columns=['pb','eb'])
                df[['date','mlt','lim']]=[img.date.values,lt,threshold]

                if len(pb)>0: df.loc[0,'pb']=90-pb[0]
                if len(eb)>0: df.loc[0,'eb']=90-eb[-1]
                
                dfs.append(df.set_index(['date','mlt','lim']))

    df = pd.concat(dfs)
    return df


def initial_boundaries(orbits):
    inpath = '/mnt/5fa6bccc-fa9d-4efc-9ddc-756f65699a0a/aohma/fuv/wic/'
    outpath = '/mnt/5fa6bccc-fa9d-4efc-9ddc-756f65699a0a/aohma/fuv/boundaries/'
    
    for orbit in orbits:
        try:
            imgs = xr.load_dataset(inpath+'wic_or'+str(orbit).zfill(4)+'.nc')
            
            bi = boundary_detection(imgs)
            bi['orbit']=orbit
            bi.to_hdf(outpath+'initial_boundaries.h5','initial',format='table',append=True,data_columns=True)
        except Exception as e: print(e)
    

    

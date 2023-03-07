#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:02:51 2023

@author: aohma
"""

import glob
import pandas as pd
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
            
            df = wic[['img','dgimg','shimg','mlat','mlt']].to_dataframe().dropna(subset='dgimg')
            df.to_hdf(outpath+'wic_or'+str(orbit).zfill(4)+'.h5','wic',format='table',append=True,data_columns=True)     
        except:
            print('I said "no no no"')
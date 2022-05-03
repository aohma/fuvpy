#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Program to remove background contamination from WIC relative to substorm onset

Created on Thu Nov 25 13:27:45 2021

@author: aohma
"""

## ALL PATHS MUST BE CHANGED ON WORKSTATION!!!

import pandas as pd
import glob
import vaex

import fuvpy as fuv


def makeSubstormFiles(inpath,outpath,hemisphere='both'):
    '''
    This script loads WIC images relative to each onset in Frey's onset list (-30 min to + 90 min)',
    removes background and store each event as a vaex compatible hdf file.
    
    Parameters
    ----------
    inpath : str
        path to the input wic files.
    outpath : str
        path to store the output files.
    hemisphere : str
    Which hemisphere to include, ['north','south','both']
    Default is 'both'

    Returns
    -------
    None.

    '''
    
    # inpath = '/mnt/0b3b8cce-3469-42cb-b694-60a7ca36e03a/IMAGE_FUV/wic/'
    # outpath = '/mnt/5fa6bccc-fa9d-4efc-9ddc-756f65699a0a/aohma/fuv/wic/'
    df = pd.DataFrame()
    df['wicfile'] = glob.glob(inpath + '*.idl')
    df['date']=pd.to_datetime(df.loc[:,'wicfile'].str.replace(inpath + 'wic','').str.replace('.idl',''),format='%Y%j%H%M')
    df = df.set_index('date')
    df = df.sort_index()
    
    onset = pd.read_pickle('../data/merged_substormlist.pd')
    frey = onset.loc['2000-03-01':'2006-01-01',:]
    frey['date'] = frey.index
    
    # con = pd.merge(df,frey,left_index=True,right_index=True)
 
    fromDates = (frey['date']-pd.Timedelta('35min')).dt.strftime('%Y-%m-%d %H:%M').astype(str).values
    toDates = (frey['date']+pd.Timedelta('95min')).dt.strftime('%Y-%m-%d %H:%M').astype(str).values
    
    files = []
    for i in range(len(frey)):
        temp = df[fromDates[i]:toDates[i]]
        # temp['sec'] = (temp.index-frey.index[i]).astype(dtype='timedelta64[s]').astype(int)
        temp['onset'] = frey.index[i]
        temp['mlat'] = frey.mlat[i]
        temp['mlt'] = frey.mlt[i]
        
        if hemisphere == 'north' and temp['mlat']>0:
            files.append(temp)
        elif hemisphere == 'south' and temp['mlat']<0:
            files.append(temp)
        elif hemisphere =='both':
            files.append(temp)
  
    for f in files:
        if not f.empty:
            if f['mlat'][0]>0:
                wic = fuv.readImg(f['wicfile'].values,dzalim=75,hemisphere='north')
            else:
                wic = fuv.readImg(f['wicfile'].values,dzalim=75,hemisphere='south')
                
            wic = fuv.makeDGmodel(wic,transform='log')
            wic = fuv.makeSHmodel(wic,4,4)
            wic = wic.to_dataframe().reset_index()[['date','row','col','mlat','mlt','img','dgimg','dgweight','shimg','shweight']]
            wic = wic.rename(columns={'row':'irow','col':'icol'})
            wic['odate']=f['onset'][0]
            wic['omlat']=f['mlat'][0]
            wic['omlt']=f['mlt'][0]
            
            rtimef = pd.DataFrame()
            rtimef['date']=pd.date_range(f['onset'][0],periods=51,freq='123s')
            rtimef['irel']=range(51)
            rtimeb = pd.DataFrame()
            rtimeb['date']=pd.date_range(f['onset'][0],periods=21,freq='-123s').sort_values()[:-1]
            rtimeb['irel']=range(-20,0)
            
            wic = pd.merge_asof(wic,pd.concat([rtimeb,rtimef]),on='date',direction='nearest',tolerance=pd.Timedelta('40s')).copy()
            wic = wic.dropna()
            vaex_df = vaex.from_pandas(wic)
            vaex_df.export_hdf5(outpath+'wic'+f.onset[0].strftime('%Y%m%d%H%M%S')+'.hdf5')
            
    return


# Field to include:
# ind,date,row,col,mlat,mlt,hemisphere,img,dgimg,shimg,onset,rind 


# def calcSuperposed(inpath,outpath):
#       Read vaex_df from hdfs
#       vaex_df = vaex.open('inpath/wic*.hdf5')
# df_names_all
#       Do superposed statistics (mean,approx median, std, skew)
#       Do on MLT,MLAT statistics
        # mean = vaex_df.mean(binby=['mlat','mlt'],limits=[[50, 90], [0, 24]], shape=(40, 24*5)))
        # median = vaex_df.median_approx(binby=['mlat','mlt'],limits=[[50, 90], [0, 24]], shape=(40, 24*5)))
        # std = vaex_df.std(binby=['mlat','mlt'],limits=[[50, 90], [0, 24]], shape=(40, 24*5)))
        # skew = vaex_df.skew(binby=['mlat','mlt'],limits=[[50, 90], [0, 24]], shape=(40, 24*5)))


#       Do on predefined equal area grid
#       store as new file to transfer from Workstation

# Only do NH?





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Program to remove background contamination from WIC relative to substorm onset

Created on Thu Nov 25 13:27:45 2021

@author: aohma
"""

## ALL PATHS MUST BE CHANGED ON WORKSTATION!!!
import numpy as np
import pandas as pd
import xarray as xr
import glob
import vaex

import fuvpy as fuv

from polplot import grids


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
        
        if not temp.empty:
            if hemisphere == 'north' and temp['mlat'][0]>0:
                files.append(temp)
            elif hemisphere == 'south' and temp['mlat'][0]<0:
                files.append(temp)
            elif hemisphere =='both':
                files.append(temp)
      
    for f in files:
        if not f.empty:
            try:
                if f['mlat'][0]>0:
                    wic = fuv.readImg(f['wicfile'].values,dzalim=75,hemisphere='north')
                else:
                    wic = fuv.readImg(f['wicfile'].values,dzalim=75,hemisphere='south')
                    
                wic = fuv.makeDGmodel(wic,transform='log',tOrder=0)
                wic = fuv.makeSHmodel(wic,4,4,order=0)
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
            except:
                print('Some error occurred. Moving on with our lives to the new substorm event.')
    return


# Field to include:
# ind,date,row,col,mlat,mlt,hemisphere,img,dgimg,shimg,onset,rind 

def addBinnumberValidation(grid,vdf):
    mlat = vdf.mlat.values
    mlt = vdf.mlt.values
    binNumber = grids.bin_number(grid,mlat,mlt)
    vdf['binNumber'] = binNumber
    return vdf

def addBinnumber(grid,vdf):
    '''
    Find the binnumber of each row in the input DataFrame
    vaex compatible version of grids.bin_number() 

    Parameters
    ----------
    grid : array
        2 x N array with mlats in the first row and mlts in the second row,
        as produced by grids.equal_area_grid() or grids.sdarngrid()
    
    vdf : vaex.DataFrame
        The Dataframe to bin. Must contain a column with mlat and mlt to bin by. 

    Returns
    -------
    vdf : vaex.DataFrame
        Return the input DataFrame with the new field 'binNumber',
        which is the bin each row map to in the input grid

    '''
        
    llat = np.unique(grid[0]) # latitude circles
    assert np.allclose(np.sort(llat) - llat, 0) # should be in sorted order automatically. If not, the algorithm will not work
    dlat = np.diff(llat)[0] # latitude step
    latbins = np.hstack(( llat, llat[-1] + dlat )) # make latitude bin edges
    vdf['latbinNumber'] = vdf.mlat.digitize(latbins) - 1 # find the latitude index for each data point

    # number of mlt bins in each latitude ring:
    nmlts = np.array([len(np.unique(grid[1][grid[0] == lat])) for lat in llat])

    vdf['latbinNumber'] = vdf.func.where(vdf['latbinNumber']<0,len(nmlts)+vdf['latbinNumber'],vdf['latbinNumber'])

    vdf2 = vaex.from_arrays(nmlts=nmlts,ind = np.arange(len(nmlts)))
    vdf=vdf.join(vdf2,left_on='latbinNumber',right_on='ind')
    # normalize all longitude bins to the equatorward ring:
    _mlt = vdf.mlt * vdf['nmlts'] / nmlts[0]
    
    # make mlt bin edges for the equatorward ring:
    lmlt = np.unique(grid[1][grid[0] == llat[0]])
    dmlt = np.diff(lmlt)[0]
    mltbins = np.hstack((lmlt, lmlt[-1] + dmlt)) # make longitude bin edges
    vdf['mltbinNumber'] = _mlt.digitize(mltbins) - 1 # find the longitude bin
    
    # map from 2D bin numbers to 1D by adding the number of bins in each row equatorward:
    
    vdf3 = vaex.from_arrays(nmltsCS=np.cumsum(np.hstack((0, nmlts))),ind2 = np.arange(len(np.cumsum(np.hstack((0, nmlts))))))    
    vdf=vdf.join(vdf3,left_on='latbinNumber',right_on='ind2')
    vdf['binNumber'] = vdf['mltbinNumber'] + vdf['nmltsCS']
    
    #Set the bin number of outside grid observations to -1
    vdf['binNumber'] = vdf.func.where(vdf['mlat']<grid[0,0],-1,vdf['binNumber'])
    
    return vdf

def addBinnumber2(vdf):
    '''
    vaex compatible version of grids.bin_number, which 
    
    This is slower than addBinNumber(), but should handle nans without crashing.
    Implement Jone fix in addBinnumber and discard this function?
    

    Parameters
    ----------
    vdf : TYPE
        DESCRIPTION.

    Returns
    -------
    vdf : TYPE
        DESCRIPTION.

    '''
    
    grid,mltres=grids.sdarngrid(dlat = 2, dlon = 2, latmin = 58)
    
    # vdf['binNumber']=0*vdf.mlat -1
    vdf['rownr'] = vaex.vrange(0, len(vdf))
    
    llat = np.unique(grid[0]) # latitude circles
    assert np.allclose(np.sort(llat) - llat, 0) # should be in sorted order automatically. If not, the algorithm will not work
    dlat = np.diff(llat)[0] # latitude step
    latbins = np.hstack(( llat, llat[-1] + dlat )) # make latitude bin edges
    
    
    ii = (vdf['mlat']>=latbins[0])&(vdf['mlat']<=latbins[-1])
    
    vdf_real = vdf[ii]
    vdf_real['latbinNumber'] = vdf_real.mlat.digitize(latbins) - 1 # find the latitude index for each data point

    # number of longitude bins in each latitude ring:
    nlons = np.array([len(np.unique(grid[1][grid[0] == lat])) for lat in llat])

    vdf2 = vaex.from_arrays(nlons=nlons,ind = np.arange(len(nlons)))
    vdf_real=vdf_real.join(vdf2,left_on='latbinNumber',right_on='ind')
    # normalize all longitude bins to the equatorward ring:
    _mlt = vdf_real.mlt * vdf_real['nlons'] / nlons[0]
    
    # make longitude bin edges for the equatorward ring:
    llon = np.unique(grid[1][grid[0] == llat[0]])
    dlon = np.diff(llon)[0]
    lonbins = np.hstack((llon, llon[-1] + dlon)) # make longitude bin edges
    vdf_real['lonbinNumber'] = _mlt.digitize(lonbins) - 1 # find the longitude bin
    
    # map from 2D bin numbers to 1D by adding the number of bins in each row equatorward:
    
    vdf3 = vaex.from_arrays(nlonsCS=np.cumsum(np.hstack((0, nlons))),ind2 = np.arange(len(np.cumsum(np.hstack((0, nlons))))))    
    vdf_real=vdf_real.join(vdf3,left_on='latbinNumber',right_on='ind2')
    vdf_real['binNumber'] = vdf_real['lonbinNumber'] + vdf_real['nlonsCS']
    vdf = vdf.join(vdf_real[['rownr','binNumber']],left_on='rownr',right_on='rownr')

    vdf['binNumber'] = vdf.func.where(ii,vdf.binNumber,-1)
    # vdf['binNumber'][vdf['binNumber'].isnan()]=-1
    return vdf

def calcSuperposed(vdf):
    vdf['rmlat'] = vdf['mlat']-vdf['omlat']
    # vdf['rmlt'] = vdf['mlt']-vdf['omlt']
    vdf['rmlt'] = np.rad2deg(np.arctan2(np.sin(np.deg2rad(15*vdf['mlt'])),np.cos(np.deg2rad(15*vdf['mlt'])))-np.arctan2(np.sin(np.deg2rad(15*vdf['omlt'])),np.cos(np.deg2rad(15*vdf['omlt']))))/15
    # df_names_all
    # Do superposed statistics (mean,approx median, std, skew)
    # Do on MLT,MLAT statistics
    mean = vdf.mean('shimg',binby=['rmlat','rmlt','irel'],limits=[[-20, 20], [-12, 12],[-15.5,30.5]], shape=(40, 24*5,46))
    median = vdf.median_approx('shimg',binby=['rmlat','rmlt','irel'],limits=[[-20, 20], [-12, 12],[-15.5,30.5]], shape=(40, 24*5,46))
    std = vdf.std('shimg',binby=['rmlat','rmlt','irel'],limits=[[-20, 20], [-12, 12],[-15.5,30.5]], shape=(40, 24*5,46))
    count = vdf.count('shimg',binby=['rmlat','rmlt','irel'],limits=[[-20, 20], [-12, 12],[-15.5,30.5]], shape=(40, 24*5,46))
        
    ds = xr.Dataset(
    data_vars=dict(
        mean=(['rlat','rlt','rind'], mean),
        median=(['rlat','rlt','rind'], median),
        std=(['rlat','rlt','rind'], std),
        count=(['rlat','rlt','rind'], count),
        ),
    coords=dict(
        rlat = np.linspace(-19.5,19.5,40),
        rlt = np.linspace(-11.9,11.9,24*5),
        rind = np.linspace(-15,30,46)
    ),
    )
    
    if 'binNumber' in vdf.column_names:
        n_bins = vdf['binNumber'].max()+1
        mean = vdf.mean('shimg',binby=['binNumber','irel'],limits=[[-0.5, n_bins-0.5],[-15.5,30.5]], shape=(n_bins,46))
        median = vdf.median_approx('shimg',binby=['binNumber','irel'],limits=[[-0.5, n_bins-0.5],[-15.5,30.5]], shape=(n_bins,46))
        std = vdf.std('shimg',binby=['binNumber','irel'],limits=[[-0.5, n_bins-0.5],[-15.5,30.5]], shape=(n_bins,46))
        count = vdf.count('shimg',binby=['binNumber','irel'],limits=[[-0.5, n_bins-0.5],[-15.5,30.5]], shape=(n_bins,46))
            # std = vaex_df.std(binby=['mlat','mlt'],limits=[[50, 90], [0, 24]], shape=(40, 24*5)))
            # skew = vaex_df.skew(binby=['mlat','mlt'],limits=[[50, 90], [0, 24]], shape=(40, 24*5)))
        
        ds2 = xr.Dataset(
        data_vars=dict(
            mean=(['binNumber','rind'], mean),
            median=(['binNumber','rind'], median),
            std=(['binNumber','rind'], std),
            count=(['binNumber','rind'], count),
            ),
        coords=dict(
            binnr = np.arange(n_bins),
            rind = np.linspace(-15,30,46)
        ),
        )
    
    
    return ds,ds2

    #       Do on predefined equal area grid
    #       store as new file to transfer from Workstation


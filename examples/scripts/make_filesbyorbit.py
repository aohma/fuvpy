#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:02:51 2023

@author: aohma
"""

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import fuvpy as fuv
from polplot import pp

import matplotlib.path as mplpath
import matplotlib.pyplot as plt

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

def background_removal(orbits,orbitpath,wicpath,outpath):
    '''
    Background removal per orbit
    
    orbits (list) : Orbit numbers
    orbitpath (str) : Path to hdf with orbit info
    wicpath (str) : Path to wic image files
    outpath (str) : Path to save the corrected images per orbit
    '''

    for orbit in orbits:
        files = pd.read_hdf(orbitpath+'wicfiles.h5',where='orbit_number=="{}"'.format(orbit))
        files['path']=wicpath

        try:
            wic = fuv.read_idl((files['path']+files['wicfile']).tolist(),dzalim=75) # Load
            wic = wic.sel(date=wic.hemisphere.date[wic.hemisphere=='north']) # Remove SH
            wic = fuv.backgroundmodel_BS(wic,sKnots=[-3.5,-0.25,0,0.25,1.5,3.5],stop=0.01,n_tKnots=5,tukeyVal=5,dampingVal=1e-3)
            wic = fuv.backgroundmodel_SH(wic,4,4,n_tKnots=5,stop=0.01,tukeyVal=5,dampingVal=1e-4)
            wic.to_netcdf(outpath+'wic_or'+str(orbit).zfill(4)+'.nc')
        except Exception as e: print(e)

def initial_boundaries(orbits,inpath,outpath):
    '''
    Initial boundary detection

    orbits (list) : Orbit numbers
    inpath (str) : Path to corrected images
    outpath (str) : Path to save identified boundaries
    '''

    for orbit in orbits:
        try:
            imgs = xr.load_dataset(inpath+'wic_or'+str(orbit).zfill(4)+'.nc')

            bi = fuv.detect_boundaries(imgs)
            bi = bi.to_dataframe()
            bi['orbit']=orbit
            bi.to_hdf(outpath+'initial_boundaries.h5','initial',format='table',append=True,data_columns=True)
        except Exception as e: print(e)

def data_coverage(imgs,window=1,dzalim=75):
    '''
    Determine if image(s) has global coverage
    imgs (xr.Dataset) : Images
    window (int) : Size of the moving window. Centered, window must be odd.
    dzalim (float) : Max viewing angle to be considered

    return (array) : Bool
    '''

    if window%2 != 1 or type(window)!=int:
        raise ValueError('window must be an odd integer')

    half = window//2

    lt = np.arange(0.5,24)
    lat = 90-(30+10*np.cos(np.pi*lt/12))

    isglobal=[]
    for t in range(len(imgs.date)):
        if t<half:
            ind = slice(None,t+half+1)
        elif t> len(imgs.date)-half-1:
            ind = slice(t-half,None)
        else:
            ind = slice(t-half,t+half+1)

        img=imgs.isel(date=ind)
        count = np.zeros_like(lt)
        for i in range(len(lt)):
            count[i]=np.sum((img['mlt'].values>(lt[i]-0.5))&(img['mlt'].values<(lt[i]+0.5))&(img['mlat'].values<lat[i])&(img['dza'].values<dzalim))
        isglobal.append((count>0).all())
    return np.array(isglobal)

def intensities(wic,bm):
    '''
    Calculate intensity inside and outside modelled aurora

    wic (xr.Dataset) : images
    bm (pd.DataFrame) : boundaries

    return bm with new columns
    '''

    bm=bm.reset_index().set_index('date')
    wic['shimg'].attrs = {'long_name': 'Counts', 'units': ''}

    a = (bm.mlt.values - 6.)/12.*np.pi

    r = (90. - np.abs(bm['pb']))
    bm['px'] =  r*np.cos(a)
    bm['py'] =  r*np.sin(a)

    r = (90. - np.abs(bm['eb']))
    bm['ex'] =  r*np.cos(a)
    bm['ey'] =  r*np.sin(a)

    r = 35+10*np.cos(np.pi*bm.mlt.values/12)
    bm['lx'] =  r*np.cos(a)
    bm['ly'] =  r*np.sin(a)

    r = (90. - np.abs(wic['mlat']))
    a = (wic.mlt.values - 6.)/12.*np.pi
    wic['x'] =  r*np.cos(a)
    wic['y'] =  r*np.sin(a)

    P_mean  = []
    P_std = []
    A_mean = []
    A_std = []
    S_mean = []
    S_std = []

    for t in wic.date:
        # Create an PB polygon
        poly = mplpath.Path(bm.loc[t.values,['px','py']].values)

        # Identify gridcell with center inside the PB polygon
        inpb = poly.contains_points(np.stack((wic.sel(date=t).x.values.flatten(),wic.sel(date=t).y.values.flatten()),axis=1))

        # Create an EB polygon
        poly = mplpath.Path(bm.loc[t.values,['ex','ey']].values)

        # Identify gridcell with center inside the EB polygon
        ineb = poly.contains_points(np.stack((wic.sel(date=t).x.values.flatten(),wic.sel(date=t).y.values.flatten()),axis=1))

        # Create an minlat polygon
        poly = mplpath.Path(bm.loc[t.values,['lx','ly']].values)

        # Identify gridcell with center inside the EB polygon
        incap = poly.contains_points(np.stack((wic.sel(date=t).x.values.flatten(),wic.sel(date=t).y.values.flatten()),axis=1))

        P_mean.append(np.nanmean(wic.sel(date=t)['shimg'].values.flatten()[inpb]))
        P_std.append(np.nanstd(wic.sel(date=t)['shimg'].values.flatten()[inpb]))
        A_mean.append(np.nanmean(wic.sel(date=t)['shimg'].values.flatten()[ineb & ~inpb]))
        A_std.append(np.nanstd(wic.sel(date=t)['shimg'].values.flatten()[ineb & ~inpb]))
        S_mean.append(np.nanmean(wic.sel(date=t)['shimg'].values.flatten()[incap & ~ineb]))
        S_std.append(np.nanstd(wic.sel(date=t)['shimg'].values.flatten()[incap & ~ineb]))
      

    # ADD TO BM
    bm = bm.reset_index().set_index(['date','mlt']).to_xarray().sortby(['date','mlt'])
    bm['P_mean'] = ('date',P_mean)
    bm['P_std'] = ('date',P_std)
    bm['A_mean'] = ('date',A_mean)
    bm['A_std'] = ('date',A_std)
    bm['S_mean'] = ('date',S_mean)
    bm['S_std'] = ('date',S_std)
    bm = bm.to_dataframe().reset_index().set_index(['date','mlt'])

    return bm

def final_bondaries(orbits,wicpath,bpath):
    '''
    Final boundary detection
    
    orbits (list) : Orbit numbers
    wicpath (str) : Path to corrected images
    bpath (str) : Path to save model boundaries
    '''

    for orbit in orbits:
        try:
            imgs = xr.load_dataset(wicpath+'wic_or'+str(orbit).zfill(4)+'.nc')
            bi = pd.read_hdf(bpath+'initial_boundaries.h5',key='initial',where='orbit=="{}"'.format(orbit))

            # Only images with identified initial boundaries
            imgs = imgs.sel(date=bi.reset_index().date.unique())

            bm = fuv.boundarymodel_BS(bi,tKnotSep=5,tLeb=1e-1,sLeb=1e-3,tLpb=1e-1,sLpb=1e-3,resample=False)
            isglobal = data_coverage(imgs,dzalim=65)
            bm['isglobal'] = ('date',isglobal)

            bm = bm.to_dataframe()
            bm['orbit']=orbit
            bm = intensities(imgs, bm)
            bm[['pb','eb','v_phi','v_theta','u_phi','u_theta','isglobal','orbit','rmse_in','rmse_out']].to_hdf(bpath+'final_boundaries.h5','final',format='table',append=True,data_columns=True)
        except Exception as e: print(e)
        
def final_bondaries_error(orbits,wicpath,bpath):
    '''
    Final boundary detection with uncertainty

    orbits (list) : Orbit numbers
    wicpath (str) : Path to corrected images
    bpath (str) : Path to save model boundaries
    '''

    for orbit in orbits:
        try:
            imgs = xr.load_dataset(wicpath+'wic_or'+str(orbit).zfill(4)+'.nc')
            bi = pd.read_hdf(bpath+'initial_boundaries.h5',key='initial',where='orbit=="{}"'.format(orbit)).to_xarray()

            # Only images with identified initial boundaries
            imgs = imgs.sel(date=bi.date)

            bms = []    
            for l in np.arange(50,201,5):
                bm = fuv.boundarymodel_BS(bi.sel(lim=l),tKnotSep=10,tLeb=1e0,sLeb=1e-2,tLpb=1e0,sLpb=1e-2)
                bm = bm.expand_dims(lim=[l])
                bms.append(bm)
            
            bms = xr.concat(bms,dim='lim')

            bm = fuv.boundarymodel_BS(bi,tKnotSep=10,tLeb=1e0,sLeb=1e-2,tLpb=1e0,sLpb=1e-2)
            keys = list(bm.keys())
            for key in keys:
                bm[key+'_err'] = bms[key].std(dim='lim')

            isglobal = data_coverage(imgs,window=5,dzalim=65)
            bm['isglobal'] = ('date',isglobal)
            bm['count'] = 0.5*(bi['pb'].rolling(date=5,min_periods=1,center=True).count().count(dim='lim')>0).sum(dim='mlt') + 0.5*(bi['eb'].rolling(date=5,min_periods=1,center=True).count().count(dim='lim')>0).sum(dim='mlt')

            bm = bm.to_dataframe()
            bm['orbit']=orbit
            bm = intensities(imgs, bm)
            bm[['pb','eb','pb_err','eb_err','ve_pb','vn_pb','ve_eb','vn_eb','dpb_dt','dpb_dp','deb_dt','deb_dp','dP','dA','dP_dt','dA_dt','isglobal','count','orbit','P_mean','P_std','A_mean','A_std','S_mean','S_std']].to_hdf(bpath+'final_boundaries_final.h5','final',format='table',append=True,data_columns=True)
        except Exception as e: print(e)

def makeGIFs(orbits,wicpath,bpath,outpath,tempdir='temp'):
    '''
    Make GIF of each orbit
    
    orbits (list) : Orbit numbers
    wicpath (str) : Path to corrected images
    bpath (str) : Path to model boundaries
    outpath (str) : PAth to save the GIFs
    tempdir (str) : Name of temp folder within outpath
    '''

    minlat = 50


    for orbit in orbits:
        try:
            wic = xr.load_dataset(wicpath+'wic_or'+str(orbit).zfill(4)+'.nc')
            wic['shimg'].attrs = {'long_name': 'Counts', 'units': ''}

            bi = pd.read_hdf(bpath+'initial_boundaries.h5',key='initial',where='orbit=="{}"'.format(orbit))
            bf = pd.read_hdf(bpath+'final_boundaries.h5',key='final',where='orbit=="{}"'.format(orbit))
            bi = bi.reset_index().set_index('date')
            bf = bf.reset_index().set_index('date')

            fig,ax = plt.subplots(figsize=(5,5))
            ax.axis('off')

            for i,t in enumerate(wic.date):
                pax = pp(ax,minlat=minlat)
                ax.set_title('Orbit: '+str(orbit).zfill(4)+'   '+t.dt.strftime('%Y-%m-%d').values.tolist()+' '+t.dt.strftime('%H:%M:%S').values.tolist())

                pax.scatter(wic.sel(date=t)['mlat'].values,wic.sel(date=t)['mlt'].values,c=wic.sel(date=t)['shimg'].values,s=2,alpha=0.5,vmin=0,vmax=500,cmap='Greens')
                try:
                    # Quality flags
                    ind0 = bf.loc[t.values,'isglobal'].all()
                    ind1 = (bf.loc[t.values,'A_mean'] > bf.loc[t.values,'P_mean']+bf.loc[t.values,'P_std']+bf.loc[t.values,'A_std']).all()
                    ind2 = (bf.loc[t.values,'A_mean'] > bf.loc[t.values,'S_mean']+bf.loc[t.values,'S_std']+bf.loc[t.values,'A_std']).all()
                    ind3 = (bf.loc[t.values,'count'] > 12).all()

                    alpha = 1 
                    linestyle = '-' if (ind0&ind1&ind2&ind3) else ':'
                    pax.scatter(bi.loc[t.values,'pb'].values,bi.loc[t.values,'mlt'].values,s=1,color='C6')
                    pax.scatter(bi.loc[t.values,'eb'].values,bi.loc[t.values,'mlt'].values,s=1,color='C9')
                    pax.plot(bf.loc[t.values,'pb'].values,bf.loc[t.values,'mlt'].values,color='C3',alpha=alpha,linestyle=linestyle)
                    pax.plot(bf.loc[t.values,'eb'].values,bf.loc[t.values,'mlt'].values,color='C0',alpha=alpha,linestyle=linestyle)
                except Exception as e: print(e)
                
                try:
                    mlat_err = np.concatenate((bf.loc[t.values,'pb'].values+bf.loc[t.values,'pb_err'].values,bf.loc[t.values,'pb'].values[[0]]+bf.loc[t.values,'pb_err'].values[[0]],bf.loc[t.values,'pb'].values[[0]]-bf.loc[t.values,'pb_err'].values[[0]],bf.loc[t.values,'pb'].values[::-1]-bf.loc[t.values,'pb_err'].values[::-1]))
                    mlt_err = np.concatenate((bf.loc[t.values,'mlt'].values,bf.loc[t.values,'mlt'].values[[0,0]],bf.loc[t.values,'mlt'].values[::-1]))
                    pax.fill(mlat_err,mlt_err,color='C3',alpha=0.3*alpha,edgecolor=None)
                    
                    mlat_err = np.concatenate((bf.loc[t.values,'eb'].values+bf.loc[t.values,'eb_err'].values,bf.loc[t.values,'eb'].values[[0]]+bf.loc[t.values,'eb_err'].values[[0]],bf.loc[t.values,'eb'].values[[0]]-bf.loc[t.values,'eb_err'].values[[0]],bf.loc[t.values,'eb'].values[::-1]-bf.loc[t.values,'eb_err'].values[::-1]))
                    pax.fill(mlat_err,mlt_err,color='C0',alpha=0.3*alpha,edgecolor=None)
                except Exception as e: print(e)
                        
                
                plt.savefig(outpath + 'temp/wic'+str(i).zfill(4)+'.png',bbox_inches='tight',dpi=150)

                ax.patches.clear()
                ax.collections.clear()
                ax.lines.clear()

            plt.close()
            os.system('convert '+outpath+'temp/wic*.png '+outpath+'imgs_or'+str(orbit).zfill(4)+'.gif')
            os.system('rm '+outpath+'temp/*.png')
        except Exception as e: print(e)



def find_reg(orbits,bpath,Ls,ind,oLs=None):
    '''
    Find regularization parameters
    
    orbits (list) : Orbit numbers
    bpath (str) : Path to model boundaries
    Ls (array) : Regularization parameters to check
    oLs (array) : Value of the remaining reg params. If none, all are zero
    ind (int) : Index of reg parameter to check. tLeb=0, sLeb=1, tLpb=2 and sLpb=3]
    '''

    if oLs is None: oLs = np.zeros(4)
    boundary = ['eb','eb','pb','pb']
    rnorms = np.full((len(orbits),len(Ls)),np.nan)
    mnorms = np.full((len(orbits),len(Ls)),np.nan)

    for i,orbit in enumerate(orbits):
        try:
            bi = pd.read_hdf(bpath+'initial_boundaries.h5',key='initial',where='orbit=="{}"'.format(orbit)).to_xarray()

            for j,l in enumerate(Ls):
                oLs[ind]=l
                bm = fuv.boundarymodel_BS(bi,tLeb=oLs[0],sLeb=oLs[1],tLpb=oLs[2],sLpb=oLs[3])
                rnorms[i,j]=bm['residualnorm_'+boundary[ind]].values
                mnorms[i,j]=bm['modelnorm_'+boundary[ind]].values
        except Exception as e: print(e)
    return rnorms,mnorms

def makeGIFs2(corenumber):
    '''
    Make GIF of each orbit
    
    orbits (list) : Orbit numbers
    wicpath (str) : Path to corrected images
    bpath (str) : Path to model boundaries
    outpath (str) : PAth to save the GIFs
    tempdir (str) : Name of temp folder within outpath
    '''
    
    wicpath = '/mnt/5fa6bccc-fa9d-4efc-9ddc-756f65699a0a/aohma/fuv/wic/'
    bpath = '/mnt/5fa6bccc-fa9d-4efc-9ddc-756f65699a0a/aohma/fuv/boundaries/'
    outpath = '/mnt/5fa6bccc-fa9d-4efc-9ddc-756f65699a0a/aohma/fuv/fig/'
    temppath = outpath + 'temp'+str(corenumber)+'/'

    orbits = [list(range(0,300)),list(range(300,500)),list(range(500,700)),list(range(700,900)),list(range(900,1100)),list(range(1100,1300)),list(range(1300,1500)),list(range(1500,1704))][corenumber-1]

    minlat = 50


    for orbit in orbits:
        try:
            wic = xr.load_dataset(wicpath+'wic_or'+str(orbit).zfill(4)+'.nc')
            wic['shimg'].attrs = {'long_name': 'Counts', 'units': ''}

            bi = pd.read_hdf(bpath+'detected_boundaries.h5',key='initial',where='orbit=="{}"'.format(orbit))
            bf = pd.read_hdf(bpath+'modeled_boundaries.h5',key='final',where='orbit=="{}"'.format(orbit))
            bi = bi.reset_index().set_index('date')
            bf = bf.reset_index().set_index('date')

            fig,ax = plt.subplots(figsize=(5,5))
            ax.axis('off')

            for i,t in enumerate(wic.date):
                pax = pp(ax,minlat=minlat)
                ax.set_title('Orbit: '+str(orbit).zfill(4)+'   '+t.dt.strftime('%Y-%m-%d').values.tolist()+' '+t.dt.strftime('%H:%M:%S').values.tolist())

                pax.scatter(wic.sel(date=t)['mlat'].values,wic.sel(date=t)['mlt'].values,c=wic.sel(date=t)['shimg'].values,s=2,alpha=0.5,vmin=0,vmax=500,cmap='Greens')
                try:
                    # Quality flags
                    ind0 = bf.loc[t.values,'isglobal'].all()
                    ind1 = (bf.loc[t.values,'A_mean'] > bf.loc[t.values,'P_mean']+2*bf.loc[t.values,'P_std']).all()
                    ind2 = (bf.loc[t.values,'A_mean'] > bf.loc[t.values,'S_mean']+2*bf.loc[t.values,'S_std']).all()
                    ind3 = (bf.loc[t.values,'count'] > 12).all()
                    ind4 = (bf.loc[t.values,'pb_err'].quantile(0.75)<1.5)
                    ind5 = (bf.loc[t.values,'eb_err'].quantile(0.75)<1.5)

                    alpha = 1 
                    linestyle = '-' if (ind0&ind1&ind2&ind3&ind4&ind5) else ':'
                    pax.scatter(bi.loc[t.values,'pb'].values,bi.loc[t.values,'mlt'].values,s=1,color='C6')
                    pax.scatter(bi.loc[t.values,'eb'].values,bi.loc[t.values,'mlt'].values,s=1,color='C9')
                    pax.plot(bf.loc[t.values,'pb'].values,bf.loc[t.values,'mlt'].values,color='C3',alpha=alpha,linestyle=linestyle)
                    pax.plot(bf.loc[t.values,'eb'].values,bf.loc[t.values,'mlt'].values,color='C0',alpha=alpha,linestyle=linestyle)
                except Exception as e: print(e)
                
                try:
                    mlat_err = np.concatenate((bf.loc[t.values,'pb'].values+bf.loc[t.values,'pb_err'].values,bf.loc[t.values,'pb'].values[[0]]+bf.loc[t.values,'pb_err'].values[[0]],bf.loc[t.values,'pb'].values[[0]]-bf.loc[t.values,'pb_err'].values[[0]],bf.loc[t.values,'pb'].values[::-1]-bf.loc[t.values,'pb_err'].values[::-1]))
                    mlt_err = np.concatenate((bf.loc[t.values,'mlt'].values,bf.loc[t.values,'mlt'].values[[0,0]],bf.loc[t.values,'mlt'].values[::-1]))
                    pax.fill(mlat_err,mlt_err,color='C3',alpha=0.3*alpha,edgecolor=None)
                    
                    mlat_err = np.concatenate((bf.loc[t.values,'eb'].values+bf.loc[t.values,'eb_err'].values,bf.loc[t.values,'eb'].values[[0]]+bf.loc[t.values,'eb_err'].values[[0]],bf.loc[t.values,'eb'].values[[0]]-bf.loc[t.values,'eb_err'].values[[0]],bf.loc[t.values,'eb'].values[::-1]-bf.loc[t.values,'eb_err'].values[::-1]))
                    pax.fill(mlat_err,mlt_err,color='C0',alpha=0.3*alpha,edgecolor=None)
                except Exception as e: print(e)
                        
                
                plt.savefig(temppath + 'wic'+str(i).zfill(4)+'.png',bbox_inches='tight',dpi=150)

                ax.patches.clear()
                ax.collections.clear()
                ax.lines.clear()

            plt.close()
            os.system('convert '+temppath+'wic*.png '+outpath+'imgs_or'+str(orbit).zfill(4)+'.gif')
            os.system('rm '+temppath+'*.png')
        except Exception as e: print(e)
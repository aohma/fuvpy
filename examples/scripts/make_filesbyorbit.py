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

import matplotlib.path as mplpath
import matplotlib.pyplot as plt

from scipy.interpolate import BSpline
from scipy.linalg import lstsq

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
            wic = fuv.makeBSmodel(wic,sKnots=[-3.5,-0.25,0,0.25,1.5,3.5],stop=0.01,n_tKnots=5,tukeyVal=5,dampingVal=1e-3)
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

def dataCoverage(imgs,dzalim=75):
    lt = np.arange(0.5,24)
    lat = 90-(30+10*np.cos(np.pi*lt/12))

    isglobal=[]
    for t in range(len(imgs.date)):
        img=imgs.isel(date=t)
        count = np.zeros_like(lt)
        for i in range(len(lt)):
            count[i]=np.sum((img['mlt'].values>(lt[i]-0.5))&(img['mlt'].values<(lt[i]+0.5))&(img['mlat'].values<lat[i])&(img['dza'].values<dzalim))
        isglobal.append((count>0).all())
    return np.array(isglobal)

def calcIntensity(wic,bm,tKnotSep=10,tVal=0,sVal=0):
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

    dfs = []
    rmse = []
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


        df = pd.DataFrame()
        df['mlat'] = wic.sel(date=t)['mlat'].values.flatten()[ineb & ~inpb]
        df['mlt'] = wic.sel(date=t)['mlt'].values.flatten()[ineb & ~inpb]
        df['I'] = wic.sel(date=t)['shimg'].values.flatten()[ineb & ~inpb]
        df['date'] = wic.sel(date=t).date.values

        dfs.append(df)

        rmse.append(np.sqrt(np.nanmean(wic.sel(date=t)['shimg'].values.flatten()[incap & ~(ineb & ~ inpb)]**2)))

    df = pd.concat(dfs).reset_index()

    # ADD RMSE TO BM
    bm = bm.reset_index().set_index(['date','mlt']).to_xarray().sortby(['date','mlt'])
    bm['rmse'] = ('date',rmse)
    bm = bm.to_dataframe().reset_index().set_index('date')

    tOrder=3



    # Start date and duration
    date0 = df['date'].min()
    duration = (df['date'].max()-date0)/ np.timedelta64(1, 'm')
    time=(df['date']-date0)/ np.timedelta64(1, 'm')
    phi = np.deg2rad(15*df['mlt'].values)


    #%% Eq boundary model

    # Equatorward boundary to radias then convert to primed coordinates
    d = df['I'].values

    # Index of finite data
    ind = np.isfinite(d)

    # Temporal knots
    tKnots = np.arange(0,duration+tKnotSep,tKnotSep)
    tKnots = np.r_[np.repeat(tKnots[0],tOrder),tKnots, np.repeat(tKnots[-1],tOrder)]

    # Number of temporal control points
    n_tcp = len(tKnots)-tOrder-1

    # Temporal design matix
    Gtime = BSpline.design_matrix(time[ind],tKnots,tOrder)


    # Spatial knots (extended)
    sOrder = 3
    mltKnots = np.arange(0,24,2)
    sKnots = np.deg2rad(15*mltKnots)
    sKnots = np.r_[sKnots-2*np.pi,sKnots,sKnots+2*np.pi]
    sKnots = np.r_[np.repeat(sKnots[0],sOrder),sKnots, np.repeat(sKnots[-1],sOrder)]

    # Spatial design matrix (extended)
    Gphi = BSpline.design_matrix(phi[ind], sKnots, sOrder)

    # Spatial design matrix (periodic)
    n_pcp = len(mltKnots)
    Gphi = Gphi[:,n_pcp:2*n_pcp]+Gphi[:,2*n_pcp:3*n_pcp]

    # Combine to full design matrix
    G_s = Gphi[:,np.repeat(np.arange(n_pcp),n_tcp)]*Gtime[:,np.tile(np.arange(n_tcp),n_pcp)]



    # 1st order Tikhonov regularization in time
    tL = np.hstack((-np.identity(n_tcp-1),np.zeros((n_tcp-1,1))))+np.hstack((np.zeros((n_tcp-1,1)),np.identity(n_tcp-1)))
    tLTL = np.zeros((n_tcp*n_pcp,n_tcp*n_pcp))
    for i in range(n_pcp): tLTL[i*n_tcp:(i+1)*n_tcp,i*n_tcp:(i+1)*n_tcp] = tL.T@tL

    # 1st order Tikhonov regularization in mlt
    sL = []
    for i in range(n_pcp): sL.append(np.roll(np.r_[-1,1,np.repeat(0,n_pcp-2)],i))
    sL=np.array(sL)
    sLTL = np.zeros((n_pcp*n_tcp,n_pcp*n_tcp))
    for t in range(n_tcp): sLTL[t:t+n_pcp*n_tcp:n_tcp,t:t+n_pcp*n_tcp:n_tcp] = sL.T@sL

    # Combined regularization
    R = tVal*tLTL + sVal*sLTL

    # Solve
    m = lstsq((G_s).T.dot(G_s)+R,(G_s).T.dot(d[ind]),lapack_driver='gelsy')[0]

    # Temporal evaluation matrix
    time_ev=(df['date'].drop_duplicates()-date0).values/ np.timedelta64(1, 'm')
    Gtime = BSpline.design_matrix(time_ev, tKnots,tOrder).toarray()

    # Spatial evaluation matrix
    phi_ev = np.arange(0,2*np.pi,2*np.pi/240)
    Gphi = BSpline.design_matrix(phi_ev, sKnots, sOrder)
    Gphi = Gphi[:,n_pcp:2*n_pcp]+Gphi[:,2*n_pcp:3*n_pcp].toarray()

    G_ev = np.tile(np.repeat(Gphi,n_tcp,axis=1),(len(time_ev),1))*np.repeat(np.tile(Gtime,(1,n_pcp)),len(phi_ev),axis=0)

    dm=G_ev.dot(m)
    bm['I']=dm
    bm = bm.reset_index().set_index(['date','mlt'])
    return bm

def calcRMSE(wic,bm):
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

    rmse_in  = []
    rmse_out = []
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

        rmse_in.append(np.sqrt(np.nanmean(wic.sel(date=t)['shimg'].values.flatten()[ineb & ~inpb]**2)))
        rmse_out.append(np.sqrt(np.nanmean(wic.sel(date=t)['shimg'].values.flatten()[incap & ~(ineb & ~ inpb)]**2)))

    

    # ADD RMSE TO BM
    bm = bm.reset_index().set_index(['date','mlt']).to_xarray().sortby(['date','mlt'])
    bm['rmse_in'] = ('date',rmse_in)
    bm['rmse_out'] = ('date',rmse_out)
    bm = bm.to_dataframe().reset_index().set_index(['date','mlt'])

    return bm

def final_bondaries(orbits):
    wicpath = '/mnt/5fa6bccc-fa9d-4efc-9ddc-756f65699a0a/aohma/fuv/wic/'
    bpath = '/mnt/5fa6bccc-fa9d-4efc-9ddc-756f65699a0a/aohma/fuv/boundaries/'


    for orbit in orbits:
        try:
            imgs = xr.load_dataset(wicpath+'wic_or'+str(orbit).zfill(4)+'.nc')
            bi = pd.read_hdf(bpath+'initial_boundaries.h5',key='initial',where='orbit=="{}"'.format(orbit))

            # Only images with identified initial boundaries
            imgs = imgs.sel(date=bi.reset_index().date.unique())

            bm = fuv.makeBoundaryModelBStest(bi,tKnotSep=5,tLeb=1e-1,sLeb=1e-3,tLpb=1e-1,sLpb=1e-3,resample=False)
            isglobal = dataCoverage(imgs,dzalim=65)
            bm['isglobal'] = ('date',isglobal)

            bm = bm.to_dataframe()
            bm['orbit']=orbit
            bm = calcRMSE(imgs, bm)
            bm[['pb','eb','v_phi','v_theta','u_phi','u_theta','isglobal','orbit','rmse_in','rmse_out']].to_hdf(bpath+'final_boundaries.h5','final',format='table',append=True,data_columns=True)
        except Exception as e: print(e)
        
def final_bondaries_error(orbits):
    wicpath = '/mnt/5fa6bccc-fa9d-4efc-9ddc-756f65699a0a/aohma/fuv/wic/'
    bpath = '/mnt/5fa6bccc-fa9d-4efc-9ddc-756f65699a0a/aohma/fuv/boundaries/'


    for orbit in orbits:
        try:
            imgs = xr.load_dataset(wicpath+'wic_or'+str(orbit).zfill(4)+'.nc')
            bi = pd.read_hdf(bpath+'initial_boundaries.h5',key='initial',where='orbit=="{}"'.format(orbit))

            # Only images with identified initial boundaries
            imgs = imgs.sel(date=bi.reset_index().date.unique())

            bms = []    
            for l in [100,150,200]:
                bm = fuv.makeBoundaryModelBStest(bi.to_xarray().sel(lim=l).to_dataframe(),tKnotSep=5,tLeb=1e-1,sLeb=1e-3,tLpb=1e-1,sLpb=1e-3)
                bm = bm.expand_dims(lim=[l])
                bms.append(bm)
            
            bms = xr.concat(bms,dim='lim')#.std(dim='sample')[['ocb','eqb']]#.rename({'ocb':'pb_err','eqb':'eb_err'})

            bm = bms.mean(dim='lim')
            bm['pb_err'] = bms['pb'].std(dim='lim')
            bm['eb_err'] = bms['eb'].std(dim='lim')

            isglobal = dataCoverage(imgs,dzalim=65)
            bm['isglobal'] = ('date',isglobal)

            bm = bm.to_dataframe()
            bm['orbit']=orbit
            bm = calcRMSE(imgs, bm)
            bm[['pb','eb','pb_err','eb_err','v_phi','v_theta','u_phi','u_theta','isglobal','orbit','rmse_in','rmse_out']].to_hdf(bpath+'final_boundaries.h5','final',format='table',append=True,data_columns=True)
        except Exception as e: print(e)

def makeGIFs(orbits):
    wicpath = '/mnt/5fa6bccc-fa9d-4efc-9ddc-756f65699a0a/aohma/fuv/wic/'
    bpath = '/mnt/5fa6bccc-fa9d-4efc-9ddc-756f65699a0a/aohma/fuv/boundaries/'
    outpath = '/mnt/5fa6bccc-fa9d-4efc-9ddc-756f65699a0a/aohma/fuv/fig/'
    minlat = 50


    for orbit in orbits:
        try:
            wic = xr.load_dataset(wicpath+'wic_or'+str(orbit).zfill(4)+'.nc')
            wic['shimg'].attrs = {'long_name': 'Counts', 'units': ''}

            bi = pd.read_hdf(bpath+'initial_boundaries.h5',key='initial',where='orbit=="{}"'.format(orbit))
            bf = pd.read_hdf(bpath+'final_boundaries.h5',key='final',where='orbit=="{}"'.format(orbit))
            bi = bi.reset_index().set_index('date')
            bf = bf.reset_index().set_index('date')


            r = (90. - np.abs(bf['pb']))/(90. - minlat)
            a = (bf.mlt.values - 6.)/12.*np.pi
            bf['px'] =  r*np.cos(a)
            bf['py'] =  r*np.sin(a)

            r = (90. - np.abs(bf['eb']))/(90. - minlat)
            a = (bf.mlt.values - 6.)/12.*np.pi
            bf['ex'] =  r*np.cos(a)
            bf['ey'] =  r*np.sin(a)

            fig,ax = plt.subplots(figsize=(5,5))
            ax.axis('off')
            pax = fuv.pp(ax,minlat=minlat)

            for i,t in enumerate(wic.date):
                pax = fuv.pp(ax,minlat=minlat)
                ax.set_title('Orbit: '+str(orbit).zfill(4)+'   '+t.dt.strftime('%Y-%m-%d').values.tolist()+' '+t.dt.strftime('%H:%M:%S').values.tolist())

                pax.scatter(wic.sel(date=t)['mlat'].values,wic.sel(date=t)['mlt'].values,c=wic.sel(date=t)['shimg'].values,s=2,alpha=0.5,vmin=0,vmax=500,cmap='Greens')
                try:
                    alpha = 1 if (bf.loc[t.values,'rmse_in']/bf.loc[t.values,'rmse_out']>2).all() else 0.2
                    linestyle = '-' if bf.loc[t.values,'isglobal'].all() else ':'
                    pax.scatter(bi.loc[t.values,'pb'].values,bi.loc[t.values,'mlt'].values,s=1,color='C7')
                    pax.scatter(bi.loc[t.values,'eb'].values,bi.loc[t.values,'mlt'].values,s=1,color='C6')
                    pax.plot(bf.loc[t.values,'pb'].values,bf.loc[t.values,'mlt'].values,color='C3',alpha=alpha,linestyle=linestyle)
                    pax.plot(bf.loc[t.values,'eb'].values,bf.loc[t.values,'mlt'].values,color='C1',alpha=alpha,linestyle=linestyle)
                except Exception as e: print(e)
                
                plt.savefig(outpath + 'temp/wic'+str(i).zfill(4)+'.png',bbox_inches='tight',dpi=150)

                ax.collections.clear()
                ax.lines.clear()

            plt.close()
            os.system('convert '+outpath+'temp/wic*.png '+outpath+'imgs_or'+str(orbit).zfill(4)+'.gif')
            # os.system('convert '+ospath+'fig/temp/binary*.png '+ospath+'fig/oval_'+e+'.gif')
            os.system('rm '+outpath+'temp/*.png')
        except Exception as e: print(e)

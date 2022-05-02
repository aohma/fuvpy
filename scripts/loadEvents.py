#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:18:55 2022

@author: aohma
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import fuvpy as fuv
from polplot import pp

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

# events = glob.glob('/Volumes/Seagate Backup Plus Drive/fuv/wic/idl/north/*')
# events.sort()

# for e in events:
#     wicfiles = glob.glob(e+'/*.idl')   
#     wic = readFUVimage(wicfiles)
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

def pplot(imgs,inImg,col_wrap = None,row=False,add_cbar = True,robust=False,cbar_kwargs={},pp_kwargs={},**kwargs):
    n_imgs = len(imgs.date)

    # Set minlat
    if 'minlat' in pp_kwargs:
        minlat = pp_kwargs['minlat']
        pp_kwargs.pop('minlat')
    else:
        minlat = 50

    # Set crange if not given
    if 'crange' in kwargs.keys():
        crange = kwargs['crange']
        kwargs.pop('crange')
    elif robust:
        crange=np.quantile(wic[inImg].values[wic['mlat'].values>minlat],[0.02,0.98])
    else:
        crange=np.quantile(wic[inImg].values[wic['mlat'].values>minlat],[0,1])
      
    # Set cbar orientation
    if 'orientation' in cbar_kwargs:
        cbar_orientation = cbar_kwargs['orientation']
        cbar_kwargs.pop('orientation')
    else:
        cbar_orientation = 'vertical'
        
    
    # Set up fig size and axes 
    cb_size = 0.5
    if row:
        n_rows = n_imgs
        n_cols=1
        f_height = 4*n_imgs
        f_width = 4
        ii = np.arange(n_imgs)[:,None]
        
        if add_cbar:
            if cbar_orientation=='horizontal':
                f_height += cb_size
            elif cbar_orientation=='verical':
                f_width += cb_size
    elif col_wrap:
        n_rows = (n_imgs-1)//col_wrap+1
        n_cols = col_wrap
        f_height = 4*n_rows
        f_width = 4*n_cols
        ii = np.pad(np.arange(n_imgs),(0,n_rows*n_cols-n_imgs),'constant',constant_values=-1).reshape(n_rows,n_cols)
        
        if add_cbar:
            if cbar_orientation=='horizontal':
                f_height += cb_size
            elif cbar_orientation=='verical':
                f_width += cb_size
    else:
        n_rows = 1
        n_cols=n_imgs
        f_height = 4
        f_width = 4*n_imgs
        ii = np.arange(n_imgs)[None,:]
        
        
        if add_cbar:
            if cbar_orientation=='horizontal':
                f_height += cb_size
            elif cbar_orientation=='verical':
                f_width += cb_size 
       
    fig = plt.figure(figsize=(f_width,f_height))
    
    if add_cbar:
        if cbar_orientation=='horizontal':
            gs0 = gridspec.GridSpec(nrows=2,ncols=1,height_ratios=[n_rows*4,cb_size],hspace=0.01)
        else:
            gs0 = gridspec.GridSpec(nrows=1,ncols=2,width_ratios=[n_cols*4,cb_size],wspace=0.01)
    else:
        gs0 = gridspec.GridSpec(nrows=2,ncols=1,height_ratios=[1,0])
    
    # IMAGES
    gs = gridspec.GridSpecFromSubplotSpec(nrows=n_rows,ncols=n_cols,subplot_spec=gs0[0],hspace=0.06,wspace=0.01)
    
    for i in range(n_imgs):
        i_row = np.where(ii==i)[0][0]
        i_col = np.where(ii==i)[1][0]
        pax = pp(plt.subplot(gs[i_row,i_col]),minlat=minlat,**pp_kwargs)
    
        mlat = imgs.isel(date=i)['mlat'].values.copy()
        mlt = imgs.isel(date=i)['mlt'].values.copy()
        image = imgs.isel(date=i)[inImg].values.copy()
        pax.plotimg(mlat,mlt,image,crange=crange,**kwargs)
        pax.ax.set_title(wic['id'].values.tolist() + ': ' + 
             imgs.isel(date=i)['date'].dt.strftime('%Y-%m-%d %H:%M:%S').values.tolist())
    
    if add_cbar:
        cax = plt.subplot(gs0[1])
        cax.axis('off')
        cbar = plt.colorbar(pax.ax.collections[0],orientation=cbar_orientation,ax=cax,fraction=1,**cbar_kwargs)
        
        # cbar name
        if len(imgs[inImg].attrs)==2:
            cbar.set_label('{} ({})'.format(wic[inImg].attrs['long_name'],wic[inImg].attrs['units']))
        elif len(imgs[inImg].attrs)==1:
            cbar.set_label('{}'.format(wic[inImg].attrs['long_name']))
        else:
            cbar.set_label(inImg)
    # gs0.tight_layout(fig)

wicfiles = glob.glob('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/python/git/fuvpy/data/wicFiles/*.idl')   
wic = fuv.readImg(wicfiles)
wic = fuv.makeDGmodel(wic,transform='log')
wic = fuv.makeSHmodel(wic,4,4)
pplot(wic.isel(date=slice(2,6)),'img',robust=True,cbar_kwargs={'extend':'both'})
pplot(wic.isel(date=slice(2,6)),'img',row=True)
pplot(wic.isel(date=slice(2,6)),'img',col_wrap=2)

pplot(wic.isel(date=slice(2,6)),'img',cbar_kwargs={'orientation':'horizontal'})
pplot(wic.isel(date=slice(2,6)),'img',row=True,cbar_kwargs={'orientation':'horizontal'})
pplot(wic.isel(date=slice(2,6)),'img',col_wrap=2,cbar_kwargs={'orientation':'horizontal','extend':'max'})

pplot(wic.isel(date=slice(2,6)),'img',add_cbar = False)
pplot(wic.isel(date=slice(2,6)),'img',row=True,add_cbar = False)
pplot(wic.isel(date=slice(2,6)),'img',col_wrap=2,add_cbar = False)









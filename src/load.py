#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 20:46:03 2022

@author: aohma
"""

import numpy as np
import pandas as pd
import xarray as xr

from datetime import datetime
from pathlib import Path

from scipy.io import idl


def readImg(filenames, dzalim = 80, minlat = 0, hemisphere = None, reflat=True):
    '''
    Load FUV images into a xarray.Dataset

    Parameters
    ----------
    filenames : str or list of str
        Path to one or several auroral fuv images stored in .idl or .sav files
    dzalim : float, optional
        Upper limit of the satellite zenith angle (viewing angle). The default is 80.
    minlat : float, optional
        Minimum abs(mlat) to include. The default is 0 (magnetic equator).
    hemisphere : str, optional
        Which hemisphere to include. 'north','south' and None.
        Default is None (automatically detected).
    reflat: bool, optional
        Reapply WIC's flatfield or keep original flatfield.
        Default is True (reapply).

    Returns
    -------
    xarray.Dataset with the following fields
    DATA:
    'img': Number of counts in each pixel
    'mlat': Magnetic latitude of each pixel
    'mlon': Magnetic longitude of each pixel
    'mlt': Magnetic local time of each pixel
    'glat': Geographic latitude of each pixel
    'glon': Geographic longitude of each pixel
    'dza': Satellite zenith angle (viewing angle) of each pixel
    'sza': Solar zenith angle of each pixel
    'hemisphere': Hemisphere of the image
    'id': Short name of the detector
    'bad': Bool of problematic pixels. True=good and False=bad
    DIMS:
    'date': Datetime of each image
    'row': rows of the detector
    'col': columns of the detector
    '''
    if isinstance(filenames, str):
        filenames = [filenames]

    filenames.sort()
    imgs = []
    for i in range(len(filenames)):
        imageinfo = idl.readsav(filenames[i])['imageinfo']

        # Read *.idl/*.sav files
        img = xr.Dataset({
        'img': (['row','col'],imageinfo['image'][0]),
        'mlat':(['row','col'],imageinfo['mlat'][0]),
        'mlon':(['row','col'],imageinfo['mlon'][0]),
        'mlt': (['row','col'],imageinfo['mlt'][0]),
        'glat':(['row','col'],imageinfo['glat'][0]),
        'glon':(['row','col'],imageinfo['glon'][0]),
        'dza': (['row','col'],imageinfo['dza'][0]),
        'sza': (['row','col'],imageinfo['sza'][0])
        })
        img = img.expand_dims(date=[pd.to_datetime(_timestampImg(imageinfo['time'][0]))])

        # Replace fill values with np.nan
        img['mlat'] = xr.where(img['mlat']==-1e+31,np.nan,img['mlat'])
        img['mlon'] = xr.where(img['mlon']==-1e+31,np.nan,img['mlon'])
        img['mlt']  = xr.where(img['mlt'] ==-1e+31,np.nan,img['mlt'])
        img['glat'] = xr.where(img['glat']==-1e+31,np.nan,img['glat'])
        img['glon'] = xr.where(img['glon']==-1e+31,np.nan,img['glon'])
        img['sza']  = xr.where(img['sza']==-1,np.nan,img['sza'])
        img['dza']  = xr.where(img['dza']==-1,np.nan,img['dza'])

        # Set mlat/mlt to np.nan where when it is outside the requested range
        img['mlt']  = xr.where(img['dza'] > dzalim,np.nan,img['mlt'])
        img['mlat'] = xr.where(img['dza'] > dzalim,np.nan,img['mlat'])

        img['mlt']  = xr.where((img['mlat'] > 90) | (img['mlat'] < -90),np.nan,img['mlt'])
        img['mlat'] = xr.where((img['mlat'] > 90) | (img['mlat'] < -90),np.nan,img['mlat'])

        img['mlt']  = xr.where(img['glat']==0,np.nan,img['mlt'])
        img['mlat'] = xr.where(img['glat']==0,np.nan,img['mlat'])

        # Select hemisphere
        if hemisphere =='south' or (hemisphere is None and np.nanmedian(img['mlat'])<0):
            img=img.assign({'hemisphere': 'south'})
            img['mlt']  = xr.where(img['mlat'] > 0,np.nan,img['mlt'])
            img['mlat'] = xr.where(img['mlat'] > 0,np.nan,img['mlat'])
        elif hemisphere == 'north' or (hemisphere is None and np.nanmedian(img['mlat'])>0):
            img=img.assign({'hemisphere': 'north'})
            img['mlt']  = xr.where(img['mlat'] < 0,np.nan,img['mlt'])
            img['mlat'] = xr.where(img['mlat'] < 0,np.nan,img['mlat'])
        else:
            img=img.assign({'hemisphere': 'na'})

        img['mlat'] = np.abs(img['mlat'])
        img['mlt']  = xr.where(img['mlat'] < minlat,np.nan,img['mlt'])
        img['mlat'] = xr.where(img['mlat'] < minlat,np.nan,img['mlat'])

        imgs.append(img)

    imgs = xr.concat(imgs, dim='date')
    imgs = imgs.assign({'id':  imageinfo['inst_id'][0].strip().decode('utf8')})

    # Add a boolerian field to flag bad pixels in the detector
    imgs = _badPixels(imgs)

    # Reapply WIC's flat field
    if (imgs['id']=='WIC')&reflat:
        imgs=_reflatWIC(imgs)

    # Add attributes
    imgs['img'].attrs = {'long_name': 'Original image'}
    imgs['mlat'].attrs = {'long_name': 'Magnetic latitude', 'units': 'deg'}
    imgs['mlon'].attrs = {'long_name': 'Magnetic longitude', 'units': 'deg'}
    imgs['mlt'].attrs = {'long_name': 'Magnetic local time', 'units': 'hrs'}
    imgs['glat'].attrs = {'long_name': 'Geographic latitude', 'units': 'deg'}
    imgs['glon'].attrs = {'long_name': 'Geographic longitude', 'units': 'deg'}
    imgs['sza'].attrs = {'long_name': 'Solar zenith angle', 'units': 'deg'}
    imgs['dza'].attrs = {'long_name': 'Viewing angle', 'units': 'deg'}

    return imgs

def _timestampImg(timestamp):
    """ returns datetime object for timestamp = imageinfo['TIME'][0] """

    hh = int(timestamp[1]/1000/60/60)
    mm = int(timestamp[1]/1000/60) - 60 * hh
    ss = int(timestamp[1]/1000) - 60**2 * hh - 60 * mm

    hh = str(hh).zfill(2)
    mm = str(mm).zfill(2)
    ss = str(ss).zfill(2)

    hhmmss = hh + mm + ss


    if len(str(timestamp[0])) != 7: # handle occasional errors in VIS image structure
        timestamp0 = int(str(timestamp[0])[:4] + str(timestamp[0])[4:].zfill(3))
    else:
        timestamp0 = timestamp[0]

    time = datetime.strptime(str(timestamp0) + hhmmss, '%Y%j%H%M%S')

    return time

def _badPixels(imgs):
    '''
    Add a data field with index of problematic pixels in the different detectors.
    The exact number will depent on the datetime, so this static approach is an approximation.
    Work could be done to get this function more accurate.

    Parameters
    ----------
    imgs : xarray.Dataset
        Dataset with the FUV images.

    Returns
    -------
    imgs : xarray.Dataset
        FUV dataset with a new field imgs['bad'], containing the
        logical indices of bad pixels (False is bad).
    '''

    if imgs['id'] == 'WIC':
        ind=np.ones((256,256),dtype=bool)
        ind[230:,:] = False # Broken boom shades the upper rows of WIC (after what date?)
        ind[:2,:] = False #
    elif imgs['id'] == 'SI13': #
        ind = np.ones((128,128),dtype=bool)
        ind[:8,:] = False
        ind[120:,:] = False
    elif imgs['id'] == 'VIS':
        ind = np.ones((256,256),dtype=bool)
        ind[:,:4]=False
        # The corner of VIS Earth have an intensity dropoff
        for jj in range(25): ind[-1-jj,:25-jj] = False
        for jj in range(25): ind[-1-jj,-25+jj:] = False
        for jj in range(25): ind[jj,-25+jj:] = False
        for jj in range(25): ind[jj,:25-jj] = False
    elif imgs['id']=='UVI': # No known problems
        ind = np.ones((228,200),dtype=bool)
    elif imgs['id']=='SI12': #
        ind = np.ones((128,128),dtype=bool)
        ind[118:,:] = False # Broken boom shades the upper rows of WIC (after what date?)
        ind[:13,:] = False
    imgs = imgs.assign({'bad':(['row','col'],ind)})
    return imgs

def _reflatWIC(wic,inImg='img',outImg='img'):
    '''
    The function changes how WIC's flatfield in implemented.
    Removes flatfield corrections, substractacts the general background noise
    from all the images, reapplies the flatfield and add background

    Parameters
    ----------
    imgs : xarray.Dataset
        Dataset with the FUV images.
    inImg : srtr, optional
        Name of the image to adjust. The default is 'image'.
    outImg : srtr, optional
        Name of the returned image. The default is 'image'.
    Returns
    -------
    imgs : xarray.Dataset
        Copy(?) of the FUV dataset with a new field containing the reflattened images.
    '''
    path = Path(__file__).parent / '../data/wic_flatfield_dbase.idl'
    flatfields = idl.readsav(path)['flatfields']
    if pd.to_datetime(wic['date'][0].values)<pd.to_datetime('2000-10-03 23:30'):
        flat = flatfields[:,0]
    else:
        flat = flatfields[:,1]

    background=np.nanmedian((wic[inImg]/flat[None,:,None]).values[wic['bad'].values&(wic['sza'].values>100|np.isnan(wic['sza'].values))])
    if np.isnan(background): background=450 # Set a reasonable value if no background pixels are available
    wic[outImg]=((wic[inImg].copy()/flat[None,:,None]-background)*flat[None,:,None]+background)
    return wic

def getRayleigh(imgs,inImg='img'):
    '''
    Convect counts to Rayleigh
    Parameters
    ----------
    imgs : xarray.Dataset
        Dataset with the FUV images.
    inImg : srtr, optional
        Name of the image to convect. The default is 'image'.
    Returns
    -------
    imgs : xarray.Dataset
        Copy(?) of the FUV dataset with a new field containing the convected images.
    '''
    if imgs['id']=='WIC':
        imgs[inImg+'R'] = imgs[inImg]/612.6
    elif imgs['id']=='VIS':
        imgs[inImg+'R'] = imgs[inImg]/4.32
    elif imgs['id']=='UVI':
        imgs[inImg+'R'] = imgs[inImg]/35.52
    elif imgs['id']=='SI13':
        imgs[inImg+'R'] = imgs[inImg]/15.3

    imgs[inImg+'R'].attrs = {'long_name': imgs[inImg].attrs['long_name'], 'units': 'kR'}
    return imgs
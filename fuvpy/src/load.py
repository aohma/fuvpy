#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 20:46:03 2022

@author: aohma
"""

import os
import numpy as np
import pandas as pd
import xarray as xr

from datetime import datetime

# from scipy.io import idl
from scipy.io import readsav

def read_idl(filenames, dzalim = 80, hemisphere = None, reflat=True, remove_bad=True):
    '''
    Load FUV images into a xarray.Dataset

    Parameters
    ----------
    filenames : str or list of str
        Path to one or several auroral fuv images stored in .idl or .sav files
    dzalim : float, optional
        Upper limit of the satellite zenith angle (viewing angle). The default is 80.
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
        imageinfo = readsav(filenames[i])['imageinfo']

        inst_id = imageinfo['inst_id'][0].strip().decode('utf8')

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
        img = img.expand_dims(date=[pd.to_datetime(_timestamp_idl(imageinfo['time'][0]))])

        coordinates = ['mlat','mlon','mlt','glat','glon','sza','dza']
        fillvals = [-1e+31,-1e+31,-1e+31,-1e+31,-1e+31,-1,-1]

        # Replace fill values with np.nan
        for coordinate,fillval in zip(coordinates, fillvals):
            img[coordinate] = xr.where(img[coordinate]==fillval,np.nan,img[coordinate])

        # Select hemisphere
        if hemisphere =='south' or (hemisphere is None and np.nanmedian(img['mlat'])<0):
            img=img.assign({'hemisphere': 'south'})
        elif hemisphere == 'north' or (hemisphere is None and np.nanmedian(img['mlat'])>0):
            img=img.assign({'hemisphere': 'north'})
        else:
            img=img.assign({'hemisphere': 'na'})

        # coordinates to include
        ind = (img['dza'] < dzalim) & (img['mlat'] < 90) & (img['mlat'] > -90) & (img['glat']!=0)
        if inst_id=='WIC': ind = ind & (img['img'] > 0) # Zero-valued pixels in WIC are actually NaNs
        if remove_bad: ind = ind & ~_bad_pixels(inst_id) # Ignore bad parts of the detectors

        for coordinate in coordinates:
            img[coordinate] = xr.where(~ind,np.nan,img[coordinate])

        imgs.append(img)

    imgs = xr.concat(imgs, dim='date')
    imgs = imgs.assign({'id':  imageinfo['inst_id'][0].strip().decode('utf8')})

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

def _timestamp_idl(timestamp):
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

def _bad_pixels(inst_id):
    '''
    Index of problematic pixels in the different detectors.
    The exact number will depent on the datetime, so this static approach is an approximation.
    Work could be done to get this function more accurate.

    Parameters
    ----------
    inst_id : str
        id of the FUV detector.

    Returns
    -------
    ind : np.array
        logical indices of bad pixels (True is bad).
    '''

    if inst_id == 'WIC':
        ind=np.zeros((256,256),dtype=bool)
        ind[230:,:] = True # Broken boom shades the upper rows of WIC (after what date?)
        ind[:2,:] = True #
    elif inst_id == 'SI13': #
        ind = np.zeros((128,128),dtype=bool)
        ind[:8,:] = True
        ind[120:,:] = True
    elif inst_id == 'VIS':
        ind = np.zeros((256,256),dtype=bool)
        ind[:,:4]=True
        # The corner of VIS Earth have an intensity dropoff
        for jj in range(25): ind[-1-jj,:25-jj] = True
        for jj in range(25): ind[-1-jj,-25+jj:] = True
        for jj in range(25): ind[jj,-25+jj:] = True
        for jj in range(25): ind[jj,:25-jj] = True
    elif inst_id=='UVI': # No known problems
        ind = np.zeros((228,200),dtype=bool)
    elif inst_id=='SI12': #
        ind = np.zeros((128,128),dtype=bool)
        ind[118:,:] = True # Broken boom shades the upper rows of WIC (after what date?)
        ind[:13,:] = True
    return ind

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
    basepath = os.path.dirname(__file__)
    flatfields = readsav(basepath+'/../utils/wic_flatfield_dbase.idl')['flatfields']
    if pd.to_datetime(wic['date'][0].values)<pd.to_datetime('2000-10-03 23:30'):
        flat = flatfields[:,0]
    else:
        flat = flatfields[:,1]

    background=np.nanmedian((wic[inImg]/flat[None,:,None]).values[wic['sza'].values>100|(np.isnan(wic['sza'].values)&(wic['img'].values>0))])
    if np.isnan(background): background=450 # Set a reasonable value if no background pixels are available
    wic[outImg]=((wic[inImg].copy()/flat[None,:,None]-background)*flat[None,:,None]+background)
    return wic

def add_rayleigh(imgs,**kwargs):
    '''
    Add data_var with intinsity converted from counts to Rayleigh

    Parameters
    ----------
    imgs : xarray.Dataset
        Dataset with the FUV images.
    inImg : str, optional
        Name of the data_var with the input image in counts. The default is 'img'.
    inplace : bool, optional
        If True, update the Dataset in place. Default is False (return new Dataset) 
    
    Returns
    -------
    imgs : xarray.Dataset, optional
        Dataset including a new data_var with the intensity in kilo Rayleigh.
    '''
    
    # Set keyword arguments to input or default values    
    inImg = kwargs.pop('inImg') if 'inImg' in kwargs.keys() else 'img'
    inplace = bool(kwargs.pop('inplace')) if 'inplace' in kwargs.keys() else False

    # Make a copy if a new Dataset should be returned
    if not inplace: imgs = imgs.copy(deep=True)

    # Add a new data_var with counts in kilo Rayleigh
    if imgs['id']=='WIC':
        imgs[inImg+'R'] = imgs[inImg]/612.6
    elif imgs['id']=='VIS':
        imgs[inImg+'R'] = imgs[inImg]/4.32
    elif imgs['id']=='UVI':
        imgs[inImg+'R'] = imgs[inImg]/35.52
    elif imgs['id']=='SI13':
        imgs[inImg+'R'] = imgs[inImg]/15.3

    # Add attributes
    imgs[inImg+'R'].attrs = {'long_name': imgs[inImg].attrs['long_name'], 'units': 'kR'}
    
    # Return the new Dataset
    if not inplace: return imgs

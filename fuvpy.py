#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 11:29:53 2022

@author: aohma
"""
import numpy as np
import pandas as pd
import xarray as xr

from datetime import datetime
from pathlib import Path

from scipy.io import idl
from scipy.interpolate import BSpline,griddata
from scipy.linalg import lstsq

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.path as path
from matplotlib.collections import PolyCollection

import apexpy
from fuvpy.utils import sh
from fuvpy.utils.sunlight import subsol
from polplot import pp
import ppigrf

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
    elif imgs['id'] == 'SI13': # No known problems
        ind = np.ones((128,128),dtype=bool)
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
    path = Path(__file__).parent / './data/wic_flatfield_dbase.idl'
    flatfields = idl.readsav(path)['flatfields']
    if pd.to_datetime(wic['date'][0].values)<pd.to_datetime('2000-10-03 23:30'):
        flat = flatfields[:,0]
    else:
        flat = flatfields[:,1]

    background=np.nanmedian((wic[inImg]/flat[None,:,None]).values[wic['bad'].values&(wic['sza'].values>100|np.isnan(wic['sza'].values))])
    if np.isnan(background): background=450 # Set a reasonable value if no background pixels are available
    wic[outImg]=((wic[inImg].copy()/flat[None,:,None]-background)*flat[None,:,None]+background)
    return wic

def getIMAGEcmap():
    """
    Make IMAGE color scale
    Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """

    r = [255,250,235,210,150, 45, 45, 70, 95,140,190,230,255,255,255,255,255,255,220,180,140,100, 60,  0]
    g = [255,250,245,230,240,225,225,235,245,255,255,250,255,238,208,174,127,  0,  0,  0,  0,  0,  0,  0]
    b = [255,250,240,230,245,210,100, 55, 10,  0,  0,  0,  0,  0,  0,  0,  0, 45, 45, 45, 45, 45, 45,  0]

    rgbarray = np.array([r, g, b]).T.astype(np.float32)/255.
    rgbseq = [tuple(s) for s in rgbarray]
    nsteps = len(rgbseq) - 1
    seq = []
    for i in range(nsteps):
        if i != nsteps - 1:
            seq.append(rgbseq[i])
            seq.append(rgbseq[i + 1])
            seq.append((i+1)*1./nsteps)
        else:
            seq.append(rgbseq[i + 1])

    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])

    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


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

def makeDGmodel(imgs,inImg='img',transform=None,sOrder=3,dampingVal=0,tukeyVal=5,stop=1e-3,minlat=0,dzalim=80,sKnots=None,tKnotSep=None,tOrder=2):
    '''
    Function to model the FUV dayglow and subtract it from the input image

    Parameters
    ----------
    imgs : xarray.Dataset
        Dataset with the FUV images, imported by readFUVimage()
    inImg : str, optional
        Name of the input image to be used in the model. The default is 'img'.
    sOrder : int, optional
        Order of the spatial spline fit. The default is 3.
    dampingVal : float, optional
        Damping (Tikhonov regularization).
        The default is 0 (no damping).
    tukeyVal : float, optional
        Determines to what degree outliers are down-weighted.
        Iterative reweights is (1-(residuals/(tukeyVal*rmse))^2)^2
        Larger tukeyVal means less down-weight
        Default is 5
    stop : float, optional
        When to stop the iteration. The default is 0.001.
    minlat : float, optional
        Lower mlat boundary to include in the model. The default is 0.
    dzalim : float, optional
        Maximum viewing angle to include. The default is 80.
    sKnots : array like, optional
        Location of the spatial Bspline knots. The default is None (default is used).
    tKnotSep : int, optional
        Approximate separation of temporal knots in minutes. The default is None (only knots at endpoints)
    tOrder : int, optional
        Order of the temporal spline fit. The default is 2.

    Returns
    -------
    imgs : xarray.Dataset
        A copy(?) of the image Dataset with three new fields:
            - imgs['dgmodel'] is the dayglow model
            - imgs['dgimg'] is the dayglow-corrected image (dayglow subtracked from the input image)
            - imgs['dgweight'] is the weights after the final iteration
    '''

    # Add temporal dimension if missing
    if len(imgs.sizes)==2: imgs = imgs.expand_dims('date')

    # Reshape the data
    sza   = imgs['sza'].stack(z=('row','col')).values
    dza   = imgs['dza'].stack(z=('row','col')).values
    if transform=='log':
        d = np.log(imgs[inImg].stack(z=('row','col')).values+1)
    else:
        d = imgs[inImg].stack(z=('row','col')).values
    mlat  = imgs['mlat'].stack(z=('row','col')).values
    remove = imgs['bad'].stack(z=('row','col')).values

    # Spatial knots and viewing angle correction
    if imgs['id'] in ['WIC','SI12','SI13','UVI']:
        fraction = np.cos(np.deg2rad(sza))/np.cos(np.deg2rad(dza))
        if sKnots is None: sKnots = [-5,-3,-1,-0.2,-0.1,0,0.1,0.2,1,3,5]
    elif imgs['id'] == 'VIS':
        fraction = np.cos(np.deg2rad(sza))
        if sKnots is None: sKnots= [-1,-0.2,-0.1,0,0.333,0.667,1]
    sKnots = np.r_[np.repeat(sKnots[0],sOrder),sKnots, np.repeat(sKnots[-1],sOrder)]

    # Minuets since first image
    date = imgs['date'].values
    time=(date-date[0])/ np.timedelta64(1, 'm')

    # temporal and spatial size
    n_t = d.shape[0]
    n_s = d.shape[1]

    # Temporal knots
    if tKnotSep==None:
        tKnots = np.linspace(time[0], time[-1], 2)
    else:
        tKnots = np.linspace(time[0], time[-1], int(np.round(time[-1]/tKnotSep)+1))
    tKnots = np.r_[np.repeat(tKnots[0],tOrder),tKnots, np.repeat(tKnots[-1],tOrder)]

    # Number of control points
    n_tcp = len(tKnots)-tOrder-1
    n_scp = len(sKnots)-sOrder-1

    # Temporal design matix
    print('Building dayglow G matrix')
    M = BSpline(tKnots, np.eye(n_tcp), tOrder)(time)

    G_g=[]
    G_s=[]
    for i in range(n_t):
        G= BSpline(sKnots, np.eye(n_scp), sOrder)(fraction[i,:]) # Spatial design matirx
        G_t = np.zeros((n_s, n_scp*n_tcp))

        for j in range(n_tcp):
            G_t[:, np.arange(j, n_scp*n_tcp, n_tcp)] = G*M[i, j]

        G_g.append(G)
        G_s.append(G_t)
    G_s=np.vstack(G_s)

    # Data
    ind = ((sza >= 0) & (dza <= dzalim) & (mlat >= minlat) & (np.isfinite(d)) & remove[None,:])
    d_s = d.flatten()
    ind = ind.flatten()
    w = np.ones(d_s.shape)

    # Damping
    damping = dampingVal*np.ones(G_s.shape[1])
    R = np.diag(damping)

    # Iterativaly solve the inverse problem
    diff = 1e10*stop
    iteration = 0
    m = None
    while (diff>stop)&(iteration < 100):
        print('Iteration:',iteration)

        m_s = np.linalg.lstsq((G_s[ind,:]*w[ind,None]).T@(G_s[ind,:]*w[ind,None])+R,(G_s[ind,:]*w[ind,None]).T@(d_s[ind]*w[ind]),rcond=None)[0]

        mNew    = M@m_s.reshape((n_scp, n_tcp)).T
        dm=[]
        for i, tt in enumerate(time):
            dm.append(G_g[i]@mNew[i, :])

        dm=np.array(dm).squeeze()
        residuals = dm.flatten()[ind] - d.flatten()[ind]
        rmse = np.sqrt(np.average(residuals**2,weights=w[ind]))

        iw = ((residuals)/(tukeyVal*rmse))**2
        iw[iw>1] = 1
        w[ind] = (1-iw)**2

        if m is not None:
            diff = np.sqrt(np.mean((mNew-m)**2))/(1+np.sqrt(np.mean(mNew**2)))
            print('Relative change model norm',diff)
        m = mNew
        iteration += 1

    # Add dayglow model and corrected image to the Dataset
    if transform=='log':
        imgs['dgmodel'] = (['date','row','col'],(np.exp(dm)).reshape((n_t,len(imgs.row),len(imgs.col))))
    else:
        imgs['dgmodel'] = (['date','row','col'],(dm).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['dgimg'] = imgs[inImg]-imgs['dgmodel']
    imgs['dgweight'] = (['date','row','col'],(w).reshape((n_t,len(imgs.row),len(imgs.col))))

    # Remove pixels outside model scope
    ind = (imgs.sza>=0)& (imgs.dza <= dzalim) & (imgs.mlat >= minlat) & imgs.bad
    imgs['dgmodel'] = xr.where(~ind,np.nan,imgs['dgmodel'])
    imgs['dgimg'] = xr.where(~ind,np.nan,imgs['dgimg'])
    imgs['dgweight'] = xr.where(~ind,np.nan,imgs['dgweight'])

    # Add attributes
    imgs['dgmodel'].attrs = {'long_name': 'Dayglow model'}
    imgs['dgimg'].attrs = {'long_name': 'Dayglow corrected image'}
    imgs['dgweight'].attrs = {'long_name': 'Dayglow model weights'}

    return imgs

def showDGmodel(img,minlat=0,pathOutput=None,**kwargs):
    '''
    Show the dayglow model for a timestep.
    Ex: showFUVdayglowModel(wic.isel(date=15))
    Parameters
    ----------
    img : xarray.Dataset
        Dataset with the FUV image.
    pathOutput : str, optional
        Path to save the figure. The default is None.
    **kwargs
    Returns
    -------
    None.
    '''

    sza = img['sza'].values.flatten()
    dza = img['dza'].values.flatten()
    image = img['img'].values.flatten()
    mlat = img['mlat'].values.flatten()
    mlt = img['mlt'].values.flatten()

    # Set functional form
    if img['id'] in ['WIC','SI12','SI13','UVI']:
        fraction = np.cos(np.deg2rad(sza))/np.cos(np.deg2rad(dza))
    elif img['id'] == 'VIS':
        fraction = np.cos(np.deg2rad(sza))

    # Figure
    fig = plt.figure(facecolor = 'white',figsize=(12,10))
    gs = gridspec.GridSpec(2,3)

    # Scatter plot of all pixels and the model fit
    ax1 = plt.subplot(gs[0,:])
    ax1.scatter(fraction[np.isfinite(mlt)],image[np.isfinite(mlt)],s=1,edgecolors='none',color='0.8')

    ax1.scatter(fraction[mlat >= minlat],img['dgmodel'].values.flatten()[mlat >= minlat],s=1,color = 'black')

    if img['id'] in ['WIC','SI13','UVI']:
        ax1.set_xlim([-3,3])
        ax1.set_xlabel(r'$\cos \theta / \cos \Phi$')
    elif img['id'] == 'VIS':
        ax1.set_xlim([-1,1])
        ax1.set_xlabel(r'$\cos $')
    ax1.set_ylabel('Counts')

    if img['id'] == 'WIC':
        cmax1 = 9000.
        cmax2 = 1000.
    elif img['id'] == 'VIS':
        cmax1 = 60.
        cmax2 = 45.
    elif img['id'] == 'SI13':
        cmax1 = 60.
        cmax2 = 30.
    elif img['id'] == 'UVI':
        cmax1 = 150.
        cmax2 = 100.

    # Polar plot of input image
    gs2 = gridspec.GridSpecFromSubplotSpec(2,1,height_ratios=[90,10],subplot_spec=gs[1,0])
    ax2 = plt.subplot(gs2[0])
    ax2c = plt.subplot(gs2[1])
    pax2 = pp(ax2)
    plotimg(img,'img',pax=pax2,crange=[0,cmax1],**kwargs)
    ax2c.axis('off')
    plt.colorbar(pax2.ax.collections[0],orientation='horizontal',ax=ax2c,fraction=1,
                 extend='max')

    # Polar plot of the dayglow model
    gs3 = gridspec.GridSpecFromSubplotSpec(2,1,height_ratios=[90,10],subplot_spec=gs[1,1])
    ax3 = plt.subplot(gs3[0])
    ax3c = plt.subplot(gs3[1])
    pax3 = pp(ax3)
    plotimg(img,'dgmodel',pax=pax3,crange=[0,cmax1],**kwargs)
    ax3c.axis('off')
    plt.colorbar(pax3.ax.collections[0],orientation='horizontal',ax=ax3c,fraction=1,
                 extend='max')

    # Polar plot of the dayglow-corrected image
    gs4 = gridspec.GridSpecFromSubplotSpec(2,1,height_ratios=[90,10],subplot_spec=gs[1,2])
    ax4 = plt.subplot(gs4[0])
    ax4c = plt.subplot(gs4[1])
    pax4 = pp(ax4)
    plotimg(img,'dgimg',pax=pax4,crange=[-cmax2,cmax2],cmap='seismic')
    ax4c.axis('off')
    plt.colorbar(pax4.ax.collections[0],orientation='horizontal',ax=ax4c,fraction=1,
                 extend='both')

    ax1.set_title(img['id'].values.tolist() + ': ' + img['date'].dt.strftime('%Y-%m-%d %H:%M:%S').values.tolist())
    ax2.set_title('Original image')
    ax3.set_title('Dayglow')
    ax4.set_title('Corrected image')

    gs.tight_layout(fig)
    if pathOutput is not None:
        filename = img['date'].dt.strftime('%Y%m%d%H%M%S').values.tolist()
        plt.savefig(pathOutput + str(img['id'].values)+filename + '_dayglowModel.png', bbox_inches='tight', dpi = 100)
        plt.clf()
        plt.close(fig)
    #plt.show()

def plotimg(img,inImg='img',pax=None,crange = None, bgcolor = None, **kwargs):
    '''
    Show a FUV image in polar coordinates coordinates.
    Wrapper to polplot's plotimg

    Parameters
    ----------
    img : xarray.Dataset
        Image dataset.
    inImg : str
        Name of the image to be show. Default is 'image'.
    pax : plt.axis object, optional
        pax to show the image. The default is None (new figure).
    crange : tuple/list, optional
        Color range, given as (lower,upper). The default is None (matplotlib default).
    bgcolor : str, optional
        Name of background color. Defaut is None (transparent).
    **kwargs :
        kwargs to Polycollection().
    Returns
    -------
    coll
        The displayed Polycollection.
    '''
    if pax is None:
        fig, ax = plt.subplots()
        pax = pp(ax)

    mlat = img['mlat'].values.copy()
    mlt = img['mlt'].values.copy()
    image = img[inImg].values.copy()
    coll = pax.plotimg(mlat,mlt,image,crange=crange,bgcolor=bgcolor,**kwargs)
    return coll

def pplot(imgs,inImg,col_wrap = None,tb=False,add_cbar = True,crange=None,robust=False,cbar_kwargs={},pp_kwargs={},**kwargs):
    '''
    Show images/data in polar coordinates coordinates.
    The images are by default plotted size-byside.
    The functionality resembles xarray 2d plot

    Parameters
    ----------
    imgs : xarray.Dataset
        Image dataset.
    inImg : str
        Name of the image/data to be show.
    col_wrap : int, optional
        If provided, the number of images before staring a new row. Ignored if tb=True.
    tb : bool, optional
        Plot images top-to-bottom. Default is False
    add_cbar : bool, optional
        Whether to include colorbar in the figure. Default is True
    crange : tuple, optional
        Minimum and maximum range of the colormap, given as (vmin,vmax).
        If not provided, crange is inferred from data and other keyword arguments (default).
    robust : bool, optional
        If True and crange is absent, the colormap range is
        computed with 2nd and 98th percentiles instead of the extreme values.
    cbar_kwargs : dict, optional
        Dictionary of keyword arguments to pass to the colorbar
        (see :meth:`matplotlib:matplotlib.figure.Figure.colorbar`).
    pp_kwargs : dict, optional
        Dictionary of keyword arguments to pass to pp
    **kwargs :
        kwargs to Polycollection().
    Returns
    -------
    None.
    '''
    n_imgs = len(imgs.date)

    # Set minlat
    if 'minlat' in pp_kwargs:
        minlat = pp_kwargs['minlat']
        pp_kwargs.pop('minlat')
    else:
        minlat = 50

    # Set crange if not given
    if crange is None and robust:
        crange=np.quantile(imgs[inImg].values[imgs['mlat'].values>minlat],[0.02,0.98])
    else:
        crange=np.quantile(imgs[inImg].values[imgs['mlat'].values>minlat],[0,1])

    # Set cbar orientation
    if 'orientation' in cbar_kwargs:
        cbar_orientation = cbar_kwargs['orientation']
        cbar_kwargs.pop('orientation')
    else:
        cbar_orientation = 'vertical'

    # Set up fig size and axes
    cb_size = 0.5
    if tb:
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
        pax.ax.set_title(imgs['id'].values.tolist() + ': ' +
             imgs.isel(date=i)['date'].dt.strftime('%Y-%m-%d %H:%M:%S').values.tolist())

    if add_cbar:
        cax = plt.subplot(gs0[1])
        cax.axis('off')
        cbar = plt.colorbar(pax.ax.collections[0],orientation=cbar_orientation,ax=cax,fraction=1,**cbar_kwargs)

        # cbar name
        if len(imgs[inImg].attrs)==2:
            cbar.set_label('{} ({})'.format(imgs[inImg].attrs['long_name'],imgs[inImg].attrs['units']))
        elif len(imgs[inImg].attrs)==1:
            cbar.set_label('{}'.format(imgs[inImg].attrs['long_name']))
        else:
            cbar.set_label(inImg)

def plotimgProj(img,inImg='img',ax=None,mltLeft=18.,mltRight=6, minlat=50,maxlat=80,crange=None, **kwargs):
    '''
    Show a section of a FUV image projected in mlt-mlat coordinates
    Ex: plotimgProj(.isel(date=15),'img',crange=(0,1500),cmap='plasma')
    Parameters
    ----------
    img : xarray.Dataset
        Image dataset.
    inImg : str
        Name of the image to be show. Default is 'image'.
    ax : plt.axis object, optional
        ax to show the image. The default is None (new figure).
    mltLeft : float, optional
        Left ("lower") MLT limit. The default is 18.
    mltRight : float, optional
        Right ("Upper") MLT limit. The default is 6.
    minlat : float, optional
        Minimum latitude. The default is 50.
    maxlat : float, optional
        Maximum latitude. The default is 80.
    crange : tuple/list, optional
        Color range, given as (lower,upper). The default is None (matplotlib default).
    **kwargs :
        kwargs to Polycollection().
    Returns
    -------
    coll
        The displayed Polycollection
    '''

    if ax is None: fig, ax = plt.subplots()

    mlat = img['mlat'].values.copy()
    mlt =img['mlt'].values.copy()
    mltRight=mltRight+0.2
    mlt[mlt>mltRight]=mlt[mlt>mltRight]-24

    if mltLeft>mltRight: mltLeft=mltLeft-24

    ll = mlt[1:,  :-1].flatten(), mlat[1:,  :-1].flatten()
    lr = mlt[1:,   1:].flatten(), mlat[1:,   1:].flatten()
    ul = mlt[:-1, :-1].flatten(), mlat[:-1, :-1].flatten()
    ur = mlt[:-1,  1:].flatten(), mlat[:-1,  1:].flatten()

    vertsx = np.vstack((ll[0], lr[0], ur[0], ul[0])).T
    vertsy = np.vstack((ll[1], lr[1], ur[1], ul[1])).T

    ii_neg = (vertsx<-5).any(axis=1)
    ii_pos = (vertsx>5).any(axis=1)
    iii = np.where(vertsx >= mltLeft, True, False)&np.where(vertsx <= mltRight, True, False)&np.where(vertsy >= minlat, True, False)&np.where(vertsy <= maxlat, True, False)
    iii = (np.any(iii, axis = 1)&(ii_neg&ii_pos==False)).nonzero()[0]

    vertsx = vertsx[iii]
    vertsy = vertsy[iii]

    verts = np.dstack((vertsx, vertsy))

    if 'cmap' in kwargs.keys():
        cmap = kwargs['cmap']
        kwargs.pop('cmap')
    else:
        cmap = plt.cm.viridis

    coll = PolyCollection(verts, array=img[inImg].values[1:, :-1].flatten()[iii], cmap = cmap, edgecolors='none', **kwargs)
    if crange is not None: coll.set_clim(crange[0], crange[1])
    ax.add_collection(coll)

    mltRight = mltRight - 0.2
    ax.set_xlim([mltLeft,mltRight])
    ax.set_ylim([minlat,maxlat])
    xlabel = np.linspace(mltLeft,mltRight,len(ax.get_xticklabels()))
    xlabel[xlabel<0] = xlabel[xlabel<0]+24
    if (xlabel[xlabel!=0]%xlabel[xlabel!=0].astype(int)==0).all():
        ax.set_xticklabels(xlabel.astype(int))
    else:
        ax.set_xticklabels(xlabel)

    return coll

def ppBoundaries(ds,boundary='ocb',pax=None):
    '''
    Polar plot showing the temporal evolution of the OCB.
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with boundaries identified.
        The coordinates must be mlt and date
    boundary : str, optional
        Name of the boundary in ds. Default is 'ocb'
    Returns
    -------
    None
    '''
    if pax is None:
        fig,ax = plt.subplots()
        pax = pp(ax)

    # Add first row to end for plotting
    ds = xr.concat([ds,ds.isel(mlt=0)],dim='mlt')

    # Date
    date = ds.date
    n_d = len(date)
    dateStr = ds.date.dt.strftime('%H:%M:%S').values
    cmap = plt.get_cmap('jet',n_d)

    for d in range(n_d):
        pax.plot(ds.isel(date=d)[boundary].values,ds.mlt.values,linestyle='-',color=cmap(d))


    norm = mcolors.Normalize(vmin=0,vmax=n_d)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm,ticks=np.arange(0,n_d,1),boundaries=np.arange(-0.5,n_d,1),ax=pax.ax)
    cbar.ax.tick_params(size=0)
    cbarlabel = n_d*['']
    cbarlabel[::n_d//10+1] = dateStr[::n_d//10+1]


    cbar.set_ticklabels(cbarlabel)
    plt.tight_layout()

# def calcFluxCS(ds,height=130,R=6371.2):
#     '''
#     Function to estimate the amount of open flux inside the given boundaries.
#     Uses CubedSphere-grid to do areas. 
    
#     Parameters
#     ----------
#     ds : xarray.Dataset
#         Dataset with the boundaries.
#         Coordinates must be 'date' and 'mlt'
#         Data variable must be 'ocb' and 'eqb'
#     height : float, optional
#         Assumed height of the given boundary
#     R : float, optional
#         Radius of Earth
#     Returns
#     -------
#     ds : xarray.Dataset
#         Copy(?) of the Dataset with calculated flux
#     '''

#     # Date for IGRF values
#     dateIGRF = ds.date[len(ds.date)//2].values

#     # Set up projection:
#     position = (-72, 80) # lon, lat
#     orientation = (0, 1) # east, north
#     projection = lompe.cs.CSprojection(position, orientation)

#     # Set up grid:
#     L, W, Lres, Wres= 1e20, 1e20, 50.e3, 50.e3 # dimensions and resolution of grid
#     csgrid  = lompe.cs.CSgrid(projection, L, W, Lres, Wres, R = (R+height)*1e3)

#     # Radial magnetic flux in grid
#     dateIGRF = ds.date[len(ds.date)//2].values # Date for IGRF values
#     Br, Btheta, Bphi = lompe.ppigrf.igrf_gc(R+height, 90-csgrid.lat, csgrid.lon, dateIGRF)
#     Br = -Br.squeeze()
#     dflux = csgrid.A*Br*1e-9

#     oFlux=[]
#     aFlux=[]
#     for t in range(len(ds.date)):
#         date = pd.to_datetime(ds.date[t].values)
#         mlt = ds.mlt.values

#         # OCB coordinates
#         ocb = ds.isel(date=t)['ocb'].values

#         # Convert boundary to geo
#         A = apexpy.Apex(date)
#         gdlat,glon = A.convert(ocb, mlt, 'mlt', 'geo', height=height,datetime=date)
#         glat = 90 - lompe.ppigrf.ppigrf.geod2geoc(gdlat,height,0,0)[0]

#         # Create an OCB polygon
#         x,y=projection.geo2cube(glon, glat)
#         poly = path.Path(np.stack((x,y),axis=1))

#         # Identify gridcell with center inside the OCB polygon
#         inocb = poly.contains_points(np.stack((csgrid.xi.flatten(),csgrid.eta.flatten()),axis=1))

#         # Summarize
#         if np.isnan(x).any(): # Boundary exceeds the grid
#             oFlux.append(np.nan)
#         else:
#             oFlux.append(np.sum(dflux.flatten()[inocb])*1e-6)

#         # EQB coordinates
#         eqb = ds.isel(date=t)['eqb'].values

#         # Convert boundary to geo
#         gdlat,glon = A.convert(eqb, mlt, 'mlt', 'geo', height=height,datetime=date)
#         glat = 90 - lompe.ppigrf.ppigrf.geod2geoc(gdlat,height,0,0)[0]

#         # Create an OCB polygon
#         x,y=projection.geo2cube(glon, glat)
#         poly = path.Path(np.stack((x,y),axis=1))

#         # Identify gridcell with center inside the OCB polygon
#         inocb = poly.contains_points(np.stack((csgrid.xi.flatten(),csgrid.eta.flatten()),axis=1))

#         # Summarize
#         if np.isnan(x).any(): # Boundary exceeds the grid
#             aFlux.append(np.nan)
#         else:
#             aFlux.append(np.sum(dflux.flatten()[inocb])*1e-6)

#     ds=ds.assign({'openFlux':('date',np.array(oFlux)),'auroralFlux':('date',np.array(aFlux)-np.array(oFlux))})
#     return ds

def calcFlux(ds,height=130):
    '''
    Function to estimate the amount of open flux inside the given boundaries.
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with the boundaries.
        Coordinates must be 'date' and 'mlt'
        Data variable must be 'ocb' and 'eqb'
    height : float, optional
        Assumed height of the given boundary
    Returns
    -------
    ds : xarray.Dataset
        Copy(?) of the Dataset with calculated flux
    '''
    
    
    # Calc km/gdlat
    lat = np.arange(29.5,90,1)
    theta,R = ppigrf.ppigrf.geod2geoc(lat,height,0,0)[0:2]
    km_per_lat = np.sqrt(np.diff(R*np.cos(np.deg2rad(theta)))**2+np.diff(R*np.sin(np.deg2rad(theta)))**2) 
    
    # Grid in geodetic
    grid_lat,grid_lon = np.meshgrid(np.arange(30,90,1),np.arange(0,360,1))
        
    # Upward magnetic flux in grid
    dateIGRF = ds.date[len(ds.date)//2].values # Date for IGRF values
    Be, Bn, Bu = ppigrf.igrf(grid_lon,grid_lat,height,dateIGRF)
    Bu = abs(Bu.squeeze())
    
    # Transform grid to cartesian
    theta = np.cumsum(km_per_lat[::-1])[::-1]
    grid_x = theta[None,:]*np.cos(np.deg2rad(grid_lon))
    grid_y = theta[None,:]*np.sin(np.deg2rad(grid_lon))
    
    xi,yi = np.meshgrid(np.arange(-10000,10001,25),np.arange(-10000,10001,25))
    xi=xi.flatten()
    yi=yi.flatten()
    Bui = griddata((grid_x.flatten(),grid_y.flatten()),Bu.flatten(),(xi,yi))
    dflux = (1e-9*Bui)*(25e3)**2

    oFlux=[]
    aFlux=[]
    for t in range(len(ds.date)):
        date = pd.to_datetime(ds.date[t].values)
        mlt = ds.mlt.values

        # OCB coordinates
        ocb = ds.isel(date=t)['ocb'].values

        # Convert boundary to geo
        A = apexpy.Apex(date)
        gdlat,glon = A.convert(ocb, mlt, 'mlt', 'geo', height=height,datetime=date)

        # Create an OCB polygon
        x,y=np.interp(gdlat,np.arange(30,90,1),theta)*np.cos(np.deg2rad(glon)),np.interp(gdlat,np.arange(30,90,1),theta)*np.sin(np.deg2rad(glon))
        poly = path.Path(np.stack((x,y),axis=1))

        # Identify gridcell with center inside the OCB polygon
        inocb = poly.contains_points(np.stack((xi,yi),axis=1))

        # Summarize
        if np.isnan(x).any(): # Boundary exceeds the grid
            oFlux.append(np.nan)
        else:
            oFlux.append(np.sum(dflux.flatten()[inocb])*1e-6)

        # EQB coordinates
        eqb = ds.isel(date=t)['eqb'].values

        # Convert boundary to geo
        gdlat,glon = A.convert(eqb, mlt, 'mlt', 'geo', height=height,datetime=date)

        # Create an OCB polygon
        x,y=np.interp(gdlat,np.arange(30,90,1),theta)*np.cos(np.deg2rad(glon)),np.interp(gdlat,np.arange(30,90,1),theta)*np.sin(np.deg2rad(glon))
        poly = path.Path(np.stack((x,y),axis=1))

        # Identify gridcell with center inside the OCB polygon
        inocb = poly.contains_points(np.stack((xi,yi),axis=1))
        
        # Summarize
        if np.isnan(x).any(): # Boundary exceeds the grid
            aFlux.append(np.nan)
        else:
            aFlux.append(np.sum(dflux.flatten()[inocb])*1e-6)

    ds=ds.assign({'openFlux':('date',np.array(oFlux)),'auroralFlux':('date',np.array(aFlux)-np.array(oFlux))})
    ds['openFlux'].attrs = {'long_name':'Open flux','unit':'MWb'}
    ds['auroralFlux'].attrs = {'long_name':'Auroral flux','unit':'MWb'}
    return ds

def makeSHmodel(imgs,Nsh,Msh,order=2,dampingVal=0,tukeyVal=5,stop=1e-3,knotSep=None):
    '''
    Function to model the FUV residual background and subtract it from the input image

    Parameters
    ----------
    imgs : xarray.Dataset
        Dataset with the FUV images
    Nsh : int
        Order of the SH
    Msh : int
        Degree of the SH
    order: int, optional
        Order of the temporal spline fit. The default is 2.
    dampingVal : TYPE, optional
        Damping to reduce the influence of the time-dependent part of the model.
        The default is 0 (no damping).
    tukeyVal : float, optional
        Determines to what degree outliers are down-weighted.
        Iterative reweights is (1-(residuals/(tukeyVal*rmse))^2)^2
        Larger tukeyVal means less down-weight
        Default is 5
    stop : float, optional
        When to stop the iteration. The default is 0.001.
    knotSep : int, optional
        Approximate separation of temporal knots in minutes. The default is None (only knots at endpoints)

    Returns
    -------
    imgs : xarray.Dataset
        A copy(?) of the image Dataset with two new fields:
            - imgs['shmodel'] is the dayglow model
            - imgs['shimg'] is the dayglow-corrected image (dayglow subtracked from the input image)
            - imgs['shweight'] is the weight if each pixel after the final iteration
    '''
    date = imgs['date'].values
    time=(date-date[0])/ np.timedelta64(1, 'm')

    glat = imgs['glat'].stack(z=('row','col')).values
    glon = imgs['glon'].stack(z=('row','col')).values
    d = imgs['dgimg'].stack(z=('row','col')).values
    dg = imgs['dgmodel'].stack(z=('row','col')).values
    wdg = imgs['dgweight'].stack(z=('row','col')).values

    # Treat dg as variance
    d = d/dg

    sslat, sslon = map(np.ravel, subsol(date))
    phi = np.deg2rad((glon - sslon[:,None] + 180) % 360 - 180)

    n_t = glat.shape[0]
    n_s = glat.shape[1]

    # Temporal knots
    if knotSep==None:
        knots = np.linspace(time[0], time[-1], 2)
    else:
        knots = np.linspace(time[0], time[-1], int(np.round(time[-1]/knotSep)+1))
    knots = np.r_[np.repeat(knots[0],order),knots, np.repeat(knots[-1],order)]

    # Number of control points
    n_cp = len(knots)-order-1

    # Temporal design matix
    M = BSpline(knots, np.eye(n_cp), order)(time)

    # Iterative (few iterations)
    skeys = sh.SHkeys(Nsh, Msh).Mge(1).MleN().setNmin(1)
    ckeys = sh.SHkeys(Nsh, Msh).MleN().setNmin(1)

    print('Building sh G matrix')
    G_g=[]
    G_s=[]
    for i in range(n_t):
        # calculate Legendre functions at glat:
        P, dP = sh.get_legendre(Nsh, Msh, 90 - glat[i,:])
        Pc = np.hstack([P[key] for key in ckeys])
        Ps = np.hstack([P[key] for key in skeys])

        Gcos = Pc * np.cos(phi[i,:].reshape((-1, 1)) * ckeys.m)
        Gsin = Ps * np.sin(phi[i,:].reshape((-1, 1)) * skeys.m)

        G = np.hstack((Gcos, Gsin))
        G = G/dg[i,:][:,None]
        n_G = np.shape(G)[1]

        G_t = np.zeros((n_s, n_G*n_cp))

        for j in range(n_cp):
            G_t[:, np.arange(j, n_G*n_cp, n_cp)] = G*M[i, j]

        G_g.append(G)
        G_s.append(G_t)

    G_s = np.array(G_s)
    G_s = G_s.reshape(-1,G_s.shape[2])

    # Data
    ind = (np.isfinite(d))&(glat>0)&(imgs['bad'].values.flatten())[None,:]
    d_s = d.flatten()
    ind = ind.flatten()
    w = wdg.flatten() # Weights from dayglow model

    # Damping
    damping = dampingVal*np.ones(G_s.shape[1])
    R = np.diag(damping)

    diff = 1e10*stop
    iteration = 0
    m = None
    while (diff>stop)&(iteration < 100):
        print('Iteration:',iteration)
        # Solve for spline amplitudes
        m_s = np.linalg.lstsq((G_s[ind,:]*w[ind,None]).T@(G_s[ind,:]*w[ind,None])+R,(G_s[ind,:]*w[ind,None]).T@(d_s[ind]*w[ind]),rcond=None)[0]

        # Retrieve B-spline smooth model paramters (coarse)
        mNew    = M@m_s.reshape((n_G, n_cp)).T
        dm=[]
        for i, tt in enumerate(time):
            dm.append(G_g[i]@mNew[i, :])

        dm=np.array(dm).squeeze()
        residuals = dm.flatten()[ind] - d.flatten()[ind]
        rmse = np.sqrt(np.average(residuals**2,weights=w[ind]))

        iw = ((residuals)/(tukeyVal*rmse))**2
        iw[iw>1] = 1
        w[ind] = (1-iw)**2

        if m is not None:
            diff = np.sqrt(np.mean( (mNew-m)**2))/(1+np.sqrt(np.mean(mNew**2)))
            print('Relative change model norm:',diff)
        m = mNew
        iteration += 1


    imgs['shmodel'] = (['date','row','col'],(dm*dg).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['shimg'] = imgs['dgimg']-imgs['shmodel']
    imgs['shweight'] = (['date','row','col'],(w).reshape((n_t,len(imgs.row),len(imgs.col))))

    # Add attributes
    imgs['shmodel'].attrs = {'long_name': 'Spherical harmonics model'}
    imgs['shimg'].attrs = {'long_name': 'Spherical harmonics corrected image'}
    imgs['shweight'].attrs = {'long_name': 'Spherical harmonics model weights'}

    return imgs

def findBoundaries(imgs,inImg='shimg',mltRes=24,limFactors=None,order=3,dampingVal=0):
    '''
    A simple function to identify the ocb of a FUV image.
    NEEDS CLEANING + COMMENTS

    Parameters
    ----------
    imgs : xarray.Dataset
        Dataset with the FUV images.
    countlim : float
        countlimit when identifying the ocbs. All values lower than countlim are set to zero.
    inImg : str, optional
        Name of the image to use in the determination. The default is 'shimage'.
    mltRes : int, optional
        Number of MLT bins. Default is 24 (1-wide sectors).
    limFactors : array_like, optional
        Array containing fractions to multipy with noise used as threshold.
        Default is None, (np.linspace(0.5,1.5,5) is used).
    order : int, optional
        Order of the B-spline fitting the intensity profile in each circular sector.
        Default is 3.
    damplingVal : float, optional
        Damping value (Tikhonov regularization) in the B-spline fit.
        Default is 0.

    Returns
    -------
    xarray.Dataset
        A Dataset containing the identified boundaries.
    '''

    # Expand date dimension if missing
    if len(imgs.sizes)==2: imgs = imgs.expand_dims('date')

    if limFactors is None: limFactors = np.linspace(0.5,1.5,5)
    edges = np.linspace(0,24,mltRes+1)

    # Circle in which all boundaries are assumed to be located
    colatMax = np.concatenate((np.linspace(40,30,mltRes+1)[1:-1:2],np.linspace(30,40,mltRes+1)[1:-1:2]))


    colatAll = 90-abs(imgs['mlat'].values.copy().flatten())
    dAll = imgs[inImg].values.copy().flatten()
    wdgAll = imgs['shweight'].values.copy().flatten()
    jjj = (np.isfinite(dAll))&(colatAll<40)
    av = np.average(dAll[jjj],weights=wdgAll[jjj])
    mae = np.average(abs(dAll[jjj]),weights=wdgAll[jjj])

    blistTime  = []
    for t in range(len(imgs['date'])):
        print('Image:',t)
        colat = 90-abs(imgs.isel(date=t)['mlat'].values.copy().flatten())
        mlt = imgs.isel(date=t)['mlt'].values.copy().flatten()
        d = imgs.isel(date=t)[inImg].values.copy().flatten()
        wDG = 0*imgs.isel(date=t)['shweight'].values.copy().flatten()+1

        blistSec=[]
        for s in range(len(edges)-1):
            ocb=[]
            eqb=[]

            colatSec = colat[(mlt>edges[s])&(mlt<edges[s+1])&(colat<40)]
            dSec = d[(mlt>edges[s])&(mlt<edges[s+1])&(colat<40)]
            wSec = wDG[(mlt>edges[s])&(mlt<edges[s+1])&(colat<40)]



            if np.nansum(wSec[colatSec<colatMax[s]])==0:
                avSec=av
            else:
                jjj = (colatSec<colatMax[s])&(np.isfinite(dSec))
                avSec = np.average(dSec[jjj],weights=wSec[jjj])

            iii = np.isfinite(colatSec)&np.isfinite(dSec)
            knots = np.linspace(0,40,21).tolist()
            knots = np.r_[np.repeat(knots[0],order),knots, np.repeat(knots[-1],order)]

            if colatSec[iii].size == 0:
                ocb.extend(len(limFactors)*[np.nan])
                eqb.extend(len(limFactors)*[np.nan])
            else:
                # Prepare the model

                # Number of control points
                n_cp = len(knots)-order-1

                # Temporal design matix
                G = BSpline(knots, np.eye(n_cp), order)(colatSec[iii])

                damping = dampingVal*np.ones(G.shape[1])
                RR = np.diag(damping)

                # Iterative estimation of model parameters
                diff = 10000

                w = 1-wSec[iii]
                m = None
                stop=1000
                while diff > stop:
                    Gw = G*w[:, np.newaxis]
                    GTG = Gw.T.dot(Gw)
                    GTd = Gw.T.dot((w*dSec[iii])[:, np.newaxis])
                    mNew = lstsq(GTG+RR, GTd)[0]
                    residuals = G.dot(mNew).flatten() - dSec[iii]
                    rmse = np.sqrt(np.average(residuals**2))
                    weights = 1.*rmse/np.abs(residuals)
                    weights[weights > 1] = 1.
                    w = weights
                    if m is not None:
                        diff = np.sqrt(np.mean((m - mNew)**2))/(1+np.sqrt(np.mean(mNew**2)))

                    m = mNew

                ev = np.linspace(0,40,401)
                G = BSpline(knots, np.eye(n_cp), order)(ev)
                dmSec = G.dot(m).flatten()
                dmSec[ev>np.max(colatSec[iii])]=np.nan

                ## identify main peak
                ind = (ev<colatMax[s])&(ev<np.max(colatSec[iii]))


                for l in range(len(limFactors)):
                    isAbove = dmSec[ind]>avSec+limFactors[l]*mae
                    if (isAbove==True).all()|(isAbove==False).all(): #All above or below
                        ocb.append(np.nan)
                        eqb.append(np.nan)
                    else:
                        isAbove2 = np.concatenate(([False],isAbove,[False]))

                        firstAbove = np.argwhere(np.diff(isAbove2.astype(int))== 1).flatten()
                        firstBelow = np.argwhere(np.diff(isAbove2.astype(int))==-1).flatten()

                        if firstAbove[0]==0:
                            ind1=1
                        else:
                            ind1=0

                        if firstBelow[-1]==len(isAbove):
                            ind2=-1
                        else:
                            ind2=len(firstBelow)
                        #     firstAbove=firstAbove[:-1]
                        #     firstBelow=firstBelow[:-1]
                        firstAbove=(firstAbove[ind1:ind2])
                        firstBelow=(firstBelow[ind1:ind2])

                        areaSec=[]
                        for k in range(len(firstAbove)):
                            areaSec.append(np.sum(dmSec[ind][firstAbove[k]:firstBelow[k]]))


                        if len(areaSec)==0:
                            ocb.append(np.nan)
                            eqb.append(np.nan)
                        elif ((s<6)|(s>=18))&(len(areaSec)>1):
                            k = np.argmax(areaSec)
                            if k>0:
                                ocb.append(ev[firstAbove[k-1]-1])
                                eqb.append(ev[firstBelow[k]])
                            else:
                                ocb.append(ev[firstAbove[k]-1])
                                eqb.append(ev[firstBelow[k+1]])
                        else:
                            k = np.argmax(areaSec)
                            ocb.append(ev[firstAbove[k]-1])
                            eqb.append(ev[firstBelow[k]])

            ds = xr.Dataset(
                data_vars=dict(
                    ocb=(['lim'], 90-np.array(ocb)),
                    eqb=(['lim'], 90-np.array(eqb)),
                    ),

                coords=dict(
                    lim = limFactors
                ),
                )
            ds = ds.expand_dims(date=[pd.to_datetime(imgs.date[t].values)])
            ds = ds.expand_dims(mlt=[s+0.5])
            blistSec.append(ds)

        blistTime.append(xr.concat(blistSec,dim='mlt'))

    ds = xr.concat(blistTime,dim='date')

    # Add attributes
    ds['mlt'].attrs = {'long_name': 'Magnetic local time','unit':'hrs'}
    ds['ocb'].attrs = {'long_name': 'Open-closed boundary','unit':'deg'}
    ds['eqb'].attrs = {'long_name': 'Equatorward boundary','unit':'deg'}
    return ds

def makeBoundaryModel(ds,stop=1e-3,dampingValE=2e0,dampingValP=2e1,n_termsE=2,n_termsP=5,order = 3,knotSep = 10):
    '''
    Function to make a spatiotemporal Fourier model of auroral boundaries.
    INCLUDE L-CURVE script

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with initial boundaries.
    stop : float, optional
        When to stop the iterations. Default is 0.001
    dampingValE : float, optional
        Damping value (Tikhonov regularization) for the equatorward boundary.
        Default is 2e0
    dampingValP : float, optional
        Damping value (Tikhonov regulatization) for the poleward boundary.
        Default is 2e1
    n_termsE : int, optional
        Number of radial Fourier terms equatorward boundary
    n_termsP : int, optional
        Number of radial Fourier terms poleward boundary
    order : int, optional
        Order of the temporal B-spline. Default is 3
    knotSep : int, optional
        Approximate knot separation in minutes. Default is 10.

    Returns
    -------
    xarray.Dataset
        Dataset with model boundaries.

    '''

    time=(ds.date-ds.date[0]).values/ np.timedelta64(1, 'm')
    mlt = np.tile(ds.mlt.values,len(ds.lim))
    n_t = len(ds.date)
    n_mlt = len(mlt)

    #%% Eq boundary model
    theta_eb = np.deg2rad(90-ds['eqb'].stack(z=('lim','mlt')).values)

    # Temporal knots
    if knotSep==None:
        knots = np.linspace(time[0], time[-1], 2)
    else:
        knots = np.linspace(time[0], time[-1], int(np.round(time[-1]/knotSep)+1))
    knots = np.r_[np.repeat(knots[0],order),knots, np.repeat(knots[-1],order)]

    # Number of control points
    n_cp = len(knots)-order-1

    # Temporal design matix
    M = BSpline(knots, np.eye(n_cp), order)(time)

    # Data kernel
    G=[]
    for i in range(n_mlt):
        phi = np.deg2rad(15*mlt[i])
        terms=[1]
        for tt in range(1,n_termsE): terms.extend([np.cos(tt*phi),np.sin(tt*phi)])
        G.append(terms)
    G = np.array(G)
    n_G = np.shape(G)[1]
    for i in range(n_t):
        G_t = np.zeros((n_mlt, n_G*n_cp))
        for j in range(n_cp):
            G_t[:, np.arange(j, n_G*n_cp, n_cp)] = G*M[i, j]

        if i == 0:
            G_s = G_t
        else:
            G_s = np.vstack((G_s, G_t))

    # Data
    d_s = np.log(theta_eb.flatten())
    ind = np.isfinite(d_s)

    # Temporal Damping
    damping = dampingValE*np.ones(G_s.shape[1])
    damping[:3*n_cp]=10*damping[:3*n_cp]

    R = np.diag(damping)

    # Iteratively estimation of model parameters
    diff = 10000
    w = np.ones(d_s[ind].shape)
    m = None
    iteration = 0
    while (diff > stop)&(iteration<100):
        print('Iteration:',iteration)
        ms = np.linalg.inv((G_s[ind,:]*w[:,None]).T@(G_s[ind,:]*w[:,None])+R)@(G_s[ind,:]*w[:,None]).T@(d_s[ind]*w)
        ms = ms.reshape((n_G, n_cp)).T

        mNew    = M@ms
        mtau=[]
        for i, tt in enumerate(time):
            mtau.append(G@mNew[i, :])
        mtau=np.array(mtau).squeeze()

        residuals = mtau.flatten()[ind] - d_s[ind]
        rmse = np.sqrt(np.average(residuals**2,weights=w))

        # Change to Tukey?
        weights = 1.5*rmse/np.abs(residuals)
        weights[weights > 1] = 1.
        w = weights
        if m is not None:
            diff = np.sqrt(np.mean((mNew - m)**2))/(1+np.sqrt(np.mean(mNew**2)))
            print('Relative change model norm', diff)

        m = mNew
        iteration += 1

    # Data kernel evaluation
    mlt_eval      = np.linspace(0,24,24*10+1)
    G=[]
    for i in range(len(mlt_eval)):
        phi = np.deg2rad(15*mlt_eval[i])
        terms=[1]
        for tt in range(1,n_termsE): terms.extend([np.cos(tt*phi),np.sin(tt*phi)])
        G.append(terms)
    G = np.array(G)
    n_G = np.shape(G)[1]

    # Model boundary in primed space
    tau1=[]
    for i, tt in enumerate(time):
        tau1.append(G@m[i, :])
    tau1=np.array(tau1).squeeze()

    # Transform to unprimed
    tau_eb = np.exp(tau1)

    ## DERIVATIVE

    # df/dt in primed space
    dt =(knots[order+1:-1]-knots[1:-order-1])
    dms_dt = (ms[1:,:] - ms[:-1,:]) * order / dt[:,None]
    knots_dt=knots[1:-1]
    n_cp_dt=len(knots_dt)-(order-1)-1

    M_dt = BSpline(knots_dt, np.eye(n_cp_dt), order-1)(time)
    dm_dt      = M_dt@dms_dt

    dtau1_dt=[]
    for i, tt in enumerate(time):
        dtau1_dt.append(G@dm_dt[i, :])
    dtau1_dt=np.array(dtau1_dt).squeeze()

    # df/d(phi) in primed space
    G_dphi=[]
    for i in range(len(mlt_eval)):
        phi = np.deg2rad(15*mlt_eval[i])
        terms=[0]
        for tt in range(1,n_termsE): terms.extend([-tt*np.sin(tt*phi),tt*np.cos(tt*phi)])
        G_dphi.append(terms)
    G_dphi = np.array(G_dphi)
    dtau1_dphi = []
    for i, tt in enumerate(time):
        dtau1_dphi.append(G_dphi@m[i, :])
    dtau1_dphi=np.array(dtau1_dphi).squeeze()

    # Transform derivatives to unprimed
    dtau_dt_eb = np.exp(tau1)*(dtau1_dt)
    dtau_dphi_eb = np.exp(tau1)*(dtau1_dphi)

    R_I = 6500e3
    dphi_dt = -( dtau_dt_eb*dtau_dphi_eb)/(np.sin(tau_eb)**2+(dtau_dphi_eb)**2)
    dtheta_dt = dtau_dt_eb*np.sin(tau_eb)**2/(np.sin(tau_eb)**2+(dtau_dphi_eb)**2)

    # Boundary velocity
    u_phi = R_I*np.sin(tau_eb)*dphi_dt/60
    u_theta = R_I*dtheta_dt/60


    #%% Poleward boundary model
    theta_pb = np.deg2rad(90 - ds['ocb'].stack(z=('lim','mlt')).values)

    theta_pb1 = theta_pb/np.exp(mtau)

    # Temporal design matix
    M = BSpline(knots, np.eye(n_cp), order)(time)

    # Data kernel
    G=[]
    for i in range(n_mlt):
        phi = np.deg2rad(15*mlt[i])
        terms=[1]
        for tt in range(1,n_termsP): terms.extend([np.cos(tt*phi),np.sin(tt*phi)])
        G.append(terms)
    G = np.array(G)
    n_G = np.shape(G)[1]
    for i in range(n_t):
        G_t = np.zeros((n_mlt, n_G*n_cp))

        for j in range(n_cp):
            G_t[:, np.arange(j, n_G*n_cp, n_cp)] = G*M[i, j]

        if i == 0:
            G_s = G_t
        else:
            G_s = np.vstack((G_s, G_t))

    # Data
    d_s = (np.log(theta_pb1)-np.log(1-theta_pb1)).flatten()
    ind = np.isfinite(d_s)

    # Temporal Damping
    damping = dampingValP*np.ones(G_s.shape[1])
    damping[:3*n_cp]=2*damping[:3*n_cp]
    R = np.diag(damping)

    # Iteratively estimation of model parameters
    diff = 10000
    w = np.ones(d_s[ind].shape)
    m = None
    iteration = 0
    while diff > stop:
        print('Iteration:',iteration)
        ms = np.linalg.inv((G_s[ind,:]*w[:,None]).T@(G_s[ind,:]*w[:,None])+R)@(G_s[ind,:]*w[:,None]).T@(d_s[ind]*w)
        ms = ms.reshape((n_G, n_cp)).T
        # Retrieve B-spline smooth model paramters (coarse)
        mNew    = M@ms

        mtau2=[]
        for i, tt in enumerate(time):
            mtau2.append(G@mNew[i, :])
        mtau2=np.array(mtau2).squeeze()

        residuals = mtau2.flatten()[ind] - d_s[ind]
        rmse = np.sqrt(np.average(residuals**2,weights=w))
        weights = 1.5*rmse/np.abs(residuals)
        weights[weights > 1] = 1.
        w = weights
        if m is not None:
            diff = np.sqrt(np.mean((mNew - m)**2))/(1+np.sqrt(np.mean(mNew**2)))
            print('Relative change model norm', diff)

        m = mNew
        iteration += 1

    # Data kernel evaluation
    G=[]
    for i in range(len(mlt_eval)):
        phi = np.deg2rad(15*mlt_eval[i])
        terms=[1]
        for tt in range(1,n_termsP): terms.extend([np.cos(tt*phi),np.sin(tt*phi)])
        G.append(terms)
    G = np.array(G)
    n_G = np.shape(G)[1]

    # Model boundary in double primed space
    tau2=[]
    for i, tt in enumerate(time):
        tau2.append(G@m[i, :])
    tau2=np.array(tau2).squeeze()

    # Transform to unprimed
    tau1 = 1/(1+np.exp(-1*tau2))
    tau_pb  = tau_eb*tau1


    # df/dt in double primed space
    dt =(knots[order+1:-1]-knots[1:-order-1])
    dms_dt = (ms[1:,:] - ms[:-1,:]) * order / dt[:,None]
    knots_dt=knots[1:-1]
    n_cp_dt=len(knots_dt)-(order-1)-1

    # Temporal design matix
    M = BSpline(knots_dt, np.eye(n_cp_dt), order-1)(time)
    dm_dt = M_dt@dms_dt

    dtau2_dt=[]
    for i, tt in enumerate(time):
        dtau2_dt.append(G@dm_dt[i, :])
    dtau2_dt=np.array(dtau2_dt).squeeze()

    # df/d(phi) in double primed coordinates
    G_dt=[]
    for i in range(len(mlt_eval)):
        phi = np.deg2rad(15*mlt_eval[i])
        terms=[0]
        for tt in range(1,n_termsP): terms.extend([-tt*np.sin(tt*phi),tt*np.cos(tt*phi)])
        G_dt.append(terms)
    G_dt = np.array(G_dt)
    dtau2_dphi = []
    for i, tt in enumerate(time):
        dtau2_dphi.append(G_dt@m[i, :])
    dtau2_dphi=np.array(dtau2_dphi).squeeze()

    # Transform derivatives to primed
    dtau1_dt = (np.exp(-tau2)/(np.exp(-tau2)+1)**2)*(dtau2_dt)
    dtau1_dphi = (np.exp(-tau2)/(np.exp(-tau2)+1)**2)*(dtau2_dphi)

    # Transform derivatives to unprimed
    dtau_dt = tau_eb*dtau1_dt  + dtau_dt_eb*tau1
    dtau_dphi = tau_eb*dtau1_dphi + dtau_dphi_eb*tau1

    # Transform from "wrapped" ionosphere to spherical ionosphere
    dphi_dt = -( dtau_dt*dtau_dphi)/(np.sin(tau_pb)**2+(dtau_dphi)**2)
    dtheta_dt = dtau_dt*np.sin(tau_pb)**2/(np.sin(tau_pb)**2+(dtau_dphi)**2)

    # Boundary velocity
    v_phi = R_I*np.sin(tau_pb)*dphi_dt/60
    v_theta = R_I*dtheta_dt/60


    # Make Dataset with modelled boundary locations and velocities
    ds2 = xr.Dataset(
    data_vars=dict(
        ocb=(['date','mlt'], 90-np.rad2deg(tau_pb)),
        eqb=(['date','mlt'], 90-np.rad2deg(tau_eb)),
        v_phi=(['date','mlt'], v_phi),
        v_theta=(['date','mlt'], v_theta),
        u_phi=(['date','mlt'], u_phi),
        u_theta=(['date','mlt'], u_theta),
        ),
    coords=dict(
        date = ds.date,
        mlt = mlt_eval
    ),
    )

    # Add attributes
    ds2['mlt'].attrs = {'long_name': 'Magnetic local time','unit':'hrs'}
    ds2['ocb'].attrs = {'long_name': 'Open-closed boundary','unit':'deg'}
    ds2['eqb'].attrs = {'long_name': 'Equatorward boundary','unit':'deg'}
    ds2['v_phi'].attrs = {'long_name': '$V_\\phi$','unit':'m/s'}
    ds2['v_theta'].attrs = {'long_name': '$V_\\theta$','unit':'m/s'}
    ds2['u_phi'].attrs = {'long_name': '$U_\\phi$','unit':'m/s'}
    ds2['u_theta'].attrs = {'long_name': '$U_\\theta$','unit':'m/s'}
    return ds2

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
from datetime import datetime

from scipy.io import idl
from scipy.interpolate import splev
from scipy.linalg import lstsq

try:
    import bspline
except:
    print("ERROR: bspline module could not be imported, makeFUVdayglowModel won't work")


# from pysymmetry.visualization.polarsubplot import Polarsubplot
# from pysymmetry.models.internalmagneticfield import igrf
# from pysymmetry.geodesy import geodeticheight2geocentricR
# from pysymmetry.visualization.grids import equal_area_grid
from pysymmetry.inverse import sh
from pysymmetry.sunlight import subsol


def readFUVimage(filenames, dzalim = 80, minlat = 0, hemisphere = None, reflatWIC=True):
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
    reflatWIC: bool, optional
        Reapply WIC's flatfield or keep original flatfield.
        Default is True (reapply).

    Returns
    -------
    xarray.Dataset with the following fields
    DATA:
    'image': Number of counts in each pixel
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
        img = img.expand_dims(date=[pd.to_datetime(_timestampFUVimage(imageinfo['time'][0]))])

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

    # Add a boolerian field to omit bad pixels in the detector
    imgs = _badPixelsFUVimage(imgs)

    # Reapply WIC's flat field
    if (imgs['id']=='WIC')&reflatWIC:
        imgs=_reflatWIC(imgs)

    return imgs

def _timestampFUVimage(timestamp):
    """ returns datetime object for timestamp = imageinfo['TIME'][0] """

    hh = int(timestamp[1]/1000/60/60)
    mm = int(timestamp[1]/1000/60) - 60 * hh
    ss = int(timestamp[1]/1000) - 60**2 * hh - 60 * mm

    hh = str(hh).zfill(2)
    mm = str(mm).zfill(2)
    ss = str(ss).zfill(2)

    hhmmss = hh + mm + ss

    # handle occasional errors in the VIS image structure:
    if len(str(timestamp[0])) != 7:
        timestamp[0] = int(str(timestamp[0])[:4] + str(timestamp[0])[4:].zfill(3))

    time = datetime.strptime(str(timestamp[0]) + hhmmss, '%Y%j%H%M%S')

    return time

def _badPixelsFUVimage(imgs):
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
        Copy(?) of the FUV dataset with a new field imgs['bad'], containing the
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
        ind[110:,:] = False # Broken boom shades the upper rows of WIC (after what date?)
        ind[:10,:] = False
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

    path = 'wic_flatfield_dbase.idl'
    flatfields = idl.readsav(path)['flatfields']
    if pd.to_datetime(wic['date'][0].values)<pd.to_datetime('2000-10-03 23:30'):
        flat = flatfields[:,0]
    else:
        flat = flatfields[:,1]

    background=np.nanmedian((wic[inImg]/flat[None,:,None]).values[wic['bad'].values&(wic['sza'].values>100|np.isnan(wic['sza'].values))])
    if np.isnan(background): background=450 # Set a reasonable value if no background pixels are available
    wic[outImg]=((wic[inImg].copy()/flat[None,:,None]-background)*flat[None,:,None]+background)
    return wic

def makeFUVdayglowModelC(imgs,inImg='img',outImg='dgimg',transform=None,model=('std',),order=3,dampingVal=0,stop=0.01,auroralZone=(0,0),minlat=0,dzalim=80,knots=None):
    '''
    Function to model the FUV dayglow and subtract it from the input image

    Parameters
    ----------
    imgs : xarray.Dataset
        Dataset with the FUV images, imported by readFUVimage()
    inImg : str, optional
        Name of the input image to be used in the model. The default is 'image'.
    outImg : str, optional
        Name of the dayglow-corrected image returned by the model. The default is 'cimage'.
    model : tuple, optional
        Functional form of the dayglow, given as a tuple.
        The first entry is the name, the following are model parameters.
        - ('std',) is cos(sza)/cos(dza)
        - ('power',c) is cos(sza)/cos(dza)^c
        - ('exp',c) is exp(c*(1-1/cos(dza)))*cos(sza)/cos(dza)
        The default is ('std',).
    order : int, optional
        Order of the spline fit. The default is 3.
    dampingVal : TYPE, optional
        Damping to reduce the influence of the time-dependent part of the model.
        The default is 0 (no damping).
    stop : float, optional
        When to stop the iteration. The default is 0.01.
    auroralZone : tuple, optional
        Pixels on the nightside (between 18 and 6 MLT) between (lower,upper) are ignored.
        The default is (60,80).
    minlat : float, optional
        Lower mlat boundary to include in the model. The default is 0.
    dzalim : float, optional
        Maximum viewing angle to include. The default is 80.
    knots : list, optional
        Location of the Bspline knots. The default is None (default is used).

    Returns
    -------
    imgs : xarray.Dataset
        A copy(?) of the image Dataset with two new fields:
            - imgs['dayglow'] is the dayglow model
            - imgs[outImg] is the dayglow-corrected image (dayglow subtracked from the input image)
        In addition, the parameters used to model the dayglow are stored
        in imgs.attrs['dayglowModel']

    '''

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

    # d[d<=0]=1

    # Select functional form
    if imgs['id'] in ['WIC','SI12','SI13','UVI']:
        if model[0] == 'power':
            c = model[1]
            fraction = np.cos(np.deg2rad(sza))/np.cos(np.deg2rad(dza))**c
            if knots is None: knots = [-5,-0.25,0,0.25,1,3,5]
        elif model[0] == 'exp':
            c = model[1]
            fraction = np.exp(c*(1. - 1/np.cos(np.deg2rad(dza))))/np.cos(np.deg2rad(dza))*np.cos(np.deg2rad(sza))
            if knots is None: knots = [-3,-0.2,0,0.2,1,2,3,4]
        else:
            fraction = np.cos(np.deg2rad(sza))/np.cos(np.deg2rad(dza))
            if knots is None: knots = [-5,-3,-1,-0.2,-0.1,0,0.1,0.2,1,3,5]

        knots = np.array([knots[0]]*order  + knots + [knots[-1]]*order )
        spline = bspline.Bspline(knots, order = order)
    elif imgs['id'] == 'VIS':
        fraction = np.cos(np.deg2rad(sza))
        if knots is None: knots= [-1,-0.2,-0.1,0,0.333,0.667,1]
        knots = np.array([knots[0]]*order  + knots + [knots[-1]]*order )
        spline = bspline.Bspline(knots, order = order)

    # Pixels to be included in the construction of the model
    iii = (sza >= 0) & (dza <= dzalim) & (mlat >= minlat) & (np.isfinite(d)) & remove
    d = d[iii]

    knots = spline.knot_vector
    bases = []
    for i in range(len(knots)):
        weights = np.zeros_like(knots)
        weights[i] = 1
        basis = splev(fraction[iii], (knots, weights, order))
        bases.append(basis)

    G = np.array(bases).T

    # Set the damping value
    damping = dampingVal*np.ones(G.shape[1])
    R = np.diag(damping)

    w = np.ones(sza.shape)
    w[(mlat <= auroralZone[0]) & (mlat >= auroralZone[1])]=0
    # First
    GTG = G.T.dot(G)
    GTd = G.T.dot((d)[:, np.newaxis])
    m = lstsq(GTG+R, GTd)[0]
    diff = np.linalg.norm(m)

    dm = G.dot(m).flatten()
    #rms = np.sqrt(np.mean((d-dm)**2))
    mae = np.average(abs(d-dm),weights=w[iii])

    iteration = 0

    while (diff > stop)&(iteration < 100):
        print(iteration)
        # w[iii] = 1.*mae/np.abs(d-dm)
        # w[w>1]=1

        iw = ((d-dm)/(8*mae))**2
        iw[iw>1] = 1
        w[iii] = (1-iw)**2

        Gw = G*w[iii, np.newaxis]
        GTG = Gw.T.dot(Gw)
        GTd = Gw.T.dot((w[iii]*d)[:, np.newaxis])

        newm = lstsq(GTG+R, GTd)[0]
        diff = np.linalg.norm(m-newm)

        m = newm

        dm = G.dot(m).flatten()
        #rms = np.sqrt(np.average((d-dm)**2,weights=w[iii]))
        mae = np.average(abs(d-dm),weights=w[iii])

        print(diff)

        iteration += 1


    # Loop through the images to add the dayglow model and the dayglow-corrected image
    dayglow = np.zeros(imgs[inImg].shape)
    cimage  = np.zeros(imgs[inImg].shape)
    wimage  = np.zeros(imgs[inImg].shape)
    w = w.reshape(imgs[inImg].shape)
    for t in range(len(imgs.date)):
        bases = []
        for j in range(len(knots)):
            weights = np.zeros_like(knots)
            weights[j] = 1
            basis = splev(fraction[t,:], (knots, weights, order))
            bases.append(basis)

        G = np.array(bases).T

        if transform=='log':
            dayglowTemp = np.exp(G.dot(m).reshape(imgs.isel(date=t)[inImg].shape))-1
        else:
            dayglowTemp = G.dot(m).reshape(imgs.isel(date=t)[inImg].shape)
        dayglow[t,:,:] = dayglowTemp
        dayglow[t,:,:] = np.where(np.isfinite(imgs.isel(date=t)['mlat']),dayglow[t,:,:],np.nan)

        cimage[t,:,:] = (imgs.isel(date=t)[inImg].values) - dayglowTemp
        cimage[t,:,:] = np.where(np.isfinite(imgs.isel(date=t)['mlat']),cimage[t,:,:],np.nan)
        cimage[t,:,:] = np.where(imgs['bad'],cimage[t,:,:],np.nan)

        wimage[t,:,:] = w[t,:,:]
        wimage[t,:,:] = np.where(np.isfinite(imgs.isel(date=t)['mlat']),wimage[t,:,:],np.nan)
        wimage[t,:,:] = np.where(imgs['bad'],wimage[t,:,:],np.nan)

    # Add the new fields to the dataset
    imgs = imgs.assign({
        'dgmodel':(['date','row','col'],dayglow),
        'dgimg': (['date','row','col'],cimage),
        'dgweight': (['date','row','col'],wimage),
        })
    return imgs

# Inline functions
def basis_fun(t_eval, degree, B_nr, knots):

    # Recursive algorithm to evaluate a specific spline at a specific point

    # Input:
    # t_eval, is where the spline is to be evaluated
    # degree, is the degree of the spline
    # B_nr, is the number of the spline, starts from 0
    # knots, are the knots

    # Output:
    # Value of the B_nr'th spline at time t_eval
    if degree == 0:
        return 1.0 if knots[B_nr] <= t_eval <= knots[B_nr+1] else 0.0

    if knots[B_nr+degree] == knots[B_nr]:
        c1 = 0.0
    else:
        c1 = (t_eval - knots[B_nr])/(knots[B_nr+degree] - knots[B_nr]) * basis_fun(t_eval, degree-1, B_nr, knots)

    if knots[B_nr+degree+1] == knots[B_nr+1]:
        c2 = 0.0
    else:
        c2 = (knots[B_nr+degree+1] - t_eval)/(knots[B_nr+degree+1] - knots[B_nr+1]) * basis_fun(t_eval, degree-1, B_nr+1, knots)

    return c1 + c2

def makeFUVshModelNew(imgs,Nsh,Msh,inImg='dgimg',dampingVal=0,stop=0.01,degree=2,knotSep=None):
    date = imgs['date'].values
    t=(date-date[0])/ np.timedelta64(1, 'm')

    glat = imgs['glat'].stack(z=('row','col')).values
    glon = imgs['glon'].stack(z=('row','col')).values
    d = imgs[inImg].stack(z=('row','col')).values
    dg = imgs['dgmodel'].stack(z=('row','col')).values
    wdg = imgs['dgweight'].stack(z=('row','col')).values

    # Treat dg as variance
    d = d/dg

    sslat, sslon = map(np.ravel, subsol(date))
    phi = np.deg2rad((glon - sslon[:,None] + 180) % 360 - 180)

    n_t = len(date)
    n_i = glat.shape[1]

    ## Define splines
    # Number of control points
    if knotSep==None:
        n_cp = degree+1
    else:
        n_cp = int(t[-1]//(knotSep))+degree+1

    # Knots
    n_k     = n_cp + degree + 1
    knots   = np.hstack((np.ones(degree)*t[0], np.linspace(t[0], t[-1], n_k-2*degree), np.ones(degree)*t[-1]))


    # Solve inverse problem with temporal spline fit

    # Evaluate all n_cp splines where there are observations
    M = np.zeros((n_t, n_cp))
    for i, tt in enumerate(t):
        for j in range(n_cp):
            M[i, j] = basis_fun(tt, degree, j, knots)

    # Quick fix, discuss with M on Monday
    M = M / np.sum(M,axis=1)[:,None]


    # Iterative (few iterations)

    skeys = sh.SHkeys(Nsh, Msh).Mge(1).MleN().setNmin(1)
    ckeys = sh.SHkeys(Nsh, Msh).MleN().setNmin(1)



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

        G_t = np.zeros((n_i, n_G*n_cp))

        for j in range(n_cp):
            G_t[:, np.arange(j, n_G*n_cp, n_cp)] = G*M[i, j]

        print(i)
        G_g.append(G)
        G_s.append(G_t)

    G_s = np.array(G_s)
    G_s = G_s.reshape(-1,G_s.shape[2])

    # Data
    ind = (np.isfinite(d))&(glat>0)&(imgs['bad'].values.flatten())[None,:]

    d_s = d.flatten()
    ind = ind.flatten()
    # Weights based on boundary fit
    # w = w.flatten()[ind]
    # w[:]=1
    w = wdg.flatten()

    # Temporal Damping
    damping = dampingVal*np.ones(G_s.shape[1])
    R = np.diag(damping)

    diff = 1e10*stop
    iteration = 0
    m = None
    while (diff>stop)&(iteration < 100):
        # Solve for spline amplitudes
        # m_s = np.linalg.inv((G_s[ind,:]*w[:,None]).T@(G_s[ind,:]*w[:,None])+R)@(G_s[ind,:]*w[:,None]).T@(d_s[ind]*w)
        m_s = np.linalg.lstsq((G_s[ind,:]*w[ind,None]).T@(G_s[ind,:]*w[ind,None])+R,(G_s[ind,:]*w[ind,None]).T@(d_s[ind]*w[ind]))[0]

        # Retrieve B-spline smooth model paramters (coarse)
        mTemp    = M@m_s.reshape((n_G, n_cp)).T
        dm=[]
        for i, tt in enumerate(t):
            dm.append(G_g[i]@mTemp[i, :])

        dm=np.array(dm).squeeze()
        residuals = dm.flatten()[ind] - d.flatten()[ind]
        # rms = np.sqrt(np.average(residuals**2,weights=w[ind]))
        mae = np.average(abs(residuals),weights=w[ind])
        # weights = .5*rms/np.abs(residuals)
        # weights[weights > 1] = 1.

        iw = ((residuals)/(8*mae))**2
        iw[iw>1] = 1
        weights = (1-iw)**2

        if m is not None:
            diff = np.sqrt(np.mean((m - mTemp)**2))
            print(np.sqrt(np.average(residuals**2, weights = weights)), diff)
        w[ind] = weights
        m = mTemp
        iteration += 1


    imgs['shmodel'] = (['date','row','col'],(dm*dg).reshape((n_t,len(imgs.row),len(imgs.col))))
    imgs['shimg'] = imgs[inImg]-imgs['shmodel']
    imgs['shweight'] = (['date','row','col'],(w).reshape((n_t,len(imgs.row),len(imgs.col))))

    return imgs

def runrun():
    inpath = '/mnt/0b3b8cce-3469-42cb-b694-60a7ca36e03a/IMAGE_FUV/wic/'
    outpath = '/mnt/5fa6bccc-fa9d-4efc-9ddc-756f65699a0a/aohma/fuv/wic/'
    df = pd.DataFrame()
    df['wicfile'] = glob.glob(inpath + '*.idl')
    print(df)
    df['date']=pd.to_datetime(df.loc[:,'wicfile'].str.replace(inpath + 'wic','').str.replace('.idl',''),format='%Y%j%H%M')
    df = df.set_index('date')
    df = df.sort_index()
    
    onset = pd.read_pickle('merged_substormlist.pd')
    frey = onset.loc['2000-03-01':'2006-01-01',:]
    frey['date'] = frey.index#.floor('min')
    
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
        # temp = temp.set_index(['onset','sec'])
        
        files.append(temp)
    # ds = pd.concat(ds)  
    for f in files:
        if not f.empty:
            if f['mlat'][0]>0:
                wic = readFUVimage(f['wicfile'].values,dzalim=75,hemisphere='north')
            else:
                wic = readFUVimage(f['wicfile'].values,dzalim=75,hemisphere='south')
                
            wic = makeFUVdayglowModelC(wic,transform='log')
            wic = makeFUVshModelNew(wic,4,4)
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

def makeTransferFile(fromDate,toDate):
    inpath = '/mnt/0b3b8cce-3469-42cb-b694-60a7ca36e03a/IMAGE_FUV/wic/'
    outpath = '/mnt/5fa6bccc-fa9d-4efc-9ddc-756f65699a0a/aohma/fuv/wic/'
    df = pd.DataFrame()
    df['wicfile'] = glob.glob(inpath + '*.idl')
    print(df)
    df['date']=pd.to_datetime(df.loc[:,'wicfile'].str.replace(inpath + 'wic','').str.replace('.idl',''),format='%Y%j%H%M')
    df = df.set_index('date')
    df = df.sort_index()
    
    df = df[fromDate:toDate]

    wic = readFUVimage(df['wicfile'].values,dzalim=75)
    wic.to_netcdf(outpath+'wic'+df.index[0].strftime('%Y%m%d%H%M%S')+'.nc')
    vaex_df.export_hdf5(outpath+'wicTemp.hdf5')
            
    return




# Field to include:
# ind,date,row,col,mlat,mlt,hemisphere,img,dgimg,shimg,onset,rind 








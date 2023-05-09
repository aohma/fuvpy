#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:28:39 2022

@author: aohma
"""

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection

from polplot import pp
    
def cmap_IMAGE():
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

def plot_backgroundmodel_BS(img,**kwargs):
    '''
    Visualize the B-spline background model for one image.
    Ex: plot_backgroundmodel_BS(imgs.isel(date=15))

    Parameters
    ----------
    img : xarray.Dataset
        Dataset with the FUV image.
    outpath : str, optional
        Path to save the figure. The default is None. 

    Returns
    -------
    None.
    '''

    # Check the number of time steps provided
    n_date = img.dims['date'] if 'date' in img.dims else 1
    if n_date != 1: raise ValueError('Only data from a single time step can be provided')

    outpath = kwargs.pop('outpath') if 'outpath' in kwargs.keys() else None

    sza = img['sza'].values.flatten()
    dza = img['dza'].values.flatten()
    image = img['img'].values.flatten()
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

    ax1.scatter(fraction,img['dgmodel'].values.flatten(),s=1,color = 'black')

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
    plotimg(img,'img',pax=pax2,crange=[0,cmax1])
    ax2c.axis('off')
    plt.colorbar(pax2.ax.collections[0],orientation='horizontal',ax=ax2c,fraction=1,
                 extend='max')

    # Polar plot of the dayglow model
    gs3 = gridspec.GridSpecFromSubplotSpec(2,1,height_ratios=[90,10],subplot_spec=gs[1,1])
    ax3 = plt.subplot(gs3[0])
    ax3c = plt.subplot(gs3[1])
    pax3 = pp(ax3)
    plotimg(img,'dgmodel',pax=pax3,crange=[0,cmax1])
    ax3c.axis('off')
    plt.colorbar(pax3.ax.collections[0],orientation='horizontal',ax=ax3c,fraction=1,
                 extend='max')

    # Polar plot of the dayglow-corrected image
    gs4 = gridspec.GridSpecFromSubplotSpec(2,1,height_ratios=[90,10],subplot_spec=gs[1,2])
    ax4 = plt.subplot(gs4[0])
    ax4c = plt.subplot(gs4[1])
    pax4 = pp(ax4)
    plotimg(img,'dgimg',pax=pax4,crange=[-cmax2,cmax2],cmap='coolwarm')
    ax4c.axis('off')
    plt.colorbar(pax4.ax.collections[0],orientation='horizontal',ax=ax4c,fraction=1,
                 extend='both')

    ax1.set_title(img['id'].values.tolist() + ': ' + img['date'].dt.strftime('%Y-%m-%d %H:%M:%S').values.tolist())
    ax2.set_title('Original image')
    ax3.set_title('Dayglow')
    ax4.set_title('Corrected image')

    gs.tight_layout(fig)

    # Save figure
    if outpath is not None:
        filename = img['date'].dt.strftime('%Y%m%d%H%M%S').values.tolist()
        plt.savefig(outpath + str(img['id'].values)+filename + '_dayglowModel.png', bbox_inches='tight', dpi = 100)
        plt.clf()
        plt.close(fig)


def plotimg(img, inImg, **kwargs):
    '''
    Show a FUV image in polar coordinates coordinates.
    Wrapper to polplot's plotimg

    Parameters
    ----------
    img : xarray.Dataset
        Image dataset.
    inImg : str
        Name of the image to be show.
    pax : Polarplot object, optional
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

    # Check the number of time steps provided
    n_date = img.dims['date'] if 'date' in img.dims else 1
    if n_date != 1: raise ValueError('Only data from a single time step can be provided')

    # Set keyword arguments to input or default values    
    pax = kwargs.pop('pax') if 'pax' in kwargs.keys() else pp(plt.subplots()[1])
    crange = kwargs.pop('crange') if 'crange' in kwargs.keys() else None
    bgcolor = kwargs.pop('bgcolor') if 'bgcolor' in kwargs.keys() else None

    return pax.plotimg(img['mlat'].values,img['mlt'].values,img[inImg].values,crange=crange,bgcolor=bgcolor,**kwargs)

def pplot(imgs,inImg,**kwargs):
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
    **kwargs : additional keyword arguments
        kwargs to Polycollection().
    Returns
    -------
    None.
    '''
    n_imgs = len(imgs.date)

    # Set keyword arguments to input or default values
    col_wrap = kwargs.pop('col_wrap') if 'col_wrap' in kwargs.keys() else None
    tb = bool(kwargs.pop('tb')) if 'tb' in kwargs.keys() else False
    add_cbar = bool(kwargs.pop('add_cbar')) if 'add_cbar' in kwargs.keys() else True
    crange = kwargs.pop('crange') if 'crange' in kwargs.keys() else None
    robust = bool(kwargs.pop('robust')) if 'robust' in kwargs.keys() else False
    cbar_kwargs = kwargs.pop('cbar_kwargs') if 'cbar_kwargs' in kwargs.keys() else {}
    pp_kwargs = kwargs.pop('pp_kwargs') if 'pp_kwargs' in kwargs.keys() else {}

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

    plt.figure(figsize=(f_width,f_height))

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

def plot_ltlat(img,inImg,**kwargs):
    '''
    Show a section of a FUV image projected in mlt-mlat coordinates
    Ex: plot_ltlat(.isel(date=15),'img',crange=(0,1500),cmap='plasma')
    Parameters
    ----------
    img : xarray.Dataset
        Image dataset.
    inImg : str
        Name of the image to be show.
    ax : plt.axis object, optional
        ax to show the image. The default is None (new figure).
    leftlt : float, optional
        Left ("lower") MLT limit. The default is 18.
    rightlt : float, optional
        Right ("Upper") MLT limit. The default is 6.
    minlat : float, optional
        Minimum latitude. The default is 50.
    maxlat : float, optional
        Maximum latitude. The default is 80.
    crange : tuple/list, optional
        Color range, given as (lower,upper). The default is None (matplotlib default).
    **kwargs : additional keyword arguments
        kwargs to Polycollection().
    Returns
    -------
    coll
        The displayed Polycollection
    '''

    ax = kwargs.pop('ax') if 'ax' in kwargs.keys() else plt.subplots()[1]
    leftlt = kwargs.pop('leftlt') if 'leftlt' in kwargs.keys() else 18
    rightlt = kwargs.pop('rightlt') if 'rightlt' in kwargs.keys() else 6
    minlat = kwargs.pop('minlat') if 'minlat' in kwargs.keys() else 50
    maxlat = kwargs.pop('maxlat') if 'maxlat' in kwargs.keys() else 80
    crange = kwargs.pop('crange') if 'crange' in kwargs.keys() else None

    mlat = img['mlat'].values.copy()
    mlt =img['mlt'].values.copy()
    rightlt=rightlt+0.2
    mlt[mlt>rightlt]=mlt[mlt>rightlt]-24

    if leftlt>rightlt:  leftlt= leftlt-24

    ll = mlt[1:,  :-1].flatten(), mlat[1:,  :-1].flatten()
    lr = mlt[1:,   1:].flatten(), mlat[1:,   1:].flatten()
    ul = mlt[:-1, :-1].flatten(), mlat[:-1, :-1].flatten()
    ur = mlt[:-1,  1:].flatten(), mlat[:-1,  1:].flatten()

    vertsx = np.vstack((ll[0], lr[0], ur[0], ul[0])).T
    vertsy = np.vstack((ll[1], lr[1], ur[1], ul[1])).T

    ii_neg = (vertsx<-5).any(axis=1)
    ii_pos = (vertsx>5).any(axis=1)
    iii = np.where(vertsx >= leftlt, True, False)&np.where(vertsx <= rightlt, True, False)&np.where(vertsy >= minlat, True, False)&np.where(vertsy <= maxlat, True, False)
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

    rightlt = rightlt - 0.2
    ax.set_xlim([leftlt,rightlt])
    ax.set_ylim([minlat,maxlat])
    xlabel = np.linspace(leftlt,rightlt,len(ax.get_xticklabels()))
    xlabel[xlabel<0] = xlabel[xlabel<0]+24
    if (xlabel[xlabel!=0]%xlabel[xlabel!=0].astype(int)==0).all():
        ax.set_xticklabels(xlabel.astype(int))
    else:
        ax.set_xticklabels(xlabel)

    return coll

def plotboundaries(ds,boundary,**kwargs):
    '''
    Polar plot showing the temporal evolution of the OCB.
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with boundaries identified.
        The coordinates must be mlt and date
    boundary : str
        Name of the boundary in ds.
    pax : Polarplot object, optional
        pax to show the image. The default is None (new figure).
    cmap : str, optional
        Colormap name. Default is Viridis
    add_cbar : bool, optional
        If True (default), a colorbar is added.
    **kwargs : additional keyword arguments
        Keywords to pyplot.plot

    Returns
    -------
    None
    '''
    
    # Set keyword arguments to input or default values    
    pax = kwargs.pop('pax') if 'pax' in kwargs.keys() else pp(plt.subplots()[1])
    cmap = kwargs.pop('cmap') if 'cmap' in kwargs.keys() else 'viridis'
    add_cbar = bool(kwargs.pop('add_cbar')) if 'add_cbar' in kwargs.keys() else True

    # Add first row to end for plotting
    ds = xr.concat([ds,ds.isel(mlt=0)],dim='mlt')

    # Date
    date = ds.date
    n_d = len(date)
    dateStr = ds.date.dt.strftime('%H:%M:%S').values
    cmap = plt.get_cmap(cmap,n_d)

    for d in range(n_d):
        pax.plot(ds.isel(date=d)[boundary].values,ds.mlt.values,color=cmap(d),**kwargs)

    if add_cbar:
        norm = mcolors.Normalize(vmin=0,vmax=n_d)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm,ticks=np.arange(0,n_d,1),boundaries=np.arange(-0.5,n_d,1),ax=pax.ax)
        cbar.ax.tick_params(size=0)
        cbarlabel = n_d*['']
        cbarlabel[::n_d//10+1] = dateStr[::n_d//10+1]
        cbar.set_ticklabels(cbarlabel)
    
    plt.tight_layout()
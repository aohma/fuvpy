#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:22:03 2022

@author: simon
"""

""" Imports """
import matplotlib.pyplot as plt
from polplot.polplot import Polarplot as polar
import fuvpy as fuv
from glob import glob
import numpy as np
import functools
import pandas as pd
import os

"""Adjustment Box"""
mlat_size= 4 # goes from clicked mlat -2 to clicked mlat +2
mlt_size= 1 # goes from clicked mlt -0.5 to clicked mlt +0.5
""" Functions that shouldn't be touched """
# For making the mlt axis showing pixel intensity at the mlat that has been chosen
class Make_MLT_ax():
    def __init__(self, ax):
        self.ax = ax
        ax.set_theta_zero_location('S')
        ax.set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21])
        ax.format_coord= self.make_format()
        ax.autoscale(enable=False)
    def plot(self, mlt, counts, **kwargs):
        return self.ax.plot(np.array(mlt)*(2*np.pi/24), counts, **kwargs)
    def make_format(current):
        # current and other are axes
        def format_coord(theta, r):
            if theta<0:
                theta+=2*np.pi
            display_coord = theta/(2*np.pi/24),r
            ax_coord= (float(i) for i in display_coord)
            string= 'mlt={:.2f}, counts={:.2f}'.format(*ax_coord)
            return (string)
        return format_coord
def datetime_to_vistime(datetime):
    year= int(str(datetime.astype('datetime64[Y]')))
    DOY= (np.datetime64(datetime)-np.datetime64(f'{year}-01-01T00:00')).astype('timedelta64[D]').astype(int)+1
    time=str(datetime).split('T')[-1]
    hour= int(time[:2])
    minute= int(time[3:5])
    seconds= int(time[6:])
    return f'vis{year}{DOY:03d}{hour:02d}{minute:02d}{seconds:02d}.idl'
def datetime_to_wictime(datetime):
    year= int(str(datetime.astype('datetime64[Y]')))
    DOY= (np.datetime64(datetime)-np.datetime64(f'{year}-01-01T00:00')).astype('timedelta64[D]').astype(int)+1
    time=str(datetime).split('T')[-1]
    hour= int(time[:2])
    minute= int(time[3:5])
    return f'wic{year}{DOY:03d}{hour:02d}{minute:02d}.idl'
def wictime_to_datetime(wic):
    wic= wic.split('wic')[-1].split('.idl')[0]
    year= wic[:4]
    DOY= wic[4:7]
    hour= wic[7:9]
    minute= wic[9:11]
    return np.datetime64(f'{year}-01-01T{hour}:{minute}')+ np.timedelta64(int(DOY)-1, 'D')
def vistime_to_datetime(vis):
    vis= vis.split('vis')[-1].split('.idl')[0]
    year=vis[:4]
    DOY= vis[4:7]
    hour= vis[7:9]
    minute= vis[9:11]
    seconds= vis[11:13]
    return np.datetime64(f'{year}-01-01T{hour}:{minute}:{seconds}')+ np.timedelta64(int(DOY)-1, 'D')
def mlt2radians(mlt):
    if not isinstance(mlt, np.ndarray):
        mlt=np.array([mlt])
    radians= mlt*(2*np.pi/24)
    radians[(radians<=2*np.pi)&(radians>=np.pi)]-=2*np.pi
    return radians
def radians2mlt(radians):
    mlt= radians *(24/(2*np.pi))
    mlt[mlt<0]+=24
    return mlt
# For showing the image file on the specified axis

# What happens when an axis is clicked
def clicked(image_axis, mlt_axis, mlat_axis, mlt, mlat):
    global marker
    marker= image_axis.scatter(mlat, mlt, marker='+', color='black', zorder=100, s=500)
    if mlt_axis and mlat_axis:
        global profile1, profile2, lines, window
        cmap=image_axis.image.get_cmap()
        mlt_lims= (mlt-mlt_size/2, mlt+mlt_size/2)
        image= image_axis.image_dat
        if mlt_lims[0]>=0 and mlt_lims[1]<24:
            image= image.where((image.mlat>=mlat-mlat_size/2)\
                                                           &(image.mlat<=mlat+mlat_size/2)\
                                                           &(image.mlt>=mlt-mlt_size/2)\
                                                           &(image.mlt<=mlt+mlt_size/2))
        elif mlt_lims[0]<0:
            image= image.where((image.mlat>=mlat-mlat_size/2)\
                                                           &(image.mlat<=mlat+mlat_size/2)\
                                                           &((image.mlt>=24+(mlt-mlt_size/2))\
                                                           |(image.mlt<=mlt+mlt_size/2)))
        elif mlt_lims[1]>24:
            image= image.where((image.mlat>=mlat-mlat_size/2)\
                                                           &(image.mlat<=mlat+mlat_size/2)\
                                                           &((image.mlt>=(mlt-mlt_size/2))\
                                                           |(image.mlt<=(mlt+mlt_size/2)-24)))
        window= image_axis.plot([mlat+mlat_size/2]*2 + [mlat-mlat_size/2]*2+ [mlat+mlat_size/2],
                                    [mlt-mlt_size/2, mlt+mlt_size/2, mlt+mlt_size/2, mlt-mlt_size/2, mlt-mlt_size/2],
                                    color='orange')
        image= image.where(image.data!=0)
        ind= image.data== np.nanmax(image.data)
        mlt_= np.nanmax(image.where(ind).mlt.values)
        mlat_= np.nanmax(image.where(ind).mlat.values)

        crange= image_axis.image.get_clim()
        profile1= [mlat_axis.scatter(image.mlat.values, image.data.values, c=image.data.values, cmap=cmap, vmin= crange[0], vmax=crange[1])]+\
                   mlat_axis.plot([mlat]*2, [np.nanmin(image.data), np.nanmax(image.data)], color='black')
        profile1.extend(mlat_axis.plot([round(mlat_, 2)]*2, [np.nanmin(image.data), np.nanmax(image.data)],
                                       color='black', linestyle='--'))
        try:
            mlat_axis.set_ylim(np.nanmin(image.data)-abs(np.nanmin(image.data))*.1,
                               np.nanmax(image.data)+ abs(np.nanmax(image.data))*.1)
        except:
            pass
        mlat_axis.set_xlim((mlat-mlat_size/2)-abs(mlat-mlat_size/2)*.01, (mlat+mlat_size/2)+abs(mlat+mlat_size/2)*.01)
        radians= mlt2radians(image.mlt.values)
        xlims= radians2mlt(np.array([np.nanmin(radians), np.nanmax(radians)]))
        xlims[0]= np.floor(xlims[0]*10)/10
        xlims[1]= np.ceil(xlims[1]*10)/10
        try:
            mlt_axis.set_xlim(*mlt2radians(xlims))
            mlt_axis.set_ylim(np.nanmin(image.data)-abs(np.nanmin(image.data))*0.1, np.nanmax(image.data)+abs(np.nanmax(image.data))*.1)
        except:
            pass

        profile2= [mlt_axis.scatter(radians, image.data.values, c=image.data.values, cmap=cmap, vmin=crange[0], vmax=crange[1])]+\
                   mlt_axis.plot([mlt2radians(mlt)]*2, [np.nanmin(image.data), np.nanmax(image.data)], color='black')

        profile2.extend(mlt_axis.plot([mlt2radians(round(mlt_, 2))]*2, [np.nanmin(image.data), np.nanmax(image.data)],
                                  color='black', linestyle='--'))
        if mlt_lims[0]<0:
            labels= np.append(np.arange(24-mlt_size/4, round(mlt_lims[0]+24, 1)-mlt_size/4, -mlt_size/4)[::-1], np.arange(0, round(mlt_lims[1], 1)+mlt_size/4, mlt_size/4))
        elif mlt_lims[1]>24:
            labels= np.append(np.arange(24-mlt_size/4, round(mlt_lims[0], 1)-mlt_size/4, -mlt_size/4)[::-1], np.arange(0, round(mlt_lims[1]-24, 1)+mlt_size/4, mlt_size/4))
        else:
            labels= np.round(np.arange(round(mlt_lims[0], 1), round(mlt_lims[1], 1)+mlt_size/4, mlt_size/4), 2)
        rads=mlt2radians(labels)
        mlt_axis.set_xticks(rads, labels)

        lines= image_axis.plot([mlat]*100, np.append(np.linspace(0, 6, 50)[::-1], np.linspace(18, 24, 50)[::-1]), zorder=100, color='orange', alpha=.7) + \
               image_axis.plot(np.linspace(50, 90, 100), [mlt]*100, zorder=100, color='orange', alpha=.7)



    plt.draw()

class Visualise():
    def __init__(self, fig, axes, caxes, MLTax=False, MLATax=False, cax_association=False):
        if not isinstance(axes, (list, np.ndarray)):
            axes= [axes]
        if not isinstance(caxes, (list, np.ndarray)):
            caxes= [caxes]
        axes= np.asarray(axes)
        caxes= np.asarray(caxes)
        onclick_wrapper=functools.partial(self.onclick, axes, MLTax, MLATax, caxes)
        cid = fig.canvas.mpl_connect('button_press_event', onclick_wrapper)
        if not cax_association:
            cax_association= [0]*len(axes)
        for ax, i in zip(axes, cax_association): ax.cax_number=i
        for ax in axes:
            ax.image= False
            ax.image_dat= False
            # ax.cax= caxes[ax.cax_number]
        self.axes= axes
        self.caxes= caxes
        self.MLTax= MLTax
        self.MLATax=MLATax
        self.cbars= [False]*len(np.unique(cax_association))
        self.figure= fig
    def show_image(self, file, axis, crange=False, cmap=False, cbar_orientation=False, in_put='img'):
        if file.endswith('.idl'):
            axis.image_dat= fuv.readImg(file).isel(date = 0).rename({in_put:'data'})
        else:
            import xarray as xr
            axis.image_dat= xr.load_dataset(file).rename({in_put:'data'})
        if axis.image:
            axis.image.remove()
            axis.image=False
            try:
                marker.remove()
                for p in profile1+ profile2+ lines+window: p.remove()
            except NameError:
                pass
            except ValueError:
                pass
        mlt= axis.image_dat.mlt
        image= axis.image_dat.where(eval(axis.mltlims)&(axis.image_dat.mlat>=(axis.minlat)))
        maxi= np.nanmax(axis.image_dat.data)
        mini= np.nanmin(axis.image_dat.data)
        cbar= self.cbars[axis.cax_number]
        cax= self.caxes[axis.cax_number]
        if not cbar:
            if not cmap:
                cmap='jet'
            if not crange:
                crange= (0, maxi)
            if not cbar_orientation:
                cbar_orientation='horizontal'
            for ax in self.axes:
                if ax.cax_number==axis.cax_number and ax!=axis:
                    if ax.image and crange!= ax.image.get_clim():
                        ax.image.set_clim(crange)
                    if ax.image and cmap!= ax.image.get_cmap().name:
                        ax.image.set_cmap(cmap)
            im= axis.plotimg(image.mlat.values, image.mlt.values, image.data.values, crange=crange, cmap=cmap, zorder=1)
            cbar=self.figure.colorbar(im, cax=cax, orientation= cbar_orientation)
            self.cbars[axis.cax_number]=cbar
        else:
            new_cbar=False
            if not crange:
                clims=[]
                for ax in self.axes:
                    if ax.image:
                        clims+=[*ax.image.get_clim()]
                if cbar.orientation=='horizontal':
                    crange=self.caxes[axis.cax_number].get_xlim()
                elif cbar.orientation=='vertical':
                    crange=self.caxes[axis.cax_number].get_ylim()
                maxi=max([maxi, *clims])
                if maxi!=crange[-1]:
                    new_cbar=True
                crange=(0, maxi)
            if not cmap:
                cmap=cbar.cmap
            elif cmap!=cbar.cmap.name:
                new_cbar=True
            if not cbar_orientation:
                cbar_orientation= cbar.orientation
            elif cbar_orientation!=cbar.orientation:
                new_cbar=True
            for ax in self.axes:
                if ax.cax_number==axis.cax_number and ax!=axis:
                    if crange!= ax.image.get_clim():
                        ax.image.set_clim(crange)
                    if cmap!= ax.image.get_cmap().name:
                        ax.image.set_cmap(cmap)
            im= axis.plotimg(image.mlat.values, image.mlt.values, image.data.values, crange=crange, cmap=cmap, zorder=1)
            if new_cbar:
                if cbar.orientation=='horizontal':
                    label= cbar.ax.get_xlabel()
                elif cbar.orientation=='vertical':
                    label= cbar.ax.get_ylabel()
                self.caxes[axis.cax_number].clear()
                self.caxes[axis.cax_number].get_xaxis().set_visible(True)
                self.caxes[axis.cax_number].get_yaxis().set_visible(True)
                cbar= self.figure.colorbar(im, orientation=cbar_orientation, cax=cax)
                self.cbars[axis.cax_number]=cbar
                cbar.set_label(label)
                for ax in self.axes:
                    if ax.cax_number== axis.cax_number and ax.image:
                        ax.image.set_clim(crange)
        axis.ax.set_title(image.date.values.astype('datetime64[s]').tolist(), y=0.1)
        axis.image= im
        return im, cbar
    # Handles which axis was clicked, where it was clicked and preparation for the clicked function
    def onclick(self, axes, prof_axis1, prof_axis2, caxes, event):
        bools_ax= np.array([axis.in_axes(event) for axis in axes])
        bools_cax= np.array([axis.in_axes(event) for axis in caxes])
        if any(bools_ax):
            global mlt, mlat
            ix, iy= event.xdata, event.ydata
            try:
                marker.remove()
                for p in profile1+ profile2+ lines+window: p.remove()
            except:
                pass
            mlat, mlt= np.array(axes)[bools_ax][0]._XYtomltMlat(ix, iy)
            clicked(np.array(axes)[bools_ax][0], prof_axis1, prof_axis2, mlt, mlat)
        elif any(bools_cax):
            ix, iy= event.xdata, event.ydata
            cax_number= np.argmax(bools_cax)
            cbar=self.cbars[cax_number]
            if cbar.orientation=='horizontal':
                crange=caxes[bools_cax][0].get_xlim()
            elif cbar.orientation=='vertical':
                crange=caxes[bools_cax][0].get_ylim()
            caxes[bools_cax][0].clear()
            if ix<sum(crange)/2:
                for ax in axes:
                    if ax.cax_number==cax_number:
                        ax.image.set_clim(round(ix, 0), crange[-1])
            if ix>=sum(crange)/2:
                for ax in axes:
                    if ax.cax_number== cax_number:
                        ax.image.set_clim(crange[0], round(ix, 0))
            try:
                if np.array(axes)[np.array([marker.axes==ax.ax for ax in axes])][0].cax_number== cax_number:
                    profile1[0].set_clim(crange)
                    profile2[0].set_clim(crange)
            except NameError:
                pass
        plt.draw()
        return

if __name__=='__main__':
    import inspect
    folder= '/'.join(inspect.getfile(fuv).split('/')[:-1])+'/data/wicFiles/'
    file=folder+'wic20002410928.idl'
    np.datetime64('2001-12-06T10:17:26')
    file2=folder+'wic20002410930.idl'
    fig= plt.figure(figsize=(20,10))
    gs= fig.add_gridspec(2, 4, height_ratios=[1, .1])
    ax= polar(fig.add_subplot(gs[0, 1]), sector='18-6')
    ax2= polar(fig.add_subplot(gs[0, 2]), sector='18-6')
    MLTax= fig.add_subplot(gs[0, 0])
    MLATax= fig.add_subplot(gs[0, 3])
    cax= fig.add_subplot(gs[1, 1])
    cax2= fig.add_subplot(gs[1,0])
    vis=Visualise(fig, np.asarray([ax, ax2]), np.asarray([cax, cax2]), MLTax, MLATax, cax_association=[0, 1])
    vis.show_image(file, ax, cmap='viridis_r', in_put='sza')
    vis.show_image(file2, ax2)

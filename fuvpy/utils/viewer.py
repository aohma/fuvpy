#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:22:03 2022

@author: simon
"""

""" Imports """
import matplotlib.pyplot as plt
from polplot import Polarplot as polar
import fuvpy as fuv
from glob import glob
import numpy as np
import functools
import pandas as pd
import os

"""Adjustment Box"""
lat_size= 4 # goes from clicked mlat -2 to clicked mlat +2
lt_size= 1 # goes from clicked mlt -0.5 to clicked mlt +0.5
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
    if wic.endswith('.idl'):
        end= '.idl'
        wic= wic.split('wic')[-1].split(end)[0]
    
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
def clicked(image_axis, lt_axis, lat_axis, lt, lat):
    global marker
    marker= image_axis.scatter(lat, lt, marker='+', color='black', zorder=100, s=500)
    if lt_axis and lat_axis:
        global profile1, profile2, lines, window
        cmap=image_axis.image.get_cmap()
        lt_lims= (lt-lt_size/2, lt+lt_size/2)
        image= image_axis.image_dat
        if lt_lims[0]>=0 and lt_lims[1]<24:
            image= image.where((image.lat>=lat-lat_size/2)\
                                                           &(image.lat<=lat+lat_size/2)\
                                                           &(image.lt>=lt-lt_size/2)\
                                                           &(image.lt<=lt+lt_size/2))
        elif lt_lims[0]<0:
            image= image.where((image.lat>=lat-lat_size/2)\
                                                           &(image.lat<=lat+lat_size/2)\
                                                           &((image.lt>=24+(lt-lt_size/2))\
                                                           |(image.lt<=lt+lt_size/2)))
        elif lt_lims[1]>24:
            image= image.where((image.lat>=lat-lat_size/2)\
                                                           &(image.lat<=lat+lat_size/2)\
                                                           &((image.lt>=(lt-lt_size/2))\
                                                           |(image.lt<=(lt+lt_size/2)-24)))
        window= image_axis.plot([lat+lat_size/2]*2 + [lat-lat_size/2]*2+ [lat+lat_size/2],
                                    [lt-lt_size/2, lt+lt_size/2, lt+lt_size/2, lt-lt_size/2, lt-lt_size/2],
                                    color='orange')
        image= image.where(image.data!=0)
        ind= image.data== np.nanmax(image.data)
        lt_= np.nanmax(image.where(ind).lt.values)
        lat_= np.nanmax(image.where(ind).lat.values)

        crange= image_axis.image.get_clim()
        profile1= [lat_axis.scatter(image.lat.values, image.data.values, c=image.data.values, cmap=cmap, vmin= crange[0], vmax=crange[1])]+\
                   lat_axis.plot([lat]*2, [np.nanmin(image.data), np.nanmax(image.data)], color='black')
        profile1.extend(lat_axis.plot([np.round(lat_, 2)]*2, [np.nanmin(image.data), np.nanmax(image.data)],
                                       color='black', linestyle='--'))
        try:
            lat_axis.set_ylim(np.nanmin(image.data)-abs(np.nanmin(image.data))*.1,
                               np.nanmax(image.data)+ abs(np.nanmax(image.data))*.1)
        except:
            pass
        lat_axis.set_xlim((lat-lat_size/2)-abs(lat-lat_size/2)*.01, (lat+lat_size/2)+abs(lat+lat_size/2)*.01)
        radians= mlt2radians(image.lt.values)
        xlims= radians2mlt(np.array([np.nanmin(radians), np.nanmax(radians)]))
        xlims[0]= np.floor(xlims[0]*10)/10
        xlims[1]= np.ceil(xlims[1]*10)/10
        try:
            lt_axis.set_xlim(*mlt2radians(xlims))
            lt_axis.set_ylim(np.nanmin(image.data)-abs(np.nanmin(image.data))*0.1, np.nanmax(image.data)+abs(np.nanmax(image.data))*.1)
        except:
            pass

        profile2= [lt_axis.scatter(radians, image.data.values, c=image.data.values, cmap=cmap, vmin=crange[0], vmax=crange[1])]+\
                   lt_axis.plot([mlt2radians(lt)]*2, [np.nanmin(image.data), np.nanmax(image.data)], color='black')

        profile2.extend(lt_axis.plot([mlt2radians(np.round(lt_, 2))]*2, [np.nanmin(image.data), np.nanmax(image.data)],
                                  color='black', linestyle='--'))
        if lt_lims[0]<0:
            labels= np.append(np.arange(24-lt_size/4, np.round(lt_lims[0]+24, 1)-lt_size/4, -lt_size/4)[::-1], np.arange(0, np.round(lt_lims[1], 1)+lt_size/4, lt_size/4))
        elif lt_lims[1]>24:
            labels= np.append(np.arange(24-lt_size/4, np.round(lt_lims[0], 1)-lt_size/4, -lt_size/4)[::-1], np.arange(0, np.round(lt_lims[1]-24, 1)+lt_size/4, lt_size/4))
        else:
            labels= np.round(np.arange(np.round(lt_lims[0], 1), np.round(lt_lims[1], 1)+lt_size/4, lt_size/4), 2)
        rads=mlt2radians(labels)
        lt_axis.set_xticks(rads, labels)

        lines= image_axis.plot([lat]*100, np.append(np.linspace(0, 6, 50)[::-1], np.linspace(18, 24, 50)[::-1]), zorder=100, color='orange', alpha=.7) + \
               image_axis.plot(np.linspace(50, 90, 100), [lt]*100, zorder=100, color='orange', alpha=.7)



    plt.draw()

class Visualise():
    def __init__(self, fig, axes, caxes, MLTax=False, MLATax=False, cax_association=False):
        """
        For initialising and setting up the visualisation tool. A tool for interacting, analysing
        and visualising data that can be displayed in polar co-ordinates with ease.
        
        Parameters
        ----------
        fig : matplotlib figure
            Figure object.
        axes : Polarplot axis or list
            Polarplot object made by the polplot package or list of Polarplot objects.
        caxes : matplotlib AxesSubplot
            AxesSubplot object or list of AxesSubplot objects where the colorbar(s) go.
        MLTax : matplotlib AxesSubplot, optional
            AxesSubplot object where MLT distributions can go when polar plots are 
            clicked on for more detail. The default is False.
        MLATax : matplotlib AxesSubplot, optional
            AxesSubplot object where MLT distributions can go when polar plots are 
            clicked on for more detail. The default is False.
        cax_association : list, optional
            List of indices that associate the colorbar subplot to the polar plot.
            The same index can be used more than once when the colorbar is shared.
            The default is False.

        Returns
        -------
        None.

        """
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
    def show_image(self, file, axis, crange=False, cmap=False, cbar_orientation=False, 
                   in_put='img', lt_val='mlt', lat_val='mlat', date=0, title_y=-0.1):
        """
        For plotting the data onto the polar plots and enabling the interactive features
        of the visualisation tool. Can be used continually each time data wants to be plotted
        as it will remove the old data and update colorbars etc.

        Parameters
        ----------
        file : string or xarray
            Must provide the path and file name of an idl or xarray file or provide an xarray object.
        axis : Polarplot axis
            Polarplot axis on which the data will be plotted.
        crange : tuple, optional
            Colorbar range for the data. The default is False in which case
            the colorbar range will be automatic based on the range of the data or if show image
            has been previously run it will reuse the previous range.
        cmap : string, optional
            Name of matplotlib colormap to be used. The default is False in which case
            the defualt matplotlib colormap is used.
        cbar_orientation : string, optional
            Orientation of the colorbar. The default is False in which case
            the orientation is horizontal or the orientation of the previous set up if show image
            has been run previously.
        in_put : string, optional
            String that is used in the file for the data that is to be plotted. 
            The default is 'img'.
        lt_val : string, optional
            String that is used in the file for the coordinate to be used as the local time. 
            The default is 'mlt'.
        lat_val : string, optional
            String that is used in the file for the coordinate to be used as the latitude. 
            The default is 'mlat'.
        date: string or datetime or integer
            Only used when input is an idl file. When using string or datetime will 
            select where the date in the file matches the date argument. Using
            an integer will select the date at that index.
            ie date=0 will pick the first date in the file. The default is 0.
        title_y: float, optional
            y coordinate for title of subplot, which is the date of the file.
            The default is -0.1.
        Returns
        -------
        Image: matplotlib Polycollection
            The polycollection object corresponding to the data plotted.
        Colorbar: matplotlib Colorbar
            The colorbar object associated with the data plotted.

        """
        axis.ax.format_coord= axis.make_format(lt_val, lat_val)
        if isinstance(file, (str, np.str_)):
            if file.endswith('.idl') or file.endswith('.sav'):
                if isinstance(date, int):
                    axis.image_dat= fuv.read_idl(file).isel(date = date).rename({in_put:'data', 
                                                                             lt_val:'lt', lat_val:'lat'})
                else:
                    axis.image_dat= fuv.read_idl(file).sel(date = date).rename({in_put:'data', 
                                                                             lt_val:'lt', lat_val:'lat'})
            else:
                import xarray as xr
                axis.image_dat= xr.load_dataset(file).rename({in_put:'data', 
                                                                         lt_val:'lt', lat_val:'lat'})
        else:
            axis.image_dat= file.rename({in_put:'data', lt_val:'lt', lat_val:'lat'})
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
        lt= axis.image_dat.lt
        image= axis.image_dat.where(eval(axis.ltlims)&(axis.image_dat.lat>=(axis.minlat)))
        maxi= np.nanmax(axis.image_dat.data)
        mini= np.nanmin(axis.image_dat.data)
        cbar= self.cbars[axis.cax_number]
        cax= self.caxes[axis.cax_number]
        if not cbar:
            if not cmap:
                cmap='viridis'
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
            im= axis.plotimg(image.lat.values, image.lt.values, image.data.values, crange=crange, cmap=cmap, zorder=1)
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
            im= axis.plotimg(image.lat.values, image.lt.values, image.data.values, crange=crange, cmap=cmap, zorder=1)
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
        try:
            axis.ax.set_title(image.date.values.astype('datetime64[s]').tolist(), y=title_y)
        except AttributeError:
            UserWarning('Failed to use date in file to make subplot title. AttributeError')
        axis.image= im
        plt.draw()
        return im, cbar
    # Handles which axis was clicked, where it was clicked and preparation for the clicked function
    def onclick(self, axes, prof_axis1, prof_axis2, caxes, event):
        bools_ax= np.array([axis.ax.in_axes(event) for axis in axes])
        bools_cax= np.array([axis.in_axes(event) for axis in caxes])
        if any(bools_ax):
            global lt, lat
            ix, iy= event.xdata, event.ydata
            try:
                marker.remove()
                for p in profile1+ profile2+ lines+window: p.remove()
            except:
                pass
            lat, lt= np.array(axes)[bools_ax][0]._xy2latlt(ix, iy)
            clicked(np.array(axes)[bools_ax][0], prof_axis1, prof_axis2, lt, lat)
        elif any(bools_cax):
            ix, iy= event.xdata, event.ydata
            cax_number= np.argmax(bools_cax)
            cbar=self.cbars[cax_number]
            if cbar.orientation=='horizontal':
                crange=caxes[bools_cax][0].get_xlim()
                coord= ix
            elif cbar.orientation=='vertical':
                crange=caxes[bools_cax][0].get_ylim()
                coord= iy
            caxes[bools_cax][0].clear()
            if coord<sum(crange)/2:
                for ax in axes:
                    if ax.cax_number==cax_number:
                        ax.image.set_clim(round(coord, 0), crange[-1])
            if coord>=sum(crange)/2:
                for ax in axes:
                    if ax.cax_number== cax_number:
                        ax.image.set_clim(crange[0], round(coord, 0))
            try:
                if marker.axes is not None and np.array(axes)[np.array([marker.axes==ax.ax for ax in axes])][0].cax_number== cax_number:
                    profile1[0].set_clim(crange)
                    profile2[0].set_clim(crange)
            except NameError:
                pass
        plt.draw()
        return

if __name__=='__main__':
    import inspect
    folder= '/'.join(inspect.getfile(fuv).split('/')[:-2])+'/examples/sample_wicfiles/'
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
    xarray= fuv.read_idl(file2).isel(date=0)
    vis.show_image(xarray, ax2)

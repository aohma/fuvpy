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
import numpy as np
import functools
import matplotlib as mpl

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
    """
    Convert datetime to a string for a file name for the vis camera.

    Parameters
    ----------
    datetime : numpy.datetime64
        Input datetime.

    Returns
    -------
    str
        Formatted string vis filename.
    """
    year = int(str(datetime.astype('datetime64[Y]')))
    DOY = (np.datetime64(datetime) - np.datetime64(f'{year}-01-01T00:00')).astype('timedelta64[D]').astype(int) + 1
    time = str(datetime).split('T')[-1]
    hour = int(time[:2])
    minute = int(time[3:5])
    seconds = int(time[6:])
    return f'vis{year}{DOY:03d}{hour:02d}{minute:02d}{seconds:02d}.idl'

def datetime_to_wictime(datetime):
    """
    Convert datetime to a string for a file name for the WIC camera.

    Parameters
    ----------
    datetime : numpy.datetime64
        Input datetime.

    Returns
    -------
    str
        Formatted string WIC filename.
    """
    year = int(str(datetime.astype('datetime64[Y]')))
    DOY = (np.datetime64(datetime) - np.datetime64(f'{year}-01-01T00:00')).astype('timedelta64[D]').astype(int) + 1
    time = str(datetime).split('T')[-1]
    hour = int(time[:2])
    minute = int(time[3:5])
    return f'wic{year}{DOY:03d}{hour:02d}{minute:02d}.idl'

def wictime_to_datetime(wic):
    """
    Convert datetime to a string for a file name for the WIC camera.

    Parameters
    ----------
    wic : str
        wic filename.

    Returns
    -------
    numpy.datetime64
        The datetime corresponding to the WIC filename.
    """
    if wic.endswith('.idl'):
        end= '.idl'
        wic= wic.split('wic')[-1].split(end)[0]
    
    year= wic[:4]
    DOY= wic[4:7]
    hour= wic[7:9]
    minute= wic[9:11]
    return np.datetime64(f'{year}-01-01T{hour}:{minute}')+ np.timedelta64(int(DOY)-1, 'D')
def vistime_to_datetime(vis):
    """
    Convert datetime to a string for a file name for the vis camera.

    Parameters
    ----------
    vis : str
        vis filename.

    Returns
    -------
    numpy.datetime64
        The datetime corresponding to the vis filename.
    """
    vis= vis.split('vis')[-1].split('.idl')[0]
    year=vis[:4]
    DOY= vis[4:7]
    hour= vis[7:9]
    minute= vis[9:11]
    seconds= vis[11:13]
    return np.datetime64(f'{year}-01-01T{hour}:{minute}:{seconds}')+ np.timedelta64(int(DOY)-1, 'D')
#Following two functions are for show the mlt distrubtion, 
#local time is converted into such a way as to display the distribution across discontinuities such as 24 to 1
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

# What happens when an axis is clicked
def clicked(vis, image_axis, lt, lat, lt_axis=False, lat_axis=False, window_lt=1, window_lat=4):
    marker= image_axis.scatter(lat, lt, marker='+', color='black', zorder=100, s=500)
    vis.plotted.update({'marker': {'plot_object':[marker], 
                                        'clear_on_show_image':True,
                                        'clear_on_click': True,
                                        'linked_to_colorbar':False}})
    if lt_axis and lat_axis:
        cmap=image_axis.image.get_cmap()
        lt_lims= (lt-window_lt/2, lt+window_lt/2)
        image= image_axis.image_dat
        if lt_lims[0]>=0 and lt_lims[1]<24:
            image= image.where((image.lat>=lat-window_lat/2)\
                                                           &(image.lat<=lat+window_lat/2)\
                                                           &(image.lt>=lt-window_lt/2)\
                                                           &(image.lt<=lt+window_lt/2))
        elif lt_lims[0]<0:
            image= image.where((image.lat>=lat-window_lat/2)\
                                                           &(image.lat<=lat+window_lat/2)\
                                                           &((image.lt>=24+(lt-window_lt/2))\
                                                           |(image.lt<=lt+window_lt/2)))
        elif lt_lims[1]>24:
            image= image.where((image.lat>=lat-window_lat/2)\
                                                           &(image.lat<=lat+window_lat/2)\
                                                           &((image.lt>=(lt-window_lt/2))\
                                                           |(image.lt<=(lt+window_lt/2)-24)))
        window= image_axis.plot([lat+window_lat/2]*2 + [lat-window_lat/2]*2+ [lat+window_lat/2],
                                    [lt-window_lt/2, lt+window_lt/2, lt+window_lt/2, lt-window_lt/2, lt-window_lt/2],
                                    color='orange')
        vis.plotted.update({f'window_{image_axis.axis_number}': {'plot_object':window, 
                                                                 'clear_on_show_image':True,
                                                                 'clear_on_click': True,
                                                                 'linked_to_colorbar':False}})
        image= image.where(image.data!=0)
        ind= image.data== np.nanmax(image.data)
        lt_= np.nanmax(image.where(ind).lt.values)
        lat_= np.nanmax(image.where(ind).lat.values)

        crange= image_axis.image.get_clim()
        scatter= lat_axis.scatter(image.lat.values, image.data.values, 
                                  c=image.data.values, cmap=cmap, vmin= crange[0], vmax=crange[1])
        vis.plotted.update({'scatter_lat': {'plot_object':[scatter], 
                                                                 'clear_on_show_image':True,
                                                                 'clear_on_click': True,
                                                                 'linked_to_colorbar':image_axis.cax_number+1}})
        profile= lat_axis.plot([lat]*2, [np.nanmin(image.data), np.nanmax(image.data)], color='black')
        profile.extend(lat_axis.plot([np.round(lat_, 2)]*2, [np.nanmin(image.data), np.nanmax(image.data)],
                                       color='black', linestyle='--'))
        vis.plotted.update({'profile_lat': {'plot_object':profile, 
                                            'clear_on_show_image':True,
                                            'clear_on_click':True,
                                            'linked_to_colorbar':False}})
        try:
            lat_axis.set_ylim(np.nanmin(image.data)-abs(np.nanmin(image.data))*.1,
                               np.nanmax(image.data)+ abs(np.nanmax(image.data))*.1)
        except:
            pass
        lat_axis.set_xlim((lat-window_lat/2)-abs(lat-window_lat/2)*.01, (lat+window_lat/2)+abs(lat+window_lat/2)*.01)
        radians= mlt2radians(image.lt.values)
        xlims= radians2mlt(np.array([np.nanmin(radians), np.nanmax(radians)]))
        xlims[0]= np.floor(xlims[0]*10)/10
        xlims[1]= np.ceil(xlims[1]*10)/10
        try:
            lt_axis.set_xlim(*mlt2radians(xlims))
            lt_axis.set_ylim(np.nanmin(image.data)-abs(np.nanmin(image.data))*0.1, np.nanmax(image.data)+abs(np.nanmax(image.data))*.1)
        except:
            pass
        scatter= lt_axis.scatter(radians, image.data.values, c=image.data.values, 
                        cmap=cmap, vmin=crange[0], vmax=crange[1])
        vis.plotted.update({'scatter_lt': {'plot_object':[scatter], 
                                            'clear_on_show_image':True,
                                            'clear_on_click': True,
                                            'linked_to_colorbar':image_axis.cax_number+1}})
        profile= lt_axis.plot([mlt2radians(lt)]*2, [np.nanmin(image.data), np.nanmax(image.data)], color='black')

        profile.extend(lt_axis.plot([mlt2radians(np.round(lt_, 2))]*2, [np.nanmin(image.data), np.nanmax(image.data)],
                                  color='black', linestyle='--'))
        vis.plotted.update({'profile_lt': {'plot_object':profile,
                                           'clear_on_click': True,
                                           'clear_on_show_image':True,
                                           'linked_to_colorbar':False}})
        if lt_lims[0]<0:
            labels= np.append(np.arange(24-window_lt/4, np.round(lt_lims[0]+24, 1)-window_lt/4, -window_lt/4)[::-1], np.arange(0, np.round(lt_lims[1], 1)+window_lt/4, window_lt/4))
        elif lt_lims[1]>24:
            labels= np.append(np.arange(24-window_lt/4, np.round(lt_lims[0], 1)-window_lt/4, -window_lt/4)[::-1], np.arange(0, np.round(lt_lims[1]-24, 1)+window_lt/4, window_lt/4))
        else:
            labels= np.round(np.arange(np.round(lt_lims[0], 1), np.round(lt_lims[1], 1)+window_lt/4, window_lt/4), 2)
        rads=mlt2radians(labels)
        lt_axis.set_xticks(rads, labels)


    plt.draw()

class Visualise():
    """
    Class for visualizing polar coordinate data.

    Attributes
    ----------
    axes : list
        List of Polarplot axis objects.
    caxes : list
        List of AxesSubplot objects where the colorbar(s) go.
    MLTax : matplotlib AxesSubplot
        AxesSubplot object where MLT distributions can go when polar plots are clicked on for more detail.
    MLATax : matplotlib AxesSubplot
        AxesSubplot object where MLT distributions can go when polar plots are clicked on for more detail.
    cbars : list
        List to store colorbar objects associated with the data plotted.
    figure : matplotlib.figure.Figure
        Figure object.
    plotted : dict
        Dictionary to store plotted objects and their properties.
    click_function : function
        Function to handle click events. Function must accept the following argument at minimum.
        def click_function(Visualise, axis, lt, lat)
            -Visualise (Visualise object)
            -axis (polarsubplot where click was made)
            -lt (local time of click)
            -lat (latitude of click)
        if anything is plotted in the function it must be included following the style of this template:
        marker= image_axis.scatter(lt, lat)
        vis.plotted.update({'marker': {'plot_object':[marker],
                                    'clear_on_click': True,
                                    'clear_on_show_image':True,
                                    'linked_to_colorbar':False}})
                        
        addtional keys can be used if needed but 'plot_object', 'clear_on_click', 'clear_on_show_image' and 'linked_to_colorbar' must be included
        the first key 'marker' should be different for each new plot object and the 'plot_object' must iterable so that for p in plot_object: p.remove() functions correctly

        key explanations:
            'plot_object': object added to the subplot (must be iterable)
            'clear_on_click': boolean, if True when a click is made in a polar subplot this will be removed before the click function is run
            'clear_on_show_image': boolean, if True will be removed when show_image is run
            'linked_to_colorbar': boolean/int , if int colour scale with be rescaled when the colourbar is altered through the click interactions if linked to the corresponding colour bar
    click_kwargs : dict
        Additional keyword arguments for the click function.
    """
    def __init__(self, fig, axes, caxes, cax_association=False, hemispheres=False, click_function=clicked, **click_kwargs):
        """
        Initialise and configure the visualisation tool for interacting, analysing,
        and visualising data displayed in polar coordinates.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object where the visualisation will be displayed.
        axes : Polarplot axis or list of Polarplot axes
            A Polarplot object created by the polplot package or a list of Polarplot objects.
        caxes : matplotlib.axes.AxesSubplot or list of AxesSubplots
            The AxesSubplot object or a list of AxesSubplot objects where the colorbars are located.
        cax_association : list, optional
            A list of indices that associate the colorbar subplot with the polar plot.
            The same index can be used more than once when the colorbar is shared.
            The default is False.
        hemispheres : list, optional
            A list specifying the hemispheres for each polar plot.
            The default is False which becomes [1]*len(axes). -1 is used for the southern hemisphere
        click_function : function, optional
            The function to be executed when a polar plot is clicked.
            The default is `clicked`: clicked(vis, image_axis, lt, lat, lt_axis=False, lat_axis=False, window_lt=1, window_lat=4).
            For user defined functions vis (Visualise object), image_axis(polar subplot where click is made), lt (lt of click) and lat (lat of click) are required and will be provided automatically.
        **click_kwargs : dict, optional
            Additional keyword arguments passed to the click function.

        Returns
        -------
        None
        """
        if not isinstance(axes, (list, np.ndarray)):
            axes= [axes]
        if not isinstance(caxes, (list, np.ndarray)):
            caxes= [caxes]
        axes= np.asarray(axes)
        caxes= np.asarray(caxes)
        onclick_wrapper=functools.partial(self.onclick, axes, caxes)
        cid = fig.canvas.mpl_connect('button_press_event', onclick_wrapper)
        if not cax_association:
            cax_association= [0]*len(axes)
        if not hemispheres:
            hemispheres= [1]*len(axes)
        for ax, i, hemisphere, j in zip(axes, cax_association, hemispheres, range(len(axes))): 
            ax.cax_number=i
            ax.hemisphere= hemisphere
            ax.axis_number= j
        for ax in axes:
            ax.image= False
            ax.image_dat= False
        self.axes= axes
        self.caxes= caxes
        self.cbars= [False]*len(np.unique(cax_association))
        self.figure= fig
        self.plotted= {}
        self.click_function= click_function
        self.click_kwargs= click_kwargs
    def show_image(self, file, axis, crange=False, cmap=False, cbar_orientation=False, 
                in_put='img', lt_val='mlt', lat_val='mlat', date=0, title_y=-0.1):
        """
        Plot data onto polar plots and enable interactive features of the visualisation tool.
        Can be used iteratively to update plots, removing old data and updating colour bars as needed.

        Parameters
        ----------
        file : str or xarray.Dataset
            Path and filename of an IDL or xarray file, or an xarray object.
        axis : Polarplot axis
            The Polarplot axis on which the data will be plotted.
        crange : tuple, optional
            Colour bar range for the data. Default is False, which results in an automatic
            range based on the data or a previously set range.
        cmap : str, optional
            Name of the matplotlib colormap to use. Default is False, which uses the
            default matplotlib colormap.
        cbar_orientation : str, optional
            Orientation of the colour bar. Default is False, resulting in horizontal
            orientation or the previous setup if `show_image` was run before.
        in_put : str, optional
            Data identifier in the file. Default is 'img'.
        lt_val : str, optional
            Local time coordinate identifier in the file. Default is 'mlt'.
        lat_val : str, optional
            Latitude coordinate identifier in the file. Default is 'mlat'.
        date : str or datetime or int, optional
            Used when input is an IDL file. A string or datetime will select a matching
            date in the file, while an integer will select the date at that index (e.g., date=0
            selects the first date). Default is 0.
        title_y : float, optional
            Y-coordinate for the subplot title, which displays the date of the file.
            Default is -0.1.

        Returns
        -------
        Image : matplotlib.collections.PolyCollection
            The PolyCollection object corresponding to the plotted data.
        Colorbar : matplotlib.colorbar.Colorbar
            The colour bar object associated with the plotted data.
        """
        axis.ax.format_coord= axis._create_coordinate_formatter(lt_val, lat_val)
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
        axis.image_dat['lat']*=axis.hemisphere
        if axis.image:
            axis.image.remove()
            axis.image=False
            for key in list(self.plotted.keys()):
                if self.plotted[key]['clear_on_show_image']:
                    for p in self.plotted.pop(key)['plot_object']: p.remove()

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
            mappable= mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=crange[0], vmax=crange[1]), cmap=cmap)
            tick_positions= self.caxes[axis.cax_number].xaxis.get_ticks_position(), self.caxes[axis.cax_number].yaxis.get_ticks_position()
            label_positions= self.caxes[axis.cax_number].xaxis.get_label_position(), self.caxes[axis.cax_number].yaxis.get_label_position() 
            cbar=self.figure.colorbar(mappable, cax=cax, orientation= cbar_orientation)
            self.caxes[axis.cax_number].xaxis.set_ticks_position(tick_positions[0])
            self.caxes[axis.cax_number].yaxis.set_ticks_position(tick_positions[1])
            self.caxes[axis.cax_number].xaxis.set_label_position(label_positions[0])
            self.caxes[axis.cax_number].yaxis.set_label_position(label_positions[1])
            self.cbars[axis.cax_number]=cbar
        else:
            new_cbar=False
            if not crange:
                clims=[]
                for ax in self.axes:
                    if ax.image and ax.cax_number==axis.cax_number:
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
                if ax.cax_number==axis.cax_number and ax!=axis and ax.image:
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
                tick_positions= self.caxes[axis.cax_number].xaxis.get_ticks_position(), self.caxes[axis.cax_number].yaxis.get_ticks_position()
                label_positions= self.caxes[axis.cax_number].xaxis.get_label_position(), self.caxes[axis.cax_number].yaxis.get_label_position() 
                self.caxes[axis.cax_number].clear()
                self.caxes[axis.cax_number].get_xaxis().set_visible(True)
                self.caxes[axis.cax_number].get_yaxis().set_visible(True)
                mappable= mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=crange[0], vmax=crange[1]), cmap=cmap)
                cbar= self.figure.colorbar(mappable, orientation=cbar_orientation, cax=cax)
                self.cbars[axis.cax_number]=cbar
                cbar.set_label(label)
                self.caxes[axis.cax_number].xaxis.set_ticks_position(tick_positions[0])
                self.caxes[axis.cax_number].yaxis.set_ticks_position(tick_positions[1])
                self.caxes[axis.cax_number].xaxis.set_label_position(label_positions[0])
                self.caxes[axis.cax_number].yaxis.set_label_position(label_positions[1])
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
    def onclick(self, axes, caxes, event):
        """
        Handles interactions with axes and colour axes upon mouse clicks.
        Prepares data and triggers associated click functions.

        Parameters
        ----------
        axes : list
            List of Polarplot axes.
        caxes : list
            List of AxesSubplot objects where colour bars are placed.
        event : MouseEvent
            The mouse event triggering the function.

        Returns
        -------
        None
        """

        bools_ax= np.array([axis.ax.in_axes(event) for axis in axes])
        bools_cax= np.array([axis.in_axes(event) for axis in caxes])
        if any(bools_ax):
            # global lt, lat
            ix, iy= event.xdata, event.ydata
            # try:
            #     marker.remove()
                # for p in profile1+ profile2+ lines+window: p.remove()
            # except:
            #     pass
            for key in list(self.plotted.keys()):
                if self.plotted[key]['clear_on_click']:
                    for p in self.plotted.pop(key)['plot_object']: p.remove()
            lat, lt= np.array(axes)[bools_ax][0]._xy2latlt(ix, iy)
            self.click_function(self, np.array(axes)[bools_ax][0], lt, lat, **self.click_kwargs)
        elif any(bools_cax):
            ix, iy= event.xdata, event.ydata
            cax_number= np.argmax(bools_cax)
            cbar=self.cbars[cax_number]
            if cbar.orientation=='horizontal':
                crange=caxes[bools_cax][0].get_xlim()
                coord= ix
                label= cbar.ax.get_xlabel()
            elif cbar.orientation=='vertical':
                crange=caxes[bools_cax][0].get_ylim()
                coord= iy
                label= cbar.ax.get_ylabel()
            tick_positions= caxes[bools_cax][0].xaxis.get_ticks_position(), caxes[bools_cax][0].yaxis.get_ticks_position()
            label_positions= caxes[bools_cax][0].xaxis.get_label_position(), caxes[bools_cax][0].yaxis.get_label_position()             
            caxes[bools_cax][0].clear()

            if coord<sum(crange)/2:
                new_crange= round(coord, 0), crange[-1]
            elif coord>=sum(crange)/2:
                new_crange= crange[0], round(coord, 0)
            for ax in axes:
                if ax.cax_number== cax_number:
                    if new_crange== ax.image.get_clim():
                        continue
                    ax.image.set_clim(new_crange)
            for key in list(self.plotted.keys()):
                if self.plotted[key]['linked_to_colorbar'] and self.plotted[key]['linked_to_colorbar']==cax_number+1:
                    for p in self.plotted[key]['plot_object']: p.set_clim(new_crange)

            mappable= mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=new_crange[0], vmax=new_crange[1]), cmap=cbar.cmap)
            self.cbars[cax_number]= self.figure.colorbar(mappable, orientation=cbar.orientation, cax=caxes[bools_cax][0])
            cbar.set_label(label)
            caxes[bools_cax][0].xaxis.set_ticks_position(tick_positions[0])
            caxes[bools_cax][0].yaxis.set_ticks_position(tick_positions[1])
            caxes[bools_cax][0].xaxis.set_label_position(label_positions[0])
            caxes[bools_cax][0].yaxis.set_label_position(label_positions[1])

        plt.draw()
        return

if __name__=='__main__':
    import inspect
    folder= '/'.join(inspect.getfile(fuv).split('/')[:-2])+'/examples/sample_wicfiles/'
    default_functionality= True
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
    def new_click_function(vis, image_axis, lt, lat, cax_y=.5, cax_color='black', **cax_scatter_kwargs):
        """
        Executes a new click function for interactive visualisation.
        Calculates the closest data point to the clicked position and plots markers on the image and colour bar axes.

        Parameters
        ----------
        vis : Visualisation object
            The main visualisation object containing plotting methods and data.
        image_axis : Polarplot axis
            Axis where the image data is plotted.
        lt : float
            Local time coordinate of the clicked position.
        lat : float
            Latitude coordinate of the clicked position.
        cax_y : float, optional
            Y-coordinate for the colour bar marker. Default is 0.5.
        cax_color : str, optional
            Colour of the colour bar marker. Default is 'black'.
        **cax_scatter_kwargs : dict
            Additional keyword arguments for the colour bar scatter plot.

        Returns
        -------
        None
        """
        print('Doing new click function')
        dist = ((image_axis.image_dat.lt.values.flatten() - lt) ** 2 + 
                (image_axis.image_dat.lat.values.flatten() - lat) ** 2) ** .5
        ind = np.nanargmin(dist)
        cax_x = image_axis.image_dat.data.values.flatten()[ind]
        print(image_axis.image_dat.lat.values.flatten()[ind], image_axis.image_dat.lt.values.flatten()[ind])
        marker = image_axis.scatter(image_axis.image_dat.lat.values.flatten()[ind], 
                                image_axis.image_dat.lt.values.flatten()[ind], 
                                marker='+', color='black', zorder=100, s=500)
        vis.plotted.update({'marker': {'plot_object':[marker], 
                                        'clear_on_show_image':True,
                                        'clear_on_click': True,
                                        'linked_to_colorbar':False}})
        marker = vis.caxes[image_axis.cax_number].scatter(cax_x, cax_y, color=cax_color, **cax_scatter_kwargs)
        vis.plotted.update({'cax_marker': {'plot_object':[marker],
                                            'clear_on_show_image':True,
                                            'clear_on_click':True,
                                            'linked_to_colorbar':False}})
        plt.draw()

    if default_functionality:
        vis=Visualise(fig, np.asarray([ax, ax2]), np.asarray([cax, cax2]), cax_association=[0, 1], lt_axis=MLTax, lat_axis= MLATax)
    else:
        vis= Visualise(fig, axes=[ax, ax2], caxes=[cax, cax2], cax_association=[0, 1], click_function=new_click_function, cax_color='red', zorder=1000, marker='d')
    vis.show_image(file, ax, cmap='viridis_r', in_put='sza')
    xarray= fuv.read_idl(file2).isel(date=0)
    vis.show_image(xarray, ax2)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:54:16 2022

@author: aohma
"""


import os
import xarray as xr

import matplotlib.pyplot as plt

from polplot import pp,grids

# path = '/Volumes/Seagate Backup Plus Drive/fuv/wic/'
# ospath  = '/Volumes/Seagate\ Backup\ Plus\ Drive/fuv/wic/'

path = '/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/data/wic/'
ospath = '/Users/aohma/BCSS-DAG\ Dropbox/Anders\ Ohma/data/wic/'

ds = xr.load_dataset(path + 'sp_mlatmlt_order0_pos.nc')
ds2 = xr.load_dataset(path + 'sp_binnumber_order0_pos.nc')

grid,mltres = grids.sdarngrid(latmin=50)

for t in range(len(ds2.rind)):
    fig,ax = plt.subplots()
    pax = pp(ax)
    fc = pax.filled_cells(grid[0,:],grid[1,:],2,mltres,ds2.isel(rind=t)['median'].values,crange=(0,600))
    plt.colorbar(fc)
    
    ax.set_title('rind = ' + str(ds2.rind[t].values),y=1.0)
    
    plt.savefig(path + 'fig/temp/equalarea'+str(t).zfill(4)+'.png',bbox_inches='tight',dpi=150)
    plt.clf()
    plt.close()
    
    fig,ax = plt.subplots(figsize=(7,6))
    ds.isel(rind=t)['median'].plot(vmin=0,vmax=1000)
    
    plt.savefig(path + 'fig/temp/mlatmlt'+str(t).zfill(4)+'.png',bbox_inches='tight',dpi=150)
    plt.clf()
    plt.close()

os.system('convert '+ospath+'fig/temp/equalarea*.png '+ospath+'fig/equalarea.gif')
os.system('convert '+ospath+'fig/temp/mlatmlt*.png '+ospath+'fig/mlatmlat.gif')
os.system('rm '+ospath+'fig/temp/*.png')

fig,ax = plt.subplots()
ds.sel(rlat=slice(-10,10))['median'].mean(dim='rlat').plot(x='rlt',y='rind',vmin=0,vmax=600)
plt.savefig(path + 'fig/timemlt.png',bbox_inches='tight',dpi=150)
plt.clf()
plt.close()
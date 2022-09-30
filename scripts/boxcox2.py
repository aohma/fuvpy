#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:13:13 2022

@author: aohma
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy import stats

wic = xr.open_dataset('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/projects/aurora_dayglow/fuv_2000-08-28-1st/wic.nc')
s12 = xr.open_dataset('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/projects/aurora_dayglow/fuv_2000-08-28-1st/s12b.nc')
s13 = xr.open_dataset('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/projects/aurora_dayglow/fuv_2000-08-28-1st/s13b.nc')

lmbdas = np.arange(-2,2.1,0.25)

## WIC ##
frac = np.cos(np.deg2rad(wic.sza))/np.cos(np.deg2rad(wic.dza))
ind = (wic.img>=0)&(frac>=0)&(frac<=4)& (wic.dza <= 80) & wic.bad #& ((wic.mlat<60)|(wic.mlat>80))
datalist = [wic.img.values[ind]]
szalist = [frac.values[ind]]

frac = np.cos(np.deg2rad(s12.sza))/np.cos(np.deg2rad(s12.dza))
ind = (s12.img>=0)&(frac>=0)&(frac<=4)&(s12.dza <= 80) & s12.bad #& ((s12.mlat<60)|(s12.mlat>80))
datalist.append(s12.img.values[ind])
szalist.append(frac.values[ind])

frac = np.cos(np.deg2rad(s13.sza))/np.cos(np.deg2rad(s13.dza))
ind = (s13.img>=0)&(frac>=0)&(frac<=4)&(s13.dza <= 80) & s13.bad #& ((s13.mlat<60)|(s13.mlat>80))
datalist.append(s13.img.values[ind])
szalist.append(frac.values[ind])

ws=[]
for ii,data in enumerate(datalist):
    # data=(data-np.quantile(data,0.01))/(np.quantile(data,0.99)-np.quantile(data,0.01))
    s = np.histogram(szalist[ii],bins=np.linspace(0,4,41))[0]
    ind = np.digitize(szalist[ii], bins=np.linspace(0,4,41))
    w = 1+0/s[ind-1]
    x = np.histogram(data,bins=np.linspace(np.quantile(data,0.01),np.quantile(data,0.99),21),weights=w)[0]
    ws.append(w)
    
    llf = np.zeros(lmbdas.shape, dtype=float)
    for ii, lmbda in enumerate(lmbdas):
        llf[ii] = stats.boxcox_llf(lmbda, x)
    
    x_most_normal, lmbda_optimal = stats.boxcox(x)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lmbdas, llf, 'b.-')
    ax.axhline(stats.boxcox_llf(lmbda_optimal, x), color='r')
    
    ax.set_xlabel('lmbda parameter')
    
    ax.set_ylabel('Box-Cox log-likelihood')
    
    
    locs = [3, 10, 4]  # 'lower left', 'center', 'lower right'
    
    for lmbda, loc in zip([-1, lmbda_optimal, 1], locs):
        xt = stats.boxcox(x, lmbda=lmbda)
        (osm, osr), (slope, intercept, r_sq) = stats.probplot(xt)
        ax_inset = inset_axes(ax, width="20%", height="20%", loc=loc)
        ax_inset.plot(osm, osr, 'c.', osm, slope*osm + intercept, 'k-')
        ax_inset.set_xticklabels([])
        ax_inset.set_yticklabels([])
        ax_inset.set_title(r'$\lambda=%1.2f$' % lmbda)



# wicprob = stats.boxcox_normplot(data/np.max(data), -5,5,N=41)
# _, wicmaxlog,wicconf = stats.boxcox(data/np.max(data),alpha=0.95)


# ind = (s12.img>0)&(s12.sza>=0)& (s12.dza <= 80) & s12.bad #& ((s12.mlat<60)|(s12.mlat>80))
# data = s12.img.values[ind]
# s12prob = stats.boxcox_normplot(data/np.max(data), -5,5,N=41)
# _, s12maxlog = stats.boxcox(data/np.max(data))

# ind = (s13.img>0)&(s13.sza>=0)& (s13.dza <= 80) & s13.bad #& ((s13.mlat<60)|(s13.mlat>80))
# data = s13.img.values[ind]
# s13prob = stats.boxcox_normplot(data/np.max(data), -5,5,N=41)
# _, s13maxlog = stats.boxcox(data/np.max(data))

# fig,axs = plt.subplots(1,3)
# axs[0].plot(wicprob[0],wicprob[1])
# axs[0].axvline(wicmaxlog, color='r')
# axs[1].plot(s12prob[0],s12prob[1])
# axs[1].axvline(s12maxlog, color='r')
# axs[2].plot(s13prob[0],s13prob[1])
# axs[2].axvline(s13maxlog, color='r')

for ii,data in enumerate(datalist):
    fig,axs = plt.subplots(1,3)
    axs[0].hist(data,bins=np.linspace(np.quantile(data,0.01),np.quantile(data,0.99),21),weights=ws[ii])
    axs[1].hist(np.sqrt(data),bins=np.linspace(np.quantile(np.sqrt(data),0.01),np.quantile(np.sqrt(data),0.99),21),weights=ws[ii])
    axs[2].hist(np.log(data),bins=np.linspace(np.quantile(np.log(data),0.01),np.quantile(np.log(data),0.99),21),weights=ws[ii])


    
    
    
    
    
    
    
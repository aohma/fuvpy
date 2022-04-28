#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:18:55 2022

@author: aohma
"""
import glob

import fuvpy.fuvpy as fuv


events = glob.glob('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/data/fuv200012/north/*')
events.sort()

for e in events[2:3]:
    wicfiles = glob.glob(e+'/*.idl')   
    wic = fuv.readFUVimage(wicfiles)
    wic = fuv.makeFUVdayglowModel(wic,transform='log')
    wic = fuv.makeFUVshModel(wic,4,4)
    
 
    # wic.to_netcdf('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/data/fuv200012/wic_'+e[63:]+'.nc')
# events = glob.glob('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/data/fuv200012/north/*')
# events.sort()

# for e in events:
#     wicfiles = glob.glob(e+'/*.idl')   
#     wic = readFUVimage(wicfiles)
#     wic = makeFUVdayglowModelC(wic,transform='log')
#     wic = makeFUVshModelNew(wic,4,4)
    
#     wic.to_netcdf('/Users/aohma/BCSS-DAG Dropbox/Anders Ohma/data/fuv200012/wic_'+e[63:]+'.nc')

# wicfiles = glob.glob('/Users/aohma/BCSS-DAG Dropbox/TMP/lompe_tutorial_2022-02-20/simon_event/wic_data/*.idl')   
# wic = readFUVimage(wicfiles)
# wic = makeFUVdayglowModelC(wic,transform='log')
# wic = makeFUVshModelNew(wic,4,4)
# wic.to_netcdf('/Users/aohma/BCSS-DAG Dropbox/TMP/lompe_tutorial_2022-02-20/simon_event/wic_20020519.nc')


# Field to include:
# ind,date,row,col,mlat,mlt,hemisphere,img,dgimg,shimg,onset,rind 

# HDD events

# events = glob.glob('/Volumes/Seagate Backup Plus Drive/fuv/wic/idl/north/*')
# events.sort()

# for e in events:
#     wicfiles = glob.glob(e+'/*.idl')   
#     wic = readFUVimage(wicfiles)
#     wic = makeFUVdayglowModelC(wic,transform='log')
#     wic = makeFUVshModelNew(wic,4,4)
    
#     wic.to_netcdf('/Volumes/Seagate Backup Plus Drive/fuv/wic/nc/wic_'+e[53:]+'.nc')

# HDD events s12

# events = glob.glob('/Volumes/Seagate Backup Plus Drive/fuv/s12/idl/north/*')
# events.sort()

# for e in events:
#     wicfiles = glob.glob(e+'/*.idl')   
#     wic = readFUVimage(wicfiles)
#     wic = makeFUVdayglowModelC(wic)
#     wic['shmodel'] = wic['dgmodel']
#     wic['shimg'] = wic['dgimg']
#     wic['shweight'] = wic['dgweight']
    
#     wic.to_netcdf('/Volumes/Seagate Backup Plus Drive/fuv/s12/nc/s12_'+e[53:]+'.nc')
import xarray as xr
from fuvpy.src.background import backgroundmodel_BS,backgroundmodel_SH
from fuvpy.src.boundaries import boundarymodel_BS,boundarymodel_F,detect_boundaries
from fuvpy.src.plotting import plotimg,plot_backgroundmodel_BS,pplot,plot_ltlat,plotboundaries

@xr.register_dataset_accessor("fuv")

class FuvAccessor:

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    # Accessors to background models
    def backgroundmodel_BS(self,**kwargs):
        inplace = bool(kwargs.pop('inplace')) if 'inplace' in kwargs.keys() else False
        if inplace:
            backgroundmodel_BS(self._obj,inplace=inplace,**kwargs)
        else:
            return backgroundmodel_BS(self._obj,**kwargs)
        
    def backgroundmodel_SH(self,Nsh,Msh,**kwargs):
        inplace = bool(kwargs.pop('inplace')) if 'inplace' in kwargs.keys() else False
        if inplace:
            backgroundmodel_SH(self._obj,Nsh,Msh,inplace=inplace,**kwargs)
        else:
            return backgroundmodel_SH(self._obj,Nsh,Msh,**kwargs)

    # Accessors to boundary models
    def detect_boundaries(self,**kwargs):
        return detect_boundaries(self._obj,**kwargs)

    def boundarymodel_BS(self,**kwargs):
        return boundarymodel_BS(self._obj,**kwargs)

    def boundarymodel_F(self,**kwargs):
        return boundarymodel_F(self._obj,**kwargs)
    
    # Accessors to plot functions
    def plotimg(self,inImg,**kwargs):
        return plotimg(self._obj,inImg,**kwargs)

    def plot_backgroundmodel_BS(self,**kwargs):
        plot_backgroundmodel_BS(self._obj,**kwargs)

    def pplot(self,inImg,**kwargs):
        pplot(self._obj,inImg,**kwargs)

    def plot_ltlat(self,inImg,**kwargs):
        return plot_ltlat(self._obj,inImg,**kwargs)

    def plotboundaries(self,boundary,**kwargs):
        plotboundaries(self._obj,boundary,**kwargs)   
    
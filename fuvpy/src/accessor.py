import xarray as xr
from fuvpy.src.background import backgroundmodel_BS,backgroundmodel_SH
from fuvpy.src.boundaries import boundarymodel_BS,boundarymodel_F,detect_boundaries

@xr.register_dataset_accessor("fuv")

class FuvAccessor:

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

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

    def detect_boundaries(self,**kwargs):
        return detect_boundaries(self._obj,**kwargs)

    def boundarymodel_BS(self,**kwargs):
        return boundarymodel_BS(self._obj,**kwargs)

    def boundarymodel_F(self,**kwargs):
        return boundarymodel_F(self._obj,**kwargs)   
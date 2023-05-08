import xarray as xr
from fuvpy.src.background import backgroundmodel_BS,backgroundmodel_SH

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
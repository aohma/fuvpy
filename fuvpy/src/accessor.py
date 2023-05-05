import xarray as xr
from fuvpy.src.background import makeBSmodel,makeSHmodel

@xr.register_dataset_accessor("fuv")

class FuvAccessor:

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def makeBSmodel(self,**kwargs):
        inplace = bool(kwargs.pop('inplace')) if 'inplace' in kwargs.keys() else False
        if inplace:
            makeBSmodel(self._obj,inplace=inplace,**kwargs)
        else:
            return makeBSmodel(self._obj,**kwargs)
        
    def makeSHmodel(self,Nsh,Msh,**kwargs):
        inplace = bool(kwargs.pop('inplace')) if 'inplace' in kwargs.keys() else False
        if inplace:
            makeSHmodel(self._obj,Nsh,Msh,inplace=inplace,**kwargs)
        else:
            return makeSHmodel(self._obj,Nsh,Msh,**kwargs)        
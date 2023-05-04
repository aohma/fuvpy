Overview
========

The fuvpy package contains code to work with global Far-Ultraviolet (FUV) images of aurora.
At present, the package includes functions to remove background emissions and to visualize images.

The code is based on xarray, using xarray.Dataset objects to store the images.
Functions to read .idl and .sav files produced by fuview3 and xvis (IMAGE and Polar mission software) into this format are provided.
Other formats need to be converted into a xarray.Dataset with coordinates ['date','row','col'] and data_vars = ['img','mlat','mlon','mlt','glat','glon','dza','sza'].
Here, 'date' is the datetime of the images and 'row' and 'col' refer to the row and column of the camera (2D images).
The data_vars are thus 3D arrays:

- 'img' is the measured intensity of each pixel
- 'mlat', 'mlon' and 'mlt' are the magnetic latitude, longitude and local time of each pixel, respectively
- 'glat' and 'glon' are the geographic latitude and longitude of each pixel, respectively
- 'dza' is the viewing angle of each pixel
- 'sza' is the solar zenith angle of each pixel

If needed, the magnetic coordinates can be calculated from the geographic coordinates (or vice versa) using e.g. apexpy (pip install apexpy).
The solar zenith angle can be calculated based on the subsolar point, which is found using fuvpy.subsol()


Dependencies
============
fuvpy depends on the following:

- matplotlib
- netcdf4
- numpy
- pandas
- scipy
- xarray
- polplot (github/klaundal/polplot)

You should also have git version >= 2.13

Install
=======
Clone the repository::

    git clone https://github.com/aohma/fuvpy

Use or create a virtual environment with ``conda`` with all dependencies installed.
A working environment is provided in the repository. In this example, the environment is called ``fuvpy-env``::

    conda env create --name fuvpy-env --file fuvpy/binder/fuvpy-env.yml
    conda activate fuvpy-env

Use pip to install in editable (development) mode:
    pip install -e ./fuvpy

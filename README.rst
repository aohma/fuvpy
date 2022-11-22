Overview
========

The fuvpy package contains code to work with global Far-Ultraviolet (FUV) images of aurora.
The package uses xarray.Dataset to store the images.
At present, the package includes functions to remove background emissions and to visualize the images.
Functions to read .idl and .sav files produced by the IMAGE and Polar missions
Lompe is a tool for estimating regional maps of ionospheric electrodynamics using measurements of plasma convection and magnetic field disturbances in space and on ground.

We recommend to use the examples to learn how to use Lompe, but the general workflow is like this:

.. code-block:: python

    >>> # prepare datasets (as many as you have - see lompe.Data doc string for how to format)
    >>> my_data1 = lompe.Data(*data1)
    >>> my_data2 = lompe.Data(*data2)
    >>> # set up grid (the parameters depend on your region, target resoultion etc):
    >>> grid = lompe.cs.CSgrid(lompe.cs.CSprojection(*projectionparams), *gridparams)
    >>> # initialize model with grid and functions to calculate Hall and Pedersen conductance
    >>> # The Hall and Pedersen functions should take (lon, lat) as parameters
    >>> model = lompe.Emodel(grid, (Hall_function, Pedersen_function))
    >>> # add data:
    >>> model.add_data(my_data1, my_data2)
    >>> # run inversion
    >>> model.run_inversion()
    >>> # now the map is ready, and we can plot plasma flows, currents, magnetic fields, ...
    >>> model.lompeplot()
    >>> # or calculate some quantity, like plasma velocity:
    >>> ve, vn = model.v(mylon, mylat)


Dependencies
============
fuvpy depends on the following:

- matplotlib
- netcdf4
- numpy
- pandas
- scipy
- xarray

You should also have git version >= 2.13

Install
=======
Start by cloning the repository, and get the code for the submodules::

    git clone https://github.com/aohma/fuvpy
    cd fuvpy
    git submodule init
    git submodule update

Then there are two options for how to make it possible to import and use lompe from any location. The first option should be fairly automatic, while the other option requires you to install all the dependencies yourself.

Option 1
--------
The first option is to create a virtual environment with ``conda``, where all dependencies are automatically installed. In this example, the environment is called ``lompe_env``::

    conda env create --name fuvpy-env --file binder/fuvpy-env.yml
    conda activate fuvpy-env
    pip install -e .

Option 2
--------
The second option is to add the path to the lompe folder to PYTHONPATH, and make sure that all dependencies are installed. This option may be good if you plan to make changes to the lompe code and have the changes available immediately.

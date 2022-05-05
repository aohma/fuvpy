""" tools that are useful for spherical harmonic analysis

    SHkeys       -- class to contain n and m - the indices of the spherical harmonic terms
    nterms       -- function which calculates the number of terms in a
                    real expansion of a poloidal (internal + external) and toroidal expansion
    get_legendre -- calculate associated legendre functions - with option for Schmidt semi-normalization
"""
import numpy as np
#from pyamps.mlt_utils import mlon_to_mlt
d2r = np.pi/180

class SHkeys(object):

    def __init__(self, Nmax, Mmax):
        """ container for n and m in spherical harmonics

            keys = SHkeys(Nmax, Mmax)

            keys will behave as a tuple of tuples, more or less
            keys['n'] will return a list of the n's
            keys['m'] will return a list of the m's
            keys[3] will return the fourth n,m tuple

            keys is also iterable

        """

        self.Nmax = Nmax
        self.Mmax = Mmax
        keys = []
        for n in range(self.Nmax + 1):
            for m in range(self.Mmax + 1):
                keys.append((n, m))

        self.keys = tuple(keys)
        self.make_arrays()

    def __getitem__(self, index):
        if index == 'n':
            return [key[0] for key in self.keys]
        if index == 'm':
            return [key[1] for key in self.keys]

        return self.keys[index]

    def __iter__(self):
        for key in self.keys:
            yield key

    def __len__(self):
        return len(self.keys)

    def __repr__(self):
        return ''.join(['n, m\n'] + [str(key)[1:-1] + '\n' for key in self.keys])[:-1]

    def __str__(self):
        return ''.join(['n, m\n'] + [str(key)[1:-1] + '\n' for key in self.keys])[:-1]

    def setNmin(self, nmin):
        """ set minimum n """
        self.keys = tuple([key for key in self.keys if key[0] >= nmin])
        self.make_arrays()
        return self

    def MleN(self):
        """ set m <= n """
        self.keys = tuple([key for key in self.keys if abs(key[1]) <= key[0]])
        self.make_arrays()
        return self

    def Mge(self, limit):
        """ set m <= n """
        self.keys = tuple([key for key in self.keys if abs(key[1]) >= limit])
        self.make_arrays()
        return self

    def NminusModd(self):
        """ remove keys if n - m is even """
        self.keys = tuple([key for key in self.keys if (key[0] - abs(key[1])) % 2 == 1])
        self.make_arrays()
        return self

    def NminusMeven(self):
        """ remove keys if n - m is odd """
        self.keys = tuple([key for key in self.keys if (key[0] - abs(key[1])) % 2 == 0])
        self.make_arrays()
        return self

    def negative_m(self):
        """ add negative m to the keys """
        keys = []
        for key in self.keys:
            keys.append(key)
            if key[1] != 0:
                keys.append((key[0], -key[1]))

        self.keys = tuple(keys)
        self.make_arrays()

        return self


    def make_arrays(self):
        """ prepare arrays with shape ( 1, len(keys) )
            these are used when making G matrices
        """

        if len(self) > 0:
            self.m = np.array(self)[:, 1][np.newaxis, :]
            self.n = np.array(self)[:, 0][np.newaxis, :]
        else:
            self.m = np.array([])[np.newaxis, :]
            self.n = np.array([])[np.newaxis, :]



def nterms(NT = 0, MT = 0, NVi = 0, MVi = 0, NVe = 0, MVe = 0):
    """ return number of coefficients in an expansion in real spherical harmonics of
        toroidal magnetic potential truncated at NT, MT
        poloidal magnetic potential truncated at NVi, MVi for internal sources
        poloidal magnetic potential truncated at NVe, MVe for external sources
    """

    return len(SHkeys(NT , MT ).setNmin(1).MleN().Mge(0)) + \
           len(SHkeys(NT , MT ).setNmin(1).MleN().Mge(1)) + \
           len(SHkeys(NVe, MVe).setNmin(1).MleN().Mge(0)) + \
           len(SHkeys(NVe, MVe).setNmin(1).MleN().Mge(1)) + \
           len(SHkeys(NVi, MVi).setNmin(1).MleN().Mge(0)) + \
           len(SHkeys(NVi, MVi).setNmin(1).MleN().Mge(1))



def get_legendre(nmax, mmax, theta, schmidtnormalize = True, negative_m = False, minlat = 0, keys = None):
    """ calculate associated Legendre functions

        nmax             -- maximum total wavenumber
        mmax             -- maximum zonal wavenumber
        theta            -- colatitude in degrees (not latitude!), with N terms
        schmidtnormalize -- True if Schmidth seminormalization is wanted, False otherwise
        negative_m       -- True if you want the functions for negative m (complex expansion)
        keys             -- pass SHkeys object to return an array instead of a dict
                            default None


        returns:
          P, dP -- dicts of legendre functions, and derivatives, with wavenumber tuple as keys

        or, if keys != None:
          PdP   -- array of size N, 2*M, where M is the number of terms. The first half of the
                   columns are P, and the second half are dP


        algorithm from "Spacecraft Attitude Determination and Control" by James Richard Wertz

        could be unstable for large nmax...

        KML 2016-04-22

    """

    theta = theta.flatten()[:, np.newaxis]

    P = {}
    dP = {}
    sinth = np.sin(d2r*theta)
    costh = np.cos(d2r*theta)

    if schmidtnormalize:
        S = {}
        S[0, 0] = 1.

    # initialize the functions:
    for n in range(nmax +1):
        for m in range(nmax + 1):
            P[n, m] = np.zeros_like(theta, dtype = np.float64)
            dP[n, m] = np.zeros_like(theta, dtype = np.float64)

    P[0, 0] = np.ones_like(theta, dtype = np.float64)
    P[0, 0][np.abs(90 - theta) < minlat] = 0
    for n in range(1, nmax +1):
        for m in range(0, min([n + 1, mmax + 1])):
            # do the legendre polynomials and derivatives
            if n == m:
                P[n, n]  = sinth * P[n - 1, m - 1]
                dP[n, n] = sinth * dP[n - 1, m - 1] + costh * P[n - 1, n - 1]
            else:

                if n == 1:
                    Knm = 0.
                    P[n, m]  = costh * P[n -1, m]
                    dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m]

                elif n > 1:
                    Knm = ((n - 1)**2 - m**2) / ((2*n - 1)*(2*n - 3))
                    P[n, m]  = costh * P[n -1, m] - Knm*P[n - 2, m]
                    dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m] - Knm * dP[n - 2, m]

            if schmidtnormalize:
                # compute Schmidt normalization
                if m == 0:
                    S[n, 0] = S[n - 1, 0] * (2.*n - 1)/n
                else:
                    S[n, m] = S[n, m - 1] * np.sqrt((n - m + 1)*(int(m == 1) + 1.)/(n + m))


    if schmidtnormalize:
        # now apply Schmidt normalization
        for n in range(1, nmax + 1):
            for m in range(0, min([n + 1, mmax + 1])):
                P[n, m]  *= S[n, m]
                dP[n, m] *= S[n, m]

    if negative_m:
        for n  in range(1, nmax + 1):
            for m in range(0, min([n + 1, mmax + 1])):
                P[n, -m]  = -1.**(-m) * factorial(n-m)/factorial(n+m) *  P[n, m]
                dP[n, -m] = -1.**(-m) * factorial(n-m)/factorial(n+m) * dP[n, m]


    if keys is None:
        return P, dP
    else:
        Pmat  = np.hstack(tuple(P[key] for key in keys))
        dPmat = np.hstack(tuple(dP[key] for key in keys))

        return np.hstack((Pmat, dPmat))


def getfield(dates, glat, glon, r, coeffs, sp_order = 1):
    """ compute magnetic field components at given dates, at given geocentric latitude and longitude:

        dates: python datetime(s). Can be an array (shape N, dim <= 1)
        glat:  geocentric latitude(s). Can be an array (shape M, dim <= 1)
        glon: geocentric longitude(s). Can be an array (shape M, dim <= 1)
        r: geocentric radius in km. Can be array (shape M, dim <= 1)
        coeffs: either tuple containing g and, h, or path to shc-file
        sp_order: order of the spline interpolation (DOESN'T WORK)

        returns: X, Y, Z (north, east, down)
                 these are arrays with shape (N, M), where N is time and M is position

    """

    if len(coeffs) == 2:
        g, h = coeffs
    else:
        g, h = read_shc(coeffs)

    # convert input to arrays in case they aren't
    glat, glon, r = map(lambda x: np.array(x, ndmin = 1), [glat, glon, r])

    # dimension tests:
    if np.ndim(glat) > 1: raise ValueError('glat must have dimension <= 1')
    if np.ndim(glon) > 1: raise ValueError('glon must have dimension <= 1')
    if np.ndim(r)    > 1: raise ValueError(   'r must have dimension <= 1')

    nglat = reduce(lambda x, y: x*y, glat.shape)
    nglon = reduce(lambda x, y: x*y, glon.shape)
    nr    = reduce(lambda x, y: x*y,    r.shape)

    glat = glat.reshape((1, nglat))
    glon = glon.reshape((1, nglon))
    r    =    r.reshape((1, nr   ))

    # check that coordinates have consistent shapes (similar or scalar):
    if not ( (nglat == nglon or nglon == 1 or nglat == 1) and
             (nglat == nr    or nr == 1 or nglat == 1) and
             (nglon == nr    or nr == 1 or nglon == 1)):
        raise ValueError('glat, glon and r must have same number of elements, or only 1')

    # if only one coordinate is passed, convert to scalar:
    if nglat == 1:
        glat = glat[0]
    if nglon == 1:
        glon = glon[0]
    if nr == 1:
        r    = r[0]





    # get maximum N and maximum M:
    N = np.max([k[0] for k in g.keys()])
    M = np.max([k[1] for k in g.keys()])


    # get the legendre functions
    P, dP = get_legendre(N, M, 90 - glat, schmidtnormalize = True)


    # Append coefficients at desired times (skip if index is already in coefficient data frame):
    index = list(g.index) + list(dates)
    g = g.reindex(index).sort_index().groupby(index).first() # reindex and skip duplicates
    h = h.reindex(index).sort_index().groupby(index).first() # reindex and skip duplicates

    # interpolate and collect the coefficients at desired times:
    g = g.interpolate(method = 'time').ix[dates]
    h = h.interpolate(method = 'time').ix[dates]

    if np.any(pd.isnull(g)) or np.any(pd.isnull(h)):
        raise Exception('Invalid values in coefficients - interpolation outside range?')

    # compute cosmlon and sinmlon:
    cosmlon = {k[1]:np.cos(k[1]*glon*d2r) for k in g.keys()}
    sinmlon = {k[1]:np.sin(k[1]*glon*d2r) for k in g.keys()}

    # compute the field values (key = (n, m))
    # down
    Z =  -np.array(reduce(lambda x, y: x + y,
                       (  P[key] * (REFRE/r)**(key[0] + 2) * (key[0] + 1) *
                        (  g[key].values[:, np.newaxis] * cosmlon[key[1]]
                         + h[key].values[:, np.newaxis] * sinmlon[key[1]])
                        for key in g.keys())))
    # north
    X =  np.array(reduce(lambda x, y: x + y,
                       (  dP[key] * (REFRE/r)**(key[0] + 1) *
                        (  g[key].values[:, np.newaxis] * cosmlon[key[1]]
                         + h[key].values[:, np.newaxis] * sinmlon[key[1]])
                        for key in g.keys())))*REFRE/r
    # east
    Y =  np.array(reduce(lambda x, y: x + y,
                       (   P[key] * (REFRE/r)**(key[0] + 1) *
                        (  g[key].values[:, np.newaxis] * sinmlon[key[1]] * key[1]
                         - h[key].values[:, np.newaxis] * cosmlon[key[1]] * key[1])
                        for key in g.keys()))) / np.cos(glat*d2r) * REFRE/r

    return X, Y, Z

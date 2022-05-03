import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from pysymmetry.utils import dates,spherical


""" function for computing subsolar point """
def subsol(datetimes):
    """ 
    calculate subsolar point at given datetime(s)

    returns:
      subsol_lat  -- latitude(s) of the subsolar point
      subsol_lon  -- longiutde(s) of the subsolar point

    The code is vectorized, so it should be fast.

    After Fortran code by: 961026 A. D. Richmond, NCAR

    Documentation from original code:
    Find subsolar geographic latitude and longitude from date and time.
    Based on formulas in Astronomical Almanac for the year 1996, p. C24.
    (U.S. Government Printing Office, 1994).
    Usable for years 1601-2100, inclusive.  According to the Almanac, 
    results are good to at least 0.01 degree latitude and 0.025 degree 
    longitude between years 1950 and 2050.  Accuracy for other years 
    has not been tested.  Every day is assumed to have exactly
    86400 seconds; thus leap seconds that sometimes occur on December
    31 are ignored:  their effect is below the accuracy threshold of
    the algorithm.

    Added by SMH 2020/04/03 (from Kalle's code stores!)
    """

    # use pandas DatetimeIndex for fast access to year, month day etc...
    if hasattr(datetimes, '__iter__'): 
        datetimes = pd.DatetimeIndex(datetimes)
    else:
        datetimes = pd.DatetimeIndex([datetimes])

    year = np.array(datetimes.year)
    # day of year:
    doy  = dates.date_to_doy(datetimes.month, datetimes.day, dates.is_leapyear(year))
    # seconds since start of day:
    ut   = datetimes.hour * 60.**2 + datetimes.minute*60. + datetimes.second 
 
    yr = year - 2000

    if year.max() >= 2100 or year.min() <= 1600:
        raise ValueError('subsol.py: subsol invalid after 2100 and before 1600')

    nleap = np.floor((year-1601)/4.)
    nleap = nleap - 99

    # exception for years <= 1900:
    ncent = np.floor((year-1601)/100.)
    ncent = 3 - ncent
    nleap[year <= 1900] = nleap[year <= 1900] + ncent[year <= 1900]

    l0 = -79.549 + (-.238699*(yr-4*nleap) + 3.08514e-2*nleap)

    g0 = -2.472 + (-.2558905*(yr-4*nleap) - 3.79617e-2*nleap)

    # Days (including fraction) since 12 UT on January 1 of IYR:
    df = (ut/86400. - 1.5) + doy

    # Addition to Mean longitude of Sun since January 1 of IYR:
    lf = .9856474*df

    # Addition to Mean anomaly since January 1 of IYR:
    gf = .9856003*df

    # Mean longitude of Sun:
    l = l0 + lf

    # Mean anomaly:
    g = g0 + gf
    grad = g*np.pi/180.

    # Ecliptic longitude:
    lmbda = l + 1.915*np.sin(grad) + .020*np.sin(2.*grad)
    lmrad = lmbda*np.pi/180.
    sinlm = np.sin(lmrad)

    # Days (including fraction) since 12 UT on January 1 of 2000:
    n = df + 365.*yr + nleap

    # Obliquity of ecliptic:
    epsilon = 23.439 - 4.e-7*n
    epsrad  = epsilon*np.pi/180.

    # Right ascension:
    alpha = np.arctan2(np.cos(epsrad)*sinlm, np.cos(lmrad)) * 180./np.pi

    # Declination:
    delta = np.arcsin(np.sin(epsrad)*sinlm) * 180./np.pi

    # Subsolar latitude:
    sbsllat = delta

    # Equation of time (degrees):
    etdeg = l - alpha
    nrot = np.round(etdeg/360.)
    etdeg = etdeg - 360.*nrot

    # Apparent time (degrees):
    aptime = ut/240. + etdeg    # Earth rotates one degree every 240 s.

    # Subsolar longitude:
    sbsllon = 180. - aptime
    nrot = np.round(sbsllon/360.)
    sbsllon = sbsllon - 360.*nrot

    return sbsllat, sbsllon

def sza(glat, glon, datetimes, degrees = True):
    """ calculate solar zenith angle at given latitude (not colat), longitude and datetimes

        handles arrays, but does not preserve shape - a flat output is returned

        the following combinations are possible:

        1) glat, glon arrays of same size and datetimes scalar
           output will be array with same size as glat, glon
        2) glat, glon, datetimes arrays of same size
           output will be array with same size as glat, glon, datetimes
        3) glat, glon scalar and datetimes array
           output will be array with same size as datetimes


        Spherical geometry is assumed

    Added by SMH 2020/04/03 (from Kalle's code stores!)
    """

    glat = np.array(glat, ndmin = 1).flatten() # turn into array and flatten
    glon = np.array(glon, ndmin = 1).flatten() # turn into array and flatten

    if glat.size != glon.size:
        raise ValueError('sza: glat and glon arrays but not of same size')

    if hasattr(datetimes, '__iter__'):
        if len(datetimes) != len(glat) and len(glat) != 1:
            raise ValueError('sza: inconsistent input size')

    if degrees:
        conv = 180/np.pi
    else:
        conv = 1.

    # compute subsolar point
    sslat, sslon = subsol(datetimes)

    # compute and return the angle
    ssr = spherical.sph_to_car(np.vstack((np.ones_like(sslat), 90. - sslat, sslon)), deg = True)
    gcr = spherical.sph_to_car(np.vstack((np.ones_like(glat ), 90. - glat , glon )), deg = True)

    # the angle is arccos of the dot product of these two vectors
    return np.arccos(np.sum(ssr*gcr, axis = 0))*conv

def terminator(datetime, sza = 90, resolution = 360):
    """ compute terminator trajectory (constant solar zenith angle contour)

        glat, glon = compute_terminator(date, sza = 90, resolution = 360)

        sza is the solar zenith angle contour, default 90 degrees

        return two arrays, geocentric latitude and longitude, which outline the sunlight terminator at given date (datetime)

        does not handle arrays - only one trajectory can be returned at the time

        Method is assuming a spherical geometry:
        - compute the subsolar point, and two approximately duskward and northward normal vectors (these will point at 90 degrees SZA)
        - rotate the northward normal around the duskward normal to get to the correct SZA
        - rotate the resulting vector about the subsolar vector a number of times to trace out the contour.
        - calculate corresponding spherical (geocentric) coordinates

    Added by SMH 2020/04/03 (from Kalle's code stores!)
    """

    sslat, sslon = subsol(datetime)
    #print sslon, sslat
    sslon = sslon[0]*np.pi/180
    sslat = sslat[0]*np.pi/180

    # make cartesian vector
    x = np.cos(sslat) * np.cos(sslon)
    y = np.cos(sslat) * np.sin(sslon) 
    z = np.sin(sslat)
    ss = np.array([x, y, z]).flatten()

    # make a cartesian vector pointing at the pole
    pole = np.array([0, 0, 1])

    # construct a vector pointing roughly towards dusk, and normalize
    t0 = np.cross(ss, pole)
    t0 = t0/np.linalg.norm(t0)

    # make a new vector pointing northward at the 90 degree SZA contour:
    sza90 = np.cross(t0, ss)

    # rotate this about the duskward vector to get specified SZA contour
    rotation_angle = -(sza - 90) * np.pi/180

    sza_vector = sza90 * np.cos(rotation_angle) + np.cross(t0, sza90) * np.sin(rotation_angle) + t0 * (np.sum(t0*sza90)) * (1 - np.cos(rotation_angle)) # (rodrigues formula)
    sza_vector = sza_vector.flatten()

    # rotate this about the sun-Earth line to trace out the trajectory:
    angles = np.r_[0 : 2*np.pi: 2*np.pi / resolution][np.newaxis, :]
    r = sza_vector[:, np.newaxis] * np.cos(angles) + np.cross(ss, sza_vector)[:, np.newaxis] * np.sin(angles) + ss[:, np.newaxis] * (np.sum(t0*sza90)) * (1 - np.cos(rotation_angle))

    # convert to spherical and return
    tsph = spherical.car_to_sph(r, deg = True)

    return 90 - tsph[1], tsph[2]


def get_max_sza(h,
                R=6371.):
    """
    h is altitude in km
    # R = 6371. # Earth radius (default)

    Added by SMH 2020/04/03
    """

    # assert not hasattr(R,'__iter__')
    hIsArray = hasattr(h,'size')

    if not hIsArray:
        h = np.array([h])

    # z = h + R
    # max_sza = 180.-np.rad2deg(np.arctan2(np.sqrt(1-(R/z)**2.), (z/R-R/z)))
    max_sza = np.rad2deg(np.pi/2+np.arccos(R/(R+h)))

    max_sza[np.isclose(h,0)] = 90.
        
    max_sza[h < 0] = np.nan
        
    if hIsArray:
        return max_sza
    else:
        return max_sza[0]


def get_t_in_darkness(alt,glat,glon,datoer,
                      tol_deg=0.15,
                      verbose=True,
                      dodiagnosticprint=False):
    """
    For a given timestamp, altitude and GEOGRAPHIC (or is it geodetic??? Ask Kalle what sza uses!) lat and lon, calculate how many seconds
    this point has been in darkness assuming no refraction of light and a perfectly spherical earth.
    
    alt           : Geodetic altitude       (km)
    glat          : Geographic(?) latitude  (degrees)
    glon          : Geographic(?) longitude (degrees)
    datoer         : Timestamps              (datetime, pandas DatetimeIndex, etc.)
    tol_deg       : Fudge factor            (degrees).
                    I find that a fraction of a degree (specifically 0.15) seems to do the job

    Added by SMH 2020/04/03
    """

    #So how do we do this?
    #
    ##########
    # 1. Get solar zenith angle (sza)
    #
    ##########
    # 2. Get maximum sza at which sun is visible for given altitude (assumes spherical Earth!)
    #    During this step we find out which points lie in darkness (taking stock of 'tol_deg' fudge factor).
    #    We mark points that are already sunlit as such, and give them t_in_darkness = 0.
    #
    ##########
    # 3. Get SZA for each altitude, latitude, and NOON longitude; see if each latitude is CURRENTLY sunlit
    #    at the local-noon longitude.
    #
    ##########
    # 4. For those alt/tstamp/lat point for which the local-noon longitude is NOT sunlit, iteratively shift
    #    tHadSun back by one day until we find that there is sunshine at local-noon longitude.
    #
    #    After this step, tHadSun will be an array of timestamps for which the sun is visible at the given 
    #    alt/lat and calculated NOON-longitude
    #
    # 4a. A check – all noonszas must be less than their corresponding maximum sza (i.e., all noon-szas must 
    #     now be sunlit) before step 5.
    #
    ##########
    # 5. Calculate the time shift (which, as a result of step 4, is at most 23.9999 hours) needed to put each 
    #    alt/lat/lon pair at noon. Subtract this time shift from tHadSun so that tHadSun corresponds to the
    #    last day on which this alt/lat/lon pair was in sunlight at local noon.
    #
    # 5a. Do some fudging here.  This fudging is necessary because, for a given latitude, the minimum sza
    # obtained over the course of a day changes.
    #
    # FREE EXAMPLE TO EXPLAIN WHY WE HAVE TO FUDGE: 
    # Consider an alt/lat/lon that is in darkness at, say, 03:00 local time. Now imagine following this point
    # along a line of constant altitude and latitude for the given timestamp (i.e. vary longitude ONLY and
    # hold all else constant) until we reach the longitude corresponding to local noon. Suppose that this
    # alt/lat is sunlit at local noon. Now, because the point we are actually interested currently lies at
    # 03:00 local time, we have to subtract 15 hours from the current time stamp to put our alt/lat/lon at
    # local noon. But this particular alt/lat pair may not have been sunlit at local noon 15 hours before the
    # present time! The necessary fudge factor appears to be of order 0.1 degrees.
    #
    ##########
    # 6. After shifting timestamps to put this longitude at local noon, all alt/lat/lon pairs are now sunlit
    # (up to the fudge factor 'tol_deg'). Now we just increment each timestamp until the alt/lat/lon pair
    # falls in darkness. The stepping in time goes by hours, then minutes, then seconds. 
    
    # 1. Get sza
    origsza = sza(glat,glon,datoer)

    # 2. Get thresh SZA for given altitude
    maxsza = get_max_sza(alt)
    
    alreadysunlit = origsza <= (maxsza+np.abs(tol_deg))
    origDark = origsza > maxsza
    nOrigDarkPts = np.sum(origDark)

    if verbose:
        print("{:d} location/time pairs in darkness ({:d} in light)!".
              format(np.sum(~alreadysunlit),
                     np.sum(alreadysunlit)))

    if nOrigDarkPts == 0:
        return np.zeros(glat.size)

    # 3. Get SZA for each altitude, latitude, and NOON longitude to see if each point is/may have been in sunlight
    noonsza = sza(glat,
                  get_noon_longitude(datoer),
                  datoer)
    
    tHadSun = datoer.copy()
    
    stillDark = noonsza > maxsza
    fixed = noonsza <= maxsza       # Keep track of which points need updating
    nStillDarkPts = stillDark.sum()
    
    if verbose:
        print("{:d} of these latitudes are in darkness at local-noon longitude ({:d} in light)!".
              format(nStillDarkPts,fixed.sum()))
    
    # 4. For each point, shift tHadSun back by one day until we find that there is sunshine at local-noon longitude.
    # After this step, tHadSun will be an array of timestamps for which the sun is visible for the given latitude
    # and altitude, and calculated NOON-longitude 
    
    daysback = 1
    totalNoonAdjusted = 0
    while nStillDarkPts > 0:
    
        thistdelta = timedelta(days=daysback)
    
        # DIAG
        if dodiagnosticprint:
            print(tHadSun[stillDark]-thistdelta)
    
        noonsza[stillDark] = sza(glat[stillDark],
                                 get_noon_longitude(tHadSun[stillDark]),
                                 tHadSun[stillDark]- thistdelta)
    
        # Calculate who is still in darkness, N stilldark points
        stillDark = noonsza > maxsza
        fixme = ~(stillDark) & ~(fixed) & ~(alreadysunlit)
        nFixme = np.sum(fixme)
    
        if nFixme > 0:
            
            totalNoonAdjusted += nFixme
            if dodiagnosticprint:
                print("Moving {:d} timestamps !".format(nFixme))
    
            tHadSun[fixme] = tHadSun[fixme]-thistdelta
    
            fixed[fixme] = True
    
        nStillDarkPts = stillDark.sum()
        daysback += 1
    
    if verbose:
        print("Moved {:d} timestamps back".format(totalNoonAdjusted))

    # 4a. A check – all noonszas should be less than their corresponding maxsza
    noonsza = sza(glat,get_noon_longitude(tHadSun),tHadSun)
    assert all(noonsza[~alreadysunlit] <= maxsza[~alreadysunlit])
    
    # 5. Calculate the time shift (which, as a result of step 4, is at most 23.9999 hours) needed to put each alt/lat/lon pair at noon.
    #    Subtract this time shift from tHadSun so that tHadSun corresponds to the last day on which this alt/lat/lon pair was in sunlight at local noon.
    shiftHours = cheap_LT_calc(tHadSun,
                               glon,
                               return_dts_too=False,
                               verbose=True)

    shiftHours = shiftHours - 12
    shiftHours[shiftHours < 0] += 24

    shiftHours[alreadysunlit] = 0

    timedeltas = pd.TimedeltaIndex(
        data=shiftHours*3600,
        unit='s').to_pytimedelta()
    
    # Don't believe this is noon for these guys? Try it:
    # print((cheap_LT_calc(tHadSun-timedeltas,
    #                      glon,
    #                      return_dts_too=False,
    #                      verbose=True)).describe())

    # RESET STILLDARK TO ACTUALDARKS
    testsza = origsza.copy()
    stillDark = origDark.copy()
    nStillDarkPts = np.sum(origDark)
    
    testsza[stillDark] = sza(glat[stillDark],
                             glon[stillDark],
                             tHadSun[stillDark]-timedeltas[stillDark])
    
    # assert all(testsza[stillDark] < maxsza[stillDark])

    # 5a. Do some fudging here.  This fudging is necessary because, for a given latitude, the minimum sza obtained over the course of a
    # day changes.
    # 
    if not all(testsza[stillDark] <= maxsza[stillDark]):

        diff = (testsza[stillDark] - maxsza[stillDark])

        if diff.max() > tol_deg:
            print("Warning! error of more than {:.2f} deg in darknesscalc!".format(tol_deg))
            breakpoint()

        badHeads = np.where(diff > 0)[0]
        badHeads = np.where(stillDark)[0][badHeads]

        maxsza[badHeads] += diff[diff > 0]
        #     print("N badheads: {:d}. Rotate 'em back a day".format(badHeads.size))
        #     tHadSun[badHeads] -= timedelta(days=1)

    dodat = stillDark & ~alreadysunlit
    tHadSun[dodat] = tHadSun[dodat]-timedeltas[dodat]
    
    # 6. After shifting timestamps to put this longitude at local noon, all alt/lat/lon pairs are now sunlit
    # (up to the fudge factor 'tol_deg'). Now we just increment each timestamp until the alt/lat/lon pair falls in darkness.
    # The stepping in time goes by hours, then minutes, then seconds. 

    #7. If some need to be fixed, go back each minute until we're where we need to be 
    # No er det berre å trylle tiden framover for å finne tidspunktet hvor mørket slår
    # tHadSun[stillDark] is filled with points that are now LIGHT
    
    daystep = 0
    hourstep = 1
    minutestep = 0
    secondstep = 0
    stepcount = 0
    
    omgangtype = 'hours'
    thistdelta = timedelta(days=daystep,hours=hourstep,minutes=minutestep,seconds=secondstep)
    
    haveSteppedDays = True
    haveSteppedHours = False
    haveSteppedMinutes = False
    haveSteppedSeconds = False
    
    testsza = origsza.copy()
    nStillLightPts = np.sum(origDark)
    stillLight = origDark.copy()
    
    while (nStillLightPts > 0) and (not haveSteppedSeconds):
    
        oldtestsza = testsza.copy()

        # Get sza for darkpoints given this timedelta
        testsza[stillLight] = sza(glat[stillLight],
                                 glon[stillLight],
                                 tHadSun[stillLight]+thistdelta)
    
        # Calculate who is still in light, N stillLight points
        # stillLight = (testsza < (maxsza-tol_deg)) & ~alreadysunlit
        stillLight = (testsza <= maxsza) & ~alreadysunlit
        nStillLightPts = stillLight.sum()

        if (omgangtype == 'hours') & (stepcount >= 24):
            print("Bogusness")
            breakpoint()

        if stepcount > 0:
            if np.where(testsza[stillLight] < oldtestsza[stillLight])[0].size > 0:
                print("BAD!")

        if dodiagnosticprint:
            print("Adjusting tstamp for {:d} points!".format(np.sum(stillLight)))
    
        # Update timestamps for those that are still light, even with this time adjustment
        tHadSun[stillLight] += thistdelta
    
        if np.where(tHadSun > datoer)[0].size > 0:
            print("Bogus!")
            breakpoint()

        print("nStillLight: ",nStillLightPts)

        # if nStillLightPts == 1:
        #     breakpoint()

        if nStillLightPts == 0:
            if dodiagnosticprint:
                print("No more lights for {:s} omgang!".format(omgangtype))
    
            if not haveSteppedDays:
                haveSteppedDays = True
    
                daystep = 0
                hourstep = 1
                minutestep = 0
                secondstep = 0
                stepcount = 0
                thistdelta = timedelta(days=daystep,
                                       hours=hourstep,
                                       minutes=minutestep,
                                       seconds=secondstep)
                omgangtype = 'hours'
    
                testsza = origsza.copy()
                nStillLightPts = np.sum(origDark)
                stillLight = origDark.copy()
    
            elif not haveSteppedHours:
                haveSteppedHours = True
    
                daystep = 0
                hourstep = 0
                minutestep = 1
                secondstep = 0
                stepcount = 0
                thistdelta = timedelta(days=daystep,
                                       hours=hourstep,
                                       minutes=minutestep,
                                       seconds=secondstep)
                omgangtype = 'minutes'
    
                testsza = origsza.copy()
                nStillLightPts = np.sum(origDark)
                stillLight = origDark.copy()
    
            elif not haveSteppedMinutes:
                haveSteppedMinutes = True
    
                daystep = 0
                hourstep = 0
                minutestep = 0
                secondstep = 1
                stepcount = 0
                thistdelta = timedelta(days=daystep,
                                       hours=hourstep,
                                       minutes=minutestep,
                                       seconds=secondstep)
                omgangtype = 'seconds'
    
                testsza = origsza.copy()
                nStillLightPts = np.sum(origDark)
                stillLight = origDark.copy()
    
            elif not haveSteppedSeconds:
                haveSteppedSeconds = True
    
        stepcount += 1
        # if dodiagnosticprint:
        print("{:4d} {:s} steps".format(stepcount,omgangtype))
    
    # sza(glat,
    #     glon,
    #     tHadSun)

    # finalcheck
    if not all(sza(glat,
                   glon,
                   tHadSun) <= maxsza):
        breakpoint()

    return pd.TimedeltaIndex(datoer-tHadSun).total_seconds()


def cheap_LT_calc(dts,gclons,
                  return_dts_too=False,
                  verbose=False):
    """

    Added by SMH 2020/04/03
    """

    # print("Convert input dts to pandas datetime index!")
    if not hasattr(dts,'__iter__'):
        dts = pd.DatetimeIndex([dts])

    elif not hasattr(dts,'hour'):
        dts = pd.DatetimeIndex(dts)

    if not hasattr(gclons,'__iter__'):
        gclons = np.array(gclons)

    gclons = (gclons+360) % 360

    if verbose:
        if gclons.size ==1:
            # relstr = "ahead of"
            reltogreenwich = gclons/15.

            relstr = "ahead of" if (gclons <= 180) else "behind"
            if reltogreenwich > 12:
                reltogreenwich -= 24

            print("Longitude {:.2f} is {:.2f} hours {:s} Greenwich!".format(gclons,reltogreenwich,relstr))

    midnightlongitude = -15*(dts.hour.values+dts.minute.values/60+dts.second.values/3600.)
    midnightlongitude = (midnightlongitude + 360) % 360

    LTs = (((gclons-midnightlongitude) + 360) % 360)/15
    if return_dts_too:
        return LTs, dts, midnightlongitude, gclons
    else:
        return LTs


def get_noon_longitude(dts,verbose=False):
    """
    
    Added by SMH 2020/04/03
    """
    # A test:
    # import datetime
    # import pandas as pd
    # from pytt.earth.sunlight import sza
    # 
    # marequinox = datetime.datetime(2015, 3, 20, 22, 45, 9, 340000)
    # junsolstice = datetime.datetime(2015, 6, 21, 16, 37, 55, 813000)
    # refdate = junsolstice
    # refdate = marequinox
    # dts = pd.date_range(start=datetime(refdate.year,refdate.month,refdate.day,0),
    #                     end=datetime(refdate.year,refdate.month,refdate.day,23),
    #                     freq='1h')
    # # FOLLOWING SHOULD ALL BE AROUND 23.44 IF JUNSOLSTICE, 0 IF MAREQUINOX
    # print(sza(np.zeros(dts.size),get_noon_longitude(dts),dts))

    if not hasattr(dts,'__iter__'):
        dts = pd.DatetimeIndex([dts])

    elif not hasattr(dts,'hour'):
        dts = pd.DatetimeIndex(dts)

    fracHour = dts.hour.values+dts.minute.values/60+dts.second.values/3600.

    assert not any((fracHour < 0) | (fracHour > 24))

    fracHour[fracHour > 12] -= 24

    fracHour *= 15

    if verbose:
        print("Min fracHour: {:.2f}".format(np.min(fracHour)))
        print("Max fracHour: {:.2f}".format(np.max(fracHour)))

    return 180 - fracHour 



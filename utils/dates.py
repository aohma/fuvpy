import datetime
from operator import attrgetter
import numpy as np
import pandas as pd

def datetime_to_doy_single(dt):
    """ Convert a datetime object to day of year
        SMH 2020-03-18
    """
    tt = dt.timetuple()
    doy = tt.tm_yday+tt.tm_hour/24.+tt.tm_min/1440.+tt.tm_sec/86400.
    return doy


def doy_from_timetuple(timetuple):
    """Convert a timetuple to a DOY. (Primarily intended to be ancillary to 'datetime_to_doy' function)
    Tuple skal være ('tm_yday','tm_hour','tm_min','tm_sec')

        SMH 2020-03-18
    """
    return timetuple[0]+timetuple[1]/24.+timetuple[2]/1440.+timetuple[3]/86400.


def datetime_to_doy(dts):
    """Convert an iterable containing datetime-like objects to a list of DOYs
        SMH 2020-03-18
    """
    tider = attrgetter('tm_yday', 'tm_hour', 'tm_min', 'tm_sec')
    return [doy_from_timetuple(tider(dt.timetuple())) for dt in dts]


def jd_to_datetime(jds):
    """Convert Julian day(s) to datetime object
        SMH 2020-03-18
    """
    noonUT_Jan2000__JD = 2451545.
    tdeltas = jds-noonUT_Jan2000__JD
    return [datetime.datetime(2000, 1, 1, 12)+datetime.timedelta(days=tdiff) for tdiff in tdeltas]


def datetimes_to_jd(dts):
    """
    Convert an array of datetimes or pandas DatetimeIndex array to Julian day.

    SMH 2020-05-12
    """

    noonUT_Jan2000__JD = 2451545.

    if not isinstance(dts,pd.DatetimeIndex):
        dts = pd.DatetimeIndex(dts)

    return (dts-datetime.datetime(2000, 1, 1, 12)).total_seconds().values/24/60/60 + noonUT_Jan2000__JD


def date_to_doy(month, day, leapyear = False):
    """ return day of year (DOY) at given month, day

        month and day -- can be arrays, but must have equal shape
        leapyear      -- can be array of equal shape or scalar

        return value  --  doy, with same shape as month and day
                          but always an array: shape (1,) if input is scalar

        The code is vectorized, so it should be relatively fast. 

        KML 2016-04-20
    """

    month = np.array(month, ndmin = 1)
    day   = np.array(day, ndmin = 1)

    if type(leapyear) == bool:
        leapyear = np.full_like(day, leapyear, dtype = bool)

    # check that shapes match
    if month.shape != day.shape:
        raise ValueError('date2ody: month and day must have the same shape')

    # check that month in [1, 12]
    if month.min() < 1 or month.max() > 12:
        raise ValueError('month not in [1, 12]')

    # check if day < 1
    if day.min() < 1:
        raise ValueError('date2doy: day must not be less than 1')

    # flatten arrays:
    shape = month.shape
    month = month.flatten()
    day   = day.flatten()

    # check if day exceeds days in months
    days_in_month    = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    days_in_month_ly = np.array([0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    if ( (np.any(day[~leapyear] > days_in_month   [month[~leapyear]])) | 
         (np.any(day[ leapyear] > days_in_month_ly[month[ leapyear]])) ):
        raise ValueError('date2doy: day must not exceed number of days in month')

    cumdaysmonth = np.cumsum(days_in_month[:-1])

    # day of year minus possibly leap day:
    doy = cumdaysmonth[month - 1] + day
    # add leap day where appropriate:
    doy[month >= 3] = doy[month >= 3] + leapyear[month >= 3]

    return doy.reshape(shape)



def is_leapyear(year):
    """ Return True if leapyear else False
    
        handles arrays (preserves shape).

        KML 2016-04-20
    """

    # if array:
    if type(year) is np.ndarray:
        out = np.full_like(year, False, dtype = bool)

        out[ year % 4   == 0] = True
        out[ year % 100 == 0] = False
        out[ year % 400 == 0] = True

        return out

    # if scalar:
    if year % 400 == 0:
        return True

    if year % 100 == 0:
        return False

    if year % 4 == 0:
        return True

    else:
        return False


def yearfrac_to_datetime(number):
    """ convert fraction of year to datetime 

        handles arrays


        example:

        >>> dates.yearfrac_to_datetime(np.array([2000.213]))
        array([datetime.datetime(2000, 3, 18, 22, 59, 31, 200000)], dtype=object)
    """
    
    year = np.uint16(number) # truncate number to get year
    # use pandas TimedeltaIndex to represent time since beginning of year: 
    delta_year = pd.TimedeltaIndex((number - year)*(365 + is_leapyear(year)), unit = 'D')
    # and DatetimeIndex to represent beginning of years:
    start_year = pd.DatetimeIndex(list(map(str, year)))
 
    # adding them produces the datetime:
    return (start_year + delta_year).to_pydatetime()
 

def datetime_to_yearfrac(dts):
    """
    Convert list of datetime-like objects to year fractions.
    For example, 2001-06-30 becomes 2001.4945364574537

    2020-04-17    SMH 
    """
    return [dt.year+datetime_to_doy_single(dt)/datetime_to_doy_single(datetime.datetime(dt.year,12,31,23,59)) for dt in dts]

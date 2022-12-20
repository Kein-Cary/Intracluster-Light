"""
use to record functions use to model the SB profiles of inner radius bin
or the ratio profiles
"""
import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc
import scipy.interpolate as interp

from scipy.stats import binned_statistic as binned
from astropy import modeling as Model


### === func.s
##. core-like funcs
def Moffat_func(R, A0, R0, n):
	mf = A0 / ( 1 + (R / R0)**2 )**n
	return mf

def KPA_func( R, A0, R_c0, R_c1 ):

	mod_F = Model.functional_models.KingProjectedAnalytic1D( amplitude = A0, r_core = R_c0, r_tide = R_c1 )
	mf = mod_F( R )
	return mf

##. Galfit mannul
def Modi_Ferrer_func( R, A0, R_bk, belta, alpha):

	mf0 = 1 - ( R / R_bk )**( 2 - belta)
	mf = mf0**alpha
	return mf * A0

def Empi_King_func( R, A0, R_t, R_c, alpha):

	mf0 = 1 + (R_t / R_c)**2
	mf1 = mf0**( 1 / alpha )

	mf2 = 1 / ( 1 - 1 / mf1 )**alpha

	mf4 = 1 + ( R / R_c)**2
	mf5 = mf4**( 1 / alpha )

	mf6 = 1 / mf5 - 1 / mf1

	mf = mf2 * mf6**alpha
	return mf * A0

def Nuker_func( R, A0, R_bk, alpha, belta, gamma):

	mf0 = 2**( (belta - gamma) / alpha )
	mf1 = mf0 / ( R / R_bk)**gamma

	mf2 = 1 + ( R / R_bk  )**alpha
	mf3 = mf2**( (gamma - belta) / alpha )

	mf = mf1 * mf3
	return mf * A0


##... power-law funciton for large radii
##... model ratio profile with " power1_func + Moffat_func or Nuker_func "
def power_func( R, R0, alpha ):
	A0 = 1.
	mf = A0 * ( (R / R0)**2.5 + 1)**alpha * 2**(-alpha)

	return mf / mf[0] - 1


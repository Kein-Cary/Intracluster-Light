import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.patches import Circle
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
from scipy import ndimage
import scipy.signal as signal


###... 
band = ['r', 'g', 'i']
load = '/home/xkchen/fig_tmp/'

#. extremely bad imgs counts
dat = pds.read_csv(load + 'photo_z_cat/photo-z_tot-r-band_bad-img_cat.csv' )
r_ra, r_dec, r_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

r_coord = SkyCoord( ra = r_ra * U.deg, dec = r_dec * U.deg,)


dat = pds.read_csv(load + 'photo_z_cat/photo-z_tot-g-band_bad-img_cat.csv' )
g_ra, g_dec, g_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

g_coord = SkyCoord( ra = g_ra * U.deg, dec = g_dec * U.deg, )


dat = pds.read_csv(load + 'photo_z_cat/photo-z_tot-i-band_bad-img_cat.csv' )
i_ra, i_dec, i_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

i_coord = SkyCoord( ra = i_ra * U.deg, dec = i_dec * U.deg, )


idx, d2d, d3d = i_coord.match_to_catalog_sky( r_coord )
id_lim = d2d.value >= 2.7e-4
ext_ra, ext_dec, ext_z = r_ra[ idx[ id_lim ] ], r_dec[ idx[ id_lim ] ], r_z[ idx[ id_lim ] ]

pi_ra, pi_dec, pi_z = np.r_[ i_ra, ext_ra ], np.r_[ i_dec, ext_dec ], np.r_[ i_z, ext_z ]

pi_coord = SkyCoord( ra = pi_ra * U.deg, dec = pi_dec * U.deg, )


idx, d2d, d3d = pi_coord.match_to_catalog_sky( g_coord )
id_lim = d2d.value >= 2.7e-4

ext_ra, ext_dec, ext_z = g_ra[ idx[ id_lim ] ], g_dec[ idx[ id_lim ] ], g_z[ idx[ id_lim ] ]

pi_ra, pi_dec, pi_z = np.r_[ pi_ra, ext_ra ], np.r_[ pi_dec, ext_dec ], np.r_[ pi_z, ext_z ]

out_arr = np.array( [pi_ra, pi_dec, pi_z] ).T
np.savetxt('/home/xkchen/total_extremely_bad_img.txt', out_arr )


#. middly bad imgs counts
dat = pds.read_csv(load + 'photo_z_cat/photo-z_r-band_tot_rule-out_cat_set_200-grid_6.0-sigma.csv')
r_ra, r_dec, r_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

r_coord = SkyCoord( ra = r_ra * U.deg, dec = r_dec * U.deg,)


dat = pds.read_csv(load + 'photo_z_cat/photo-z_g-band_tot_rule-out_cat_set_200-grid_6.0-sigma.csv')
g_ra, g_dec, g_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

g_coord = SkyCoord( ra = g_ra * U.deg, dec = g_dec * U.deg, )


dat = pds.read_csv(load + 'photo_z_cat/photo-z_i-band_tot_rule-out_cat_set_200-grid_6.0-sigma.csv')
i_ra, i_dec, i_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

i_coord = SkyCoord( ra = i_ra * U.deg, dec = i_dec * U.deg, )


idx, d2d, d3d = i_coord.match_to_catalog_sky( r_coord )
id_lim = d2d.value >= 2.7e-4
ext_ra, ext_dec, ext_z = r_ra[ idx[ id_lim ] ], r_dec[ idx[ id_lim ] ], r_z[ idx[ id_lim ] ]

pi_ra, pi_dec, pi_z = np.r_[ i_ra, ext_ra ], np.r_[ i_dec, ext_dec ], np.r_[ i_z, ext_z ]

pi_coord = SkyCoord( ra = pi_ra * U.deg, dec = pi_dec * U.deg, )


idx, d2d, d3d = pi_coord.match_to_catalog_sky( g_coord )
id_lim = d2d.value >= 2.7e-4

ext_ra, ext_dec, ext_z = g_ra[ idx[ id_lim ] ], g_dec[ idx[ id_lim ] ], g_z[ idx[ id_lim ] ]

pi_ra, pi_dec, pi_z = np.r_[ pi_ra, ext_ra ], np.r_[ pi_dec, ext_dec ], np.r_[ pi_z, ext_z ]

out_arr = np.array( [pi_ra, pi_dec, pi_z] ).T
np.savetxt('/home/xkchen/total_middly_bad_img.txt', out_arr )


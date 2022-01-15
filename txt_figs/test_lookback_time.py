import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy.stats import binned_statistic as binned
from scipy import interpolate as interp
import scipy.integrate as integrate

### === ### lookback time estimate
def inves_E_func( z, Om0, Ode0, Ogamma0, Ok0):

	inv_Ef = np.sqrt( Ogamma0 * (1 + z)**4 + Om0 * (1 + z)**3 + Ok0 * (1 + z)**2 + Ode0 )
	integ_F = ( 1 + z) * inv_Ef

	return 1 / integ_F

def integ_t_back_func( z, H0, Om0, Ode0, Ogamma0, Ok0):

	params = (Om0, Ode0, Ogamma0, Ok0)

	az = 1 / ( 1 + z)
	af = inves_E_func( az, Om0, Ode0, Ogamma0, Ok0)

	Ns = len(az)
	pre_integ_t = np.zeros( Ns,)

	t_H0 = 1 / (H0 * (U.km / (U.Mpc * U.s) ).to( 1 / U.s) )
	t_to_Gyr = U.s.to(U.Gyr)

	for oo in range( Ns ):
		pre_integ_t[ oo ] = integrate.quad( inves_E_func, 0, z[ oo ], args = params,)[0]

	return pre_integ_t * t_H0 * t_to_Gyr

### === ### origin mass-bin sample
path = '/home/xkchen/mywork/ICL/data/'

cat = pds.read_csv( path + 'cat_z_form/clslowz_z0.17-0.30_bc03_cat.csv' )
ra, dec, z = np.array(cat['ra']), np.array(cat['dec']), np.array(cat['z'])
rich, z_form = np.array(cat['lambda']), np.array(cat['z_form'])
lg_Mstar = np.array( cat['lg_M*_photo_z'] )

#. cosmology
# T0_model = apcy.Planck15.clone( H0 = 67.8, Om0 = 0.31,)
T0_model = apcy.Planck15.clone( H0 = 100, Om0 = 0.3,)
H0, Om0, Ode0, Ogamma0, Ok0 = T0_model.H0.value, T0_model.Om0, T0_model.Ode0, T0_model.Ogamma0, T0_model.Ok0

lb_time_0 = T0_model.lookback_time( z ).value
lb_time_1 = T0_model.lookback_time( z_form ).value
age_T0 = lb_time_1 - lb_time_0

#. integrate
dt0 = integ_t_back_func( z, H0, Om0, Ode0, Ogamma0, Ok0)
dt1 = integ_t_back_func( z_form, H0, Om0, Ode0, Ogamma0, Ok0)
age_T2 = dt1 - dt0


plt.figure()
plt.hist( age_T0, bins = 55, density = True, color = 'b', histtype = 'step',)
plt.hist( age_T2, bins = 55, density = True, color = 'g', histtype = 'step',)

plt.xlabel( 'age' )
plt.ylabel( 'PDF' )

plt.savefig('/home/xkchen/age_compare_test.png', dpi = 300)
plt.close()
raise

##.. mass-bin samples
M_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/low_star-Mass_cat.csv')
lo_ra, lo_dec, lo_z = np.array( M_dat['ra'] ), np.array( M_dat['dec'] ), np.array( M_dat['z'] )

M_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/high_star-Mass_cat.csv')
hi_ra, hi_dec, hi_z = np.array( M_dat['ra'] ), np.array( M_dat['dec'] ), np.array( M_dat['z'] )

cp_ra = np.r_[ lo_ra, hi_ra ]
cp_dec = np.r_[ lo_dec, hi_dec ]
cp_z = np.r_[ lo_z, hi_z ]

coord_0 = SkyCoord( ra = ra * U.deg, dec = dec * U.deg,)
cp_coord = SkyCoord( ra = cp_ra * U.deg, dec = cp_dec * U.deg,)

idx, sep, d3d = cp_coord.match_to_catalog_sky( coord_0 )
id_lim = sep.value < 2.7e-4


mp_ra, mp_dec, mp_z = ra[ idx[ id_lim ] ], dec[ idx[ id_lim ] ], z[ idx[ id_lim ] ]
lim_ra, lim_dec, lim_z = cp_ra[ id_lim ], cp_dec[ id_lim ], cp_z[ id_lim ]

plt.figure()
plt.plot( mp_ra, mp_dec, 'ro', alpha = 0.5, label = 'z_form list')
plt.plot( lim_ra, lim_dec, 'g*', alpha = 0.5, label = 'mass-bin list')
plt.show()


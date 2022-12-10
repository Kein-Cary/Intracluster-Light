import h5py
import numpy as np
import pandas as pds
import astropy.constants as C
import astropy.units as U
import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.io.ascii as asc

from astropy import cosmology as apcy
from scipy import signal
from scipy import interpolate as interp
from scipy import ndimage
from scipy import integrate as integ
from io import StringIO

#constant
rad2arcsec = U.rad.to(U.arcsec)

# cosmology model
vc = C.c.to(U.km/U.s).value
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
DH = vc / H0

# band information of SDSS
band = ['r', 'g', 'i']
l_wave = np.array([6166, 4686, 7480])
mag_add = np.array([0, 0, 0 ])

Mag_sun = [ 4.65, 5.11, 4.53 ]  ## Abs_magnitude of the Sun (for SDSS filters)


### === ### coordinate or position translation
def WCS_to_pixel_func(ra, dec, header_inf):
	"""
	according to SDSS Early Data Release paper (section 4.2.2 wcs)
	"""
	Ra0 = header_inf['CRVAL1']
	Dec0 = header_inf['CRVAL2']

	row_0 = header_inf['CRPIX2']
	col_0 = header_inf['CRPIX1']

	af = header_inf['CD1_1']
	bf = header_inf['CD1_2']

	cf = header_inf['CD2_1']
	df = header_inf['CD2_2']

	y1 = (ra - Ra0) * np.cos( Dec0 * np.pi / 180 )
	y2 = dec - Dec0

	delt_col = (bf * y2 - df * y1) / ( bf * cf - af * df )
	delt_row = (af * y2 - cf * y1) / ( af * df - bf * cf )

	id_col = col_0 + delt_col
	id_row = row_0 + delt_row

	return id_col, id_row

def pixel_to_WCS_func(x, y, header_inf):

	Ra0 = header_inf['CRVAL1']
	Dec0 = header_inf['CRVAL2']

	row_0 = header_inf['CRPIX2']
	col_0 = header_inf['CRPIX1']

	af = header_inf['CD1_1']
	bf = header_inf['CD1_2']

	cf = header_inf['CD2_1']
	df = header_inf['CD2_2']

	_delta = bf * cf - af * df

	delta_x = x - col_0
	delta_y = y - row_0

	delta_ra = _delta * ( delta_x * af + delta_y * bf ) / _delta
	delta_dec = _delta * ( delta_x * cf + delta_y * df ) / _delta

	dec = Dec0 + delta_dec
	ra = Ra0 + delta_ra / np.cos( Dec0 * np.pi / 180 )

	return ra, dec


### === 
def zref_sat_pos_func( cat_file, z_ref, out_file, pix_size ):
	"""
	this part use for calculate BCG position after pixel resampling. 
	"""
	dat = pds.read_csv( cat_file )
	bcg_ra, bcg_dec, bcg_z = np.array(dat['bcg_ra']), np.array(dat['bcg_dec']), np.array(dat['bcg_z'])

	ra, dec = np.array(dat['sat_ra']), np.array(dat['sat_dec'])
	z = bcg_z + 0.

	sat_x, sat_y = np.array(dat['cut_cx']), np.array(dat['cut_cy'])

	Da_z = Test_model.angular_diameter_distance(z).value
	Da_ref = Test_model.angular_diameter_distance(z_ref).value

	L_ref = Da_ref * pix_size / rad2arcsec
	L_z = Da_z * pix_size / rad2arcsec
	eta = L_ref / L_z

	ref_satx = sat_x / eta
	ref_saty = sat_y / eta

	keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_x', 'sat_y' ]
	values = [ bcg_ra, bcg_dec, bcg_z, ra, dec, ref_satx, ref_saty ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_file )

	return

##.
def arr_zref_pos_func( sat_x, sat_y, sat_z, z_ref, pix_size ):
	"""
	this part use for calculate BCG position after pixel resampling. 
	"""

	z_g = sat_z + 0.

	Da_z = Test_model.angular_diameter_distance( z_g ).value
	Da_ref = Test_model.angular_diameter_distance( z_ref ).value

	L_ref = Da_ref * pix_size / rad2arcsec
	L_z = Da_z * pix_size / rad2arcsec
	eta = L_ref / L_z

	ref_satx = sat_x / eta
	ref_saty = sat_y / eta

	return ref_satx, ref_saty


### ===
def arr_jack_func(SB_array, R_array, N_sample):
	"""
	SB_array : y-data for jackknife resampling
	R_array : x-data for jackknife resampling
	( SB_array, R_array : list type )

	N_sample : number of sub-samples
	"""
	dx_r = np.array(R_array)
	dy_sb = np.array(SB_array)

	n_r = dx_r.shape[1]
	Len = np.zeros( n_r, dtype = np.float32)
	for nn in range( n_r ):
		tmp_I = dy_sb[:,nn]
		idnn = np.isnan(tmp_I)
		Len[nn] = N_sample - np.sum(idnn)

	Stack_R = np.nanmean(dx_r, axis = 0)
	Stack_SB = np.nanmean(dy_sb, axis = 0)
	std_Stack_SB = np.nanstd(dy_sb, axis = 0)

	###. only calculate r bins in which sub-sample number larger than one
	id_one = Len > 1
	Stack_R = Stack_R[ id_one ]
	Stack_SB = Stack_SB[ id_one ]
	std_Stack_SB = std_Stack_SB[ id_one ]
	N_img = Len[ id_one ]
	jk_Stack_err = np.sqrt(N_img - 1) * std_Stack_SB

	###. limit the radius bin contribution at least 1/10 * N_sample

	id_min = N_img >= np.int(N_sample / 10)
	lim_r = Stack_R[id_min]
	lim_R = np.nanmax(lim_r)

	return Stack_R, Stack_SB, jk_Stack_err, lim_R


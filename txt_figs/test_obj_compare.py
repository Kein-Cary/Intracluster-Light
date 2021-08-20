import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import astropy.io.fits as fits

import wget
import mechanize
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.wcs as awc
import astropy.io.ascii as asc

from io import StringIO
from scipy import optimize
from scipy import interpolate as interp
from scipy import integrate as integ
from astropy import cosmology as apcy

from img_pre_selection import WCS_to_pixel_func
from light_measure import light_measure_Z0_weit, light_measure_weit

### === ### cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

band = ['r', 'g', 'i']
Mag_sun = [ 4.65, 5.11, 4.53 ]
L_wave = np.array([ 6166, 4686, 7480 ])
psf_FWHM = [ 1.56, 1.67, 1.50 ] # arcsec

pixel = 0.396

ref_Rii = np.array([0.23,  0.68,  1.03,   1.76,   3.00, 
					4.63,  7.43,  11.42,  18.20,  28.20, 
					44.21, 69.00, 107.81, 168.20, 263.00]) # in unit 'arcsec'
ref_R_pix = ref_Rii / pixel

### data load
dat = pds.read_csv('/home/xkchen/mywork/ICL/data/cat_random/match_2_28/random_i-band_tot_remain_cat_set_200-grid_6.0-sigma.csv')
ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

img_path = '/home/xkchen/figs/i_mag_test/img_cat/'
cat_path = '/home/xkchen/figs/i_mag_test/source_cat/'
sql_path = '/home/xkchen/figs/i_mag_test/SDSS_objs/'

band_str = 'i'

for ii in range( 0,1 ):

	ra_g, dec_g, z_g = ra[ii], dec[ii], z[ii]

	img_file = img_path + 'rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band_str, ra_g, dec_g, z_g)
	data = fits.open( img_file )
	img = data[0].data
	wcs_lis = awc.WCS( data[0].header )

	#.obj_mag
	c_dat = pds.read_csv( img_path + 
		'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_3-R-Kron_pix-sum_re-scaled-cmag.csv' % (band_str, ra_g, dec_g, z_g),)
	obj_mag = np.array( c_dat['cMag'] )
	cx, cy = np.array( c_dat['cx'] ), np.array( c_dat['cy'] )
	A, B, phi = np.array( c_dat['A'] ), np.array( c_dat['B'] ), np.array( c_dat['phi'] )
	obj_ra, obj_dec = np.array( c_dat['ra'] ), np.array( c_dat['dec'] )
	order_lis = np.array( c_dat['order'] )
	N_obj = len( obj_ra )

	#.sdss_list
	sdss_dat = pds.read_csv('/home/xkchen/figs/i_mag_test/photoobj_All_match/SDSS_obj_list.csv')
	sdss_ra, sdss_dec = np.array( sdss_dat['ra'] ), np.array( sdss_dat['dec'] )
	sdss_imag = np.array( sdss_dat['cModelMag_i'] )
	sdss_type = np.array( sdss_dat['type'] )

	#.SDSS_list with clean information
	c_sdss_dat = pds.read_csv('/home/xkchen/figs/i_mag_test/photoobj_All_match/SDSS_obj_list_clean.csv')
	c_sdss_ra = np.array( c_sdss_dat['ra'] )
	c_sdss_dec = np.array( c_sdss_dat['dec'] )
	c_clean = np.array( c_sdss_dat['clean'] )
	c_sdss_imag = np.array( c_sdss_dat['cModelMag_i'] )

	id_clean = c_clean > 0
	sdss_ra_1 = c_sdss_ra[ id_clean ]
	sdss_dec_1 = c_sdss_dec[ id_clean ]
	sdss_imag_1 = c_sdss_imag[ id_clean ]

	cx, cy = wcs_lis.all_world2pix( obj_ra * U.deg, obj_dec * U.deg, 0)
	com_cx, com_cy = wcs_lis.all_world2pix( sdss_ra * U.deg, sdss_dec * U.deg, 0)
	com_cx_1, com_cy_1 = wcs_lis.all_world2pix( sdss_ra_1 * U.deg, sdss_dec_1 * U.deg, 0)

	#. distance min value
	dl_G0 = []
	id_ref_cat = []
	for jj in range( N_obj):
		dr = np.sqrt( (cx[jj] - com_cx)**2 + (cy[jj] - com_cy)**2 )
		dl_G0.append( np.min(dr) )

		id_g = np.where( dr == np.min(dr) )[0]
		id_ref_cat.append( id_g[0] )

	dl_G0 = np.array( dl_G0 )
	id_ref_cat = np.array( id_ref_cat )

	ref_type = sdss_type[ id_ref_cat ]
	ref_mag = sdss_imag[ id_ref_cat ] 

	id_match = dl_G0 < 3.8
	id_type = ref_type == 'GALAXY'

	# rule out two obj( close to bright star, presudo objs)
	id_rule = obj_mag > 14

	id_lim = id_match & id_type & id_rule

	mag_match = obj_mag[ id_lim ]
	pair_mag = ref_mag[ id_lim ]

	tt_mag = mag_match - pair_mag
	sigma = np.sqrt( np.sum( tt_mag**2 ) / len(tt_mag) )

	mag_bins = np.linspace(16.5, 24, 35)

	fig = plt.figure( figsize = (10.6, 4.8) )
	ax0 = fig.add_axes([0.09, 0.13, 0.40, 0.82])
	ax1 = fig.add_axes([0.58, 0.13, 0.40, 0.82])

	ax0.hist( mag_match, bins = mag_bins, density = True, histtype = 'step', ls = '-', color = 'r', label = 'Mine')
	ax0.axvline( x = np.nanmedian(mag_match), ls = '-', color = 'r', ymin = 0.0, ymax = 0.35, alpha = 0.5,)
	ax0.hist( pair_mag, bins = mag_bins, density = True, histtype = 'step', ls = '--', color = 'k', label = 'SDSS')
	ax0.axvline( x = np.nanmedian(pair_mag), ls = '--', color = 'k', ymin = 0.0, ymax = 0.35, alpha = 0.5)

	ax0.legend( loc = 2, frameon = False, fontsize = 15,)
	ax0.set_xlim( 15, 24 )
	ax0.set_xlabel('${ \\rm i \\_ cmag } \; [mag]$', fontsize = 15,)
	ax0.set_ylabel('$ \\mathcal{pdf} $', fontsize = 15,)
	ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	ax1.plot( mag_match, pair_mag, 'ko', markersize = 3, alpha = 0.75)
	ax1.plot( mag_match, mag_match, 'r-', alpha = 0.75)
	ax1.plot( mag_match, mag_match - sigma, 'r:', alpha = 0.75)
	ax1.plot( mag_match, mag_match + sigma, 'r:', alpha = 0.75)

	ax1.set_ylabel('$ { \\rm i \\_ cmag }_{ \; SDSS} \; [mag] $', fontsize = 15,)
	ax1.set_xlabel('$ { \\rm i \\_ cmag }_{ \; This \; work} \; [mag] $', fontsize = 15,)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	plt.savefig('/home/xkchen/cmag_compare.png', dpi = 300)
	plt.close()

	raise

	# cc_dl_G0 = []
	# for jj in range( len(sdss_ra) ):
	# 	dr = np.sqrt( (cx - com_cx[jj])**2 + (cy - com_cy[jj])**2 )
	# 	cc_dl_G0.append( np.min(dr) )
	# cc_dl_G0 = np.array( cc_dl_G0 )

	dl_G1 = []
	for jj in range( N_obj):
		dr = np.sqrt( (cx[jj] - com_cx_1)**2 + (cy[jj] - com_cy_1)**2 )
		dl_G1.append( np.min(dr) )
	dl_G1 = np.array( dl_G1 )


	fig = plt.figure( figsize = (10.6, 4.8) )
	ax0 = fig.add_axes([0.05, 0.13, 0.40, 0.82])
	ax1 = fig.add_axes([0.55, 0.13, 0.40, 0.82])

	ax0.set_title('PhotoObj, All')
	ax0.imshow( img, origin = 'lower', cmap = 'Greys', norm = mpl.colors.LogNorm(),)
	ax0.scatter( cx, cy, marker = 'o', s = 30, facecolors = 'none', edgecolors = 'r', linewidth = 0.75,)
	ax0.scatter( com_cx, com_cy, marker = 's', s = 15, facecolors = 'none', edgecolors = 'b', linewidth = 0.75,)

	ax1.set_title('PhotoObj, Clean only')
	ax1.imshow( img, origin = 'lower', cmap = 'Greys', norm = mpl.colors.LogNorm(),)
	ax1.scatter( com_cx_1, com_cy_1, marker = 's', s = 5, facecolors = 'none', edgecolors = 'g', linewidth = 0.75,)
	ax1.scatter( cx, cy, marker = 'o', s = 30, facecolors = 'none', edgecolors = 'r', linewidth = 0.75,)

	plt.savefig('/home/xkchen/rematch_objs.png', dpi = 300)
	plt.close()

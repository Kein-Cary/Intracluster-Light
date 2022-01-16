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

def sdss_obj_cmag( cat_ra, cat_dec, out_file):

	import mechanize
	from io import StringIO

	url = 'http://skyserver.sdss.org/dr12/en/tools/search/sql.aspx'

	Nz = len( cat_ra )

	for kk in range( 443,Nz):

		ra_g = cat_ra[kk]
		dec_g = cat_dec[kk]

		data_set = """
		SELECT
			g.objID, g.cModelMag_g, g.cModelMag_r, g.cModelMag_i,
			g.ra, g.dec, g.clean

		FROM Galaxy as g
		JOIN dbo.fGetNearbyObjEq(%.5f, %.5f, 0.026) AS GN 
				ON G.objID = GN.objID
		WHERE
			g.clean = 1
		ORDER BY distance
		""" % (ra_g, dec_g)

		br = mechanize.Browser()
		resp = br.open(url)
		resp.info()

		br.select_form(name = "sql")
		br['cmd'] = data_set
		br['format'] = ['csv']
		response = br.submit()
		s = str(response.get_data(), encoding = 'utf-8')
		doc = open( out_file % (ra_g, dec_g), 'w')
		print(s, file = doc)
		doc.close()

		# print( kk )
	print( 'down!' )

	return

### === ### data load
dat = pds.read_csv('/home/xkchen/mywork/ICL/data/cat_random/match_2_28/random_i-band_tot_remain_cat_set_200-grid_6.0-sigma.csv')
ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
imgx, imgy = np.array( dat['bcg_x'] ), np.array( dat['bcg_y'] )

img_path = '/home/xkchen/tmp_run/data_files/figs/i_mag_test/img_cat/'
cat_path = '/home/xkchen/tmp_run/data_files/figs/i_mag_test/source_cat/'
sql_path = '/home/xkchen/tmp_run/data_files/figs/i_mag_test/SDSS_objs/'

band_str = 'i'

"""
for ii in range( 0,1 ):

	ra_g, dec_g, z_g = ra[ii], dec[ii], z[ii]

	c_dat = pds.read_csv( img_path + 'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_cmag.csv' % (band_str, ra_g, dec_g, z_g),)
	obj_mag = np.array( c_dat['cMag'] )
	cx, cy = np.array( c_dat['cx'] ), np.array( c_dat['cy'] )
	A, B, phi = np.array( c_dat['A'] ), np.array( c_dat['B'] ), np.array( c_dat['phi'] )
	s_ra, s_dec = np.array( c_dat['ra'] ), np.array( c_dat['dec'] )
	out_file = sql_path + 'SDSS_match_ra%.3f_dec%.3f_obj.csv'

	sdss_obj_cmag( s_ra, s_dec, out_file)
"""

for ii in range( 0,1 ):

	ra_g, dec_g, z_g = ra[ii], dec[ii], z[ii]

	img_file = img_path + 'rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band_str, ra_g, dec_g, z_g)
	data = fits.open( img_file )
	img = data[0].data
	wcs_lis = awc.WCS( data[0].header )

	# c_dat = pds.read_csv( img_path + 'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_re-scaled-cmag.csv' % (band_str, ra_g, dec_g, z_g),)
	# c_dat = pds.read_csv( img_path + 'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_3-R-Kron_re-scaled-cmag.csv' % (band_str, ra_g, dec_g, z_g),)
	c_dat = pds.read_csv( img_path + 'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_3-R-Kron_pix-sum_re-scaled-cmag.csv' % (band_str, ra_g, dec_g, z_g),)

	# c_dat = pds.read_csv( img_path + 'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_cmag.csv' % (band_str, ra_g, dec_g, z_g),)

	obj_mag = np.array( c_dat['cMag'] )
	cx, cy = np.array( c_dat['cx'] ), np.array( c_dat['cy'] )
	A, B, phi = np.array( c_dat['A'] ), np.array( c_dat['B'] ), np.array( c_dat['phi'] )
	s_ra, s_dec = np.array( c_dat['ra'] ), np.array( c_dat['dec'] )
	order_lis = np.array( c_dat['order'] )

	out_file = sql_path + 'SDSS_match_ra%.3f_dec%.3f_obj.csv'
	Ns = len( s_ra )

	com_imag = []
	com_rmag = []

	com_dex = []
	com_ra, com_dec = [], []
	dl = []

	for tt in range( Ns ):

		c_dat = pds.read_csv( out_file % (s_ra[tt], s_dec[tt]), skiprows = 1)
		c_ra, c_dec = np.array(  c_dat['ra'] ), np.array(  c_dat['dec'] ) 
		c_imag = np.array( c_dat['cModelMag_i'] )
		c_rmag = np.array( c_dat['cModelMag_r'] )

		_nl_ = len( c_ra )

		if _nl_ == 0:
			continue

		if _nl_ == 1:

			delt_ra = np.abs( s_ra[tt] - c_ra )
			delt_dec = np.abs( s_dec[tt] - c_dec )

			delt = np.sqrt( delt_dec**2 + delt_ra**2 )
			dl.append( delt[0] )

			if np.min(delt) < 4e-4:
				com_dex.append( tt )
				com_imag.append( c_imag[0] )
				com_ra.append( c_ra )
				com_dec.append( c_dec )

				com_rmag.append( c_rmag[0] )

		if _nl_ > 1:

			delt_ra = np.abs( s_ra[tt] - c_ra )
			delt_dec = np.abs( s_dec[tt] - c_dec )

			delt = np.sqrt( delt_dec**2 + delt_ra**2 )
			dl.append( delt[ delt == delt.min() ][0] )

			if np.min(delt) < 4e-4:
				com_dex.append( tt )
				com_imag.append( c_imag[ delt == delt.min() ][0] )
				com_ra.append( c_ra[ delt == delt.min() ][0] )
				com_dec.append( c_dec[ delt == delt.min() ][0] )

				com_rmag.append( c_rmag[ delt == delt.min() ][0] )

	com_imag = np.array( com_imag )
	com_ra = np.array( com_ra )
	com_dec = np.array( com_dec )
	com_cx, com_cy = wcs_lis.all_world2pix( com_ra * U.deg, com_dec * U.deg, 0)

	tt_mag = obj_mag[ com_dex ] - com_imag
	idm = tt_mag <= -1

	tt_order = order_lis[ com_dex ]
	tt_cx, tt_cy = cx[ com_dex ], cy[ com_dex ]
	tt_A, tt_B, tt_phi = A[ com_dex ], B[ com_dex ], phi[ com_dex ]
	tt_ra, tt_dec = s_ra[ com_dex ], s_dec[ com_dex ]

	sigma = np.sqrt( np.sum( tt_mag**2 ) / len(tt_mag) )

	mag_bins = np.linspace(16.5, 24, 35)
	diff_list = list( set( order_lis ).difference( set(com_dex) ) )
	diff_mag = obj_mag[diff_list]

	#. save the compare arr
	# keys = ['cx', 'cy', 'A', 'B', 'phi', 'ra', 'dec', 'delt_Mag', 'order', 'mag_sdss']
	# values = [ tt_cx, tt_cy, tt_A, tt_B, tt_phi, tt_ra, tt_dec, tt_mag, tt_order, com_imag ]
	# fill = dict(zip( keys, values) )
	# out_data = pds.DataFrame( fill )

	# out_data.to_csv( img_path + 'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_re-scaled-cmag_SDSS_compare.csv' % (band_str, ra_g, dec_g, z_g) )
	# out_data.to_csv( img_path + 
	# 	'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_3-R-Kron_re-scaled-cmag_SDSS_compare.csv' % (band_str, ra_g, dec_g, z_g) )


	# plt.figure()
	# plt.plot( s_ra[ com_dex], s_dec[com_dex], 'ro', alpha = 0.5,)
	# plt.plot( com_ra, com_dec, 'g*', alpha = 0.5,)
	# plt.xlim( 197.20, 197.40 )
	# plt.xlabel( 'RA' )
	# plt.ylim( 21.15, 21.45 )
	# plt.ylabel( 'DEC' )
	# plt.savefig('/home/xkchen/position_compare.png', dpi = 300)
	# plt.close()

	fig = plt.figure( figsize = (10.6, 4.8) )
	ax0 = fig.add_axes([0.09, 0.13, 0.40, 0.82])
	ax1 = fig.add_axes([0.58, 0.13, 0.40, 0.82])

	# ax0.hist( obj_mag, bins = mag_bins, density = True, histtype = 'step', ls = '-', color = 'r', label = 'Mine')
	# ax0.axvline( x = np.nanmedian(obj_mag), ls = '-', color = 'r', ymin = 0.0, ymax = 0.35, alpha = 0.5,)

	ax0.hist( diff_mag, bins = mag_bins, density = True, histtype = 'step', ls = '-', color = 'b', label = 'Mine (only)')
	ax0.axvline( x = np.nanmedian(diff_mag), ls = '-', color = 'b', ymin = 0.0, ymax = 0.35, alpha = 0.5,)

	ax0.hist( obj_mag[com_dex], bins = mag_bins, density = True, histtype = 'step', ls = '-', color = 'r', label = 'Mine (SDSS)')
	ax0.axvline( x = np.nanmedian(obj_mag[com_dex]), ls = '-', color = 'r', ymin = 0.0, ymax = 0.35, alpha = 0.5,)

	ax0.hist( com_imag, bins = mag_bins, density = True, histtype = 'step', ls = '--', color = 'k', label = 'SDSS')
	ax0.axvline( x = np.nanmedian(com_imag), ls = '--', color = 'k', ymin = 0.0, ymax = 0.35, alpha = 0.5)

	ax0.legend( loc = 2, frameon = False, fontsize = 15,)
	ax0.set_xlim( 15, 24 )
	ax0.set_xlabel('${ \\rm i \\_ cmag } \; [mag]$', fontsize = 15,)
	ax0.set_ylabel('$ \\mathcal{pdf} $', fontsize = 15,)
	ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	ax1.plot( obj_mag[ com_dex ], com_imag, 'ko', markersize = 3, alpha = 0.75)
	ax1.plot( obj_mag[ com_dex ], obj_mag[ com_dex ], 'r-', alpha = 0.75)
	ax1.plot( obj_mag[ com_dex ], obj_mag[ com_dex ] - sigma, 'r:', alpha = 0.75)
	ax1.plot( obj_mag[ com_dex ], obj_mag[ com_dex ] + sigma, 'r:', alpha = 0.75)

	ax1.set_ylabel('$ { \\rm i \\_ cmag }_{ \; SDSS} \; [mag] $', fontsize = 15,)
	ax1.set_xlabel('$ { \\rm i \\_ cmag }_{ \; This \; work} \; [mag] $', fontsize = 15,)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	plt.savefig('/home/xkchen/i_cmag_compare.png', dpi = 300)
	plt.close()

	raise

	plt.figure()
	plt.plot( com_imag, tt_mag, 'bo', markersize = 1, alpha = 0.5)
	plt.axhline( y = 0, ls = '-', color = 'k', alpha = 0.5, linewidth = 5,)
	plt.axhline( y = np.nanmedian(obj_mag[ com_dex ] - com_imag), ls = '--', color = 'r',)

	plt.axhline( y = np.nanmedian(obj_mag[ com_dex ] - com_imag) - sigma, ls = ':', color = 'r')
	plt.axhline( y = np.nanmedian(obj_mag[ com_dex ] - com_imag) + sigma, ls = ':', color = 'r')

	plt.ylabel('$ mag_{i \,;\, Mine} - mag_{i \,;\, SDSS} \; [mag] $')
	plt.xlabel('$ mag_{i \,;\, SDSS} \; [mag] $')
	plt.savefig('/home/xkchen/i_cmag_compare.png', dpi = 300)
	plt.close()

	plt.figure()
	plt.plot( obj_mag[ com_dex ], com_imag, 'bo', markersize = 3, alpha = 0.5)
	plt.plot( obj_mag[ com_dex ][idm], com_imag[idm], 'k+', markersize = 5,)

	plt.plot( obj_mag[ com_dex ], obj_mag[ com_dex ], 'r-', alpha = 0.5)
	plt.plot( obj_mag[ com_dex ], obj_mag[ com_dex ] - sigma, 'r:', alpha = 0.5)
	plt.plot( obj_mag[ com_dex ], obj_mag[ com_dex ] + sigma, 'r:', alpha = 0.5)

	plt.ylabel('$ mag_{i \,;\, SDSS} \; [mag] $')
	plt.xlabel('$ mag_{i \,;\, Mine} \; [mag] $')
	plt.savefig('/home/xkchen/i_cmag_compare_1.png', dpi = 300)
	plt.close()


	fig = plt.figure( figsize = (10.6, 4.8) )
	ax0 = fig.add_axes([0.05, 0.13, 0.40, 0.82])
	ax1 = fig.add_axes([0.55, 0.13, 0.40, 0.82])

	ax0.imshow( img, origin = 'lower', cmap = 'Greys', norm = mpl.colors.LogNorm(),)
	ax0.scatter( cx, cy, marker = 'o', s = 30, facecolors = 'none', edgecolors = 'r', linewidth = 0.75,)
	ax0.scatter( com_cx, com_cy, marker = 's', s = 15, facecolors = 'none', edgecolors = 'b', linewidth = 0.75,)

	ax1.imshow( img, origin = 'lower', cmap = 'Greys', norm = mpl.colors.LogNorm(),)

	plt.savefig('/home/xkchen/objs_pos_compare.png', dpi = 300)
	plt.close()


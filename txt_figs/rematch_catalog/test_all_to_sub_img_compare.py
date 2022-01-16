import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C
import scipy.signal as signal

from astropy.table import Table
from astropy import cosmology as apcy
from scipy import interpolate as interp
from scipy import integrate as integ
from astropy.coordinates import SkyCoord
from scipy import ndimage
from scipy import optimize
from scipy import stats as sts


##... jack-sub images
def sersic_func(r, Ie, re, ndex):
	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir


from img_random_SB_fit import random_SB_fit_func, clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func
from light_measure import light_measure_weit
from img_cat_param_match import match_func
from img_pre_selection import extra_match_func

# cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']

psf_FWHM = 1.32 # arcsec
Da_ref = Test_model.angular_diameter_distance( z_ref ).value
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec

L_pix = Da_ref * 10**3 * pixel / rad2asec
n500 = 500 / L_pix
n1000 = 1000 / L_pix
n900 = 900 / L_pix


path = '/home/xkchen/figs/extend_bcgM_cat/SBs/'
img_path = '/home/xkchen/Downloads/'
out_path = '/home/xkchen/figs/extend_bcgM_cat/BGs/'
rand_path = '/home/xkchen/figs/re_measure_SBs/random_ref_SB/'


##... sub-28
pat = np.loadtxt('/home/xkchen/sub-28_half_1.txt')
p_ra, p_dec, p_z = pat[0], pat[1], pat[2]

pat = np.loadtxt('/home/xkchen/sub-28_half_0.txt')
p_ra, p_dec, p_z = np.r_[ p_ra, pat[0] ], np.r_[ p_dec, pat[1] ], np.r_[ p_z, pat[2] ]

p_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )


dat = pds.read_csv('/home/xkchen/gri_diff_cat.csv')
# dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/compare_to_pre/' + 
# 					'low_BCG_star-Mass_r-band_pre-diffi_BCG_cat.csv')
ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )


# re_coord = SkyCoord( ra = ra * U.deg, dec = dec * U.deg )

# idx, sep, d3d = p_coord.match_to_catalog_sky( re_coord )
# id_lim = sep.value < 2.7e-4
# mp_ra, mp_dec, mp_z = ra[ idx[id_lim ] ], dec[ idx[id_lim ] ], z[ idx[id_lim ] ]

# out_arr = np.array([ mp_ra, mp_dec, mp_z])
# np.savetxt('/home/xkchen/sub-28_dmp-cat.txt', out_arr )


Nz = len( z )
Nep = Nz // 11

da0 = np.int( Nep ) * 10
set_ra, set_dec, set_z = ra[da0:], dec[da0:], z[da0:]

out_arr = np.array( [ set_ra, set_dec, set_z ] )
np.savetxt('/home/xkchen/extra-500_sub-10.txt', out_arr )


raise

kk = 0

with h5py.File( '/home/xkchen/rgi-differ-cat_%s-band_stack_test_img.h5' % band[kk], 'r') as f:
	tot_img = np.array( f['a'] )

for pp in range( 11 ):

	# with h5py.File( '/home/xkchen/rgi-differ-cat_%s-band_stack_test_img.h5' % band[kk], 'r') as f:

	with h5py.File( '/home/xkchen/Downloads/rgi-differ-cat_r-band_stack_test_img_%d-rank.h5' % pp, 'r') as f:
		tmp_img = np.array( f['a'])
	
	cen_x, cen_y = np.int( tmp_img.shape[1] / 2 ), np.int( tmp_img.shape[0] / 2 )

	idnn = np.isnan( tmp_img )
	idy_lim, idx_lim = np.where(idnn == False)
	x_lo_lim, x_up_lim = idx_lim.min(), idx_lim.max()
	y_lo_lim, y_up_lim = idy_lim.min(), idy_lim.max()


	## random imgs
	with h5py.File( rand_path + 'random_%s-band_rand-stack_Mean_jack_img_z-ref-aveg.h5' % band[kk], 'r') as f:
		rand_img = np.array( f['a'])
	xn, yn = np.int( rand_img.shape[1] / 2 ), np.int( rand_img.shape[0] / 2 )

	idnn = np.isnan( rand_img )
	idy_lim, idx_lim = np.where( idnn == False)
	x_lo_eff, x_up_eff = idx_lim.min(), idx_lim.max()
	y_lo_eff, y_up_eff = idy_lim.min(), idy_lim.max()


	## BG-estimate params
	BG_file = out_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-profile_params_diag-fit.csv' % band[kk]

	cat = pds.read_csv( BG_file )
	offD, I_e, R_e = np.array(cat['offD'])[0], np.array(cat['I_e'])[0], np.array(cat['R_e'])[0]
	sb_2Mpc = sersic_func( 2e3, I_e, R_e, 2.1)

	shift_rand_img = rand_img / pixel**2 - offD + sb_2Mpc
	BG_sub_img = tmp_img / pixel**2 - shift_rand_img

	idnn = np.isnan( BG_sub_img )
	idy_lim, idx_lim = np.where( idnn == False)
	x_lo_cut, x_up_cut = idx_lim.min(), idx_lim.max()
	y_lo_cut, y_up_cut = idy_lim.min(), idy_lim.max()


	#. differ to total image
	diff_img = ( tot_img - tmp_img ) / pixel**2

	cut_BG_sub_img = BG_sub_img[ y_lo_cut: y_up_cut + 1, x_lo_cut: x_up_cut + 1 ]
	id_nan = np.isnan( cut_BG_sub_img )
	cut_BG_sub_img[id_nan] = 0.


	y_peak, x_peak = np.where( cut_BG_sub_img == np.nanmax( cut_BG_sub_img ) )
	y_peak, x_peak = y_peak[0], x_peak[0]
	cen_region = cut_BG_sub_img[ y_peak - np.int( 3 * n500 ) : y_peak + np.int( 3 * n500 ) + 1, 
									x_peak - np.int( 3 * n500 ) : x_peak + np.int( 3 * n500 ) + 1 ]

	filt_img_4 = ndimage.gaussian_filter( cen_region, sigma = 15,)
	mag_map_4 = 22.5 - 2.5 * np.log10( filt_img_4 )


	cut_diff_img = diff_img[ y_lo_cut: y_up_cut + 1, x_lo_cut: x_up_cut + 1 ]
	id_nan = np.isnan( cut_diff_img )
	cut_diff_img[id_nan] = 0.

	cen_diff_img = cut_diff_img[ y_peak - np.int( 3 * n500 ) : y_peak + np.int( 3 * n500 ) + 1, 
							 	 x_peak - np.int( 3 * n500 ) : x_peak + np.int( 3 * n500 ) + 1 ]

	filt_diff = ndimage.gaussian_filter( cen_diff_img, sigma = 15,)


	dl0 = 250
	dl1 = 450

	xc, yc = np.int( 3 * n500 ), np.int( 3 * n500 )

	plt.figure()
	ax = plt.subplot(111)
	ax.imshow( filt_diff, origin = 'lower', cmap = 'bwr', vmin = -1e-2, vmax = 1e-2,)

	Box = Rectangle( xy = (xc - dl0, yc - dl0), width = dl0, height = dl0, fill = False, 
						ec = 'r', ls = '--', linewidth = 1, )
	ax.add_patch( Box )

	Box = Rectangle( xy = (xc - dl1, yc - dl1), width = dl1, height = dl1, fill = False, 
						ec = 'r', ls = '-', linewidth = 1, )
	ax.add_patch( Box )

	plt.savefig('/home/xkchen/extra-500_diff_contour_%d.png' % pp, dpi = 300)
	plt.close()


	color_lis = []

	for jj in np.arange( 11 ):
		color_lis.append( mpl.cm.rainbow_r( jj / 10) )

	dd_lis = np.arange(26, 33, 1)


	plt.figure()
	ax2 = plt.subplot(111)
	ax2.imshow( cen_region, origin = 'lower', cmap = 'Greys', vmin = -2e-2, vmax = 2e-2,)
	ax2.contour( mag_map_4, origin = 'lower', levels = dd_lis, 
							colors = [color_lis[0], color_lis[2], color_lis[3], color_lis[4], 
										color_lis[6], color_lis[8], color_lis[10] ],)

	Box = Rectangle( xy = (xc - dl0, yc - dl0), width = dl0, height = dl0, fill = False, 
						ec = 'r', ls = '--', linewidth = 1,)
	ax2.add_patch( Box )

	Box = Rectangle( xy = (xc - dl1, yc - dl1), width = dl1, height = dl1, fill = False, 
						ec = 'r', ls = '-', linewidth = 1,)
	ax2.add_patch( Box )

	plt.savefig('/home/xkchen/extra-500_sub-%d.png' % pp, dpi = 300)
	plt.close()

raise



##... total
N_samples = 30

for kk in range( 1 ):

	## all clusters
	with h5py.File( img_path + 
		'photo-z_match_tot-BCG-star-Mass_%s-band_Mean_jack_img_z-ref.h5' % band[kk], 'r') as f:
		tot_img = np.array( f['a'] )

	cen_x, cen_y = np.int( tot_img.shape[1] / 2 ), np.int( tot_img.shape[0] / 2 )


	with h5py.File( '/home/xkchen/figs/re_measure_SBs/SBs/' + 
					'photo-z_match_tot-BCG-star-Mass_%s-band_Mean_jack_img_z-ref.h5' % band[kk], 'r') as f:	
		pre_img = np.array( f['a'] )

	diff_img = tot_img - pre_img

	cut_diff_img = diff_img[ cen_y - np.int( 3 * n500 ): cen_y + np.int( 3 * n500 ) + 1,
							 cen_x - np.int( 3 * n500 ): cen_x + np.int( 3 * n500 ) + 1 ]

	filt_diff = ndimage.gaussian_filter( cut_diff_img, sigma = 15,)


	fig = plt.figure()
	ax = fig.add_axes([0.02, 0.10, 0.80, 0.80])

	tf = ax.imshow( filt_diff / pixel**2, origin = 'lower', cmap = 'bwr', vmin = -1e-3, vmax = 1e-3,)

	dl0 = 250
	dl1 = 450

	xc, yc = np.int( 3 * n500 ), np.int( 3 * n500 )

	Box = Rectangle( xy = (xc - dl0, yc - dl0), width = dl0, height = dl0, fill = False, 
						ec = 'r', ls = '--', linewidth = 1, alpha = 0.75)
	ax.add_patch( Box )

	Box = Rectangle( xy = (xc - dl1, yc - dl1), width = dl1, height = dl1, fill = False, 
						ec = 'r', ls = '-', linewidth = 1, alpha = 0.75)
	ax.add_patch( Box )

	plt.colorbar( tf, ax = ax, fraction = 0.035, pad = 0.01,)
	plt.savefig('/home/xkchen/total_diff_img.png', dpi = 300)
	plt.close()

	raise


	for jj in range( 28,29 ):

		## flux imgs
		with h5py.File( img_path + 
			'photo-z_match_tot-BCG-star-Mass_%s-band_jack-sub-%d_img_z-ref.h5' % (band[kk], jj), 'r') as f:
			tmp_img = np.array( f['a'])
		cen_x, cen_y = np.int( tmp_img.shape[1] / 2 ), np.int( tmp_img.shape[0] / 2 )

		idnn = np.isnan( tmp_img )
		idy_lim, idx_lim = np.where(idnn == False)
		x_lo_lim, x_up_lim = idx_lim.min(), idx_lim.max()
		y_lo_lim, y_up_lim = idy_lim.min(), idy_lim.max()

		## random imgs
		with h5py.File( rand_path + 'random_%s-band_rand-stack_Mean_jack_img_z-ref-aveg.h5' % band[kk], 'r') as f:
			rand_img = np.array( f['a'])
		xn, yn = np.int( rand_img.shape[1] / 2 ), np.int( rand_img.shape[0] / 2 )

		idnn = np.isnan( rand_img )
		idy_lim, idx_lim = np.where( idnn == False)
		x_lo_eff, x_up_eff = idx_lim.min(), idx_lim.max()
		y_lo_eff, y_up_eff = idy_lim.min(), idy_lim.max()


		## BG-estimate params
		BG_file = out_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-profile_params_diag-fit.csv' % band[kk]

		cat = pds.read_csv( BG_file )
		offD, I_e, R_e = np.array(cat['offD'])[0], np.array(cat['I_e'])[0], np.array(cat['R_e'])[0]
		sb_2Mpc = sersic_func( 2e3, I_e, R_e, 2.1)

		shift_rand_img = rand_img / pixel**2 - offD + sb_2Mpc
		BG_sub_img = tmp_img / pixel**2 - shift_rand_img


		cut_tot_img = tot_img / pixel**2 - shift_rand_img
		cut_tot_img = cut_tot_img[ cen_y - np.int( 3 * n500 ): cen_y + np.int( 3 * n500 ) + 1,
								cen_x - np.int( 3 * n500 ): cen_x + np.int( 3 * n500 ) + 1 ]

		cut_sub_img = BG_sub_img[ cen_y - np.int( 3 * n500 ): cen_y + np.int( 3 * n500 ) + 1,
								cen_x - np.int( 3 * n500 ): cen_x + np.int( 3 * n500 ) + 1 ]


		filt_img0 = ndimage.gaussian_filter( cut_tot_img, sigma = 15,) # 21
		mag_0 = 22.5 - 2.5 * np.log10( filt_img0 )

		filt_img1 = ndimage.gaussian_filter( cut_sub_img, sigma = 15,)
		mag_1 = 22.5 - 2.5 * np.log10( filt_img1 )

		smooth_diff = filt_img0 - filt_img1


		xp, yp = np.int( cut_sub_img.shape[1] / 2 ), np.int( cut_sub_img.shape[0] / 2 )

		fig = plt.figure( figsize = (19.84, 4.8) )
		ax0 = fig.add_axes([0.03, 0.09, 0.30, 0.85])
		ax1 = fig.add_axes([0.38, 0.09, 0.30, 0.85])
		ax2 = fig.add_axes([0.73, 0.09, 0.25, 0.85])

		ax0.set_title('total')
		ax0.imshow( filt_img0, origin = 'lower', cmap = 'Greys', vmin = -1e-4, vmax = 1e0, 
					norm = mpl.colors.SymLogNorm( linthresh = 1e-4, linscale = 5e-4, base = 10),)
		ax0.plot( np.int( 3 * n500 ), np.int( 3 * n500 ), 'cP', )

		ax0.set_xlim( xp - np.ceil(1.11 * n1000), xp + np.ceil(1.11 * n1000 ) )
		ax0.set_ylim( yp - np.ceil(1.11 * n1000), yp + np.ceil(1.11 * n1000 ) )

		ax1.set_title('jack-sub-%d' % jj)
		ax1.imshow( filt_img1, origin = 'lower', cmap = 'Greys', vmin = -1e-4, vmax = 1e0, 
					norm = mpl.colors.SymLogNorm( linthresh = 1e-4, linscale = 5e-4, base = 10),)
		ax1.plot( np.int( 3 * n500 ), np.int( 3 * n500 ), 'cP', )

		ax1.set_xlim( xp - np.ceil(1.11 * n1000), xp + np.ceil(1.11 * n1000 ) )
		ax1.set_ylim( yp - np.ceil(1.11 * n1000), yp + np.ceil(1.11 * n1000 ) )

		ax2.set_title('difference')
		ax2.imshow( smooth_diff, origin = 'lower', cmap = 'bwr', vmin = -5e-4, vmax = 5e-4,)
		ax2.plot( np.int( 3 * n500 ), np.int( 3 * n500 ), 'cP', )

		ax2.set_xlim( xp - np.ceil(1.11 * n1000), xp + np.ceil(1.11 * n1000 ) )
		ax2.set_ylim( yp - np.ceil(1.11 * n1000), yp + np.ceil(1.11 * n1000 ) )


		plt.savefig('/home/xkchen/differ_img_jk-sub-%d.png' % jj, dpi = 300)
		plt.close()

"""
		idnn = np.isnan( BG_sub_img )
		idy_lim, idx_lim = np.where( idnn == False)
		x_lo_cut, x_up_cut = idx_lim.min(), idx_lim.max()
		y_lo_cut, y_up_cut = idy_lim.min(), idy_lim.max()

		## 2D signal
		cut_img = tmp_img[ y_lo_lim: y_up_lim + 1, x_lo_lim: x_up_lim + 1 ] / pixel**2
		id_nan = np.isnan( cut_img )
		cut_img[id_nan] = 0.

		cut_rand = rand_img[ y_lo_eff: y_up_eff + 1, x_lo_eff: x_up_eff + 1 ] / pixel**2
		id_nan = np.isnan( cut_rand )
		cut_rand[id_nan] = 0.

		cut_off_rand = shift_rand_img[ y_lo_eff: y_up_eff + 1, x_lo_eff: x_up_eff + 1 ]
		id_nan = np.isnan( cut_off_rand )
		cut_off_rand[id_nan] = 0.

		cut_BG_sub_img = BG_sub_img[ y_lo_cut: y_up_cut + 1, x_lo_cut: x_up_cut + 1 ]
		id_nan = np.isnan( cut_BG_sub_img )
		cut_BG_sub_img[id_nan] = 0.

		cen_y, cen_x = np.where( cut_BG_sub_img == np.nanmax( cut_BG_sub_img ) )
		cen_y, cen_x = cen_y[0], cen_x[0]
		cen_region = cut_BG_sub_img[ cen_y - np.int( 3 * n500 ) : cen_y + np.int( 3 * n500 ) + 1, cen_x - np.int( 3 * n500 ) : cen_x + np.int( 3 * n500 ) + 1 ]

		filt_img_0 = ndimage.gaussian_filter( cen_region, sigma = 3,)
		mag_map_0 = 22.5 - 2.5 * np.log10( filt_img_0 )

		filt_img_1 = ndimage.gaussian_filter( cen_region, sigma = 7,)
		mag_map_1 = 22.5 - 2.5 * np.log10( filt_img_1 )

		filt_img_2 = ndimage.gaussian_filter( cen_region, sigma = 11,)
		mag_map_2 = 22.5 - 2.5 * np.log10( filt_img_2 )

		filt_img_3 = ndimage.gaussian_filter( cen_region, sigma = 17,)
		mag_map_3 = 22.5 - 2.5 * np.log10( filt_img_3 )

		filt_img_4 = ndimage.gaussian_filter( cen_region, sigma = 21,)  # sigma = 21, 27, 15
		mag_map_4 = 22.5 - 2.5 * np.log10( filt_img_4 )

		filt_BG_sub_img = ndimage.gaussian_filter( cut_BG_sub_img, sigma = 21,)  # sigma = 21
		filt_BG_sub_mag = 22.5 - 2.5 * np.log10( filt_BG_sub_img )


		#.figs
		color_lis = []

		for ll in np.arange( 11 ):
			color_lis.append( mpl.cm.rainbow_r( ll / 10) )

		dd_lis = np.arange(26, 33, 1)
		_mag_img = 22.5 - 2.5 * np.log10( cut_BG_sub_img )		


		fig = plt.figure( )
		ax2 = fig.add_axes( [0.155, 0.11, 0.75, 0.75] )

		ax2.pcolormesh( cut_BG_sub_img, cmap = 'Greys', vmin = -2e-2, vmax = 2e-2, alpha = 0.75,)

		ax2.contour( mag_map_0, origin = 'lower', levels = [26, 100], colors = [ color_lis[0], 'w'], 
			extent = (cen_x - np.int( 3 * n500 ), cen_x + np.int( 3 * n500 ) + 1, cen_y - np.int( 3 * n500 ), cen_y + np.int( 3 * n500 ) + 1), )

		ax2.contour( mag_map_1, origin = 'lower', levels = [27, 100], colors = [ color_lis[2], 'w'], 
			extent = (cen_x - np.int( 3 * n500 ), cen_x + np.int( 3 * n500 ) + 1, cen_y - np.int( 3 * n500 ), cen_y + np.int( 3 * n500 ) + 1), )

		ax2.contour( mag_map_2, origin = 'lower', levels = [28, 100], colors = [ color_lis[3], 'w'], 
			extent = (cen_x - np.int( 3 * n500 ), cen_x + np.int( 3 * n500 ) + 1, cen_y - np.int( 3 * n500 ), cen_y + np.int( 3 * n500 ) + 1), )

		ax2.contour( mag_map_3, origin = 'lower', levels = [29, 100], colors = [ color_lis[4], 'w'], 
			extent = (cen_x - np.int( 3 * n500 ), cen_x + np.int( 3 * n500 ) + 1, cen_y - np.int( 3 * n500 ), cen_y + np.int( 3 * n500 ) + 1), )

		ax2.contour( mag_map_4, origin = 'lower', levels = [30, 100], colors = [ color_lis[6], 'w'], 
			extent = (cen_x - np.int( 3 * n500 ), cen_x + np.int( 3 * n500 ) + 1, cen_y - np.int( 3 * n500 ), cen_y + np.int( 3 * n500 ) + 1), )

		ax2.contour( filt_BG_sub_mag, origin = 'lower',  levels = [31, 32, 100], colors = [ color_lis[8], color_lis[10], 'w'], 
			extent = (0, x_up_cut + 1 - x_lo_cut, 0, y_up_cut + 1 - y_lo_cut), )

		# ax2.set_xticklabels( labels = [] )
		# ax2.set_yticklabels( labels = [] )

		# x_ticks_0 = np.arange( xn - x_lo_cut, 0, -1 * n500)
		# x_ticks_1 = np.arange( xn - x_lo_cut, cut_rand.shape[1], n500)
		# x_ticks = np.r_[ x_ticks_0[::-1], x_ticks_1[1:] ]

		# tick_R = np.r_[ np.arange( -(len(x_ticks_0) - 1 ) * 500, 0, 500), np.arange(0, 500 * ( len(x_ticks_1) ), 500) ]
		# tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

		# ax2.set_xticks( x_ticks, minor = True, )
		# ax2.set_xticklabels( labels = tick_lis, minor = True, fontsize = 15,)
		# ax2.xaxis.set_ticks_position('bottom')
		# ax2.set_xlabel( '$\\mathrm{X} \; [\\mathrm{M}pc] $', fontsize = 15, )

		# y_ticks_0 = np.arange( yn - y_lo_cut, 0, -1 * n500)
		# y_ticks_1 = np.arange( yn - y_lo_cut, cut_rand.shape[0], n500)
		# y_ticks = np.r_[ y_ticks_0[::-1], y_ticks_1[1:] ]

		# tick_R = np.r_[ np.arange( -(len(y_ticks_0) - 1 ) * 500, 0, 500), np.arange(0, 500 * ( len(y_ticks_1) ), 500) ]
		# tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

		# ax2.set_yticks( y_ticks, minor = True )
		# ax2.set_yticklabels( labels = tick_lis, minor = True, fontsize = 15,)
		# ax2.set_ylabel( '$\\mathrm{Y} \; [\\mathrm{M}pc] $', fontsize = 15,)
		# ax2.tick_params( axis = 'both', which = 'major', direction = 'in', bottom = False, left = False, top = False, labelsize = 15,)

		xp, yp = np.int( cut_BG_sub_img.shape[1] / 2 ), np.int( cut_BG_sub_img.shape[0] / 2 )
		ax2.set_xlim( xp - np.ceil(1.11 * n1000), xp + np.ceil(1.11 * n1000 ) )
		ax2.set_ylim( yp - np.ceil(1.11 * n1000), yp + np.ceil(1.11 * n1000 ) )

		plt.savefig('/home/xkchen/%s-band_%d-jk-sub_img.png' % (band[kk], jj), dpi = 300)
		plt.show()

"""


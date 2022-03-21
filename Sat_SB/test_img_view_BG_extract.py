from re import T
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Ellipse

import h5py
import numpy as np
import pandas as pds

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc
import scipy.stats as sts

from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord

from light_measure import light_measure_weit
from img_sat_BG_extract import BG_build_func
from img_sat_BG_extract import sat_BG_extract_func


##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1. - Omega_m
Omega_k = 1. - (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']


### === local test (Mock 2D background image and located satellites at z_ref)
def local_test():

	# BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BGs/'
	# cat_lis = ['inner-mem', 'outer-mem']

	##. fixed i_Mag_10
	BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/BGs/'
	cat_lis = ['inner', 'middle', 'outer']


	##... SB profile of BCG+ICL+BG (as the background)
	tmp_bR, tmp_BG, tmp_BG_err = [], [], []
	tmp_img = []

	for mm in range( 3 ):

		_sub_bg_R, _sub_bg_sb, _sub_bg_err = [], [], []
		_sub_img = []

		for kk in range( 3 ):

			#. 1D profile
			with h5py.File( BG_path + 
					'photo-z_match_tot-BCG-star-Mass_%s_%s-band_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mm], band[kk]), 'r') as f:

				tt_r = np.array(f['r'])
				tt_sb = np.array(f['sb'])
				tt_err = np.array(f['sb_err'])

			#. 2D image
			R_max = 3e3

			out_file = '/home/xkchen/%s_%s-band_BG_img.fits' % (cat_lis[mm], band[kk])
			# BG_build_func( tt_r, tt_sb, z_ref, pixel, R_max, out_file)

			tt_img = fits.open( out_file )
			tt_img_arr = tt_img[0].data

			_sub_bg_R.append( tt_r )
			_sub_bg_sb.append( tt_sb )
			_sub_bg_err.append( tt_err )
			_sub_img.append( tt_img_arr )

		tmp_bR.append( _sub_bg_R )
		tmp_BG.append( _sub_bg_sb )
		tmp_BG_err.append( _sub_bg_err )
		tmp_img.append( _sub_img )


	for mm in range( 2,3 ):

		for kk in range( 3 ):

			sub_img = tmp_img[mm][kk] + 0.
			sub_cont = np.ones( ( sub_img.shape[0], sub_img.shape[1] ), )
			r_bins = np.logspace( 0, 3.48, 55 )

			xn, yn = np.int( sub_img.shape[1] / 2 ), np.int( sub_img.shape[0] / 2 )

			Intns, phy_r, Intns_err, npix, nratio = light_measure_weit( sub_img, sub_cont, pixel, xn, yn, z_ref, r_bins)

			id_vx = npix > 0
			_kk_R, _kk_sb, _kk_err = phy_r[ id_vx ], Intns[ id_vx ], Intns_err[ id_vx ]


			fig = plt.figure( figsize = (10.0, 4.8) )

			ax2 = fig.add_axes([0.05, 0.12, 0.39, 0.80])
			ax3 = fig.add_axes([0.59, 0.12, 0.39, 0.80])
			cb_ax2 = fig.add_axes( [0.44, 0.12, 0.02, 0.8] )

			ax2.pcolormesh( np.log10( tmp_img[mm][kk] / pixel**2 ), cmap = 'rainbow', vmin = -3, vmax = 0,)

			cmap = mpl.cm.rainbow
			norm = mpl.colors.Normalize( vmin = -3, vmax = 0 )
			c_ticks = np.array( [-3, -2, -1, 0] )
			cbs = mpl.colorbar.ColorbarBase( ax = cb_ax2, cmap = cmap, norm = norm, extend = 'neither', ticks = c_ticks, 
											orientation = 'vertical',)

			cbs.set_label( '$\\lg \, \\mu $',)

			ax3.plot( tmp_bR[mm][kk], tmp_BG[mm][kk], ls = '-', color = 'k', alpha = 0.5,)
			ax3.plot( _kk_R, _kk_sb, ls = '--', color = 'r', alpha = 0.5, )

			ax3.set_xlim( 3e0, 3e3)
			ax3.set_xscale('log')
			ax3.set_xlabel('$R \; [kpc]$')

			ax3.set_ylabel('$\\mu_{%s} \; [nanomaggies \, / \, arcsec^{2}]$' % band[kk],)
			ax3.set_yscale('log')

			plt.savefig('/home/xkchen/%s_%s-band_BG_img.png' % (cat_lis[mm],band[kk]), dpi = 300)
			plt.close()

# local_test()
# raise


### === take the stacked 2D image as background (then cutout background images baed on satellite location at z_ref)
def BG_2D_with_stack_img():

	##. fixed i_Mag_10, N_g weighted stacked cluster image
	# BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/BGs/'
	# cat_lis = ['inner', 'middle', 'outer']

	# for mm in range( 3 ):

	# 	for kk in range( 3 ):
	# 		with h5py.File( BG_path + 
	# 				'photo-z_match_tot-BCG-star-Mass_%s_%s-band_Mean_jack_img_z-ref.h5' % (cat_lis[mm], band[kk]), 'r') as f:
	# 			tmp_img = np.array( f['a'] )

	# 		Nx, Ny = tmp_img.shape[1], tmp_img.shape[0]
	# 		ref_pix_x, ref_pix_y = Nx / 2, Ny / 2

	# 		#. save the img file
	# 		out_file = '/home/xkchen/stacked_cluster_%s_%s-band_img.fits' % (cat_lis[mm], band[kk])

	# 		keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2', 'CENTER_X','CENTER_Y', 'Z', 'P_SCALE' ]
	# 		value = ['T', 32, 2, Nx, Ny, ref_pix_x, ref_pix_y, z_ref, pixel ]
	# 		ff = dict( zip( keys, value) )
	# 		fill = fits.Header( ff )
	# 		fits.writeto( out_file, tmp_img, header = fill, overwrite = True)


	##. use the entire cluster sample stacked image as background (without N_g weight)
	for kk in range( 3 ):

		with h5py.File('/home/xkchen/figs/extend_bcgM_cat/SBs/photo-z_match_tot-BCG-star-Mass_%s-band_Mean_jack_img_z-ref.h5' % band[kk], 'r') as f:
			tmp_img = np.array( f['a'] )

		Nx, Ny = tmp_img.shape[1], tmp_img.shape[0]
		ref_pix_x, ref_pix_y = Nx / 2, Ny / 2

		#. save the img file
		out_file = '/home/xkchen/stacked_all_cluster_%s-band_img.fits' % band[kk]

		keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2', 'CENTER_X','CENTER_Y', 'Z', 'P_SCALE' ]
		value = ['T', 32, 2, Nx, Ny, ref_pix_x, ref_pix_y, z_ref, pixel ]
		ff = dict( zip( keys, value) )
		fill = fits.Header( ff )
		fits.writeto( out_file, tmp_img, header = fill, overwrite = True)

# BG_2D_with_stack_img()


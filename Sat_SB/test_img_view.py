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


### === satellite image extract test
def sat_cut_img():

	from img_sat_extract import sate_Extract_func
	from img_sat_extract import sate_surround_mask_func

	home = '/home/xkchen/data/SDSS/'
	load = '/home/xkchen/fig_tmp/'

	dat = pds.read_csv( home + 'member_files/sat_cat_z02_03/Extend-BCGM_rgi-common_cat.csv')
	ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
	clus_ID = np.array( dat['clust_ID'])

	Ns = len( ra )

	dat = pds.read_csv( home + 'member_files/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-limit_member-cat.csv')
	s_ra, s_dec, s_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z_spec'] )
	s_host_ID = np.array( dat['clus_ID'] )
	s_host_ID = s_host_ID.astype( int )


	for tt in range( 1 ):

		band_str = band[ tt ]

		for kk in range( 1 ):

			ra_g, dec_g, z_g = sub_ra[ kk ], sub_dec[ kk ], sub_z[ kk]

			kk_ID = sub_clusID[ kk ]

			id_vx = s_host_ID == kk_ID
			lim_ra, lim_dec, lim_z = s_ra[ id_vx ], s_dec[ id_vx ], s_z[ id_vx ]

			# R_cut = 2.5  ## scaled case
			R_cut = 320

			d_file = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
			gal_file = home + 'photo_files/detect_source_cat/photo-z_img_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
			offset_file = home + 'photo_files/pos_offset_correct_imgs/offset/%s-band_ra%.3f_dec%.3f_z%.3f_star-pos-offset.csv'

			##... pre cutout
			out_file = '/home/xkchen/figs/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_img.fits'
			sate_Extract_func( d_file, ra_g, dec_g, z_g, lim_ra, lim_dec, lim_z, band_str, gal_file, out_file, R_cut, offset_file = offset_file)

			##... image mask
			cat_file = home + 'photo_files/star_cats/source_SQL_Z%.3f_ra%.3f_dec%.3f.csv'
			out_mask_file = '/home/xkchen/figs/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_mask-img.fits'

			if band_str == 'r':
				extra_cat = [ home + 'photo_files/detect_source_cat/photo-z_img_g-band_mask_ra%.3f_dec%.3f_z%.3f.cat', 
							  home + 'photo_files/detect_source_cat/photo-z_img_i-band_mask_ra%.3f_dec%.3f_z%.3f.cat']

				extra_img = [ home + 'photo_data/frame-g-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',
							  home + 'photo_data/frame-i-ra%.3f-dec%.3f-redshift%.3f.fits.bz2']

			if band_str == 'g':
				extra_cat = [ home + 'photo_files/detect_source_cat/photo-z_img_r-band_mask_ra%.3f_dec%.3f_z%.3f.cat', 
							  home + 'photo_files/detect_source_cat/photo-z_img_i-band_mask_ra%.3f_dec%.3f_z%.3f.cat']

				extra_img = [ home + 'photo_data/frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',
							  home + 'photo_data/frame-i-ra%.3f-dec%.3f-redshift%.3f.fits.bz2']

			if band_str == 'i':
				extra_cat = [ home + 'photo_files/detect_source_cat/photo-z_img_r-band_mask_ra%.3f_dec%.3f_z%.3f.cat',
							  home + 'photo_files/detect_source_cat/photo-z_img_g-band_mask_ra%.3f_dec%.3f_z%.3f.cat']

				extra_img = [ home + 'photo_data/frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',
							  home + 'photo_data/frame-g-ra%.3f-dec%.3f-redshift%.3f.fits.bz2']

			tt2 = time.time()

			sate_surround_mask_func(d_file, cat_file, ra_g, dec_g, z_g, lim_ra, lim_dec, lim_z, band_str, gal_file, out_mask_file, R_cut, 
									offset_file = offset_file, extra_cat = extra_cat, extra_img = extra_img, stack_info = stack_cat )

			print( time.time() - tt2 )

	return

def sat_cut_test():

	home = '/home/xkchen/data/SDSS/'
	load = '/home/xkchen/figs/'

	#. cluster cat
	dat = pds.read_csv( home + 'member_files/sat_cat_z02_03/Extend-BCGM_rgi-common_cat.csv')
	ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
	clus_ID = np.array( dat['clust_ID'])

	N_clus = len( ra )


	dat = pds.read_csv( home + 'member_files/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-limit_member-cat.csv')
	s_ra, s_dec, s_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z_spec'] )
	s_host_ID = np.array( dat['clus_ID'] )
	s_host_ID = s_host_ID.astype( int )


	### ... image cut out
	band_str = band[0]

	d_file = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
	out_file = '/home/xkchen/figs/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_img.fits'
	out_mask_file = '/home/xkchen/figs/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_mask-img.fits'

	for kk in range( 1 ):

		ra_g, dec_g, z_g = ra[ kk ], dec[ kk ], z[ kk]

		kk_ID = clus_ID[ kk ]
		id_vx = s_host_ID == kk_ID
		lim_ra, lim_dec, lim_z = s_ra[ id_vx ], s_dec[ id_vx ], s_z[ id_vx ]

		img_data = fits.open( d_file % (band_str, ra_g, dec_g, z_g), )
		img_arr = img_data[0].data
		Header = img_data[0].header

		wcs_lis = awc.WCS( Header )

		pos_x, pos_y = wcs_lis.all_world2pix( lim_ra, lim_dec, 0 )
		cen_x, cen_y = wcs_lis.all_world2pix( ra_g, dec_g, 0 )

		N_sat = len( lim_ra )


		#. view on source detection
		source = asc.read( home + 
					'photo_files/detect_source_cat/photo-z_img_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat' % (band_str, ra_g, dec_g, z_g),)
		Numb = np.array(source['NUMBER'][-1])
		A = np.array(source['A_IMAGE'])
		B = np.array(source['B_IMAGE'])
		theta = np.array(source['THETA_IMAGE'])
		
		cx = np.array(source['X_IMAGE'])
		cy = np.array(source['Y_IMAGE'])

		pk_cx = np.array(source['XPEAK_IMAGE'])
		pk_cy = np.array(source['YPEAK_IMAGE'])

		a = 10 * A
		b = 10 * B


		id_xm = A == np.max(A)
		t_cx, t_cy = cx[ id_xm ], cy[ id_xm ]
		t_a, t_b = a[ id_xm ], b[ id_xm ]
		t_chi = theta[ id_xm ]


		plt.figure()
		ax = plt.subplot(111)

		ax.imshow( img_arr, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm(),)
		ax.scatter( pos_x, pos_y, s = 20, edgecolors = 'r', facecolors = 'none',)
		ax.scatter( cen_x, cen_y, s = 20, marker = 's', edgecolors = 'b', facecolors = 'none',)

		# for ll in range( Numb ):
		# 	ellips = Ellipse( xy = (cx[ll], cy[ll]), width = a[ll], height = b[ll], angle = theta[ll], fill = False, 
		# 		ec = 'm', ls = '--', linewidth = 0.75, )
		# 	ax.add_patch( ellips )

		ellips = Ellipse( xy = (t_cx, t_cy), width = t_a, height = t_b, angle = t_chi, fill = False, 
			ec = 'm', ls = '-', linewidth = 0.75, )
		ax.add_patch( ellips )

		ax.set_xlim( 0, 2048 )
		ax.set_ylim( 0, 1489 )

		plt.savefig('/home/xkchen/figs/cluster_%s-band_ra%.3f_dec%.3f_z%.3f.png' % (band_str, ra_g, dec_g, z_g), dpi = 300)
		plt.close()


		for pp in range( N_sat ):

			kk_ra, kk_dec = lim_ra[ pp ], lim_dec[ pp ] 

			cut_img = fits.open( out_file % (band_str, ra_g, dec_g, z_g, kk_ra, kk_dec),)
			cut_img_arr = cut_img[0].data
			kk_px, kk_py = cut_img[0].header['CENTER_X'], cut_img[0].header['CENTER_Y']
			_pkx, _pky = cut_img[0].header['PEAK_X'], cut_img[0].header['PEAK_Y']

			cut_mask = fits.open( out_mask_file % (band_str, ra_g, dec_g, z_g, kk_ra, kk_dec),)
			cut_mask_arr = cut_mask[0].data
			cp_px, cp_py = cut_mask[0].header['CENTER_X'], cut_mask[0].header['CENTER_Y']

			fig = plt.figure( figsize = (13.12, 4.8) )
			ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
			ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

			ax0.imshow( cut_img_arr, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm(),)
			ax0.scatter( kk_px, kk_py, s = 20, marker = 's', edgecolors = 'b', facecolors = 'none',)
			ax0.scatter( _pkx, _pky, s = 20, marker = 'o', edgecolors = 'r', facecolors = 'none',)

			ax0.set_xlim( kk_px - 75, kk_px + 75 )
			ax0.set_ylim( kk_py - 75, kk_py + 75 )

			ax1.imshow( cut_mask_arr, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm(),)
			ax1.scatter( kk_px, kk_py, s = 20, marker = 's', edgecolors = 'b', facecolors = 'none',)
			ax1.scatter( cp_px, cp_py, s = 20, marker = 'o', edgecolors = 'r', facecolors = 'none',)

			ax1.set_xlim( kk_px - 75, kk_px + 75 )
			ax1.set_ylim( kk_py - 75, kk_py + 75 )

			plt.savefig('/home/xkchen/figs/' + 
					'cluster_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.3f_dec%.3f.png' % (band_str, ra_g, dec_g, z_g, kk_ra, kk_dec), dpi = 300)
			plt.close()

# sat_cut_img()
# sat_cut_test()


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
			# with h5py.File( BG_path + 
			# 		'photo-z_match_tot-BCG-star-Mass_%s_%s-band_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mm], band[kk]), 'r') as f:

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



### === SDSS origin image cut (for satellite background estimation)
img_file = '/media/xkchen/My Passport/data/SDSS/photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

##. origin image location of satellites
cat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/' + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv')

clus_ID = np.array( cat['clus_ID'] )

set_IDs = np.array( list( set( clus_ID ) ) )
set_IDs = set_IDs.astype( int )

N_ss = len( set_IDs )


##. shuffle cluster IDs
rand_IDs = np.loadtxt('/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/shufle_test/img_tract_cat/' + 
					'Extend-BCGM_rgi-common_frame-lim_Pm-cut_mem_shuffle-clus_cat.txt')
rand_mp_IDs = rand_IDs[0].astype( int )   ## order will have error ( treat as the fiducial order list)


band_str = 'i'

##. error raise up ordex (use row_0 in shuffle list)
err_dat = pds.read_csv( '/home/xkchen/err_in_%s-band_position_shuffle.csv' % band_str,)
err_IDs = np.array( err_dat['clus_ID'] )
err_ra, err_dec, err_z = np.array( err_dat['ra'] ), np.array( err_dat['dec'] ), np.array( err_dat['z'] )

ordex = []
for kk in range( len(err_ra) ):

	id_vx = np.where( set_IDs == err_IDs[ kk ] )[0][0]
	ordex.append( id_vx )

##. combine shuffle
t0_rand_arr = rand_IDs[0].astype( int )
t1_rand_arr = rand_IDs[2].astype( int )

cp_rand_arr = t0_rand_arr + 0
cp_rand_arr[ ordex ] = t1_rand_arr[ ordex ]

np.savetxt('/home/xkchen/Extend-BCGM_rgi-common_frame-lim_Pm-cut_mem_%s-band_extra-shuffle-clus_cat.txt' % band_str, cp_rand_arr)
raise

dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/shufle_test/img_tract_cat/' + 
			'Extend-BCGM_rgi-common_frame-limit_Pm-cut_exlu-BCG_Sat_%s-band_origin-img_position.csv' % band_str )

bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
bcg_x, bcg_y = np.array( dat['bcg_x'] ), np.array( dat['bcg_y'] )
sat_x, sat_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )
tt_clus_ID = np.array( dat['clus_ID'] )

err_ra, err_dec, err_z = [], [], []
err_IDs = []

# for kk in range( N_ss ):
for kk in range( len( ordex ) ):

	#. target cluster satellites
	# id_vx = tt_clus_ID == set_IDs[ kk ]
	id_vx = tt_clus_ID == set_IDs[ ordex[ kk ] ]

	sub_ra, sub_dec = sat_ra[ id_vx ], sat_dec[ id_vx ]
	ra_g, dec_g, z_g = bcg_ra[ id_vx ][0], bcg_dec[ id_vx ][0], bcg_z[ id_vx ][0]

	img = fits.open( img_file % (band_str, ra_g, dec_g, z_g),)
	img_arr = img[0].data


	x_cen, y_cen = bcg_x[ id_vx ][0], bcg_y[ id_vx ][0]
	x_sat, y_sat = sat_x[ id_vx ], sat_y[ id_vx ]

	sat_R = np.sqrt( (x_sat - x_cen)**2 + (y_sat - y_cen)**2 )
	sat_theta = np.arctan2( (y_sat - y_cen), (x_sat - x_cen) )

	off_x, off_y = sat_R * np.cos( sat_theta ), sat_R * np.sin( sat_theta )


	#. shuffle clusters and matched images
	# id_ux = clus_ID == rand_mp_IDs[ kk ]
	id_ux = clus_ID == t1_rand_arr[ ordex[ kk ] ]

	cp_ra_g, cp_dec_g, cp_z_g = bcg_ra[ id_ux ][0], bcg_dec[ id_ux ][0], bcg_z[ id_ux ][0]

	cp_img = fits.open( img_file % (band_str, cp_ra_g, cp_dec_g, cp_z_g),)
	cp_img_arr = cp_img[0].data

	#. image center
	pix_cx, pix_cy = cp_img_arr.shape[1] / 2, cp_img_arr.shape[0] / 2

	cp_cx, cp_cy = bcg_x[ id_ux ][0], bcg_y[ id_ux ][0]
	cp_sx_1, cp_sy_1 = cp_cx + off_x, cp_cy + off_y


	#. identify satellites beyond the image frame of shuffle cluster
	Lx, Ly = cp_img_arr.shape[1], cp_img_arr.shape[0]

	id_x_lim = ( cp_sx_1 < 0 ) | ( cp_sx_1 >= 2047 )
	id_y_lim = ( cp_sy_1 < 0 ) | ( cp_sy_1 >= 1488 )

	id_lim = id_x_lim + id_y_lim

	tp_sx, tp_sy = cp_sx_1[ id_lim ], cp_sy_1[ id_lim ]
	tp_chi = sat_theta[ id_lim ]
	tp_Rs = sat_R[ id_lim ]


	#. loop for satellites position adjust 
	N_pot = np.sum( id_lim )

	tm_sx, tm_sy = np.zeros( N_pot,), np.zeros( N_pot,)

	id_err = np.zeros( N_pot,)  ##. records points cannot located in image frame

	for tt in range( N_pot ):

		tm_phi = np.array( [ np.pi + tp_chi[ tt ], np.pi - tp_chi[ tt ], np.pi * 2 - tp_chi[tt] ] )

		tm_off_x = tp_Rs[ tt ] * np.cos( tm_phi )
		tm_off_y = tp_Rs[ tt ] * np.sin( tm_phi )

		tt_sx, tt_sy = cp_cx + tm_off_x, cp_cy + tm_off_y

		id_ux = ( tt_sx >= 0 ) & ( tt_sx < 2047 )
		id_uy = ( tt_sy >= 0 ) & ( tt_sy < 1488 )

		id_up = id_ux & id_uy

		if np.sum( id_up ) > 0:

			tm_sx[ tt ] = tt_sx[ id_up ][0]
			tm_sy[ tt ] = tt_sy[ id_up ][0]

		else:
			id_err[ tt ] = 1.

	#. 
	cp_sx, cp_sy = cp_sx_1 + 0., cp_sy_1 + 0.
	cp_sx[ id_lim ] = tm_sx
	cp_sy[ id_lim ] = tm_sy


	##. if there is somepoint is always can not located in image frame
	##. then take the symmetry points of entire satellites sample
	print( np.sum( id_err ) )

	if np.sum( id_err ) > 0:

		cp_sx, cp_sy = 2 * pix_cx - x_sat, 2 * pix_cy - y_sat
		_sm_cp_cx, _sm_cp_cy = 2 * pix_cx - x_cen, 2 * pix_cy - y_cen

		# err_IDs.append( set_IDs[ kk ] )
		# err_ra.append( ra_g )
		# err_dec.append( dec_g )
		# err_z.append( z_g )


	plt.figure( figsize = (10, 5),)
	ax0 = plt.subplot(121)
	ax1 = plt.subplot(122)

	ax0.imshow( img_arr, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e0, norm = mpl.colors.LogNorm(),)
	ax0.scatter( x_sat, y_sat, s = 10, marker = 'o', facecolors = 'none', edgecolors = 'r',)
	ax0.scatter( x_cen, y_cen, s = 5, marker = 's', facecolors = 'none', edgecolors = 'b',)

	ax1.imshow( cp_img_arr, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e0, norm = mpl.colors.LogNorm(),)
	ax1.scatter( cp_cx, cp_cy, s = 5, marker = 's', facecolors = 'none', edgecolors = 'r',)

	if np.sum( id_err ) > 0:
		ax1.scatter( _sm_cp_cx, _sm_cp_cy, s = 5, marker = 's', facecolors = 'none', edgecolors = 'b',)

	ax1.scatter( cp_sx_1, cp_sy_1, s = 12, marker = 'o', facecolors = 'none', edgecolors = 'g', label = 'relative offset')
	ax1.scatter( tp_sx, tp_sy, s = 8, marker = '*', facecolors = 'none', edgecolors = 'r',)
	ax1.scatter( tm_sx, tm_sy, s = 8, marker = 'd', facecolors = 'none', edgecolors = 'r',)

	# plt.savefig('/home/xkchen/random_sat-Pos_set.png', dpi = 300)
	plt.savefig('/home/xkchen/random_sat-Pos_set_%d.png' % kk, dpi = 300)
	plt.close()


raise

##. record for row_0 in shuffle list
err_IDs = np.array( err_IDs )
err_ra = np.array( err_ra )
err_dec = np.array( err_dec )
err_z = np.array( err_z )

keys = [ 'ra', 'dec', 'z', 'clus_ID' ]
values = [ err_ra, err_dec, err_z, err_IDs ]
fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/err_in_position_shuffle.csv')


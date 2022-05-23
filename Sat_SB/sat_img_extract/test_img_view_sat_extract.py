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



"""
Modelling the BCGs to get their position angle relative to longer frame side~( N_pix = 2048)
"""
import sys 
# sys.path.append('/home/xkchen/tool/Conda/Tools/normitems')
sys.path.append('/home/xkchen/Tools/normitems')

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpathes
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import h5py
import random
import numpy as np
import pandas as pds

import scipy.stats as scists
import astropy.stats as astrosts

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.io.ascii as asc

from scipy import interpolate as interp
from astropy import cosmology as apcy
from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.coordinates import Angle

#.
from tqdm import tqdm
import subprocess as subpro

import warnings
from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyUserWarning
from astropy.modeling import Parameter

#.
from astropy.convolution import interpolate_replace_nans
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve

#.
from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


#.code-mine
from img_segmap import array_mask_func
from img_segmap import simp_finder_func
import galfit_init


### === cosmology
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

##. constants
pixel = 0.396
rad2arcsec = U.rad.to(U.arcsec)
z_ref = 0.25


### === ### use obj_detection to map BCG position angle
home = '/home/xkchen/data/SDSS/'
cat_path = '/home/xkchen/data/SDSS/extend_Zphoto_cat/region_sql_cat/'
out_path = '/home/xkchen/project/tmp/'

band = ['r', 'g', 'i']

##.
# for kk in range( 3 ):
for kk in range( rank, rank + 1 ):

	band_str = band[ kk ]

	##. location and z_obs
	dat = pds.read_csv( cat_path + 
				'redMaPPer_z-phot_0.2-0.3_%s-band_selected_clust_sql_size.csv' % band_str,)

	bcg_ra, bcg_dec, bcg_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

	sub_IDs = np.array( dat['clus_ID'] )
	sub_IDs = sub_IDs.astype( int )

	N_g = len( bcg_ra )

	##.
	kk_cx = np.zeros( N_g,)
	kk_cy = np.zeros( N_g,)

	kk_b_fit = np.zeros( N_g,)
	kk_a_fit = np.zeros( N_g,)
	kk_PA_fit = np.zeros( N_g,)

	for nn in range( N_g ):

		##... BCG centered image cutout
		ra_g, dec_g, z_g = bcg_ra[ nn ], bcg_dec[ nn ], bcg_z[ nn ]

		nn_IDs = sub_IDs[ nn ]

		ref_file = '/home/xkchen/data/SDSS/photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

		data_ref = fits.open( ref_file % (band_str, ra_g, dec_g, z_g),)
		ref_img = data_ref[0].data

		Head = data_ref[0].header

		wcs_ref = awc.WCS( Head )

		k_nMGY = Head['NMGY']

		xn, yn = wcs_ref.all_world2pix( ra_g, dec_g, 0 )

		##. source detection catalog
		obj_file = home + 'photo_files/detect_source_cat/photo-z_img_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'

		source = asc.read( obj_file % (band_str, ra_g, dec_g, z_g),)
		Numb = np.array(source['NUMBER'][-1])
		A = np.array(source['A_IMAGE'])
		B = np.array(source['B_IMAGE'])
		theta = np.array(source['THETA_IMAGE'])
		cx = np.array(source['X_IMAGE'])
		cy = np.array(source['Y_IMAGE'])
		p_type = np.array(source['CLASS_STAR'])

		Kron = 7   ## 10, 9, 8, 7, 6, 5
		a = Kron * A
		b = Kron * B

		##.
		dR = np.sqrt( (cx - xn)**2 + (cy - yn)**2 )

		id_mx = dR == dR.min()

		mp_cx, mp_cy = cx[ id_mx ], cy[ id_mx ]
		mp_ar, mp_br = a[ id_mx ], b[ id_mx ]
		mp_chi = theta[ id_mx ]

		#.
		kk_cx[ nn ] = mp_cx + 0.
		kk_cy[ nn ] = mp_cy + 0.

		kk_b_fit[ nn ] = mp_br + 0.
		kk_a_fit[ nn ] = mp_ar + 0.
		kk_PA_fit[ nn ] = mp_chi + 0.

		if nn <= 20:

			fig = plt.figure( )
			ax0 = fig.add_axes([0.11, 0.11, 0.80, 0.85])

			tf = ax0.imshow( ref_img, origin = 'lower', cmap = 'Greys', 
							norm = mpl.colors.SymLogNorm( vmin = -1e-1, vmax = 5e0, linthresh = 1e-3,),)
			plt.colorbar( tf, ax = ax0, pad = 0.01, fraction = 0.047,)

			ax0.annotate( s = '%.3f' % mp_chi, xy = (0.65, 0.05), xycoords = 'axes fraction', fontsize = 12, color = 'r',)

			out_lines = mpathes.Ellipse( xy = (mp_cx, mp_cy), width = mp_ar, height = mp_br, angle = mp_chi, fc = 'none', ec = 'r')
			ax0.add_artist( out_lines )

			ax0.set_xlim( mp_cx - 50, mp_cx + 50 )
			ax0.set_ylim( mp_cy - 50, mp_cy + 50 )

			ax0.set_xticklabels( [] )
			ax0.set_yticklabels( [] )

			plt.savefig('/home/xkchen/figs/BCG_located_%s-band_ra%.3f_dec%.3f_z%.3f.png' %
						(band_str, ra_g, dec_g, z_g), dpi = 300)
			plt.close()

		# else:
		# 	break

	##... save the fitting parameters
	keys = [ 'ra', 'dec', 'z', 'clus_ID', 'cx', 'cy', 'a_pix', 'b_pix', 'PA' ]
	values = [ bcg_ra, bcg_dec, bcg_z, sub_IDs, kk_cx, kk_cy, kk_a_fit, kk_b_fit, kk_PA_fit ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( out_path + 'BCG_located-params_sub-%d_rank.csv' % rank,)

	print('%d-rank, done!' % rank)


raise

### === ### cluster catalog
cat_path = '/home/xkchen/data/SDSS/extend_Zphoto_cat/region_sql_cat/'
out_path = '/home/xkchen/project/tmp/'

band = ['r', 'g', 'i']

##.
# for kk in range( 3 ):
for kk in range( rank, rank + 1 ):

	band_str = band[ kk ]

	##. location and z_obs
	dat = pds.read_csv( cat_path + 
				'redMaPPer_z-phot_0.2-0.3_%s-band_selected_clust_sql_size.csv' % band_str,)

	bcg_ra, bcg_dec, bcg_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

	sub_IDs = np.array( dat['clus_ID'] )
	sub_IDs = sub_IDs.astype( int )

	N_g = len( bcg_ra )

	##.
	kk_Re_fit = np.zeros( N_g,)
	kk_ne_fit = np.zeros( N_g,)
	kk_b2a_fit = np.zeros( N_g,)
	kk_PA_fit = np.zeros( N_g,)

	for nn in range( N_g ):

		##... BCG centered image cutout
		ra_g, dec_g, z_g = bcg_ra[ nn ], bcg_dec[ nn ], bcg_z[ nn ]

		nn_IDs = sub_IDs[ nn ]

		ref_file = '/home/xkchen/data/SDSS/photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

		data_ref = fits.open( ref_file % (band_str, ra_g, dec_g, z_g),)
		ref_img = data_ref[0].data

		Head = data_ref[0].header

		wcs_ref = awc.WCS( Head )

		k_nMGY = Head['NMGY']

		xn, yn = wcs_ref.all_world2pix( ra_g, dec_g, 0 )


		##. source detection and mask_array
		d_file = ref_file % (band_str, ra_g, dec_g, z_g)
		out_file = out_path + 'ref_frame_cat_%d.cat' % rank

		config_file = '/home/xkchen/project/sat_SB/SEX_param/default_adjust.sex'
		params_file = '/home/xkchen/project/sat_SB/SEX_param/default_mask_A.param'
		tmp_file = out_path + 'pre_detect_%d.fits' % rank

		simp_finder_func( d_file, out_file, config_file, params_file, tmp_file = tmp_file )

		cmd = 'rm %s' % tmp_file
		APRO = subpro.Popen( cmd, shell = True )
		APRO.wait()


		##. mask_array
		obj_cat = out_path + 'ref_frame_cat_%d.cat' % rank

		source = asc.read( obj_cat,)
		Numb = np.array(source['NUMBER'][-1])
		A = np.array(source['A_IMAGE'])
		B = np.array(source['B_IMAGE'])
		theta = np.array(source['THETA_IMAGE'])
		cx = np.array(source['X_IMAGE'])
		cy = np.array(source['Y_IMAGE'])
		p_type = np.array(source['CLASS_STAR'])

		Kron = 7   ## 10, 9, 8, 7, 6, 5
		a = Kron * A
		b = Kron * B

		#. source masked and BCG region cut
		f_eta = 2
		dR = np.sqrt( (cx - xn)**2 + (cy - yn)**2 )

		id_tag = np.where( dR == np.min( dR ) )[0][0]
		tag_arr = [ cx[ id_tag ], cy[ id_tag ], A[ id_tag ] * f_eta, B[ id_tag ] * f_eta, theta[ id_tag ] ]

		mask_img = array_mask_func( ref_img, cx, cy, a, b, theta, tag_arr = tag_arr )

		#.
		wdx = 50

		cut_arr = np.zeros( ( np.int( 2 * wdx + 1 ), np.int( 2 * wdx + 1 ) ), dtype = np.float32 ) + np.nan

		d_x0 = np.max( [ xn - wdx, 0 ] )
		d_x1 = np.min( [ xn + wdx, ref_img.shape[1] - 1 ] )

		d_y0 = np.max( [ yn - wdx, 0 ] )
		d_y1 = np.min( [ yn + wdx, ref_img.shape[0] - 1 ] )

		d_x0 = np.int( d_x0 )
		d_x1 = np.int( d_x1 )

		d_y0 = np.int( d_y0 )
		d_y1 = np.int( d_y1 )

		pre_cut = mask_img[d_y0: d_y1, d_x0: d_x1]

		pre_cut_cx = xn - d_x0
		pre_cut_cy = yn - d_y0

		pre_cx = np.int( pre_cut_cx )
		pre_cy = np.int( pre_cut_cy )

		#.
		p_xn, p_yn = wdx, wdx

		pa0 = np.int( p_xn - pre_cx )
		pa1 = np.int( p_xn - pre_cx + pre_cut.shape[1] )

		pb0 = np.int( p_yn - pre_cy )
		pb1 = np.int( p_yn - pre_cy + pre_cut.shape[0] )

		cut_arr[ pb0 : pb1, pa0 : pa1 ] = pre_cut + 0.

		#.. sigma-clipping and sampling to fill mask holes
		id_NN = np.isnan( cut_arr )
		res_flux = cut_arr[ id_NN == False ]

		clip_arr = astrosts.sigma_clip( res_flux, sigma = 3, maxiters = None,)

		id_mx = clip_arr.mask
		clip_flux = clip_arr.data[ id_mx == False ]

		n_clip = len( clip_flux )
		dt0 = np.random.choice( n_clip, np.sum(id_NN), )

		cp_cut_arr = cut_arr.copy()
		cp_cut_arr[ id_NN ] = clip_flux[ dt0 ]

		##. save image cuts~( covert to counts )
		dNx, dNy = cut_arr.shape[1], cut_arr.shape[0]

		keys = [ 'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXPTIME',
				 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'P_SCALE', 'NMGY', 'GAIN']

		value = [ 'T', 32, 2, dNx, dNy, 1, 
					dNx / 2, dNy / 2, ra_g, dec_g, pixel, k_nMGY, 1 ]

		ff = dict( zip(keys,value) )
		fil = fits.Header(ff)
		fits.writeto( out_path + 'BCG_filled_cut_img_%d.fits' % rank, cp_cut_arr / k_nMGY, header = fil, overwrite = True)		


		##... BCG centered image fitting with astropy modeling fitting
		fit_file = out_path + 'BCG_filled_cut_img_%d.fits' % rank

		fit_data = fits.open( fit_file )
		fit_img = fit_data[0].data

		Nx, Ny = fit_img.shape[1], fit_img.shape[0]

		nx = np.linspace(0, Nx - 1, Nx)
		ny = np.linspace(0, Ny - 1, Ny)
		grid_x, grid_y = np.meshgrid( nx, ny )

		xcen, ycen = fit_img.shape[1] / 2, fit_img.shape[0] / 2

		P_init = models.Sersic2D( amplitude = 3, r_eff = 4.5, n = 2, x_0 = xcen, y_0 = ycen, ellip = 0.5, theta = 45 * U.deg,)
		# P_init = models.Gaussian2D( amplitude = 2, x_mean = xcen, y_mean = ycen, x_stddev = 2.5, y_stddev = 2.5, theta = 45 * U.deg,)

		fit_P = fitting.LevMarLSQFitter()

		with warnings.catch_warnings():

			warnings.filterwarnings('ignore', message = 'Model is linear in parameters', category = AstropyUserWarning)
			PF = fit_P( P_init, grid_x, grid_y, fit_img )

		mod_img = PF( grid_x, grid_y )
		mod_res = fit_img - mod_img

		#.
		xo_fit = PF.x_0.value
		yo_fit = PF.y_0.value
		re_fit = PF.r_eff.value
		ne_fit = PF.n.value
		b2a_fit = PF.ellip.value
		PA_fit = PF.theta.value * 180 / np.pi

		#.
		kk_Re_fit[ nn ] = re_fit + 0.
		kk_ne_fit[ nn ] = ne_fit + 0.
		kk_b2a_fit[ nn ] = b2a_fit + 0.
		kk_PA_fit[ nn ] = PA_fit + 0.

		##.	
		if nn <= 20:

			fig = plt.figure( figsize = (13.2, 4.8) )
			ax0 = fig.add_axes([0.02, 0.10, 0.28, 0.84])
			ax1 = fig.add_axes([0.35, 0.10, 0.28, 0.84])
			ax2 = fig.add_axes([0.68, 0.10, 0.28, 0.84])

			ax0.set_title('BCG')
			tf = ax0.imshow( fit_img, origin = 'lower', cmap = 'Greys', 
							norm = mpl.colors.SymLogNorm( vmin = -5e0, vmax = 5e1, linthresh = 1e-1,),)
			plt.colorbar( tf, ax = ax0, pad = 0.01, fraction = 0.047,)

			# ax0.contour( fit_img, levels = 7, alpha = 0.75, cmap = 'rainbow')

			out_lines = mpathes.Ellipse( xy = (xo_fit, yo_fit), width = 2 * re_fit, height = 2 * re_fit * b2a_fit, 
										angle = PA_fit, fc = 'none', ec = 'r')
			ax0.add_artist( out_lines )

			ax0.set_xticklabels( [] )
			ax0.set_yticklabels( [] )


			ax1.set_title('Galfit modelling')
			tf = ax1.imshow( mod_img, origin = 'lower', cmap = 'Greys',
							norm = mpl.colors.SymLogNorm( vmin = -5e0, vmax = 5e1, linthresh = 1e-1,),)
			plt.colorbar( tf, ax = ax1, pad = 0.01, fraction = 0.047,)

			out_lines = mpathes.Ellipse( xy = (xo_fit, yo_fit), width = 2 * re_fit, height = 2 * re_fit * b2a_fit, 
										angle = PA_fit, fc = 'none', ec = 'r')
			ax1.add_artist( out_lines )

			ax1.set_xticklabels( [] )
			ax1.set_yticklabels( [] )


			ax2.set_title('Residual / np.std( residual )')
			tf = ax2.imshow( mod_res / np.std( mod_res ), origin = 'lower', cmap = 'bwr', vmin = -2, vmax = 2,)
			plt.colorbar( tf, ax = ax2, pad = 0.01, fraction = 0.047,)

			ax2.set_xticklabels( [] )
			ax2.set_yticklabels( [] )

			plt.savefig('/home/xkchen/figs/BCG_modelling_%s-band_ra%.3f_dec%.3f_z%.3f.png' %
						(band_str, ra_g, dec_g, z_g), dpi = 300)
			plt.close()


	##... save the fitting parameters
	keys = [ 'ra', 'dec', 'z', 'clus_ID', 'Re_pix', 'n', 'b/a', 'PA' ]
	values = [ bcg_ra, bcg_dec, bcg_z, sub_IDs, kk_Re_fit, kk_ne_fit, kk_b2a_fit, kk_PA_fit ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( out_path + 'BCG_fit_params_sub-%d_rank.csv' % rank,)

	print('%d-rank, done!' % rank)


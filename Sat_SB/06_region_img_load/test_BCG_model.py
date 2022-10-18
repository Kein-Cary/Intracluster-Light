"""
Modelling the BCGs to get their position angle relative to longer frame side~( N_pix = 2048)
"""
import sys 
sys.path.append('/home/xkchen/tool/Conda/Tools/normitems')

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
from img_segmap import array_mask_func
from img_segmap import simp_finder_func
import galfit_init


##. cosmology
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

##. constants
# u,g,r,i,z == 0,1,2,3,4
band_id = [ 0, 1, 2, 3, 4 ]
band = ['u', 'g', 'r', 'i', 'z']

#.
band_str = 'r'
pixel = 0.396

rad2arcsec = U.rad.to(U.arcsec)



### === ### cluster catalog
#.
dat = pds.read_csv('/home/xkchen/figs_cp/SDSS_img_load/cat_file/' + 
				'redMaPPer_z-phot_0.1-0.33_clust_sql_size.csv',)

bcg_ra, bcg_dec, bcg_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

sub_IDs = np.array( dat['clus_ID'] )
sub_IDs = sub_IDs.astype( int )

N_g = len( bcg_ra )



### === ### BCG modelling and compare
out_path = '/home/xkchen/figs_cp/SDSS_img_load/sat_SB_compare/obj_file/'

"""
### === image cut
for nn in range( 4, 5 ):

	##. ref_point images
	ra_g, dec_g, z_g = bcg_ra[ nn ], bcg_dec[ nn ], bcg_z[ nn ]

	ref_file = '/home/xkchen/figs_cp/SDSS_img_load/tmp_file/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

	data_ref = fits.open( ref_file % (band_str, ra_g, dec_g, z_g),)
	ref_img = data_ref[0].data

	Head = data_ref[0].header

	wcs_ref = awc.WCS( Head )

	k_nMGY = Head['NMGY']

	xn, yn = wcs_ref.all_world2pix( ra_g, dec_g, 0 )


	##. PSF read and save
	psf_file = '/home/xkchen/figs_cp/SDSS_img_load/tmp_file/psfField_ra%.3f_dec%.3f_z%.3f.fit' % (ra_g, dec_g, z_g)
	out_psf = '/home/xkchen/figs_cp/SDSS_img_load/tmp_file/BCG_psf_ra%.3f_dec%.3f_z%.3f.fit' % (ra_g, dec_g, z_g)

	pass_id = band.index( band_str )

	cmd = 'read_PSF %s %d %d %d %s' % (psf_file, pass_id, np.around(xn), np.around(yn), out_psf)
	APRO = subpro.Popen( cmd, shell = True )
	APRO.wait()

	#. normalize psf
	p_data = fits.open( out_psf )
	psf_arr = p_data[0].data
	psf_head = p_data[0].header

	norm_psf = (psf_arr - 1000) / np.sum( (psf_arr - 1000) )

	##. save psf
	keys = [ 'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2' ]

	value = [ 'T', 32, 2, norm_psf.shape[1], norm_psf.shape[0] ]

	ff = dict( zip(keys,value) )
	fil = fits.Header(ff)
	fits.writeto( out_path + 
		'BCG_norm-psf_ra%.3f_dec%.3f_z%.3f.fits' % (ra_g, dec_g, z_g), norm_psf, header = fil, overwrite = True)


	##. source detection
	d_file = ref_file % (band_str, ra_g, dec_g, z_g)
	out_file = out_path + 'ref_frame_cat.cat'

	config_file = '/home/xkchen/mywork/Sat_SB/code/SEX_param/default_adjust.sex'
	params_file = '/home/xkchen/mywork/Sat_SB/code/SEX_param/default_mask_A.param'
	tmp_file = '/home/xkchen/pre_detect.fits'

	simp_finder_func( d_file, out_file, config_file, params_file, tmp_file = tmp_file )

	cmd = 'rm %s' % tmp_file
	APRO = subpro.Popen( cmd, shell = True )
	APRO.wait()


	##. mask_array
	obj_cat = out_path + 'ref_frame_cat.cat'

	source = asc.read( obj_cat,)
	Numb = np.array(source['NUMBER'][-1])
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])
	theta = np.array(source['THETA_IMAGE'])
	cx = np.array(source['X_IMAGE'])
	cy = np.array(source['Y_IMAGE'])
	p_type = np.array(source['CLASS_STAR'])

	Kron = 9
	a = Kron * A
	b = Kron * B

	#. source masked and BCG region cut
	f_eta = Kron / 2
	dR = np.sqrt( (cx - xn)**2 + (cy - yn)**2 )

	id_tag = np.where( dR == np.min( dR ) )[0][0]
	tag_arr = [ cx[ id_tag ], cy[ id_tag ], A[ id_tag ] * f_eta, B[ id_tag ] * f_eta, theta[ id_tag ] ]

	mask_img = array_mask_func( ref_img, cx, cy, a, b, theta, tag_arr = tag_arr )

	dx0, dx1 = np.int( xn - 55 ), np.int( xn + 55 )
	dy0, dy1 = np.int( yn - 55 ), np.int( yn + 55 )

	cut_arr = mask_img[dy0: dy1, dx0: dx1]

	#. sigma-clipping and sampling to fill mask holes
	id_NN = np.isnan( cut_arr )
	res_flux = cut_arr[ id_NN == False ]

	clip_arr = astrosts.sigma_clip( res_flux, sigma = 3, maxiters = None,)

	id_mx = clip_arr.mask
	clip_flux = clip_arr.data[ id_mx == False ]

	n_clip = len( clip_flux )
	dt0 = np.random.choice( n_clip, np.sum(id_NN), )

	cp_cut_arr = cut_arr.copy()
	cp_cut_arr[ id_NN ] = clip_flux[ dt0 ]

	# #.
	# f_bins = np.linspace( res_flux.min(), res_flux.max(), 105)
	# fig = plt.figure()
	# ax = fig.add_axes([0.12, 0.11, 0.80, 0.85,])

	# ax.hist( res_flux, bins = f_bins, density = False, histtype = 'step', color = 'b', label = 'Before clipping')
	# ax.hist( clip_flux, bins = f_bins, density = False, histtype = 'step', color = 'r', ls = '--', label = 'After clipping')
	# ax.legend( loc = 1, frameon = False, fontsize  = 12,)
	# ax.set_xlabel('flux [nanomaggies]', fontsize = 12,)

	# ax.set_yscale('log')

	# ax.xaxis.set_minor_locator( ticker.AutoMinorLocator() )
	# ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	# plt.savefig('/home/xkchen/cliping_test.png', dpi = 300)
	# plt.close()


	##. smoothing out masks
	kernel = Gaussian2DKernel( x_stddev = 3 )   ## 3, 5, 7
	conv_fill_arr = convolve( cut_arr, kernel )

	# kernel = Gaussian2DKernel( x_stddev = 3 )
	# conv_fill_arr = interpolate_replace_nans(cut_arr, kernel)


	##. save image cuts~( covert to counts )
	dNx, dNy = cut_arr.shape[1], cut_arr.shape[0]

	keys = [ 'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXPTIME',
			 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'P_SCALE', 'NMGY', 'GAIN']

	value = [ 'T', 32, 2, dNx, dNy, 1, 
				dNx / 2, dNy / 2, ra_g, dec_g, pixel, k_nMGY, 1 ]

	#.
	ff = dict( zip(keys,value) )
	fil = fits.Header(ff)
	fits.writeto( out_path + 
		'BCG_cut_img_ra%.3f_dec%.3f_z%.3f.fits' % (ra_g, dec_g, z_g), cut_arr / k_nMGY, header = fil, overwrite = True)

	#.
	ff = dict( zip(keys,value) )
	fil = fits.Header(ff)
	fits.writeto( out_path + 
		'BCG_smooth_cut_img_ra%.3f_dec%.3f_z%.3f.fits' % (ra_g, dec_g, z_g), conv_fill_arr / k_nMGY, header = fil, overwrite = True)

	#.
	ff = dict( zip(keys,value) )
	fil = fits.Header(ff)
	fits.writeto( out_path + 
		'BCG_filled_cut_img_ra%.3f_dec%.3f_z%.3f.fits' % (ra_g, dec_g, z_g), cp_cut_arr / k_nMGY, header = fil, overwrite = True)


	#.
	fig = plt.figure( figsize = (13.12, 9.84) )
	ax0 = fig.add_axes([0.03, 0.55, 0.40, 0.43])
	ax1 = fig.add_axes([0.50, 0.55, 0.40, 0.43])
	ax2 = fig.add_axes([0.03, 0.05, 0.40, 0.43])
	ax3 = fig.add_axes([0.50, 0.05, 0.40, 0.43])

	tf = ax0.imshow( ref_img, origin = 'lower', cmap = 'Greys', norm = mpl.colors.LogNorm( vmin = 1e-4, vmax = 1e0),)
	plt.colorbar( tf, ax = ax0, pad = 0.01, fraction = 0.041, label = 'flux [nanomaggy]')
	ax0.scatter( xn, yn, s = 35, marker = 'o', facecolors = 'none', edgecolors = 'r',)

	for mm in range( Numb ):

		ellips = mpathes.Ellipse( xy = (cx[ mm ], cy[ mm ]), width = a[ mm ], height = b[ mm ], 
					angle = theta[ mm ], fill = False, ec = 'k', ls = '-', alpha = 0.75,)
		ax0.add_patch( ellips )

	ax0.set_xlim( xn - 55, xn + 55 )
	ax0.set_ylim( yn - 55, yn + 55 )


	tf = ax1.imshow( mask_img, origin = 'lower', cmap = 'Greys', vmin = -1e-1, vmax = 1e-1,)
	plt.colorbar( tf, ax = ax1, pad = 0.01, fraction = 0.041, label = 'flux [nanomaggy]')
	ax1.set_xlim( xn - 55, xn + 55 )
	ax1.set_ylim( yn - 55, yn + 55 )


	ax2.set_title('astropy.convolution')
	tf = ax2.imshow( conv_fill_arr, origin = 'lower', cmap = 'Greys', vmin = -1e-1, vmax = 1e-1,)
	plt.colorbar( tf, ax = ax2, pad = 0.01, fraction = 0.041, label = 'flux [nanomaggy]')


	ax3.set_title('$3 \\sigma \,$ clipping filling')
	tf = ax3.imshow( cp_cut_arr, origin = 'lower', cmap = 'Greys', vmin = -1e-1, vmax = 1e-1,)
	plt.colorbar( tf, ax = ax3, pad = 0.01, fraction = 0.041, label = 'flux [nanomaggy]')

	plt.savefig('/home/xkchen/BCG_filling_cut_test.png', dpi = 300)
	plt.close()

# raise
"""


### === ... modelling and fitting

for nn in range( 4, 5 ):

	##. ref_point images
	ra_g, dec_g, z_g = bcg_ra[ nn ], bcg_dec[ nn ], bcg_z[ nn ]

	##.
	psf_file = out_path + 'BCG_norm-psf_ra%.3f_dec%.3f_z%.3f.fits' % (ra_g, dec_g, z_g)

	##.
	fit_file = out_path + 'BCG_filled_cut_img_ra%.3f_dec%.3f_z%.3f.fits' % (ra_g, dec_g, z_g)
	# fit_file = out_path + 'BCG_cut_img_ra%.3f_dec%.3f_z%.3f.fits' % (ra_g, dec_g, z_g)

	##.
	bcg_data = fits.open( fit_file )
	pre_img = bcg_data[0].data

	#.
	Nx, Ny = pre_img.shape[1], pre_img.shape[0]

	nx = np.linspace(0, Nx - 1, Nx)
	ny = np.linspace(0, Ny - 1, Ny)
	grid_x, grid_y = np.meshgrid( nx, ny )

	# ##. points
	# id_nan = np.isnan( pre_img )
	# dd_py, dd_px = np.where( id_nan == False )

	# N_pont = len( dd_py )

	# N00 = np.int( N_pont * 0.90 )
	# tt0 = np.random.choice( N_pont, N00, replace = False) 

	# pt_x = dd_px[ tt0 ]
	# pt_y = dd_py[ tt0 ]
	# pt_arr = pre_img[ pt_y, pt_x ]

	xcen, ycen = pre_img.shape[1] / 2, pre_img.shape[0] / 2

	P_init = models.Sersic2D( amplitude = 3, r_eff = 4.5, n = 2, x_0 = xcen, y_0 = ycen, ellip = 0.5, theta = 45 * U.deg,)
	# P_init = models.Gaussian2D( amplitude = 2, x_mean = xcen, y_mean = ycen, x_stddev = 2.5, y_stddev = 2.5, theta = 45 * U.deg,)

	fit_P = fitting.LevMarLSQFitter()

	with warnings.catch_warnings():
		warnings.filterwarnings('ignore', message = 'Model is linear in parameters', category = AstropyUserWarning)

		# PF = fit_P( P_init, pt_x, pt_y, pt_arr )
		PF = fit_P( P_init, grid_x, grid_y, pre_img )

	mod_img = PF( grid_x, grid_y )
	diff_img = pre_img - mod_img

	##. params
	xo_fit = PF.x_0.value
	yo_fit = PF.y_0.value
	re_fit = PF.r_eff.value
	ne_fit = PF.n.value
	b2a_fit = PF.ellip.value
	PA_fit = PF.theta.value * 180 / np.pi


	##.
	fig = plt.figure( figsize = (13.2, 4.8) )

	ax0 = fig.add_axes([0.02, 0.10, 0.28, 0.84])
	ax1 = fig.add_axes([0.35, 0.10, 0.28, 0.84])
	ax2 = fig.add_axes([0.68, 0.10, 0.28, 0.84])

	ax0.set_title('BCG')
	tf = ax0.imshow( pre_img, origin = 'lower', cmap = 'Greys', 
					norm = mpl.colors.SymLogNorm( vmin = -5e0, vmax = 5e1, linthresh = 1e-1,),)
	plt.colorbar( tf, ax = ax0, pad = 0.01, fraction = 0.047,)

	ax0.contour( pre_img, levels = 7, alpha = 0.75, cmap = 'rainbow')

	ax0.set_xticklabels( [] )
	ax0.set_yticklabels( [] )

	ax1.set_title('Model')
	tf = ax1.imshow( mod_img, origin = 'lower', cmap = 'Greys',
					norm = mpl.colors.SymLogNorm( vmin = -5e0, vmax = 5e1, linthresh = 1e-1,),)
	plt.colorbar( tf, ax = ax1, pad = 0.01, fraction = 0.047,)

	out_lines = mpathes.Ellipse( xy = (xo_fit, yo_fit), width = 2 * re_fit, height = 2 * re_fit * b2a_fit, 
								angle = PA_fit, fc = 'none', ec = 'r')
	ax1.add_artist( out_lines )

	ax1.set_xticklabels( [] )
	ax1.set_yticklabels( [] )

	ax2.set_title('Residual / np.std( residual )')
	tf = ax2.imshow( diff_img / np.nanstd( diff_img ), origin = 'lower', cmap = 'bwr', vmin = -2, vmax = 2,)
	plt.colorbar( tf, ax = ax2, pad = 0.01, fraction = 0.047,)
	ax2.set_xticklabels( [] )
	ax2.set_yticklabels( [] )

	plt.savefig('/home/xkchen/BCG_model_test.png', dpi = 300)
	plt.close()

raise



for nn in range( 4, 5 ):

	##. ref_point images
	ra_g, dec_g, z_g = bcg_ra[ nn ], bcg_dec[ nn ], bcg_z[ nn ]

	##.
	psf_file = out_path + 'BCG_norm-psf_ra%.3f_dec%.3f_z%.3f.fits' % (ra_g, dec_g, z_g)

	##.
	fit_file = out_path + 'BCG_filled_cut_img_ra%.3f_dec%.3f_z%.3f.fits' % (ra_g, dec_g, z_g)
	# fit_file = out_path + 'BCG_cut_img_ra%.3f_dec%.3f_z%.3f.fits' % (ra_g, dec_g, z_g)


	##.
	fit_data = fits.open( fit_file )
	pre_img = fit_data[0].data

	fit_img = pre_img + 0.

	xcen, ycen = fit_img.shape[1] / 2, fit_img.shape[0] / 2

	xmin, xmax = 20, fit_img.shape[1] - 20
	ymin, ymax = 20, fit_img.shape[0] - 20

	cov_x, cov_y = fit_img.shape[1] - 5, fit_img.shape[0] - 5

	zero_point = 22.5
	init_sky = np.median( fit_img )

	out_file = out_path + 'BCG_galfit_ra%.3f_dec%.3f_z%.3f.fits' % (ra_g, dec_g, z_g)


	##. initial parameters
	magtot = 19.5

	##. with PSF
	# init_str = galfit_init.dps_wPSF % ( fit_file, out_file, psf_file, 
	# 								xmin, xmax, ymin, ymax, 
	# 								cov_x, cov_y, 
	# 								zero_point, 
	# 								pixel, pixel,
	# 								xcen, ycen,
	# 								magtot )

	##. without PSF
	init_str = galfit_init.dps_woPSF % ( fit_file, out_file, 
									xmin, xmax, ymin, ymax, 
									cov_x, cov_y, 
									zero_point, 
									pixel, pixel,
									xcen, ycen,
									magtot )

	init_file = open('/home/xkchen/test.feedfit', 'w')
	print( init_str, file = init_file )
	init_file.close()


	##.
	fit_txt = '/home/xkchen/test.feedfit'
	cmd = 'galfit -noskyest %s' % fit_txt   ##. fitting without sky

	APRO = subpro.Popen( cmd, shell = True )
	APRO.wait()

	##. compare
	mod_data = fits.open( out_file )
	mod_img = mod_data[2].data
	mod_res = mod_data[3].data

	#.
	xo_fit = np.float( mod_data[2].header['1_XC'].split(' ')[0] )
	yo_fit = np.float( mod_data[2].header['1_yC'].split(' ')[0] )
	re_fit = np.float( mod_data[2].header['1_RE'].split(' ')[0] )
	ne_fit = np.float( mod_data[2].header['1_N'].split(' ')[0] )
	b2a_fit = np.float( mod_data[2].header['1_AR'].split(' ')[0] )
	PA_fit = np.float( mod_data[2].header['1_PA'].split(' ')[0] )

	##.
	fig = plt.figure( figsize = (13.2, 4.8) )
	ax0 = fig.add_axes([0.02, 0.10, 0.28, 0.84])
	ax1 = fig.add_axes([0.35, 0.10, 0.28, 0.84])
	ax2 = fig.add_axes([0.68, 0.10, 0.28, 0.84])

	ax0.set_title('BCG')
	tf = ax0.imshow( fit_img, origin = 'lower', cmap = 'Greys', 
					norm = mpl.colors.SymLogNorm( vmin = -5e0, vmax = 5e1, linthresh = 1e-1,),)
	plt.colorbar( tf, ax = ax0, pad = 0.01, fraction = 0.047,)
	ax0.contour( fit_img, levels = 7, alpha = 0.75, cmap = 'rainbow')

	out_lines = mpathes.Ellipse( xy = (xo_fit, yo_fit), width = 2 * re_fit, height = 2 * re_fit * b2a_fit, 
								angle = PA_fit + 90, fc = 'none', ec = 'r')
	ax0.add_artist( out_lines )

	ax0.set_xticklabels( [] )
	ax0.set_yticklabels( [] )


	ax1.set_title('Galfit modelling')
	tf = ax1.imshow( mod_img, origin = 'lower', cmap = 'Greys',
					norm = mpl.colors.SymLogNorm( vmin = -5e0, vmax = 5e1, linthresh = 1e-1,),)
	plt.colorbar( tf, ax = ax1, pad = 0.01, fraction = 0.047,)

	out_lines = mpathes.Ellipse( xy = (xo_fit, yo_fit), width = 2 * re_fit, height = 2 * re_fit * b2a_fit, 
								angle = PA_fit + 90, fc = 'none', ec = 'r')
	ax1.add_artist( out_lines )

	ax1.set_xticklabels( [] )
	ax1.set_yticklabels( [] )


	ax2.set_title('Residual / np.std( residual )')
	tf = ax2.imshow( mod_res / np.std( mod_res ), origin = 'lower', cmap = 'bwr', vmin = -2, vmax = 2,)
	plt.colorbar( tf, ax = ax2, pad = 0.01, fraction = 0.047,)

	ax2.set_xticklabels( [] )
	ax2.set_yticklabels( [] )

	plt.savefig('/home/xkchen/BCG_modelling_test.png', dpi = 300)
	plt.close()


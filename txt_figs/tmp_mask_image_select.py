import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp

import astropy.units as U
import astropy.constants as C
import scipy.stats as sts
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
from scipy import ndimage

from img_random_SB_fit import random_SB_fit_func, clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func
from img_pre_selection import extra_match_func
from light_measure import flux_recal
from resample_modelu import sum_samp, down_samp

# cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']

Da_ref = Test_model.angular_diameter_distance(z_ref).value
L_ref = Da_ref * pixel / rad2asec

#### ==== #### gravity (image overview)

### masking, stacking, BG-estimate figs
home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'

### photo-z sample
# origin_files = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
# mask_files = home + 'photo_files/mask_imgs/photo-z_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
# resamp_files = home + 'photo_files/resample_imgs/photo-z_resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'

# source_cat = home + 'photo_files/detect_source_cat/photo-z_img_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'

# dat = pds.read_csv( load + 'photo_z_cat/photo-z_r-band_tot_remain_cat_set_200-grid_6.0-sigma.csv')
# ra, dec, z = np.array(dat['ra']), np.array(dat['dec']), np.array(dat['z'])
# imgx, imgy = np.array(dat['bcg_x']), np.array(dat['bcg_y'])

### spec-z sample
origin_files = home + 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
mask_files = home + 'mask_imgs/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
resamp_files = home + 'pix_resamp_imgs/z_ref_0.25/resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'

source_cat = home + 'source_detect_cat/cluster_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'

# dat = pds.read_csv( load + 'img_cat_2_28/cluster_r-band_tot_remain_cat_set_200-grid_6.0-sigma.csv')
dat = pds.read_csv( load + 'img_cat_2_28/cluster_tot-r-band_norm-img_cat.csv')
ra, dec, z = np.array(dat['ra']), np.array(dat['dec']), np.array(dat['z'])
imgx, imgy = np.array(dat['bcg_x']), np.array(dat['bcg_y'])


band_str = band[0]

Nz = len( z )

# np.random.seed( 5 )
# tt0 = np.random.choice( Nz, Nz, replace = False)
# set_ra, set_dec, set_z = ra[tt0], dec[tt0], z[tt0]
# set_x, set_y = imgx[tt0], imgy[tt0]

set_ra, set_dec, set_z = ra, dec, z
set_x, set_y = imgx, imgy

for tt in range( Nz ):

	ra_g, dec_g, z_g = set_ra[tt], set_dec[tt], set_z[tt]
	cen_x, cen_y = set_x[tt], set_y[tt]

	data_o = fits.open( origin_files % (band_str, ra_g, dec_g, z_g),)
	img_o = data_o[0].data

	data_m = fits.open( mask_files % (band_str, ra_g, dec_g, z_g),)
	img_m = data_m[0].data

	Da_z = Test_model.angular_diameter_distance( z_g ).value
	L_pix = Da_z * 10**3 * pixel / rad2asec
	R1Mpc = 1e3 / L_pix
	R200kpc = 2e2 / L_pix

	## check BCG region
	xn, yn = np.int(cen_x), np.int(cen_y)
	cen_region = img_m[ yn - np.int(100 / L_pix): yn + np.int(100 / L_pix) + 1, xn - np.int(100 / L_pix): xn + np.int(100 / L_pix) + 1]
	N_pix = cen_region.shape[0] * cen_region.shape[1]
	idnn = np.isnan( cen_region )

	eta = np.sum( idnn ) / N_pix

	if eta >= 0.35: 
		continue

	else:
		## obj_cat
		source = asc.read( source_cat % (band_str, ra_g, dec_g, z_g),)
		Numb = np.array(source['NUMBER'][-1])
		A = np.array(source['A_IMAGE'])
		B = np.array(source['B_IMAGE'])
		theta = np.array(source['THETA_IMAGE'])
		cx = np.array(source['X_IMAGE'])
		cy = np.array(source['Y_IMAGE'])
		p_type = np.array(source['CLASS_STAR'])

		Kron = 16
		a = Kron * A
		b = Kron * B


		fig = plt.figure( figsize = (19.84, 4.8) )
		ax0 = fig.add_axes([0.03, 0.09, 0.28, 0.85])
		ax1 = fig.add_axes([0.36, 0.09, 0.28, 0.85])
		ax2 = fig.add_axes([0.68, 0.09, 0.28, 0.85])

		ax0.imshow( img_o, origin = 'lower', cmap = 'Greys', vmin = -3e-2, vmax = 3e-2,)

		clust = Circle( xy = (cen_x, cen_y), radius = R1Mpc, fill = False, ec = 'r', ls = '--', linewidth = 1, alpha = 0.75,)
		ax0.add_patch(clust)

		clust = Circle( xy = (cen_x, cen_y), radius = R200kpc, fill = False, ec = 'r', ls = '-', linewidth = 1, alpha = 0.75,)
		ax0.add_patch(clust)

		ax0.set_xlim( -100, img_o.shape[1] + 100 )
		ax0.set_ylim( -100, img_o.shape[0] + 100 )

		ax1.imshow( img_o, origin = 'lower', cmap = 'Greys', vmin = -3e-2, vmax = 3e-2,)

		clust = Circle( xy = (cen_x, cen_y), radius = R1Mpc, fill = False, ec = 'r', ls = '--', linewidth = 1, alpha = 0.75,)
		ax1.add_patch(clust)

		clust = Circle( xy = (cen_x, cen_y), radius = R200kpc, fill = False, ec = 'r', ls = '-', linewidth = 1, alpha = 0.75,)
		ax1.add_patch(clust)

		for ll in range( Numb ):

			ellips = Ellipse( xy = (cx[ll], cy[ll]), width = a[ll], height = b[ll], angle = theta[ll], fill = False, 
				ec = 'g', ls = '-', linewidth = 1, alpha = 0.35,)
			ax1.add_patch( ellips )

		ax1.set_xlim( -100, img_o.shape[1] + 100 )
		ax1.set_ylim( -100, img_o.shape[0] + 100 )

		ax2.imshow( img_m, origin = 'lower', cmap = 'Greys', vmin = -3e-2, vmax = 3e-2,)

		clust = Circle( xy = (cen_x, cen_y), radius = R1Mpc, fill = False, ec = 'r', ls = '--', linewidth = 1, alpha = 0.75,)
		ax2.add_patch(clust)

		clust = Circle( xy = (cen_x, cen_y), radius = R200kpc, fill = False, ec = 'r', ls = '-', linewidth = 1, alpha = 0.75,)
		ax2.add_patch(clust)

		ax2.set_xlim( -100, img_o.shape[1] + 100 )
		ax2.set_ylim( -100, img_o.shape[0] + 100 )

		# plt.savefig('/home/xkchen/figs/photo-z_%s-band_ra%.3f-dec%.3f-z%.3f_compare.png' % (band_str, ra_g, dec_g, z_g), dpi = 300)
		plt.savefig('/home/xkchen/figs/spec-z_%s-band_ra%.3f-dec%.3f-z%.3f_compare.png' % (band_str, ra_g, dec_g, z_g), dpi = 300)
		plt.close()

print('done!')


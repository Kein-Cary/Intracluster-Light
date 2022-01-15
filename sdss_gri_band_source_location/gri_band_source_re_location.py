import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc

from astropy.coordinates import SkyCoord
from scipy import interpolate as interp
from astropy import cosmology as apcy

from light_measure import jack_SB_func
from BCG_SB_pro_stack import BCG_SB_pros_func
from img_pre_selection import cat_match_func, get_mu_sigma
from fig_out_module import zref_BCG_pos_func
from img_cat_param_match import match_func
from img_mask import source_detect_func, mask_func
from img_mask_adjust import adjust_mask_func

##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

##### constant
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
pc2cm = U.pc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)

pixel = 0.396

#**********# local

load = '/home/xkchen/mywork/ICL/data/'

### masking adjust test
lo_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/band_match/low_BCG_star-Mass_g-band_remain_cat.csv')
lo_ra, lo_dec, lo_z = np.array(lo_dat.ra), np.array(lo_dat.dec), np.array(lo_dat.z)

d_file = load + 'sdss_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
cat_file = '/home/xkchen/mywork/ICL/data/corrected_star_cat/dr12/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'

z_set, ra_set, dec_set = lo_z[:20], lo_ra[:20], lo_dec[:20]

out_file0 = load + 'tmp_img/2012_mask_test/cluster_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
out_file1 = load + 'tmp_img/2012_mask_test/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'

bcg_mask = 0
'''
for kk in range( 3 ):
	#mask_func(d_file, cat_file, z_set, ra_set, dec_set, band[kk], out_file0, out_file1, bcg_mask, stack_info = None, pixel = 0.396, source_det = True,)
	source_detect_func(d_file, z_set, ra_set, dec_set, band[kk], out_file0,)
'''
extra_cat = [load + 'tmp_img/2012_mask_test/cluster_r-band_mask_ra%.3f_dec%.3f_z%.3f.cat',
			 load + 'tmp_img/2012_mask_test/cluster_g-band_mask_ra%.3f_dec%.3f_z%.3f.cat', 
			 load + 'tmp_img/2012_mask_test/cluster_i-band_mask_ra%.3f_dec%.3f_z%.3f.cat',]

bcg_photo_file = '/media/xkchen/My Passport/data/SDSS/BCG_photometric/BCG_photo_Z%.3f_ra%.3f_dec%.3f.txt'

#adjust_mask_func(d_file, cat_file, z_set, ra_set, dec_set, band[1], out_file0, out_file1, bcg_mask, bcg_photo_file, extra_cat,)

for kk in range( 20 ):

	ra_g, dec_g, z_g = ra_set[kk], dec_set[kk], z_set[kk]

	r_data = fits.open( d_file % ('r', ra_g, dec_g, z_g),)
	r_img = r_data[0].data
	r_wcs = awc.WCS(r_data[0].header)
	r_cx, r_cy = r_wcs.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

	r_gal_cat = asc.read( out_file0 % ('r', ra_g, dec_g, z_g),)
	r_gal_x = np.array( r_gal_cat['X_IMAGE'])
	r_gal_y = np.array( r_gal_cat['Y_IMAGE'])
	r_gal_A = np.array( r_gal_cat['A_IMAGE'])
	r_gal_B = np.array( r_gal_cat['B_IMAGE'])
	r_gal_chi = np.array( r_gal_cat['THETA_IMAGE'])

	r_gal_ra, r_gal_dec = r_wcs.all_pix2world(r_gal_x, r_gal_y, 1)


	g_data = fits.open( d_file % ('g', ra_g, dec_g, z_g),)
	g_img = g_data[0].data
	g_wcs = awc.WCS(g_data[0].header)
	g_cx, g_cy = g_wcs.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

	g_gal_cat = asc.read( out_file0 % ('g', ra_g, dec_g, z_g),)
	g_gal_x = np.array( g_gal_cat['X_IMAGE'])
	g_gal_y = np.array( g_gal_cat['Y_IMAGE'])
	g_gal_A = np.array( g_gal_cat['A_IMAGE'])
	g_gal_B = np.array( g_gal_cat['B_IMAGE'])
	g_gal_chi = np.array( g_gal_cat['THETA_IMAGE'])

	g_gal_ra, g_gal_dec = g_wcs.all_pix2world(g_gal_x, g_gal_y, 1)


	i_data = fits.open( d_file % ('i', ra_g, dec_g, z_g),)
	i_img = i_data[0].data
	i_wcs = awc.WCS(i_data[0].header)
	i_cx, i_cy = i_wcs.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

	i_gal_cat = asc.read( out_file0 % ('i', ra_g, dec_g, z_g),)
	i_gal_x = np.array( i_gal_cat['X_IMAGE'])
	i_gal_y = np.array( i_gal_cat['Y_IMAGE'])
	i_gal_A = np.array( i_gal_cat['A_IMAGE'])
	i_gal_B = np.array( i_gal_cat['B_IMAGE'])
	i_gal_chi = np.array( i_gal_cat['THETA_IMAGE'])

	i_gal_ra, i_gal_dec = i_wcs.all_pix2world(i_gal_x, i_gal_y, 1)


	BCG_photo_cat = pds.read_csv( bcg_photo_file % (z_g, ra_g, dec_g), skiprows = 1)
	r_Reff = np.array(BCG_photo_cat['deVRad_r'])[0]
	g_Reff = np.array(BCG_photo_cat['deVRad_g'])[0]
	i_Reff = np.array(BCG_photo_cat['deVRad_i'])[0]

	## sources location in different filter comparison
	r2g_x, r2g_y = g_wcs.all_world2pix( r_gal_ra * U.deg, r_gal_dec * U.deg, 1)
	i2g_x, i2g_y = g_wcs.all_world2pix( i_gal_ra * U.deg, i_gal_dec * U.deg, 1)

	g2r_x, g2r_y = r_wcs.all_world2pix( g_gal_ra * U.deg, g_gal_dec * U.deg, 1)
	i2r_x, i2r_y = r_wcs.all_world2pix( i_gal_ra * U.deg, i_gal_dec * U.deg, 1)

	r2i_x, r2i_y = i_wcs.all_world2pix( r_gal_ra * U.deg, r_gal_dec * U.deg, 1)
	g2i_x, g2i_y = i_wcs.all_world2pix( g_gal_ra * U.deg, g_gal_dec * U.deg, 1)

	Kron = 6
	DL = 100

	fig = plt.figure( figsize = (19.84, 4.8) )
	fig.suptitle('ra%.3f dec%.3f z%.3f' % (ra_g, dec_g, z_g),)
	ax0 = fig.add_axes([0.03, 0.09, 0.30, 0.80])
	ax1 = fig.add_axes([0.36, 0.09, 0.30, 0.80])
	ax2 = fig.add_axes([0.69, 0.09, 0.30, 0.80])

	ax0.set_title('r band')
	ax1.set_title('g band')
	ax2.set_title('i band')

	ax0.imshow(r_img, origin = 'lower', cmap = 'seismic', vmin = -5e-1, vmax = 5e-1,)
	ax0.scatter(r_cx, r_cy, s = 25, marker = 'X', edgecolors = 'b', facecolors = 'none', linewidth = 0.75, alpha = 0.5,)

	for mm in range( len(r_gal_x) ):
		ellips = Ellipse(xy = (r_gal_x[mm], r_gal_y[mm]), width = 0.5 * Kron * r_gal_A[mm], height = 0.5 * Kron * r_gal_B[mm], 
			angle = r_gal_chi[mm], fill = False, ec = 'r', ls = '-', alpha = 0.5,)
		ax0.add_patch(ellips)

	for mm in range( len(g_gal_x) ):
		ellips = Ellipse(xy = (g2r_x[mm], g2r_y[mm]), width = 0.6 * Kron * g_gal_A[mm], height = 0.6 * Kron * g_gal_B[mm], 
			angle = g_gal_chi[mm], fill = False, ec = 'g', ls = '-', alpha = 0.5,)
		ax0.add_patch(ellips)

	for mm in range( len(i_gal_x) ):
		ellips = Ellipse(xy = (i2r_x[mm], i2r_y[mm]), width = Kron * i_gal_A[mm], height = Kron * i_gal_B[mm], 
			angle = i_gal_chi[mm], fill = False, ec = 'b', ls = '-', alpha = 0.5,)
		ax0.add_patch(ellips)

	ax0.set_xlim( r_cx - DL, r_cx + DL)
	ax0.set_ylim( r_cy - DL, r_cy + DL)

	ax1.imshow(g_img, origin = 'lower', cmap = 'seismic', vmin = -5e-1, vmax = 5e-1,)
	ax1.scatter( g_cx, g_cy, s = 25, marker = 'X', edgecolors = 'b', facecolors = 'none', linewidth = 0.75, alpha = 0.5,)

	for mm in range( len(r_gal_x) ):
		ellips = Ellipse(xy = (r2g_x[mm], r2g_y[mm]), width = 0.5 * Kron * r_gal_A[mm], height = 0.5 * Kron * r_gal_B[mm], 
			angle = r_gal_chi[mm], fill = False, ec = 'r', ls = '-', alpha = 0.5,)
		ax1.add_patch(ellips)

	for mm in range( len(g_gal_x) ):
		ellips = Ellipse(xy = (g_gal_x[mm], g_gal_y[mm]), width = 0.6 * Kron * g_gal_A[mm], height = 0.6 * Kron * g_gal_B[mm], 
			angle = g_gal_chi[mm], fill = False, ec = 'g', ls = '-', alpha = 0.5,)
		ax1.add_patch(ellips)

	for mm in range( len(i_gal_x) ):
		ellips = Ellipse(xy = (i2g_x[mm], i2g_y[mm]), width = Kron * i_gal_A[mm], height = Kron * i_gal_B[mm], 
			angle = i_gal_chi[mm], fill = False, ec = 'b', ls = '-', alpha = 0.5,)
		ax1.add_patch(ellips)

	ax1.set_xlim( r_cx - DL, r_cx + DL)
	ax1.set_ylim( r_cy - DL, r_cy + DL)

	ax2.imshow(i_img, origin = 'lower', cmap = 'seismic', vmin = -5e-1, vmax = 5e-1,)
	ax2.scatter( i_cx, i_cy, s = 25, marker = 'X', edgecolors = 'b', facecolors = 'none', linewidth = 0.75, alpha = 0.5,)

	for mm in range( len(r_gal_x) ):
		ellips = Ellipse(xy = (r2i_x[mm], r2i_y[mm]), width = 0.5 * Kron * r_gal_A[mm], height = 0.5 * Kron * r_gal_B[mm], 
			angle = r_gal_chi[mm], fill = False, ec = 'r', ls = '-', alpha = 0.5,)
		ax2.add_patch(ellips)

	for mm in range( len(g_gal_x) ):
		ellips = Ellipse(xy = (g2i_x[mm], g2i_y[mm]), width = 0.6 * Kron * g_gal_A[mm], height = 0.6 * Kron * g_gal_B[mm], 
			angle = g_gal_chi[mm], fill = False, ec = 'g', ls = '-', alpha = 0.5,)
		ax2.add_patch(ellips)

	for mm in range( len(i_gal_x) ):
		ellips = Ellipse(xy = (i_gal_x[mm], i_gal_y[mm]), width = Kron * i_gal_A[mm], height = Kron * i_gal_B[mm], 
			angle = i_gal_chi[mm], fill = False, ec = 'b', ls = '-', alpha = 0.5,)
		ax2.add_patch(ellips)

	ax2.set_xlim( r_cx - DL, r_cx + DL)
	ax2.set_ylim( r_cy - DL, r_cy + DL)

	plt.savefig('img_ra%.3f_dec%.3f_z%.3f.png' % (ra_g, dec_g, z_g), dpi = 300)
	plt.close()

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
name_lis = ['low $M_{\\ast}$', 'high $M_{\\ast}$']

img_file = load + 'sdss_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

tmp_r_offD, tmp_g_offD, tmp_i_offD = [], [], []
tmp_r_Reff, tmp_g_Reff, tmp_i_Reff = [], [], []

for ll in range( 2 ):

	for kk in range( 3 ):

		dat = pds.read_csv('/home/xkchen/mywork/ICL/%s_%s-band_off-D_BCG-eff-R.csv' % (cat_lis[ll], band[kk]),)
		ra, dec, z = np.array(dat['ra']), np.array(dat['dec']), np.array(dat['z'])

		r_offD, g_offD, i_offD = np.array(dat['r_off_D']), np.array(dat['g_off_D']), np.array(dat['i_off_D'])
		r_Reff, g_Reff, i_Reff = np.array(dat['r_Reff']), np.array(dat['g_Reff']), np.array(dat['i_Reff'])

		idrx = r_offD > r_Reff
		idgx = g_offD > g_Reff
		idix = i_offD > i_Reff

		tmp_r_offD.append( r_offD )
		tmp_g_offD.append( g_offD )
		tmp_i_offD.append( i_offD )

		tmp_r_Reff.append( r_Reff )
		tmp_g_Reff.append( g_Reff )
		tmp_i_Reff.append( i_Reff )

		## offset hist
		plt.figure()
		plt.title('%s %s' % (band[kk], name_lis[ll]),)
		plt.hist(r_offD, bins = 50, density = True, histtype = 'step', color = 'r', alpha = 0.5, 
			label = 'offset between redMapper catalog position' + '\n' + ' ' * 18 + 'and the detected cloest source')
		plt.hist(g_offD, bins = 50, density = True, histtype = 'step', color = 'g', alpha = 0.5, )
		plt.hist(i_offD, bins = 50, density = True, histtype = 'step', color = 'b', alpha = 0.5, )

		plt.hist(r_Reff, bins = 50, density = True, color = 'r', alpha = 0.25, label = '$R^{r}_{eff} [N(D_{off} >= R_{eff}) = %d]$' % np.sum(idrx),)
		plt.hist(g_Reff, bins = 50, density = True, color = 'g', alpha = 0.25, label = '$R^{g}_{eff} [N(D_{off} >= R_{eff}) = %d]$' % np.sum(idgx),)
		plt.hist(i_Reff, bins = 50, density = True, color = 'b', alpha = 0.25, label = '$R^{i}_{eff} [N(D_{off} >= R_{eff}) = %d]$' % np.sum(idix),)

		plt.axvline(x = np.median(r_Reff), color = 'r', ls = '--', linewidth = 1, alpha = 0.5, label = 'median',)
		plt.axvline(x = np.median(g_Reff), color = 'g', ls = '--', linewidth = 1, alpha = 0.5, )
		plt.axvline(x = np.median(i_Reff), color = 'b', ls = '--', linewidth = 1, alpha = 0.5, )
		plt.xlabel('$D_{off}$ (step) or BCG $R_{eff}$ (bar) [pixels]',)
		plt.ylabel('PDF')
		plt.yscale('log')
		#plt.xscale('log')
		plt.ylim(6e-3, 3e0)
		plt.xlim(0, 30)
		plt.legend(loc = 1, fontsize = 9)
		plt.tick_params(axis = 'both', which = 'both', direction = 'in')
		plt.savefig('%s_%s-band_offD_hist.png' % (cat_lis[ll], band[kk]), dpi = 300)
		plt.close()


import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc

from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy.stats import binned_statistic as binned
from fig_out_module import cc_grid_img, grid_img

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
z_ref = 0.25
band = ['r', 'g', 'i']

def simple_match(ra_lis, dec_lis, z_lis, ref_file, id_choose = True):

	ref_dat = pds.read_csv( ref_file )
	tt_ra, tt_dec, tt_z = np.array(ref_dat.ra), np.array(ref_dat.dec), np.array(ref_dat.z)

	dd_ra, dd_dec, dd_z = [], [], []
	order_lis = []

	for kk in range( len(tt_z) ):
		identi = ('%.3f' % tt_ra[kk] in ra_lis) * ('%.3f' % tt_dec[kk] in dec_lis) * ('%.3f' % tt_z[kk] in z_lis)

		if id_choose == True:
			if identi == True:
				dd_ra.append( tt_ra[kk])
				dd_dec.append( tt_dec[kk])
				dd_z.append( tt_z[kk])
				order_lis.append( kk )

			else:
				continue
		else:
			if identi == True:
				continue
			else:
				dd_ra.append( tt_ra[kk])
				dd_dec.append( tt_dec[kk])
				dd_z.append( tt_z[kk])
				order_lis.append( kk )

	dd_ra = np.array( dd_ra)
	dd_dec = np.array( dd_dec)
	dd_z = np.array( dd_z)
	order_lis = np.array( order_lis )

	return dd_ra, dd_dec, dd_z, order_lis

home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'
"""
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

## image check
for kk in range( 3 ):

	dat_1 = pds.read_csv('/home/xkchen/photo-z_tot-%s-band_norm-img_cat.csv' % band[kk] )
	ra_1, dec_1, z_1 = np.array(dat_1['ra']), np.array(dat_1['dec']), np.array(dat_1['z'])

	Ns = len( z_1 )
	tt0 = np.random.choice( Ns, 50, replace = False,)

	set_ra, set_dec, set_z = ra_1[tt0], dec_1[tt0], z_1[tt0]

	for pp in range( 50 ):

		ra_g, dec_g, z_g = set_ra[pp], set_dec[pp], set_z[pp]

		d_fits1 = fits.open( home + 
			'tmp_stack/photo_z_img/photo-z_mask_%s_ra%.3f_dec%.3f_z%.3f.fits' % (band[kk], ra_g, dec_g, z_g),)
		img_1 = d_fits1[0].data

		fig = plt.figure( figsize = (13.12, 9.84) )
		ax0 = fig.add_axes([0.05, 0.50, 0.40, 0.40])
		ax1 = fig.add_axes([0.55, 0.50, 0.40, 0.40])
		ax2 = fig.add_axes([0.05, 0.05, 0.40, 0.40])
		ax3 = fig.add_axes([0.55, 0.05, 0.40, 0.40])

		ax0.set_title('grid size: 100 * 100')
		ax1.set_title('grid size: 150 * 150')
		ax2.set_title('grid size: 200 * 200')
		ax3.set_title('%s, ra%.3f dec%.3f z%.3f' % (band[kk], ra_g, dec_g, z_g),)

		patch_f0 = cc_grid_img( img_1, 100, 100)[0]
		patch_f1 = cc_grid_img( img_1, 150, 150)[0]
		patch_f2 = cc_grid_img( img_1, 200, 200)[0]

		tf0 = ax0.imshow(patch_f0 / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -5e-2, vmax = 5e-2,)
		cb0 = plt.colorbar( tf0, ax = ax0, fraction = 0.034, pad = 0.01,)
		cb0.formatter.set_powerlimits((0,0))

		tf1 = ax1.imshow(patch_f1 / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -5e-2, vmax = 5e-2,)
		cb1 = plt.colorbar( tf1, ax = ax1, fraction = 0.034, pad = 0.01,)
		cb1.formatter.set_powerlimits((0,0))

		tf2 = ax2.imshow(patch_f2 / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -5e-2, vmax = 5e-2,)
		cb2 = plt.colorbar( tf2, ax = ax2, fraction = 0.034, pad = 0.01,)
		cb2.formatter.set_powerlimits((0,0))

		tf3 = ax3.imshow(img_1 / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -5e-1, vmax = 5e-1,)
		cb3 = plt.colorbar( tf3, ax = ax3, fraction = 0.035, pad = 0.01,)

		plt.savefig(
			'/home/xkchen/figs/img_%s-band_ra%.3f_dec%.3f_z%.3f.png' % (band[kk], ra_g, dec_g, z_g), dpi = 300)
		plt.close()

raise
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

def pdf_func(data_arr, bins_arr,):

	N_pix, edg_f = binned(data_arr, data_arr, statistic = 'count', bins = bins_arr)[:2]
	pdf_pix = (N_pix / np.sum(N_pix) ) / (edg_f[1] - edg_f[0])
	pdf_err = ( np.sqrt(N_pix) / np.sum(N_pix) ) / (edg_f[1] - edg_f[0])
	f_cen = 0.5 * ( edg_f[1:] + edg_f[:-1])

	id_zero = N_pix < 1.
	pdf_arr = pdf_pix[ id_zero == False]
	err_arr = pdf_err[ id_zero == False]
	pdf_x = f_cen[ id_zero == False]

	return pdf_arr, err_arr, pdf_x

def mode_func(data_arr, bins_arr):

	pdf_arr, err_arr, pdf_x = pdf_func( data_arr, bins_arr)

	mean_f = np.mean( data_arr )
	std_f = np.std( data_arr )
	medi_f = np.median( data_arr )

	id_max = pdf_arr == np.max( pdf_arr )

	# if return gives more than one values, choose the one which closed to median 
	if np.sum( id_max ) > 1:
		S_medi = np.abs( pdf_x[id_max] - medi_f )
		idvx = S_medi == np.min( S_medi )
		mode_f = pdf_x[ id_max ][ idvx ]

	else:
		mode_f = pdf_x[ id_max ]

	return mode_f
'''
## (mu,sigma) histogram compare
for kk in range( 3 ):

	dat_0 = pds.read_csv('/home/xkchen/mywork/ICL/data/photo_cat/photo-z_%s-band_200-grid-img_mu-sigma.csv' % band[kk],)
	ra_0, dec_0, z_0 = np.array(dat_0['ra']), np.array(dat_0['dec']), np.array(dat_0['z'])
	cen_mu_0 = np.array( dat_0['cen_mu'])
	cen_sigma_0 = np.array( dat_0['cen_sigma'])
	img_mu_0 = np.array( dat_0['img_mu'])
	img_sigma_0 = np.array( dat_0['img_sigma'])

	dat_1 = pds.read_csv('/home/xkchen/mywork/ICL/data/photo_cat/photo-z_%s-band_100-grid-img_mu-sigma.csv' % band[kk],)
	ra_1, dec_1, z_1 = np.array(dat_1['ra']), np.array(dat_1['dec']), np.array(dat_1['z'])
	cen_mu_1 = np.array( dat_1['cen_mu'])
	cen_sigma_1 = np.array( dat_1['cen_sigma'])
	img_mu_1 = np.array( dat_1['img_mu'])
	img_sigma_1 = np.array( dat_1['img_sigma'])


	fig = plt.figure( figsize = (13.12, 4.8) )
	ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
	ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

	ax0.set_title('mean of image SB [%s band]' % band[kk])
	ax1.set_title('scatter of image SB [%s band]' % band[kk])

	sb0 = np.min([ np.min(img_mu_0), np.min(img_mu_1) ]) / pixel**2
	sb1 = np.max([ np.max(img_mu_0), np.max(img_mu_1) ]) / pixel**2

	if kk == 0:
		bin_w = np.std( img_mu_0 / pixel**2) / 10
	else:
		bin_w = np.std( img_mu_0 / pixel**2) / 30
	mu_bins = np.arange(sb0, sb1, bin_w)
	mu_bins = np.r_[ mu_bins, mu_bins[-1] + bin_w]

	## pdf
	mode_mu_0 = mode_func( img_mu_0 / pixel**2, mu_bins)
	mode_mu_1 = mode_func( img_mu_1 / pixel**2, mu_bins)

	print(mode_mu_0)
	print(mode_mu_1)

	mean_f_0 = np.mean(img_mu_0 / pixel**2)
	medi_f_0 = np.median(img_mu_0 / pixel**2)

	mean_f_1 = np.mean(img_mu_1 / pixel**2)
	medi_f_1 = np.median(img_mu_1 / pixel**2)


	ax0.hist(img_mu_0 / pixel**2, bins = mu_bins, density = True, color = 'b', alpha = 0.45,
		label = 'grid size = 200',)
	ax0.axvline( x = mean_f_0, ls = '-', color = 'b', alpha = 0.5, label = 'mean',)
	ax0.axvline( x = medi_f_0, ls ='--', color = 'b', alpha = 0.5, label = 'median',)
	ax0.axvline( x = mode_mu_0, ls = ':', color = 'b', alpha = 0.5, label = 'mode',)

	ax0.hist(img_mu_1 / pixel**2, bins = mu_bins, density = True, color = 'r', alpha = 0.45,
		label = 'grid size = 100')
	ax0.axvline( x = mean_f_1, ls = '-', color = 'r', alpha = 0.5,)
	ax0.axvline( x = medi_f_1, ls = '--', color = 'r', alpha = 0.5,)
	ax0.axvline( x = mode_mu_1, ls = ':', color = 'r', alpha = 0.5,)

	ax0.legend( loc = 1,)
	ax0.set_xlabel('SB [nanomaggies / $ arcsec^2 $]')
	ax0.set_ylabel('pdf')
	#ax0.set_xlim( np.mean(img_mu_0 / pixel**2) - 5 * np.std(img_mu_0 / pixel**2),
	#			np.mean(img_mu_0 / pixel**2) + 5 * np.std(img_mu_0 / pixel**2))
	ax0.set_xlim( -0.04, 0.04 )

	sb2 = np.min([ np.min(img_sigma_0), np.min(img_sigma_0) ]) / pixel**2
	sb3 = np.max([ np.max(img_sigma_1), np.max(img_sigma_1) ]) / pixel**2

	bin_w = np.std( img_sigma_0 / pixel**2) / 10
	sigma_bins = np.arange( sb2, sb3, bin_w)
	sigma_bins = np.r_[ sigma_bins, sigma_bins[-1] + bin_w ]

	## pdf compare
	mode_sigma_0 = mode_func( img_sigma_0 / pixel**2, sigma_bins)
	mode_sigma_1 = mode_func( img_sigma_1 / pixel**2, sigma_bins)

	mean_std_0 = np.mean(img_sigma_0 / pixel**2)
	medi_std_0 = np.median(img_sigma_0 / pixel**2)

	mean_std_1 = np.mean(img_sigma_1 / pixel**2)
	medi_std_1 = np.median(img_sigma_1 / pixel**2)


	ax1.hist(img_sigma_0 / pixel**2, bins = sigma_bins, density = True, color = 'b', alpha = 0.45,
		label = 'grid size = 200')
	ax1.axvline( x = mean_std_0, ls = '-', color = 'b', alpha = 0.5, label = 'mean',)
	ax1.axvline( x = medi_std_0, ls = '--', color ='b', alpha = 0.5, label = 'median',)
	ax1.axvline( x = mode_sigma_0, ls = ':', color = 'b', alpha = 0.5, label = 'mode',)

	ax1.hist(img_sigma_1 / pixel**2, bins = sigma_bins, density = True, color = 'r', alpha = 0.45,
		label = 'grid size = 100')
	ax1.axvline( x = mean_std_1, ls = '-', color = 'r', alpha = 0.5,)
	ax1.axvline( x = medi_std_1, ls = '--', color ='r', alpha = 0.5,)
	ax1.axvline( x = mode_sigma_1, ls = ':', color = 'r', alpha = 0.5,)

	ax1.legend( loc = 1,)
	ax1.set_xlabel('SB [nanomaggies / $ arcsec^2 $]')
	ax1.set_ylabel('pdf')
	ax1.set_xlim(0, np.mean(img_sigma_0 / pixel**2) + 5 * np.std(img_sigma_0 / pixel**2),)

	plt.savefig('/home/xkchen/photo-z_%s-band_img-mu-sigma_compare.png' % band[kk], dpi = 300)
	plt.close()
'''


import matplotlib as mpl
import matplotlib.pyplot as plt
from img_pre_selection import cat_match_func
from img_cat_param_match import match_func

lo_dat = pds.read_csv( '/home/xkchen/mywork/ICL/data/' + 'BCG_stellar_mass_cat/low_star-Mass_cat.csv')
lo_ra, lo_dec, lo_z = np.array(lo_dat.ra), np.array(lo_dat.dec), np.array(lo_dat.z)
lo_rich, lo_M_star = np.array(lo_dat.rich), np.array(lo_dat.lg_Mass)
C_lo = SkyCoord(ra = lo_ra * U.degree, dec = lo_dec * U.degree)

idlx = (lo_z >= 0.2) & (lo_z <= 0.3)
print('low, N = ', np.sum(idlx) )


hi_dat = pds.read_csv( '/home/xkchen/mywork/ICL/data/' + 'BCG_stellar_mass_cat/high_star-Mass_cat.csv')
hi_ra, hi_dec, hi_z = np.array(hi_dat.ra), np.array(hi_dat.dec), np.array(hi_dat.z)
hi_rich, hi_M_star = np.array(hi_dat.rich), np.array(hi_dat.lg_Mass)
C_hi = SkyCoord(ra = hi_ra * U.degree, dec = hi_dec * U.degree)

idhx = (hi_z >= 0.2) & (hi_z <= 0.3)
print('high, N = ', np.sum(idhx) )


cat_file = '/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits'
sf_len = 5

out_file_0 = 'low_star-Mass_macth_test_cat.csv'
match_func( lo_ra, lo_dec, lo_z, cat_file, out_file_0, sf_len, id_spec = False,)

out_file_1 = 'high_star-Mass_macth_test_cat.csv'
match_func( hi_ra, hi_dec, hi_z, cat_file, out_file_1, sf_len, id_spec = False,)


dat_0 = pds.read_csv( 'low_star-Mass_macth_test_cat.csv' )
ra_0 = np.array(dat_0['ra'])
print( 'lo match', len(ra_0) )

dat_1 = pds.read_csv( 'high_star-Mass_macth_test_cat.csv' )
ra_1 = np.array(dat_1['ra'])
print( 'hi match', len(ra_1) )


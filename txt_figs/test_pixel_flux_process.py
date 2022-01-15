import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C
import scipy.signal as signal

from astropy import cosmology as apcy
from scipy import interpolate as interp
from scipy import integrate as integ
from astropy.coordinates import SkyCoord

from light_measure import flux_recal
from light_measure import light_measure_Z0_weit, light_measure_weit

from resample_modelu import sum_samp, down_samp
from img_resample import resamp_func

### === ### cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

pixel = 0.396
band = ['r', 'g', 'i']
L_wave = np.array([ 6166, 4686, 7480 ])
Mag_sun = [ 4.65, 5.11, 4.53 ]
z_ref = 0.25

### === ### catalog and images
d_Path = '/home/xkchen/mywork/ICL/data/'
img_file = d_Path + 'sdss_data/frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

dat = pds.read_csv( d_Path + 'BCG_stellar_mass_cat/match_compare/high_star-Mass_z-spec_macth_cat.csv')
ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

for ii in range( 1,2 ):

	ra_g, dec_g, z_g = ra[ii], dec[ii], z[ii]

	img_data = fits.open( img_file % (ra_g, dec_g, z_g) )

	r_img = img_data[0].data
	Head = img_data[0].header
	wcs_lis = awc.WCS( Head )

	cen_x, cen_y = wcs_lis.all_world2pix( ra_g * U.deg, dec_g * U.deg, 0 )

	Ny, Nx = r_img.shape

	weit_data = np.ones( (Ny, Nx), dtype = np.float32)

	# plt.figure()
	# plt.imshow( r_img, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm(),)
	# plt.plot( cen_x, cen_y, 'r+')
	# plt.show()

	R_bins = np.logspace( 0, 3, 50)
	mu_z0, angl_r0, mu0_err, npix, nratio = light_measure_Z0_weit( r_img, weit_data, pixel, cen_x, cen_y, R_bins)


	#. current resampling method
	# Da_g = Test_model.angular_diameter_distance(z_g).value # unit Mpc
	# Da_ref = Test_model.angular_diameter_distance(z_ref).value

	# L_ref = Da_ref * pixel / rad2asec
	# L_z0 = Da_g * pixel / rad2asec
	# _pix_f = L_ref / L_z0

	_pix_f = 0.5

	pre_img = r_img / _pix_f**2

	bf_pix = pixel / _pix_f

	mu_z1, angl_r1, mu1_err, npix, nratio = light_measure_Z0_weit( pre_img, weit_data, bf_pix, cen_x, cen_y, R_bins )


	eta_pix = _pix_f + 0.

	if eta_pix > 1:
		resp_img, resp_x, resp_y = sum_samp( eta_pix, eta_pix, pre_img, cen_x, cen_y )
	else:
		resp_img, resp_x, resp_y = down_samp( eta_pix, eta_pix, pre_img, cen_x, cen_y )

	dev_05_x = resp_x - np.int( resp_x )
	dev_05_y = resp_y - np.int( resp_y )

	if dev_05_x > 0.5:
		xn = np.int( resp_x ) + 1
	else:
		xn = np.int( resp_x )

	if dev_05_y > 0.5:
		yn = np.int( resp_y + 1 )
	else:
		yn = np.int( resp_y )

	Ny_resp, Nx_resp = resp_img.shape
	weit_arr = np.ones( (Ny_resp, Nx_resp), dtype = np.float32)

	mu_z2, angl_r2, mu2_err, npix, nratio = light_measure_Z0_weit( resp_img, weit_arr, pixel, xn, yn, R_bins )


	#.. previous calculation
	if eta_pix > 1:
		cc_resp_img, cc_x1, cc_y1 = sum_samp( eta_pix, eta_pix, r_img, cen_x, cen_y )
	else:
		cc_resp_img, cc_x1, cc_y1 = down_samp( eta_pix, eta_pix, r_img, cen_x, cen_y )

	mu_z3, angl_r3, mu3_err, npix, nratio = light_measure_Z0_weit( cc_resp_img, weit_arr, pixel, xn, yn, R_bins )		


	fig = plt.figure( figsize = (13.12, 9.84) )
	ax0 = fig.add_axes([0.05, 0.55, 0.40, 0.45])
	ax1 = fig.add_axes([0.55, 0.55, 0.40, 0.45])
	ax2 = fig.add_axes([0.05, 0.05, 0.40, 0.45])
	ax3 = fig.add_axes([0.55, 0.08, 0.40, 0.40])

	ax0.set_title('Origin image')
	ax0.imshow( r_img, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm(),)

	ax1.set_title('Shifted image')
	ax1.imshow( pre_img, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm(),)

	ax2.set_title('Resampled image')
	ax2.imshow( resp_img, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm(),)

	ax3.plot( angl_r0[:-1], mu_z0[:-1], 'k:', alpha = 0.75, label = 'Origin image')
	ax3.plot( angl_r1[:-1], mu_z1[:-1], 'b--', alpha = 0.75, label = 'Shifted image')
	ax3.plot( angl_r2[:-1], mu_z2[:-1], 'r-', alpha = 0.75, label = 'Resampled image')

	ax3.plot( angl_r3[:-1], mu_z3[:-1], 'g-.', label = 'previous resampled image')

	ax3.legend( loc = 1, frameon = False,)
	ax3.set_xlim( 1e0, 1e3)
	ax3.set_xscale('log')
	ax3.set_xlabel('R [arcsec]')
	ax3.set_yscale('log')
	ax3.set_ylabel('$\\mu \; [ nanomaggies \, / \, arcsec^2 ]$')

	plt.savefig('/home/xkchen/image_trans_check.png', dpi = 300)
	plt.close()


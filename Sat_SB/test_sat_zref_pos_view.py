import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Ellipse

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
import scipy.signal as signal
import scipy.stats as sts
from scipy import ndimage


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

rad2asec = U.rad.to(U.arcsec)


### === satellite location point
Da_ref = Test_model.angular_diameter_distance( z_ref ).value
L_pix = Da_ref * 10**3 * pixel / rad2asec
n500 = 500 / L_pix
n200 = 200 / L_pix

cat_lis = ['inner-mem', 'outer-mem']
fig_name = ['$R_{Sat} \, \\leq \, 0.191 R_{200m}$', '$R_{Sat} \, > \, 0.191 R_{200m}$']

tmp_Rs = []

for mm in range( 2 ):

	for kk in range( 1 ):

		band_str = band[ kk ]

		#. background image
		BG_file = '/home/xkchen/figs/extend_bcgM_cat_Sat/Radii_bin/BG_2D_file/%s_%s-band_BG_img.fits' % (cat_lis[mm], band_str)
		BG_data = fits.open( BG_file )
		BG_img = BG_data[0].data

		BG_xn, BG_yn = BG_data[0].header['CENTER_X'], BG_data[0].header['CENTER_Y']
		Nx, Ny = BG_img.shape[1], BG_img.shape[0]


		#. relative position
		dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/mock_BG_tract_cat/' + 
						'Extend-BCGM_rgi-common_frame-lim_Pm-cut_%s_%s-band_BG-tract_cat.csv' % (cat_lis[mm], band_str),)

		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

		R_sat = np.array( dat['R_sat'] )   ## in units of kpc
		sat_PA = np.array( dat['sat_PA'] )  ## position angle (relative to BCG)

		Da_x = Test_model.angular_diameter_distance( bcg_z ).value # Mpc

		R_pix = ( R_sat * 1e-3 / Da_x * rad2asec ) / pixel
		off_xn, off_yn = R_pix * np.cos( sat_PA ), R_pix * np.sin( sat_PA )
		tag_xn, tag_yn = BG_xn + off_xn, BG_yn + off_yn


		Ns_bin = sts.binned_statistic_2d( tag_xn, tag_yn, values = tag_yn, statistic = 'count', bins = 100 )[0]
		sm_Ns_bin = ndimage.gaussian_filter( Ns_bin, sigma = 1,)

		tmp_Rs.append( R_sat )


		##. figs
		ns_level = np.array( [1, 3, 5, 10, 15, 20, 30, 40 ] )

		mu_level_0 = np.linspace( -2.4, -1, 9)
		mu_level_1 = np.linspace( -2.5, -2.4, 11)
		mu_level = np.r_[ mu_level_1[:-1], mu_level_0 ]


		fig = plt.figure( figsize = (5, 5) )
		ax = fig.add_axes([ 0.12, 0.12, 0.80, 0.80])

		ax.set_title( fig_name[mm] )

		# ax.imshow( np.log10( BG_img / pixel**2 ), origin = 'lower', cmap = 'rainbow', vmin = -2.5, vmax = 0,)

		ax.scatter( tag_xn, tag_yn, marker = '.', s = 5, c = R_sat / R_sat.max(), cmap = 'winter', alpha = 0.1,)

		ax.contour( np.log10( BG_img / pixel**2 ), origin = 'lower', cmap = 'rainbow', 
					levels = mu_level_1, linestyles = '--', linewidths = 1, extent = (0, Nx, 0, Ny),)

		# tf = ax.contour( sm_Ns_bin.T, levels = ns_level, cmap = 'Greys_r', extent = (0, Nx, 0, Ny),)
		# plt.clabel( tf, colors = 'k', fontsize = 8, fmt = '%.0f', inline = False,)

		ax.set_xlim( 0, Nx )
		ax.set_ylim( 0, Ny )

		ax.set_xticklabels( labels = [], major = True,)
		ax.set_yticklabels( labels = [], major = True,)


		x_ticks_0 = np.arange( BG_xn, 0, -1 * n500 )
		x_ticks_1 = np.arange( BG_xn, Nx, n500 )
		x_ticks = np.r_[ x_ticks_0[::-1], x_ticks_1[1:] ]

		y_ticks_0 = np.arange( BG_yn, 0, -1 * n500)
		y_ticks_1 = np.arange( BG_yn, Ny, n500 )
		y_ticks = np.r_[ y_ticks_0[::-1], y_ticks_1[1:] ]

		tick_R = np.r_[ np.arange( -( len(x_ticks_0) - 1 ) * 500, 0, 500), np.arange(0, 500 * ( len(x_ticks_1) ), 500) ]
		tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

		ax.set_xticks( x_ticks, minor = True, )
		ax.set_xticklabels( labels = tick_lis, minor = True,)
		ax.set_xlabel('Mpc')

		tick_R = np.r_[ np.arange( -( len(y_ticks_0) - 1 ) * 500, 0, 500), np.arange(0, 500 * ( len(y_ticks_1) ), 500) ]
		tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

		ax.set_yticks( y_ticks, minor = True, )
		ax.set_yticklabels( labels = tick_lis, minor = True,)
		ax.set_ylabel('Mpc')

		ax.tick_params( axis = 'both', which = 'major', direction = 'in', bottom = False, left = False, top = False)

		plt.savefig('/home/xkchen/%s_%s-band_sat_pos_view.png' % (cat_lis[mm], band_str), dpi = 300)
		plt.close()

raise

plt.figure()
plt.hist( tmp_Rs[0], bins = 55, density = True, histtype = 'step', color = 'b', label = fig_name[0],)
plt.axvline( x = np.median( tmp_Rs[0] ), ls = '--', color = 'b', alpha = 0.75, label = 'Median')
plt.axvline( x = np.mean( tmp_Rs[0] ), ls = '-', color = 'b', alpha = 0.75, label = 'Mean')

plt.hist( tmp_Rs[1], bins = 55, density = True, histtype = 'step', color = 'r', label = fig_name[1],)
plt.axvline( x = np.median( tmp_Rs[1] ), ls = '--', color = 'r', alpha = 0.75,)
plt.axvline( x = np.mean( tmp_Rs[1] ), ls = '-', color = 'r', alpha = 0.75,)

plt.legend( loc = 1,)
plt.xlabel('$R_{Sat} \; [kpc]$')

plt.savefig('/home/xkchen/R_sat_pdf.png', dpi = 300)
plt.close()


"""
record the most closed galaxy and its size (~3 * R_kron from source detection)
PS : for comparison, 
the most closed ===> R_pix or R_phy~( given by redMaPPer)
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc
import scipy.stats as sts

from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy.stats import binned_statistic as binned


from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


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


### === get line of two points
def getLinearEquation(p1x, p1y, p2x, p2y):

	sign = 1
	a = p2y - p1y
	if a < 0:
		sign = -1
		a = sign * a
	b = sign * (p1x - p2x)
	c = sign * (p1y * p2x - p1x * p2y)

	return [a, b, c]


### === partial cut test, associated with the closed satellite
#.
cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/part_cut_BCG_stack/closed_sat_cat/'

# keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'bcg_x', 'bcg_y', 'clus_ID', 

# 		 'min_Rpix_sat_x', 'min_Rpix_sat_y', 'min_Rpix_sat_ra', 'min_Rpix_sat_dec', 
# 		 'min_Rpix_Rsat', 'min_Rpix_sR_sat', 'min_Rpix_PA2bcg',
		 
# 		 'min_Rphy_sat_x', 'min_Rphy_sat_y', 'min_Rphy_sat_ra', 'min_Rphy_sat_dec', 
# 		 'min_Rphy_Rsat', 'min_Rphy_sR_sat', 'min_Rphy_PA2bcg' ]

dat = pds.read_csv( cat_path + 
		'Extend-BCGM_rgi-common_frame-limit_exlu-BCG_Sat_r-band_closed-mem_position.csv')

bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
clus_ID = np.array( dat['clus_ID'] )

#.
min_Rpix_ra, min_Rpix_dec = np.array( dat['min_Rpix_sat_ra'] ), np.array( dat['min_Rpix_sat_dec'] )

min_Rpix_R_sat = np.array( dat['min_Rpix_Rsat'] )
min_Rpix_sR_sat = np.array( dat['min_Rpix_sR_sat'] )

min_Rpix_xs = np.array( dat['min_Rpix_sat_x'] )
min_Rpix_ys = np.array( dat['min_Rpix_sat_y'] )
min_Rpix_chi2bcg = np.array( dat['min_Rpix_PA2bcg'] )

min_Rpix_sat_ar = np.array( dat['min_Rpix_sat_ar'] )
min_Rpix_sat_br = np.array( dat['min_Rpix_sat_br'] )

min_Rpix_sat_chi = np.array( dat['min_Rpix_sat_chi'] )
min_Rpix_sat_chi = min_Rpix_sat_chi * np.pi / 180



##.
cat_1 = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/shufl_with_BCG_PA/BCG_PA_cat/' + 
						'BCG_located-params_r-band.csv')
ra_1, dec_1, z_1 = np.array( cat_1['ra'] ), np.array( cat_1['dec'] ), np.array( cat_1['z'] )

IDs_1 = np.array( cat_1['clus_ID'] )
IDs_1 = IDs_1.astype( np.int )

#. -90 ~ 90
PA_1 = np.array( cat_1['PA'] )
#. change to rad
PA_1 = PA_1 * np.pi / 180


### === ### partial cut image test
img_files = ( '/media/xkchen/My Passport/data/SDSS/photo_data/' + 
			'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',)[0]

N_s = len( bcg_ra )

band_str = 'r'


###... half BCG cut
"""
for kk in range( 10 ):

	ra_g, dec_g, z_g = bcg_ra[ kk ], bcg_dec[ kk ], bcg_z[ kk ]
	kk_ID = clus_ID[ kk ]

	##.
	id_ux = IDs_1 == kk_ID
	ref_PA = PA_1[ id_ux ][0]

	##. closed satellite
	kk_s_ra, kk_s_dec = min_Rpix_ra[ kk ], min_Rpix_dec[ kk ]
	kk_s_x, kk_s_y = min_Rpix_xs[ kk ], min_Rpix_ys[ kk ] 

	kk_s_PA = min_Rpix_chi2bcg[ kk ]

	##.
	img_data = fits.open( img_files % (band_str, ra_g, dec_g, z_g),)
	img_arr = img_data[0].data
	wcs_lis = awc.WCS( img_data[0].header )

	x_cen, y_cen = wcs_lis.all_world2pix( ra_g, dec_g, 0 )
	Ny, Nx = img_arr.shape


	##. lines of major-axis
	xx_1 = np.linspace( x_cen - 500, x_cen + 500, 200 )

	k1 = np.tan( ref_PA )
	b1 = y_cen - k1 * x_cen
	l1 = k1 * xx_1 + b1

	##. identify the relative position of points to the major-axis
	dy = kk_s_y - ( k1 * kk_s_x + b1 )

	nx = np.linspace( 0, Nx - 1, Nx )
	ny = np.linspace( 0, Ny - 1, Ny )
	grid_nx, grid_ny = np.meshgrid( nx, ny )

	#.
	delt_y = grid_ny - ( k1 * grid_nx + b1 )
	sign_y = dy * delt_y

	id_sign = sign_y > 0.

	copy_img = img_arr.copy()
	copy_img[ id_sign ] = np.nan


	##. align with frame
	fig = plt.figure( figsize = (10.4, 4.8) )
	ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
	ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

	ax0.imshow( img_arr, origin = 'lower', cmap = 'Greys', 
		norm = mpl.colors.SymLogNorm( linthresh = 1e-3, linscale = 1e-2, vmin = -1e-1, vmax = 5e0, base=10),)

	#.
	rect = mpathes.Ellipse(xy = (x_cen, y_cen), width = 30, height = 20, angle = ref_PA * 180 / np.pi, 
							ec = 'r', fc = 'none', ls = '-', alpha = 0.5,)
	ax0.add_patch( rect )

	ax0.scatter( kk_s_x, kk_s_y, marker = 'o', s = 30, facecolors = 'none', edgecolors = 'b', alpha = 0.5)

	ax0.plot( xx_1, l1, ls = '--', color = 'r', alpha = 0.5,)

	ax0.set_xlim( x_cen + 300, x_cen - 300 )
	ax0.set_ylim( y_cen + 300, y_cen - 300 )


	ax1.imshow( copy_img, origin = 'lower', cmap = 'Greys', 
		norm = mpl.colors.SymLogNorm( linthresh = 1e-3, linscale = 1e-2, vmin = -1e-1, vmax = 5e0, base=10),)

	rect = mpathes.Ellipse(xy = (x_cen, y_cen), width = 30, height = 20, angle = ref_PA * 180 / np.pi, 
							ec = 'r', fc = 'none', ls = '-', alpha = 0.5,)
	ax1.add_patch( rect )

	ax1.scatter( kk_s_x, kk_s_y, marker = 'o', s = 30, facecolors = 'none', edgecolors = 'b', alpha = 0.5)

	ax1.plot( xx_1, l1, ls = '--', color = 'r', alpha = 0.5,)

	ax1.set_xlim( x_cen + 300, x_cen - 300 )
	ax1.set_ylim( y_cen + 300, y_cen - 300 )

	plt.savefig('/home/xkchen/partial_cut_image_%d.png' % kk, dpi = 300)
	plt.close()

"""


###... Perpendicular cut~(relative to the line link BCG and satellite)
for kk in range( 10 ):

	ra_g, dec_g, z_g = bcg_ra[ kk ], bcg_dec[ kk ], bcg_z[ kk ]
	kk_ID = clus_ID[ kk ]

	##.
	id_ux = IDs_1 == kk_ID
	ref_PA = PA_1[ id_ux ][0]

	##. closed satellite
	kk_s_ra, kk_s_dec = min_Rpix_ra[ kk ], min_Rpix_dec[ kk ]
	kk_s_x, kk_s_y = min_Rpix_xs[ kk ], min_Rpix_ys[ kk ] 

	kk_s_PA = min_Rpix_chi2bcg[ kk ]

	##.
	img_data = fits.open( img_files % (band_str, ra_g, dec_g, z_g),)
	img_arr = img_data[0].data
	wcs_lis = awc.WCS( img_data[0].header )

	x_cen, y_cen = wcs_lis.all_world2pix( ra_g, dec_g, 0 )
	Ny, Nx = img_arr.shape


	##. lines of major-axis
	xx_1 = np.linspace( x_cen - 500, x_cen + 500, 200 )

	k1 = np.tan( ref_PA )
	b1 = y_cen - k1 * x_cen
	l1 = k1 * xx_1 + b1

	##. line link satellite and BCG
	k2 = (kk_s_y - y_cen) / (kk_s_x - x_cen)
	b2 = kk_s_y - kk_s_x * k2
	l2 = k2 * xx_1 + b2

	##. perpendocular to l2
	k3 = -1/k2
	b3 = y_cen - k3 * x_cen
	l3 = k3 * xx_1 + b3

	##. identify the relative position of points to the major-axis
	nx = np.linspace( 0, Nx - 1, Nx )
	ny = np.linspace( 0, Ny - 1, Ny )
	grid_nx, grid_ny = np.meshgrid( nx, ny )

	#.
	dy = kk_s_y - ( k3 * kk_s_x + b3 )

	delt_y = grid_ny - ( k3 * grid_nx + b3 )
	sign_y = dy * delt_y

	id_sign = sign_y > 0.

	copy_img = img_arr.copy()
	copy_img[ id_sign ] = np.nan


	##. align with frame
	fig = plt.figure( figsize = (10.4, 4.8) )
	ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
	ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

	ax0.imshow( img_arr, origin = 'lower', cmap = 'Greys', 
		norm = mpl.colors.SymLogNorm( linthresh = 1e-3, linscale = 1e-2, vmin = -1e-1, vmax = 5e0, base=10),)

	#.
	rect = mpathes.Ellipse(xy = (x_cen, y_cen), width = 30, height = 20, angle = ref_PA * 180 / np.pi, 
							ec = 'r', fc = 'none', ls = '-', alpha = 0.5,)
	ax0.add_patch( rect )

	ax0.scatter( kk_s_x, kk_s_y, marker = 'o', s = 30, facecolors = 'none', edgecolors = 'b', alpha = 0.5)

	ax0.plot( xx_1, l1, ls = '--', color = 'r', alpha = 0.5,)
	ax0.plot( xx_1, l2, ls = ':', color = 'c', lw = 2,)
	ax0.plot( xx_1, l3, ls = '-', color = 'b', alpha = 0.5,)

	ax0.set_xlim( x_cen + 300, x_cen - 300 )
	ax0.set_ylim( y_cen + 300, y_cen - 300 )


	ax1.imshow( copy_img, origin = 'lower', cmap = 'Greys', 
		norm = mpl.colors.SymLogNorm( linthresh = 1e-3, linscale = 1e-2, vmin = -1e-1, vmax = 5e0, base=10),)

	rect = mpathes.Ellipse(xy = (x_cen, y_cen), width = 30, height = 20, angle = ref_PA * 180 / np.pi, 
							ec = 'r', fc = 'none', ls = '-', alpha = 0.5,)
	ax1.add_patch( rect )

	ax1.scatter( kk_s_x, kk_s_y, marker = 'o', s = 30, facecolors = 'none', edgecolors = 'b', alpha = 0.5)

	ax1.plot( xx_1, l1, ls = '--', color = 'r', alpha = 0.5,)
	ax1.plot( xx_1, l2, ls = ':', color = 'c', lw = 2,)
	ax1.plot( xx_1, l3, ls = '-', color = 'b', alpha = 0.5,)

	ax1.set_xlim( x_cen + 300, x_cen - 300 )
	ax1.set_ylim( y_cen + 300, y_cen - 300 )

	plt.savefig('/home/xkchen/partial_cut_image_%d.png' % kk, dpi = 300)
	plt.close()

raise


###... mask pixels in angle region~(correlated with the size of satellite)
for kk in range( 10 ):

	ra_g, dec_g, z_g = bcg_ra[ kk ], bcg_dec[ kk ], bcg_z[ kk ]
	kk_ID = clus_ID[ kk ]

	##.
	id_ux = IDs_1 == kk_ID
	ref_PA = PA_1[ id_ux ][0]

	##. closed satellite
	kk_s_ra, kk_s_dec = min_Rpix_ra[ kk ], min_Rpix_dec[ kk ]
	kk_s_x, kk_s_y = min_Rpix_xs[ kk ], min_Rpix_ys[ kk ]
	kk_s_PA = min_Rpix_chi2bcg[ kk ]


	##.
	img_data = fits.open( img_files % (band_str, ra_g, dec_g, z_g),)
	img_arr = img_data[0].data
	wcs_lis = awc.WCS( img_data[0].header )

	x_cen, y_cen = wcs_lis.all_world2pix( ra_g, dec_g, 0 )
	Ny, Nx = img_arr.shape


	##. lines align major-axis of BCG
	xx_1 = np.linspace( x_cen - 500, x_cen + 500, 200 )

	k1 = np.tan( ref_PA )
	b1 = y_cen - k1 * x_cen
	l1 = k1 * xx_1 + b1


	##. lines align major-axis of satellite
	r_fact = 2.

	sat_chi = min_Rpix_sat_chi[ kk ]
	cen_off_x = r_fact * min_Rpix_sat_ar[ kk ] * np.cos( sat_chi )

	#.
	k2 = np.tan( sat_chi )
	b2 = kk_s_y - k2 * kk_s_x

	lx_0, lx_1 = kk_s_x - cen_off_x, kk_s_x + cen_off_x
	ly_0 = k2 * lx_0 + b2
	ly_1 = k2 * lx_1 + b2

	#. lines link to BCG center
	kl_0 = ( ly_0 - y_cen ) / ( lx_0 - x_cen )
	bl_0 = ly_0 - lx_0 * kl_0

	kl_1 = ( ly_1 - y_cen ) / ( lx_1 - x_cen )
	bl_1 = ly_1 - lx_1 * kl_1


	##. identify the relative position of points to the major-axis
	nx = np.linspace( 0, Nx - 1, Nx )
	ny = np.linspace( 0, Ny - 1, Ny )
	grid_nx, grid_ny = np.meshgrid( nx, ny )


	##.
	dy = kk_s_y - ( k1 * kk_s_x + b1 )
	delt_y = grid_ny - ( k1 * grid_nx + b1 )
	sign_y = dy * delt_y
	id_sign = sign_y > 0.


	dy_0 = kk_s_y - ( kl_0 * kk_s_x + bl_0 )
	delt_y0 = grid_ny - ( kl_0 * grid_nx + bl_0 )
	sign_y0 = dy_0 * delt_y0
	id_sign_0 = sign_y0 > 0.


	dy_1 = kk_s_y - ( kl_1 * kk_s_x + bl_1 )
	delt_y1 = grid_ny - ( kl_1 * grid_nx + bl_1 )
	sign_y1 = dy_1 * delt_y1
	id_sign_1 = sign_y1 > 0.

	#.
	id_msx = id_sign * id_sign_0 * id_sign_1

	copy_img = img_arr.copy()
	copy_img[ id_msx ] = np.nan


	##. align with frame
	fig = plt.figure( figsize = (10.4, 4.8) )
	ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
	ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

	ax0.imshow( img_arr, origin = 'lower', cmap = 'Greys', 
		norm = mpl.colors.SymLogNorm( linthresh = 1e-3, linscale = 1e-2, vmin = -1e-1, vmax = 5e0, base=10),)

	#.
	rect = mpathes.Ellipse(xy = (x_cen, y_cen), width = 30, height = 20, angle = ref_PA * 180 / np.pi, 
							ec = 'r', fc = 'none', ls = '-', alpha = 0.5,)
	ax0.add_patch( rect )

	ax0.scatter( kk_s_x, kk_s_y, marker = 'o', s = 30, facecolors = 'none', edgecolors = 'b', alpha = 0.5)

	ax0.plot( xx_1, l1, ls = '--', color = 'r', alpha = 0.5,)

	ax0.set_xlim( x_cen + 300, x_cen - 300 )
	ax0.set_ylim( y_cen + 300, y_cen - 300 )


	ax1.imshow( copy_img, origin = 'lower', cmap = 'Greys', 
		norm = mpl.colors.SymLogNorm( linthresh = 1e-3, linscale = 1e-2, vmin = -1e-1, vmax = 5e0, base=10),)

	rect = mpathes.Ellipse(xy = (x_cen, y_cen), width = 30, height = 20, angle = ref_PA * 180 / np.pi, 
							ec = 'r', fc = 'none', ls = '-', alpha = 0.5,)
	ax1.add_patch( rect )

	ax1.scatter( kk_s_x, kk_s_y, marker = 'o', s = 30, facecolors = 'none', edgecolors = 'b', alpha = 0.5)

	ax1.plot( xx_1, l1, ls = '--', color = 'r', alpha = 0.5,)

	ax1.set_xlim( x_cen + 300, x_cen - 300 )
	ax1.set_ylim( y_cen + 300, y_cen - 300 )

	plt.savefig('/home/xkchen/partial_cut_image_%d.png' % kk, dpi = 300)
	plt.close()


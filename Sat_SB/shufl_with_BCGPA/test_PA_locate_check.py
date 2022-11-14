import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes

import h5py
import numpy as np
import pandas as pds

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc

from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord

from img_sat_resamp import resamp_func
from img_sat_resamp import BG_resamp_func
from img_sat_BG_extract_tmp import origin_img_cut_func

from img_sat_fast_stack import sat_img_fast_stack_func
from img_sat_fast_stack import sat_BG_fast_stack_func

#.
import time
from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


##### cosmology model
Test_model = apcy.Planck15.clone( H0 = 67.74, Om0 = 0.311 )
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']
rad2arcsec = U.rad.to( U.arcsec )


### === Position Angle record
##. SDSS redMaPPer member catalog
red_cat = fits.open('/home/xkchen/mywork/ICL/data/redmapper/' + 
					'redmapper_dr8_public_v6.3_members.fits')

red_table = red_cat[1].data

all_ra = np.array( red_table['RA'] )
all_dec = np.array( red_table['DEC'] )
all_clus_ID = np.array( red_table['ID'] )


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


##.
cat_0 = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/shufle_test/img_tract_cat/' + 
					'Extend-BCGM_rgi-common_frame-limit_exlu-BCG_Sat_r-band_origin-img_position.csv')

ra_0, dec_0, z_0 = np.array( cat_0['bcg_ra'] ), np.array( cat_0['bcg_dec'] ), np.array( cat_0['bcg_z'] )

sat_ra, sat_dec = np.array( cat_0['sat_ra'] ), np.array( cat_0['sat_dec'] )
sat_x, sat_y = np.array( cat_0['sat_x'] ), np.array( cat_0['sat_y'] )

##. -pi ~ pi
PA_0 = np.array( cat_0['sat_PA2bcg'] )
IDs_0 = np.array( cat_0['clus_ID'] )


### === image load and align points test
band_str = 'r'

img_files = ( '/media/xkchen/My Passport/data/SDSS/photo_data/' + 
			'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',)[0]

R_pix = 200


##. relocated satellite and applied original PA to the longer side ~ (2048) of image frame

#.
rand_dex = 100
tt_Ng = 15

for kk in range( 20 ):

	ra_g, dec_g, z_g = ra_1[ kk ], dec_1[ kk ], z_1[ kk ]

	ref_ID = IDs_1[ kk ]
	ref_PA = PA_1[ kk ]

	try:
		#.
		id_ux = IDs_0 == ref_ID
		m_ra, m_dec = sat_ra[ id_ux ], sat_dec[ id_ux ]
		
		x_s, y_s = sat_x[ id_ux ], sat_y[ id_ux ]
		m_PAs = PA_0[ id_ux ]


		#.
		data = fits.open( img_files % (band_str, ra_g, dec_g, z_g), )
		img_arr = data[0].data

		wcs_lis = awc.WCS( data[0].header )
		x_cen, y_cen = wcs_lis.all_world2pix( ra_g, dec_g, 0 )


		#.
		dR_pix = np.sqrt( (x_s - x_cen)**2 + (y_s - y_cen)**2 )
		ordex = np.argsort( dR_pix )

		cp_xs, cp_ys = x_s[ ordex ][:tt_Ng], y_s[ ordex ][:tt_Ng]
		cp_PA2bcg = m_PAs[ ordex ][:tt_Ng]

		Da_z0 = Test_model.angular_diameter_distance( z_g ) 
		Rs_phy = dR_pix * pixel * Da_z0 / rad2arcsec

		cp_Rphy = Rs_phy[ ordex ][:tt_Ng]


		##.
		rand_ID = IDs_0[ rand_dex ]
		rand_ra, rand_dec, rand_z = ra_1[ rand_dex ], dec_1[ rand_dex ], z_1[ rand_dex ]

		rand_data = fits.open( img_files % (band_str, rand_ra, rand_dec, rand_z),)

		rand_img = rand_data[0].data
		rand_wcs = awc.WCS( rand_data[0].header )

		rand_cx, rand_cy = rand_wcs.all_world2pix( rand_ra, rand_dec, 0 )
		rand_PA = PA_1[ rand_dex ]

		Da_zn = Test_model.angular_diameter_distance( rand_z ) 
		nR_pix = ( cp_Rphy / Da_zn * rad2arcsec ) / pixel

		off_x, off_y = nR_pix * np.cos( cp_PA2bcg ), nR_pix * np.sin( cp_PA2bcg )
		rand_sx, rand_sy = rand_cx + off_x, rand_cy + off_y   ## new satellite location

		##. all member match and compare
		id_px = all_clus_ID == ref_ID

		tcp_ra, tcp_dec = all_ra[ id_px ], all_dec[ id_px ]
		tcp_x, tcp_y = wcs_lis.all_world2pix( tcp_ra, tcp_dec, 0 )


		##. align with frame
		fig = plt.figure( figsize = (10.4, 4.8) )
		ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
		ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

		ax0.imshow( img_arr, origin = 'lower', cmap = 'Greys', 
			norm = mpl.colors.SymLogNorm( linthresh = 1e-3, linscale = 1e-2, vmin = -1e-1, vmax = 5e0, base=10),)

		for nn in range( tt_Ng ):

			rect = mpathes.Circle( (cp_xs[nn], cp_ys[nn]), radius = 10, ec = mpl.cm.rainbow( nn / tt_Ng), fc = 'none',)
			ax0.add_patch( rect )

		#.
		rect = mpathes.Ellipse(xy = (x_cen, y_cen), width = 30, height = 20, angle = ref_PA, 
								ec = 'r', fc = 'none', ls = '-', alpha = 0.5,)
		ax0.add_patch( rect )

		# ax0.scatter( tcp_x, tcp_y, s = 25, marker = 'o', facecolors = 'none', edgecolors = 'c', alpha = 0.5, ls = '--')
		ax0.scatter( x_s, y_s, s = 25, marker = 'o', facecolors = 'none', edgecolors = 'c', alpha = 0.5, ls = '--')

		ax0.set_xlim( x_cen - 300, x_cen + 300 )
		ax0.set_ylim( y_cen - 300, y_cen + 300 )

		#.
		ax1.imshow( rand_img, origin = 'lower', cmap = 'Greys', 
			norm = mpl.colors.SymLogNorm( linthresh = 1e-3, linscale = 1e-2, vmin = -1e-1, vmax = 5e0, base=10),)

		for nn in range( tt_Ng ):

			rect = mpathes.Circle( (rand_sx[nn], rand_sy[nn]), radius = 10, ec = mpl.cm.rainbow( nn / tt_Ng), fc = 'none',)
			ax1.add_patch( rect )

		#.
		rect = mpathes.Ellipse(xy = (rand_cx, rand_cy), width = 30, height = 20, angle = rand_PA, ec = 'r', fc = 'none', ls = '--')
		ax1.add_patch( rect )

		ax1.set_xlim( rand_cx - 300, rand_cx + 300 )
		ax1.set_ylim( rand_cy - 300, rand_cy + 300 )

		plt.savefig('/home/xkchen/sat_random_located_pos_%d.png' % kk, dpi = 300)
		plt.close("all")

	except:
		pass

raise


##. relocated satellite and applied original PA to the major axis of BCG

#.
rand_dex = 135
tt_Ng = 15

for kk in range( 20 ):

	ra_g, dec_g, z_g = ra_1[ kk ], dec_1[ kk ], z_1[ kk ]

	ref_ID = IDs_1[ kk ]
	ref_PA = PA_1[ kk ]

	try:
		#.
		id_ux = IDs_0 == ref_ID
		m_ra, m_dec = sat_ra[ id_ux ], sat_dec[ id_ux ]
		
		x_s, y_s = sat_x[ id_ux ], sat_y[ id_ux ]
		m_PAs = PA_0[ id_ux ]

		delta_chi = m_PAs - ref_PA


		#.
		data = fits.open( img_files % (band_str, ra_g, dec_g, z_g), )
		img_arr = data[0].data

		wcs_lis = awc.WCS( data[0].header )
		x_cen, y_cen = wcs_lis.all_world2pix( ra_g, dec_g, 0 )


		#.
		dR_pix = np.sqrt( (x_s - x_cen)**2 + (y_s - y_cen)**2 )
		ordex = np.argsort( dR_pix )

		cp_xs, cp_ys = x_s[ ordex ][:tt_Ng], y_s[ ordex ][:tt_Ng]
		
		cp_PA2bcg = m_PAs[ ordex ][:tt_Ng]
		cp_delta_chi = delta_chi[ ordex ][:tt_Ng]

		Da_z0 = Test_model.angular_diameter_distance( z_g ) 
		Rs_phy = dR_pix * pixel * Da_z0 / rad2arcsec

		cp_Rphy = Rs_phy[ ordex ][:tt_Ng]


		##. relocated satellite and applied original PA to the longer side of image frame
		rand_ID = IDs_0[ rand_dex ]
		rand_ra, rand_dec, rand_z = ra_1[ rand_dex ], dec_1[ rand_dex ], z_1[ rand_dex ]

		rand_data = fits.open( img_files % (band_str, rand_ra, rand_dec, rand_z),)

		rand_img = rand_data[0].data
		rand_wcs = awc.WCS( rand_data[0].header )

		rand_cx, rand_cy = rand_wcs.all_world2pix( rand_ra, rand_dec, 0 )
		rand_PA = PA_1[ rand_dex ]

		Da_zn = Test_model.angular_diameter_distance( rand_z ) 
		nR_pix = ( cp_Rphy / Da_zn * rad2arcsec ) / pixel

		# off_x, off_y = nR_pix * np.cos( cp_delta_chi ), nR_pix * np.sin( cp_delta_chi )
		off_x, off_y = nR_pix * np.cos( cp_delta_chi + rand_PA ), nR_pix * np.sin( cp_delta_chi + rand_PA )

		rand_sx, rand_sy = rand_cx + off_x, rand_cy + off_y   ## new satellite location


		##. lines of major-axis
		xx_1 = np.linspace( x_cen - 500, x_cen + 500, 200 )

		k1 = np.tan( ref_PA )
		b1 = y_cen - k1 * x_cen
		l1 = k1 * xx_1 + b1

		xx_2 = np.linspace( rand_cx - 500, rand_cx + 500, 200 )

		k2 = np.tan( rand_PA )
		b2 = rand_cy - k2 * rand_cx
		l2 = k2 * xx_2 + b2


		##. align with frame
		fig = plt.figure( figsize = (10.4, 4.8) )
		ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
		ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

		ax0.imshow( img_arr, origin = 'lower', cmap = 'Greys', 
			norm = mpl.colors.SymLogNorm( linthresh = 1e-3, linscale = 1e-2, vmin = -1e-1, vmax = 5e0, base=10),)

		for nn in range( tt_Ng ):
			rect = mpathes.Circle( (cp_xs[nn], cp_ys[nn]), radius = 10, ec = mpl.cm.rainbow( nn / tt_Ng), fc = 'none',)
			ax0.add_patch( rect )

		#.
		rect = mpathes.Ellipse(xy = (x_cen, y_cen), width = 30, height = 20, angle = ref_PA * 180 / np.pi, 
								ec = 'r', fc = 'none', ls = '-', alpha = 0.5,)
		ax0.add_patch( rect )

		ax0.plot( xx_1, l1, ls = '--', color = 'r', alpha = 0.5,)

		ax0.set_xlim( x_cen + 300, x_cen - 300 )
		ax0.set_ylim( y_cen + 300, y_cen - 300 )

		#.
		ax1.imshow( rand_img, origin = 'lower', cmap = 'Greys', 
			norm = mpl.colors.SymLogNorm( linthresh = 1e-3, linscale = 1e-2, vmin = -1e-1, vmax = 5e0, base=10),)

		for nn in range( tt_Ng ):
			rect = mpathes.Circle( (rand_sx[nn], rand_sy[nn]), radius = 10, ec = mpl.cm.rainbow( nn / tt_Ng), fc = 'none',)
			ax1.add_patch( rect )

		#.
		rect = mpathes.Ellipse(xy = (rand_cx, rand_cy), width = 30, height = 20, 
								angle = rand_PA * 180 / np.pi, ec = 'r', fc = 'none', ls = '-')
		ax1.add_patch( rect )

		ax1.plot( xx_2, l2, ls = '--', color = 'r', alpha = 0.5,)

		ax1.set_xlim( rand_cx + 300, rand_cx - 300 )
		ax1.set_ylim( rand_cy + 300, rand_cy - 300 )

		plt.savefig('/home/xkchen/sat_random_located_pos_%d.png' % kk, dpi = 300)
		plt.close("all")

	except:
		pass


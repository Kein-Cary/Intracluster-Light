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

from scipy import optimize
from scipy import stats as sts

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

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

### ===== ### overall cluster sample P_cen view
def overviw_figs():

	sdss_file = '/home/xkchen/data/SDSS/redmapper/redmapper_dr8_public_v6.3_catalog.fits'
	cat_arr = fits.open( sdss_file )

	ra, dec, z_photo = cat_arr[1].data['RA'], cat_arr[1].data['DEC'], cat_arr[1].data['Z_LAMBDA']

	P_cen = cat_arr[1].data['P_CEN']
	RA_cen = cat_arr[1].data['RA_CEN']
	DEC_cen = cat_arr[1].data['DEC_CEN']

	idx = ( 0.2 <= z_photo ) & ( z_photo <= 0.3 )

	lim_ra, lim_dec, lim_z = ra[idx], dec[idx], z_photo[idx]
	lim_Pcen, lim_Ra_cen, lim_Dec_cen = P_cen[idx], RA_cen[idx], DEC_cen[idx]

	ref_coord = SkyCoord( ra = lim_ra * U.deg, dec = lim_dec * U.deg,)


	cat = pds.read_csv('/home/xkchen/fig_tmp/BCG_R_lim_M_cat/r-band_BCG-mag_cat.csv')
	ra_g, dec_g, z_g = np.array( cat['ra'] ), np.array( cat['dec'] ), np.array( cat['z'] )

	coord_g = SkyCoord( ra = ra_g * U.deg, dec = dec_g * U.deg,)

	idx_0, sep_0, d3d_0 = coord_g.match_to_catalog_sky( ref_coord )
	id_lim_0 = sep_0.value < 2.7e-4

	m_Pcen_0 = lim_Pcen[:,0][ idx_0[ id_lim_0 ] ]
	m_Pcen_1 = lim_Pcen[:,1][ idx_0[ id_lim_0 ] ]
	m_Pcen_2 = lim_Pcen[:,2][ idx_0[ id_lim_0 ] ]
	m_Pcen_3 = lim_Pcen[:,3][ idx_0[ id_lim_0 ] ]
	m_Pcen_4 = lim_Pcen[:,4][ idx_0[ id_lim_0 ] ]

	m_Ra_cen_0 = lim_Ra_cen[:,0][ idx_0[ id_lim_0 ] ]
	m_Ra_cen_1 = lim_Ra_cen[:,1][ idx_0[ id_lim_0 ] ]
	m_Ra_cen_2 = lim_Ra_cen[:,2][ idx_0[ id_lim_0 ] ]
	m_Ra_cen_3 = lim_Ra_cen[:,3][ idx_0[ id_lim_0 ] ]
	m_Ra_cen_4 = lim_Ra_cen[:,4][ idx_0[ id_lim_0 ] ]

	m_dec_cen_0 = lim_Dec_cen[:,0][ idx_0[ id_lim_0 ] ]
	m_dec_cen_1 = lim_Dec_cen[:,1][ idx_0[ id_lim_0 ] ]
	m_dec_cen_2 = lim_Dec_cen[:,2][ idx_0[ id_lim_0 ] ]
	m_dec_cen_3 = lim_Dec_cen[:,3][ idx_0[ id_lim_0 ] ]
	m_dec_cen_4 = lim_Dec_cen[:,4][ idx_0[ id_lim_0 ] ]

	m_ra, m_dec, m_z = ra_g[ id_lim_0 ], dec_g[ id_lim_0 ], z_g[ id_lim_0 ]
	m_P_cen = [ m_Pcen_0, m_Pcen_1, m_Pcen_2, m_Pcen_3, m_Pcen_4 ]


	#. High P_cen catalog
	hi_Pc = 0.9
	id_Px = m_Pcen_0 >= hi_Pc
	idP_ra, idP_dec, idP_z = m_ra[ id_Px ], m_dec[ id_Px ], m_z[ id_Px ]

	idP_ra_0, idP_dec_0 = m_Ra_cen_0[ id_Px ], m_dec_cen_0[ id_Px ]
	idP_ra_1, idP_dec_1 = m_Ra_cen_1[ id_Px ], m_dec_cen_1[ id_Px ]
	idP_ra_2, idP_dec_2 = m_Ra_cen_2[ id_Px ], m_dec_cen_2[ id_Px ]
	idP_ra_3, idP_dec_3 = m_Ra_cen_3[ id_Px ], m_dec_cen_3[ id_Px ]
	idP_ra_4, idP_dec_4 = m_Ra_cen_4[ id_Px ], m_dec_cen_4[ id_Px ]

	sub_N0 = len( idP_ra )

	for jj in range( sub_N0 ):

		ra_x, dec_x, z_x = idP_ra[jj], idP_dec[jj], idP_z[jj]

		data = fits.open('/home/xkchen/data/SDSS/photo_data/' + 
			'frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (ra_x, dec_x, z_x),)
		img = data[0].data
		Header = data[0].header

		wcs_lis = awc.WCS( Header )

		bcg_x0, bcg_y0 = wcs_lis.all_world2pix( idP_ra_0[jj] * U.deg, idP_dec_0[jj] * U.deg, 0 )
		bcg_x1, bcg_y1 = wcs_lis.all_world2pix( idP_ra_1[jj] * U.deg, idP_dec_1[jj] * U.deg, 0 )
		bcg_x2, bcg_y2 = wcs_lis.all_world2pix( idP_ra_2[jj] * U.deg, idP_dec_2[jj] * U.deg, 0 )
		bcg_x3, bcg_y3 = wcs_lis.all_world2pix( idP_ra_3[jj] * U.deg, idP_dec_3[jj] * U.deg, 0 )
		bcg_x4, bcg_y4 = wcs_lis.all_world2pix( idP_ra_4[jj] * U.deg, idP_dec_4[jj] * U.deg, 0 )

		plt.figure()
		plt.title('ra %.3f, dec %.3f, z %.3f [$P_{cen} \\leq %.2f$]' % (ra_x, dec_x, z_x, hi_Pc),)
		plt.imshow( img, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm(),)

		plt.scatter( bcg_x0, bcg_y0, s = 10, marker = 'o', edgecolors = 'r', facecolors = 'none', linewidth = 0.5,)
		plt.scatter( bcg_x1, bcg_y1, s = 10, marker = 's', edgecolors = mpl.cm.rainbow_r(4/5), facecolors = 'none', linewidth = 0.5,)
		plt.scatter( bcg_x2, bcg_y2, s = 10, marker = 's', edgecolors = mpl.cm.rainbow_r(3/5), facecolors = 'none', linewidth = 0.5,)
		plt.scatter( bcg_x3, bcg_y3, s = 10, marker = 's', edgecolors = mpl.cm.rainbow_r(2/5), facecolors = 'none', linewidth = 0.5,)
		plt.scatter( bcg_x4, bcg_y4, s = 10, marker = 's', edgecolors = mpl.cm.rainbow_r(1/5), facecolors = 'none', linewidth = 0.5,)

		plt.savefig('/home/xkchen/figs/hi_P-cen_ra-%.3f_dec-%.3f_z-%.3f.png' % (ra_x, dec_x, z_x), dpi = 300)
		plt.close()

	#. Low P_cen catalog
	lo_Pc = 0.9
	id_Px_1 = m_Pcen_0 < lo_Pc
	cc_dP_ra, cc_dP_dec, cc_dP_z = m_ra[ id_Px_1 ], m_dec[ id_Px_1 ], m_z[ id_Px_1 ]

	cc_dP_ra_0, cc_dP_dec_0 = m_Ra_cen_0[ id_Px_1 ], m_dec_cen_0[ id_Px_1 ]
	cc_dP_ra_1, cc_dP_dec_1 = m_Ra_cen_1[ id_Px_1 ], m_dec_cen_1[ id_Px_1 ]
	cc_dP_ra_2, cc_dP_dec_2 = m_Ra_cen_2[ id_Px_1 ], m_dec_cen_2[ id_Px_1 ]
	cc_dP_ra_3, cc_dP_dec_3 = m_Ra_cen_3[ id_Px_1 ], m_dec_cen_3[ id_Px_1 ]
	cc_dP_ra_4, cc_dP_dec_4 = m_Ra_cen_4[ id_Px_1 ], m_dec_cen_4[ id_Px_1 ]

	sub_N1 = len( cc_dP_ra )

	for kk in range( sub_N1 ):

		ra_x, dec_x, z_x = cc_dP_ra[kk], cc_dP_dec[kk], cc_dP_z[kk]
		data = fits.open('/home/xkchen/data/SDSS/photo_data/' + 
			'frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (ra_x, dec_x, z_x),)
		img = data[0].data
		Header = data[0].header

		wcs_lis = awc.WCS( Header )

		bcg_x0, bcg_y0 = wcs_lis.all_world2pix( cc_dP_ra_0[kk] * U.deg, cc_dP_dec_0[kk] * U.deg, 0 )
		bcg_x1, bcg_y1 = wcs_lis.all_world2pix( cc_dP_ra_1[kk] * U.deg, cc_dP_dec_1[kk] * U.deg, 0 )
		bcg_x2, bcg_y2 = wcs_lis.all_world2pix( cc_dP_ra_2[kk] * U.deg, cc_dP_dec_2[kk] * U.deg, 0 )
		bcg_x3, bcg_y3 = wcs_lis.all_world2pix( cc_dP_ra_3[kk] * U.deg, cc_dP_dec_3[kk] * U.deg, 0 )
		bcg_x4, bcg_y4 = wcs_lis.all_world2pix( cc_dP_ra_4[kk] * U.deg, cc_dP_dec_4[kk] * U.deg, 0 )

		plt.figure()
		plt.title('ra %.3f, dec %.3f, z %.3f [$P_{cen} \\leq %.2f$]' % (ra_x, dec_x, z_x, lo_Pc),)
		plt.imshow( img, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm(),)

		plt.scatter( bcg_x0, bcg_y0, s = 10, marker = 'o', edgecolors = 'r', facecolors = 'none', linewidth = 0.5,)
		plt.scatter( bcg_x1, bcg_y1, s = 10, marker = 's', edgecolors = mpl.cm.rainbow_r(4/5), facecolors = 'none', linewidth = 0.5,)
		plt.scatter( bcg_x2, bcg_y2, s = 10, marker = 's', edgecolors = mpl.cm.rainbow_r(3/5), facecolors = 'none', linewidth = 0.5,)
		plt.scatter( bcg_x3, bcg_y3, s = 10, marker = 's', edgecolors = mpl.cm.rainbow_r(2/5), facecolors = 'none', linewidth = 0.5,)
		plt.scatter( bcg_x4, bcg_y4, s = 10, marker = 's', edgecolors = mpl.cm.rainbow_r(1/5), facecolors = 'none', linewidth = 0.5,)

		plt.savefig('/home/xkchen/figs/low_P-cen_ra-%.3f_dec-%.3f_z-%.3f.png' % (ra_x, dec_x, z_x), dpi = 300)
		plt.close()

	return

# overviw_figs()


### ===== ### P_cen histogram and comparison
sdss_file = '/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits'
cat_arr = fits.open( sdss_file )

ra, dec, z_photo = cat_arr[1].data['RA'], cat_arr[1].data['DEC'], cat_arr[1].data['Z_LAMBDA']

P_cen = cat_arr[1].data['P_CEN']
RA_cen = cat_arr[1].data['RA_CEN']
DEC_cen = cat_arr[1].data['DEC_CEN']

idx = ( 0.2 <= z_photo ) & ( z_photo <= 0.3 )

lim_ra, lim_dec, lim_z = ra[idx], dec[idx], z_photo[idx]
lim_Pcen, lim_Ra_cen, lim_Dec_cen = P_cen[idx], RA_cen[idx], DEC_cen[idx]

ref_coord = SkyCoord( ra = lim_ra * U.deg, dec = lim_dec * U.deg,)


def P_cen_compare_fig():

	#. stacked catalog
	cat = pds.read_csv('/home/xkchen/figs/BCG_aper_M/r-band_BCG-mag_cat.csv')
	ra_g, dec_g, z_g = np.array( cat['ra'] ), np.array( cat['dec'] ), np.array( cat['z'] )

	coord_g = SkyCoord( ra = ra_g * U.deg, dec = dec_g * U.deg,)

	idx_0, sep_0, d3d_0 = coord_g.match_to_catalog_sky( ref_coord )
	id_lim_0 = sep_0.value < 2.7e-4

	m_Pcen_0 = lim_Pcen[:,0][ idx_0[ id_lim_0 ] ]
	m_Pcen_1 = lim_Pcen[:,1][ idx_0[ id_lim_0 ] ]
	m_Pcen_2 = lim_Pcen[:,2][ idx_0[ id_lim_0 ] ]
	m_Pcen_3 = lim_Pcen[:,3][ idx_0[ id_lim_0 ] ]
	m_Pcen_4 = lim_Pcen[:,4][ idx_0[ id_lim_0 ] ]

	m_Ra_cen_0 = lim_Ra_cen[:,0][ idx_0[ id_lim_0 ] ]
	m_Ra_cen_1 = lim_Ra_cen[:,1][ idx_0[ id_lim_0 ] ]
	m_Ra_cen_2 = lim_Ra_cen[:,2][ idx_0[ id_lim_0 ] ]
	m_Ra_cen_3 = lim_Ra_cen[:,3][ idx_0[ id_lim_0 ] ]
	m_Ra_cen_4 = lim_Ra_cen[:,4][ idx_0[ id_lim_0 ] ]

	m_dec_cen_0 = lim_Dec_cen[:,0][ idx_0[ id_lim_0 ] ]
	m_dec_cen_1 = lim_Dec_cen[:,1][ idx_0[ id_lim_0 ] ]
	m_dec_cen_2 = lim_Dec_cen[:,2][ idx_0[ id_lim_0 ] ]
	m_dec_cen_3 = lim_Dec_cen[:,3][ idx_0[ id_lim_0 ] ]
	m_dec_cen_4 = lim_Dec_cen[:,4][ idx_0[ id_lim_0 ] ]

	m_ra, m_dec, m_z = ra_g[ id_lim_0 ], dec_g[ id_lim_0 ], z_g[ id_lim_0 ]
	m_P_cen = [ m_Pcen_0, m_Pcen_1, m_Pcen_2, m_Pcen_3, m_Pcen_4 ]


	plt.figure()

	for kk in range( 5 ):

		plt.hist( m_P_cen[kk], bins = 55, density = False, histtype = 'step', color = mpl.cm.rainbow_r(kk/5), 
				label = 'order = %d' % kk, alpha = 0.5,)
		plt.axvline( x = np.median( m_P_cen[kk] ), ls = '--', color = mpl.cm.rainbow_r(kk/5), ymin = 0.8, ymax = 1.0,)

	plt.xlabel('$P_{cen}$')
	plt.ylabel('$ \\# $ of clusters')
	plt.yscale('log')
	plt.legend( loc = 'upper center')
	plt.savefig('/home/xkchen/P_cen_hist.png', dpi = 300)
	plt.close()


	#. fraction estimate
	p_cen_arr = np.arange(0.1, 1, 0.1)
	p_eta = []

	plt.figure()
	for jj in range( len( p_cen_arr) ):

		_sub_idx = m_Pcen_0 >= p_cen_arr[jj]
		_sub_f = np.sum( _sub_idx ) / len( m_ra )

		p_eta.append( _sub_f )

	plt.plot( p_cen_arr, p_eta, '-',)
	plt.axvline( x= 0.6, ls = ':', color = 'b',)
	plt.axvline( x= 0.7, ls = '-.', color = 'b',)
	plt.axvline( x= 0.75, ls = '--', color = 'b',)
	plt.axvline( x= 0.9, ls = '-', color = 'b',)
	plt.xlabel('$P_{c}$')
	plt.ylabel('$ \\# \,(P_{cen} >= P_{c}) \, / \, total \, number$')
	plt.savefig('/home/xkchen/P-cen_fraction.png', dpi = 300)
	plt.close()


	#. High P_cen catalog
	hi_Pc = 0.9
	id_Px = m_Pcen_0 >= hi_Pc
	idP_ra, idP_dec, idP_z = m_ra[ id_Px ], m_dec[ id_Px ], m_z[ id_Px ]

	idP_ra_0, idP_dec_0 = m_Ra_cen_0[ id_Px ], m_dec_cen_0[ id_Px ]
	idP_ra_1, idP_dec_1 = m_Ra_cen_1[ id_Px ], m_dec_cen_1[ id_Px ]
	idP_ra_2, idP_dec_2 = m_Ra_cen_2[ id_Px ], m_dec_cen_2[ id_Px ]
	idP_ra_3, idP_dec_3 = m_Ra_cen_3[ id_Px ], m_dec_cen_3[ id_Px ]
	idP_ra_4, idP_dec_4 = m_Ra_cen_4[ id_Px ], m_dec_cen_4[ id_Px ]


	tt0 = np.random.choice( len(idP_ra), 50, replace = False,)
	for jj in ( tt0 ):

		ra_x, dec_x, z_x = idP_ra[jj], idP_dec[jj], idP_z[jj]

		data = fits.open('/media/xkchen/My Passport/data/SDSS/photo_data/' + 
						'frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (ra_x, dec_x, z_x),)

		img = data[0].data
		Header = data[0].header

		wcs_lis = awc.WCS( Header )

		bcg_x0, bcg_y0 = wcs_lis.all_world2pix( idP_ra_0[jj] * U.deg, idP_dec_0[jj] * U.deg, 0 )
		bcg_x1, bcg_y1 = wcs_lis.all_world2pix( idP_ra_1[jj] * U.deg, idP_dec_1[jj] * U.deg, 0 )
		bcg_x2, bcg_y2 = wcs_lis.all_world2pix( idP_ra_2[jj] * U.deg, idP_dec_2[jj] * U.deg, 0 )
		bcg_x3, bcg_y3 = wcs_lis.all_world2pix( idP_ra_3[jj] * U.deg, idP_dec_3[jj] * U.deg, 0 )
		bcg_x4, bcg_y4 = wcs_lis.all_world2pix( idP_ra_4[jj] * U.deg, idP_dec_4[jj] * U.deg, 0 )


		plt.figure()
		plt.title('ra %.3f, dec %.3f, z %.3f [$P_{cen} \\geq %.2f$]' % (ra_x, dec_x, z_x, hi_Pc),)
		plt.imshow( img, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm(),)

		plt.scatter( bcg_x0, bcg_y0, s = 10, marker = 'o', edgecolors = 'r', facecolors = 'none', linewidth = 0.5,)
		plt.scatter( bcg_x1, bcg_y1, s = 10, marker = 's', edgecolors = mpl.cm.rainbow_r(4/5), facecolors = 'none', linewidth = 0.5,)
		plt.scatter( bcg_x2, bcg_y2, s = 10, marker = 's', edgecolors = mpl.cm.rainbow_r(3/5), facecolors = 'none', linewidth = 0.5,)
		plt.scatter( bcg_x3, bcg_y3, s = 10, marker = 's', edgecolors = mpl.cm.rainbow_r(2/5), facecolors = 'none', linewidth = 0.5,)
		plt.scatter( bcg_x4, bcg_y4, s = 10, marker = 's', edgecolors = mpl.cm.rainbow_r(1/5), facecolors = 'none', linewidth = 0.5,)

		plt.xlim( bcg_x0 - 700, bcg_x0 + 700 )
		plt.ylim( bcg_y0 - 700, bcg_y0 + 700 )
		plt.savefig('/home/xkchen/hi_P-cen_ra-%.3f_dec-%.3f_z-%.3f.png' % (ra_x, dec_x, z_x), dpi = 300)
		plt.close()


	#. Low P_cen catalog
	lo_Pc = 0.8
	id_Px_1 = m_Pcen_0 < lo_Pc
	cc_dP_ra, cc_dP_dec, cc_dP_z = m_ra[ id_Px_1 ], m_dec[ id_Px_1 ], m_z[ id_Px_1 ]

	cc_dP_ra_0, cc_dP_dec_0 = m_Ra_cen_0[ id_Px_1 ], m_dec_cen_0[ id_Px_1 ]
	cc_dP_ra_1, cc_dP_dec_1 = m_Ra_cen_1[ id_Px_1 ], m_dec_cen_1[ id_Px_1 ]
	cc_dP_ra_2, cc_dP_dec_2 = m_Ra_cen_2[ id_Px_1 ], m_dec_cen_2[ id_Px_1 ]
	cc_dP_ra_3, cc_dP_dec_3 = m_Ra_cen_3[ id_Px_1 ], m_dec_cen_3[ id_Px_1 ]
	cc_dP_ra_4, cc_dP_dec_4 = m_Ra_cen_4[ id_Px_1 ], m_dec_cen_4[ id_Px_1 ]

	for kk in range( 50 ):

		ra_x, dec_x, z_x = cc_dP_ra[kk], cc_dP_dec[kk], cc_dP_z[kk]
		data = fits.open('/media/xkchen/My Passport/data/SDSS/photo_data/' + 
						'frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (ra_x, dec_x, z_x),)

		img = data[0].data
		Header = data[0].header

		wcs_lis = awc.WCS( Header )

		bcg_x0, bcg_y0 = wcs_lis.all_world2pix( cc_dP_ra_0[kk] * U.deg, cc_dP_dec_0[kk] * U.deg, 0 )
		bcg_x1, bcg_y1 = wcs_lis.all_world2pix( cc_dP_ra_1[kk] * U.deg, cc_dP_dec_1[kk] * U.deg, 0 )
		bcg_x2, bcg_y2 = wcs_lis.all_world2pix( cc_dP_ra_2[kk] * U.deg, cc_dP_dec_2[kk] * U.deg, 0 )
		bcg_x3, bcg_y3 = wcs_lis.all_world2pix( cc_dP_ra_3[kk] * U.deg, cc_dP_dec_3[kk] * U.deg, 0 )
		bcg_x4, bcg_y4 = wcs_lis.all_world2pix( cc_dP_ra_4[kk] * U.deg, cc_dP_dec_4[kk] * U.deg, 0 )


		plt.figure()
		plt.title('ra %.3f, dec %.3f, z %.3f [$P_{cen} \\leq %.2f$]' % (ra_x, dec_x, z_x, lo_Pc),)
		plt.imshow( img, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm(),)

		plt.scatter( bcg_x0, bcg_y0, s = 10, marker = 'o', edgecolors = 'r', facecolors = 'none', linewidth = 0.5,)
		plt.scatter( bcg_x1, bcg_y1, s = 10, marker = 's', edgecolors = mpl.cm.rainbow_r(4/5), facecolors = 'none', linewidth = 0.5,)
		plt.scatter( bcg_x2, bcg_y2, s = 10, marker = 's', edgecolors = mpl.cm.rainbow_r(3/5), facecolors = 'none', linewidth = 0.5,)
		plt.scatter( bcg_x3, bcg_y3, s = 10, marker = 's', edgecolors = mpl.cm.rainbow_r(2/5), facecolors = 'none', linewidth = 0.5,)
		plt.scatter( bcg_x4, bcg_y4, s = 10, marker = 's', edgecolors = mpl.cm.rainbow_r(1/5), facecolors = 'none', linewidth = 0.5,)

		plt.xlim( bcg_x0 - 700, bcg_x0 + 700 )
		plt.ylim( bcg_y0 - 700, bcg_y0 + 700 )
		plt.savefig('/home/xkchen/low_P-cen_ra-%.3f_dec-%.3f_z-%.3f.png' % (ra_x, dec_x, z_x), dpi = 300)
		plt.close()

	return

# P_cen_compare_fig()


### === ### sub-sample match and select
tot_lo_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/' + 'low_star-Mass_cat.csv')
tot_lo_lgM = np.array( tot_lo_dat['lg_Mass'] )
tot_lo_rich = np.array( tot_lo_dat['rich'] )
tot_lo_ra, tot_lo_dec, tot_lo_z = np.array( tot_lo_dat['ra'] ), np.array( tot_lo_dat['dec'] ), np.array( tot_lo_dat['z'] )

tot_hi_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/' + 'high_star-Mass_cat.csv')
tot_hi_lgM = np.array( tot_hi_dat['lg_Mass'] )
tot_hi_rich = np.array( tot_hi_dat['rich'] )
tot_hi_ra, tot_hi_dec, tot_hi_z = np.array( tot_hi_dat['ra'] ), np.array( tot_hi_dat['dec'] ), np.array( tot_hi_dat['z'] )

tot_ra = np.r_[ tot_lo_ra, tot_hi_ra ]
tot_dec = np.r_[ tot_lo_dec, tot_hi_dec ]
tot_z = np.r_[ tot_lo_z, tot_hi_z ]
tot_rich = np.r_[ tot_lo_rich, tot_hi_rich ]

#. change the mass unit from M_sun / h^2 to M_sun
tot_lgM = np.r_[ tot_lo_lgM, tot_hi_lgM ] - 2 * np.log10( h )

tot_coord = SkyCoord( tot_ra * U.deg, tot_dec * U.deg )

# id_zx = (tot_z >= 0.2) & (tot_z <= 0.3)
# stk_ra, stk_dec, stk_z = tot_ra[ id_zx ], tot_dec[ id_zx ], tot_z[ id_zx ]
# stk_rich, stk_lgM = tot_rich[ id_zx ], tot_lgM[ id_zx ]

# tot_coord = SkyCoord( stk_ra * U.deg, stk_dec * U.deg )

# idx, d2d, d3d = tot_coord.match_to_catalog_sky( ref_coord )
# id_lim = d2d.value < 2.7e-4
# mp_Pcen = lim_Pcen[ idx[ id_lim ] ]

# plt.figure()
# plt.plot( stk_lgM, mp_Pcen[:,0], 'bo', markersize = 1,)
# plt.xlim( 10, 12.5,)
# plt.yscale('log')
# plt.show()


# Pcen_lim = 0.945 # rule about half of cluster samples
Pcen_lim = 0.85
'''
for kk in range( 3 ):

	#... sample list
	# cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
	# fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} $', 'High $ M_{\\ast}^{\\mathrm{BCG}} $']
	# hi_dat = pds.read_csv( '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_match/' + 
	# 						'high_BCG_star-Mass_%s-band_photo-z-match_BCG-pos_cat.csv' % band[kk] )
	# lo_dat = pds.read_csv( '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_match/' + 
	# 						'low_BCG_star-Mass_%s-band_photo-z-match_BCG-pos_cat.csv' % band[kk] )

	# cat_lis = ['low-lgM10', 'hi-lgM10']
	# fig_name = ['$Low \; M_{\\ast, \, 10}$', '$High \; M_{\\ast, \, 10}$']
	# hi_dat = pds.read_csv( '/home/xkchen/figs/BCG_aper_M/uniform_M2L_cat/' + 
	# 						'photo-z_match_%s-band_hi-lgM10_cluster_cat.csv' % band[kk] )
	# lo_dat = pds.read_csv( '/home/xkchen/figs/BCG_aper_M/uniform_M2L_cat/' + 
	# 						'photo-z_match_%s-band_low-lgM10_cluster_cat.csv' % band[kk] )

	cat_lis = ['low-lgM20', 'hi-lgM20']
	fig_name = ['$Low \; M_{\\ast, \, 20}$', '$High \; M_{\\ast, \, 20}$']
	hi_dat = pds.read_csv( '/home/xkchen/figs/BCG_aper_M/uniform_M2L_cat/' + 
							'photo-z_match_r-band_hi-lgM20_cluster_cat.csv' )
	lo_dat = pds.read_csv( '/home/xkchen/figs/BCG_aper_M/uniform_M2L_cat/' + 
							'photo-z_match_r-band_low-lgM20_cluster_cat.csv' )


	lo_ra, lo_dec, lo_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )
	lo_coord = SkyCoord( ra = lo_ra * U.deg, dec = lo_dec * U.deg,)

	ordex, d2d, d3d = lo_coord.match_to_catalog_sky( ref_coord )
	id_lim = d2d.value < 2.7e-4

	lo_Pcen_0 = lim_Pcen[:,0][ ordex[ id_lim ] ]
	lo_Pcen_1 = lim_Pcen[:,1][ ordex[ id_lim ] ]
	lo_Pcen_2 = lim_Pcen[:,2][ ordex[ id_lim ] ]
	lo_Pcen_3 = lim_Pcen[:,3][ ordex[ id_lim ] ]
	lo_Pcen_4 = lim_Pcen[:,4][ ordex[ id_lim ] ]

	lo_P_cen = [ lo_Pcen_0, lo_Pcen_1, lo_Pcen_2, lo_Pcen_3, lo_Pcen_4 ]

	
	##. save the properties of subsamples
	keys = ['ra', 'dec', 'z', 'P_cen_0', 'P_cen_1', 'P_cen_2', 'P_cen_3', 'P_cen_4' ]
	values = [ lo_ra, lo_dec, lo_z, lo_Pcen_0, lo_Pcen_1, lo_Pcen_2, lo_Pcen_3, lo_Pcen_4 ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( '/home/xkchen/%s_%s-band_Pcen_cat.csv' % (cat_lis[0], band[kk]), )


	#. save the P_cen selected samples
	id_Px = lo_Pcen_0 >= Pcen_lim

	print( np.sum(id_Px) / len(lo_ra) )
	bin_ra, bin_dec, bin_z = lo_ra[ id_Px ], lo_dec[ id_Px ], lo_z[ id_Px ]

	# keys = ['ra', 'dec', 'z']
	# values = [ bin_ra, bin_dec, bin_z ]
	# fill = dict( zip( keys, values ) )
	# data = pds.DataFrame( fill )
	# data.to_csv( '/home/xkchen/%s_%s-band_P-cen_lim_cat.csv' % (cat_lis[0], band[kk]), )


	#. save the properties
	p_coord = SkyCoord( ra = bin_ra * U.deg, dec = bin_dec * U.deg,)

	p_idx, p_d2d, p_d3d = p_coord.match_to_catalog_sky( tot_coord )
	id_p_lim = p_d2d.value < 2.7e-4

	bin_rich, bin_lgM_bcg = tot_rich[ p_idx[ id_p_lim ] ], tot_lgM[ p_idx[ id_p_lim ] ]

	if 'BCG_star-Mass' in cat_lis[0]:
		bin_ra, bin_dec, bin_z = bin_ra[ id_p_lim ], bin_dec[ id_p_lim ], bin_z[ id_p_lim ]

	# keys = ['ra', 'dec', 'z', 'rich', 'lg_Mbcg']
	# values = [ bin_ra, bin_dec, bin_z, bin_rich, bin_lgM_bcg ]
	# fill = dict( zip( keys, values ) )
	# data = pds.DataFrame( fill )
	# data.to_csv( '/home/xkchen/%s_%s-band_P-cen_lim_cat_params.csv' % (cat_lis[0], band[kk]), )

	tt_rich_0 = bin_rich
	tt_lgM_0 = bin_lgM_bcg


	hi_ra, hi_dec, hi_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )
	hi_coord = SkyCoord( ra = hi_ra * U.deg, dec = hi_dec * U.deg,)

	ordex_1, d2d_1, d3d_1 = hi_coord.match_to_catalog_sky( ref_coord )
	id_lim_1 = d2d_1.value < 2.7e-4

	hi_Pcen_0 = lim_Pcen[:,0][ ordex_1[ id_lim_1 ] ]
	hi_Pcen_1 = lim_Pcen[:,1][ ordex_1[ id_lim_1 ] ]
	hi_Pcen_2 = lim_Pcen[:,2][ ordex_1[ id_lim_1 ] ]
	hi_Pcen_3 = lim_Pcen[:,3][ ordex_1[ id_lim_1 ] ]
	hi_Pcen_4 = lim_Pcen[:,4][ ordex_1[ id_lim_1 ] ]

	hi_P_cen = [ hi_Pcen_0, hi_Pcen_1, hi_Pcen_2, hi_Pcen_3, hi_Pcen_4 ]

	
	##. save the properties of subsamples
	keys = ['ra', 'dec', 'z', 'P_cen_0', 'P_cen_1', 'P_cen_2', 'P_cen_3', 'P_cen_4' ]
	values = [ hi_ra, hi_dec, hi_z, hi_Pcen_0, hi_Pcen_1, hi_Pcen_2, hi_Pcen_3, hi_Pcen_4 ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( '/home/xkchen/%s_%s-band_Pcen_cat.csv' % (cat_lis[1], band[kk]), )
	

	#. save the P_cen selected samples
	id_Px = hi_Pcen_0 >= Pcen_lim

	print( np.sum(id_Px) / len(hi_ra) )
	bin_ra, bin_dec, bin_z = hi_ra[ id_Px ], hi_dec[ id_Px ], hi_z[ id_Px ]

	# keys = ['ra', 'dec', 'z']
	# values = [ bin_ra, bin_dec, bin_z ]
	# fill = dict( zip( keys, values ) )
	# data = pds.DataFrame( fill )
	# data.to_csv( '/home/xkchen/%s_%s-band_P-cen_lim_cat.csv' % (cat_lis[1], band[kk]), )


	#. save the properties
	p_coord = SkyCoord( ra = bin_ra * U.deg, dec = bin_dec * U.deg,)

	p_idx, p_d2d, p_d3d = p_coord.match_to_catalog_sky( tot_coord )
	id_p_lim = p_d2d.value < 2.7e-4

	bin_rich, bin_lgM_bcg = tot_rich[ p_idx[ id_p_lim ] ], tot_lgM[ p_idx[ id_p_lim ] ]

	# keys = ['ra', 'dec', 'z', 'rich', 'lg_Mbcg']
	# values = [ bin_ra, bin_dec, bin_z, bin_rich, bin_lgM_bcg ]
	# fill = dict( zip( keys, values ) )
	# data = pds.DataFrame( fill )
	# data.to_csv( '/home/xkchen/%s_%s-band_P-cen_lim_cat_params.csv' % (cat_lis[1], band[kk]), )

	tt_rich_1 = bin_rich
	tt_lgM_1 = bin_lgM_bcg

	plt.figure()
	plt.hist( tt_rich_0, bins = 55, density = True, histtype = 'step', color = 'b', label = fig_name[0] )
	plt.axvline( x = np.median( tt_rich_0), ls = '--', color = 'b', label = 'median', ymin = 0.0, ymax = 0.45)
	plt.axvline( x = np.mean( tt_rich_0), ls = ':', color = 'b', label = 'mean', ymin = 0.0, ymax = 0.45)

	plt.hist( tt_rich_1, bins = 55, density = True, histtype = 'step', color = 'r', label = fig_name[1] )
	plt.axvline( x = np.median( tt_rich_1), ls = '--', color = 'r', ymin = 0.0, ymax = 0.45)
	plt.axvline( x = np.mean( tt_rich_1), ls = ':', color = 'r', ymin = 0.0, ymax = 0.45)

	plt.legend( loc = 1)
	plt.xlabel('$\\lambda$')
	plt.ylabel('PDF')
	plt.xscale('log')
	plt.yscale('log')
	plt.savefig('/home/xkchen/%s-band_richness_compare.png' % band[kk], dpi = 300)
	plt.close()


	plt.figure()
	plt.hist( tt_lgM_0, bins = 55, density = True, histtype = 'step', color = 'b', label = fig_name[0] )
	plt.axvline( x = np.median( tt_lgM_0), ls = '--', color = 'b', label = 'median', ymin = 0.0, ymax = 0.45)
	plt.axvline( x = np.mean( tt_lgM_0), ls = ':', color = 'b', label = 'mean', ymin = 0.0, ymax = 0.45)

	plt.hist( tt_lgM_1, bins = 55, density = True, histtype = 'step', color = 'r', label = fig_name[1] )
	plt.axvline( x = np.median( tt_lgM_1), ls = '--', color = 'r', ymin = 0.0, ymax = 0.45)
	plt.axvline( x = np.mean( tt_lgM_1), ls = ':', color = 'r', ymin = 0.0, ymax = 0.45)

	plt.legend( loc = 1)
	plt.xlabel('$\\lg \, M_{\\ast} \; [M_{\\odot}]$')
	plt.ylabel('PDF')
	plt.xlim( 10.5, 12.5)
	plt.savefig('/home/xkchen/%s-band_lgMbcg_compare.png' % band[kk], dpi = 300)
	plt.close()

raise
'''

for kk in range( 1 ):

	#... sample list

	# cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
	# fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} $', 'High $ M_{\\ast}^{\\mathrm{BCG}} $']
	# hi_dat = pds.read_csv( '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_gri_common/' + 
	# 						'high_BCG_star-Mass_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' )
	# lo_dat = pds.read_csv( '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_gri_common/' + 
	# 						'low_BCG_star-Mass_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' )


	# cat_lis = ['low-lgM10', 'hi-lgM10']
	# fig_name = ['$Low \; M_{\\ast, \, 10}$', '$High \; M_{\\ast, \, 10}$']
	# hi_dat = pds.read_csv( '/home/xkchen/figs/BCG_aper_M/uniform_M2L_cat/' + 
	# 						'photo-z_match_hi-lgM10_gri-common_cluster_cat.csv' )
	# lo_dat = pds.read_csv( '/home/xkchen/figs/BCG_aper_M/uniform_M2L_cat/' + 
	# 						'photo-z_match_low-lgM10_gri-common_cluster_cat.csv' )


	cat_lis = ['low-lgM20', 'hi-lgM20']
	fig_name = ['$Low \; M_{\\ast, \, 20}$', '$High \; M_{\\ast, \, 20}$']
	hi_dat = pds.read_csv( '/home/xkchen/figs/BCG_aper_M/uniform_M2L_cat/' + 
							'photo-z_match_hi-lgM20_gri-common_cluster_cat.csv' )
	lo_dat = pds.read_csv( '/home/xkchen/figs/BCG_aper_M/uniform_M2L_cat/' + 
							'photo-z_match_low-lgM20_gri-common_cluster_cat.csv' )


	lo_ra, lo_dec, lo_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )
	lo_coord = SkyCoord( ra = lo_ra * U.deg, dec = lo_dec * U.deg,)

	ordex, d2d, d3d = lo_coord.match_to_catalog_sky( ref_coord )
	id_lim = d2d.value < 2.7e-4

	lo_Pcen_0 = lim_Pcen[:,0][ ordex[ id_lim ] ]
	lo_Pcen_1 = lim_Pcen[:,1][ ordex[ id_lim ] ]
	lo_Pcen_2 = lim_Pcen[:,2][ ordex[ id_lim ] ]
	lo_Pcen_3 = lim_Pcen[:,3][ ordex[ id_lim ] ]
	lo_Pcen_4 = lim_Pcen[:,4][ ordex[ id_lim ] ]

	lo_P_cen = [ lo_Pcen_0, lo_Pcen_1, lo_Pcen_2, lo_Pcen_3, lo_Pcen_4 ]


	#. save the properties of subsamples
	keys = ['ra', 'dec', 'z', 'P_cen_0', 'P_cen_1', 'P_cen_2', 'P_cen_3', 'P_cen_4' ]
	values = [ lo_ra, lo_dec, lo_z, lo_Pcen_0, lo_Pcen_1, lo_Pcen_2, lo_Pcen_3, lo_Pcen_4 ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( '/home/xkchen/%s_gri-common_Pcen_cat.csv' % cat_lis[0], )


	#. save the P_cen selected samples
	id_Px = lo_Pcen_0 >= Pcen_lim

	print( np.sum(id_Px) / len(lo_ra) )
	bin_ra, bin_dec, bin_z = lo_ra[ id_Px ], lo_dec[ id_Px ], lo_z[ id_Px ]

	# keys = ['ra', 'dec', 'z']
	# values = [ bin_ra, bin_dec, bin_z ]
	# fill = dict( zip( keys, values ) )
	# data = pds.DataFrame( fill )
	# data.to_csv( '/home/xkchen/%s_gri-common_P-cen_lim_cat.csv' % cat_lis[0], )


	#. save the properties
	p_coord = SkyCoord( ra = bin_ra * U.deg, dec = bin_dec * U.deg,)

	p_idx, p_d2d, p_d3d = p_coord.match_to_catalog_sky( tot_coord )
	id_p_lim = p_d2d.value < 2.7e-4

	bin_rich, bin_lgM_bcg = tot_rich[ p_idx[ id_p_lim ] ], tot_lgM[ p_idx[ id_p_lim ] ]

	# keys = ['ra', 'dec', 'z', 'rich', 'lg_Mbcg']
	# values = [ bin_ra, bin_dec, bin_z, bin_rich, bin_lgM_bcg ]
	# fill = dict( zip( keys, values ) )
	# data = pds.DataFrame( fill )
	# data.to_csv( '/home/xkchen/%s_gri-common_P-cen_lim_params.csv' % cat_lis[0], )

	tt_rich_0 = bin_rich
	tt_lgM_0 = bin_lgM_bcg



	hi_ra, hi_dec, hi_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )
	hi_coord = SkyCoord( ra = hi_ra * U.deg, dec = hi_dec * U.deg,)

	ordex_1, d2d_1, d3d_1 = hi_coord.match_to_catalog_sky( ref_coord )
	id_lim_1 = d2d_1.value < 2.7e-4

	hi_Pcen_0 = lim_Pcen[:,0][ ordex_1[ id_lim_1 ] ]
	hi_Pcen_1 = lim_Pcen[:,1][ ordex_1[ id_lim_1 ] ]
	hi_Pcen_2 = lim_Pcen[:,2][ ordex_1[ id_lim_1 ] ]
	hi_Pcen_3 = lim_Pcen[:,3][ ordex_1[ id_lim_1 ] ]
	hi_Pcen_4 = lim_Pcen[:,4][ ordex_1[ id_lim_1 ] ]

	hi_P_cen = [ hi_Pcen_0, hi_Pcen_1, hi_Pcen_2, hi_Pcen_3, hi_Pcen_4 ]


	#. save the properties of subsamples
	keys = ['ra', 'dec', 'z', 'P_cen_0', 'P_cen_1', 'P_cen_2', 'P_cen_3', 'P_cen_4' ]
	values = [ hi_ra, hi_dec, hi_z, hi_Pcen_0, hi_Pcen_1, hi_Pcen_2, hi_Pcen_3, hi_Pcen_4 ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( '/home/xkchen/%s_gri-common_Pcen_cat.csv' % cat_lis[1], )


	#. save the P_cen selected samples
	id_Px = hi_Pcen_0 >= Pcen_lim

	print( np.sum(id_Px) / len(hi_ra) )
	bin_ra, bin_dec, bin_z = hi_ra[ id_Px ], hi_dec[ id_Px ], hi_z[ id_Px ]

	# keys = ['ra', 'dec', 'z']
	# values = [ bin_ra, bin_dec, bin_z ]
	# fill = dict( zip( keys, values ) )
	# data = pds.DataFrame( fill )
	# data.to_csv( '/home/xkchen/%s_gri-common_P-cen_lim_cat.csv' % cat_lis[1], )


	#. save the properties
	p_coord = SkyCoord( ra = bin_ra * U.deg, dec = bin_dec * U.deg,)

	p_idx, p_d2d, p_d3d = p_coord.match_to_catalog_sky( tot_coord )
	id_p_lim = p_d2d.value < 2.7e-4

	bin_rich, bin_lgM_bcg = tot_rich[ p_idx[ id_p_lim ] ], tot_lgM[ p_idx[ id_p_lim ] ]

	# keys = ['ra', 'dec', 'z', 'rich', 'lg_Mbcg']
	# values = [ bin_ra, bin_dec, bin_z, bin_rich, bin_lgM_bcg ]
	# fill = dict( zip( keys, values ) )
	# data = pds.DataFrame( fill )
	# data.to_csv( '/home/xkchen/%s_gri-common_P-cen_lim_params.csv' % cat_lis[1], )

	tt_rich_1 = bin_rich
	tt_lgM_1 = bin_lgM_bcg


	plt.figure()
	plt.hist( tt_rich_0, bins = 55, density = True, histtype = 'step', color = 'b', label = fig_name[0] )
	plt.axvline( x = np.median( tt_rich_0), ls = '--', color = 'b', label = 'median', ymin = 0.0, ymax = 0.45)
	plt.axvline( x = np.mean( tt_rich_0), ls = ':', color = 'b', label = 'mean', ymin = 0.0, ymax = 0.45)

	plt.hist( tt_rich_1, bins = 55, density = True, histtype = 'step', color = 'r', label = fig_name[1] )
	plt.axvline( x = np.median( tt_rich_1), ls = '--', color = 'r', ymin = 0.0, ymax = 0.45)
	plt.axvline( x = np.mean( tt_rich_1), ls = ':', color = 'r', ymin = 0.0, ymax = 0.45)

	plt.legend( loc = 1)
	plt.xlabel('$\\lambda$')
	plt.ylabel('PDF')
	plt.xscale('log')
	plt.yscale('log')
	plt.savefig('/home/xkchen/richness_compare.png', dpi = 300)
	plt.close()


	plt.figure()
	plt.hist( tt_lgM_0, bins = 55, density = True, histtype = 'step', color = 'b', label = fig_name[0] )
	plt.axvline( x = np.median( tt_lgM_0), ls = '--', color = 'b', label = 'median', ymin = 0.0, ymax = 0.45)
	plt.axvline( x = np.mean( tt_lgM_0), ls = ':', color = 'b', label = 'mean', ymin = 0.0, ymax = 0.45)

	plt.hist( tt_lgM_1, bins = 55, density = True, histtype = 'step', color = 'r', label = fig_name[1] )
	plt.axvline( x = np.median( tt_lgM_1), ls = '--', color = 'r', ymin = 0.0, ymax = 0.45)
	plt.axvline( x = np.mean( tt_lgM_1), ls = ':', color = 'r', ymin = 0.0, ymax = 0.45)

	plt.legend( loc = 1)
	plt.xlabel('$\\lg \, M_{\\ast} \; [M_{\\odot}]$')
	plt.ylabel('PDF')
	plt.xlim( 10.5, 12.5)
	plt.savefig('/home/xkchen/lgMbcg_compare.png', dpi = 300)
	plt.close()



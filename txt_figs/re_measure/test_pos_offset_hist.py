import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle

import time
import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits
import subprocess as subpro

import astropy.units as U
import astropy.constants as C
import astropy.wcs as awc
import astropy.io.ascii as asc

from io import StringIO
from astropy import cosmology as apcy

from img_pre_selection import WCS_to_pixel_func
from fig_out_module import zref_BCG_pos_func

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

###.cosmology
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

### === ###
def cc_star_pos_func( star_cat, Head_info):

	wcs_lis = awc.WCS( Head_info )

	## stars catalog
	p_cat = pds.read_csv( star_cat, skiprows = 1)
	set_ra = np.array( p_cat['ra'])
	set_dec = np.array( p_cat['dec'])
	set_mag = np.array( p_cat['r'])
	OBJ = np.array( p_cat['type'])
	xt = p_cat['Column1']
	flags = [str(qq) for qq in xt]

	# x, y = wcs_lis.all_world2pix( set_ra * U.deg, set_dec * U.deg, 1, ra_dec_order = True,) ## previous
	x, y = wcs_lis.all_world2pix( set_ra * U.deg, set_dec * U.deg, 0, ra_dec_order = True,)

	set_A = np.array( [ p_cat['psffwhm_r'] , p_cat['psffwhm_g'], p_cat['psffwhm_i']]) / pixel
	set_B = np.array( [ p_cat['psffwhm_r'] , p_cat['psffwhm_g'], p_cat['psffwhm_i']]) / pixel
	set_chi = np.zeros(set_A.shape[1], dtype = np.float32)

	lln = np.array([len(set_A[:,ll][set_A[:,ll] > 0 ]) for ll in range(set_A.shape[1]) ])
	lr_iso = np.array([np.max(set_A[:,ll]) for ll in range(set_A.shape[1]) ])
	sr_iso = np.array([np.max(set_B[:,ll]) for ll in range(set_B.shape[1]) ])

	# normal stars
	iq = lln >= 2 ## at lest observed in 2 band
	ig = OBJ == 6
	ie = (set_mag <= 20)

	ic = (ie & ig & iq)
	sub_x0 = x[ic]
	sub_y0 = y[ic]

	sub_A0 = lr_iso[ic] * 15
	sub_B0 = sr_iso[ic] * 15
	sub_chi0 = set_chi[ic]

	sub_ra0, sub_dec0 = set_ra[ic], set_dec[ic]

	# saturated source(may not stars)
	xa = ['SATURATED' in qq for qq in flags]
	xv = np.array(xa)
	idx = xv == True
	ipx = (idx)

	sub_x2 = x[ipx]
	sub_y2 = y[ipx]

	sub_A2 = lr_iso[ipx] * 5
	sub_B2 = sr_iso[ipx] * 5
	sub_chi2 = set_chi[ipx]

	## for stars
	ddx = np.around( sub_x0 )
	d_x0 = np.array( list( set( ddx ) ) )

	m_A0, m_B0, m_chi0 = [], [], []
	m_x0, m_y0 = [], []

	for jj in range( len( d_x0 ) ):
		dex_0 = list( ddx ).index( d_x0[jj] )

		m_x0.append( sub_x0[dex_0] )
		m_y0.append( sub_y0[dex_0] )
		m_A0.append( sub_A0[dex_0] )
		m_B0.append( sub_B0[dex_0] )
		m_chi0.append( sub_chi0[dex_0] )

	m_A0, m_B0, m_chi0 = np.array( m_A0 ), np.array( m_B0 ), np.array( m_chi0 )
	m_x0, m_y0 = np.array( m_x0 ), np.array( m_y0 )

	# cm_x0, cm_y0 = m_x0 + 0, m_y0 + 0
	# cm_A0, cm_B0, cm_chi0 = m_A0 + 0, m_B0 + 0, m_chi0 + 0

	cm_x0, cm_y0 = sub_x0 + 0, sub_y0 + 0
	cm_A0, cm_B0, cm_chi0 = sub_A0 + 0, sub_B0 + 0, sub_chi0 + 0

	# for jj in range( len( d_x0 ) - 1 ):

	# 	fill_s = np.ones( len(m_x0), )

	# 	for pp in range( jj + 1, len( d_x0 ) ):

	# 		l_dx = np.abs( cm_x0[jj] - m_x0[pp] )
	# 		l_dy = np.abs( cm_y0[jj] - m_y0[pp] )

	# 		if ( l_dx < 1) & ( l_dy < 1):

	# 			fill_s[pp] = 0

	# 		else:
	# 			fill_s[pp] = 1

	# 	id_nul = np.where( fill_s < 1 )[0]

	# 	cm_x0[id_nul] = np.nan
	# 	cm_y0[id_nul] = np.nan
	# 	cm_A0[id_nul] = np.nan
	# 	cm_B0[id_nul] = np.nan
	# 	cm_chi0[id_nul] = np.nan

	return cm_x0, cm_y0, cm_A0, cm_B0, cm_chi0, sub_ra0, sub_dec0

def star_pos_func( star_cat, Head_info):

	## stars catalog
	p_cat = pds.read_csv( star_cat, skiprows = 1)
	set_ra = np.array( p_cat['ra'])
	set_dec = np.array( p_cat['dec'])
	set_mag = np.array( p_cat['r'])
	OBJ = np.array( p_cat['type'])
	xt = p_cat['Column1']
	flags = [str(qq) for qq in xt]

	x, y = WCS_to_pixel_func( set_ra, set_dec, Head_info ) ## SDSS EDR paper relation

	set_A = np.array( [ p_cat['psffwhm_r'] , p_cat['psffwhm_g'], p_cat['psffwhm_i']]) / pixel
	set_B = np.array( [ p_cat['psffwhm_r'] , p_cat['psffwhm_g'], p_cat['psffwhm_i']]) / pixel
	set_chi = np.zeros(set_A.shape[1], dtype = np.float32)

	lln = np.array([len(set_A[:,ll][ set_A[:,ll] > 0 ]) for ll in range(set_A.shape[1]) ])
	lr_iso = np.array([np.max(set_A[:,ll]) for ll in range(set_A.shape[1]) ])
	sr_iso = np.array([np.max(set_B[:,ll]) for ll in range(set_B.shape[1]) ])

	# normal stars
	iq = lln >= 2 ## at lest observed in 2 band
	ig = OBJ == 6
	ie = (set_mag <= 20)

	ic = (ie & ig & iq)
	sub_x0 = x[ic]
	sub_y0 = y[ic]

	sub_A0 = lr_iso[ic] * 15
	sub_B0 = sr_iso[ic] * 15
	sub_chi0 = set_chi[ic]

	sub_ra0, sub_dec0 = set_ra[ic], set_dec[ic]

	# saturated source(may not stars)
	xa = ['SATURATED' in qq for qq in flags]
	xv = np.array(xa)
	idx = xv == True
	ipx = (idx)

	sub_x2 = x[ipx]
	sub_y2 = y[ipx]

	sub_A2 = lr_iso[ipx] * 5
	sub_B2 = sr_iso[ipx] * 5
	sub_chi2 = set_chi[ipx]

	## for stars
	ddx = np.around( sub_x0 )
	d_x0 = np.array( list( set( ddx ) ) )

	m_A0, m_B0, m_chi0 = [], [], []
	m_x0, m_y0 = [], []

	for jj in range( len( d_x0 ) ):
		dex_0 = list( ddx ).index( d_x0[jj] )

		m_x0.append( sub_x0[dex_0] )
		m_y0.append( sub_y0[dex_0] )
		m_A0.append( sub_A0[dex_0] )
		m_B0.append( sub_B0[dex_0] )
		m_chi0.append( sub_chi0[dex_0] )

	m_A0, m_B0, m_chi0 = np.array( m_A0 ), np.array( m_B0 ), np.array( m_chi0 )
	m_x0, m_y0 = np.array( m_x0 ), np.array( m_y0 )

	# cm_x0, cm_y0 = m_x0 + 0, m_y0 + 0
	# cm_A0, cm_B0, cm_chi0 = m_A0 + 0, m_B0 + 0, m_chi0 + 0

	cm_x0, cm_y0 = sub_x0 + 0, sub_y0 + 0
	cm_A0, cm_B0, cm_chi0 = sub_A0 + 0, sub_B0 + 0, sub_chi0 + 0

	# for jj in range( len( d_x0 ) - 1 ):

	# 	fill_s = np.ones( len(m_x0), )

	# 	for pp in range( jj + 1, len( d_x0 ) ):

	# 		l_dx = np.abs( cm_x0[jj] - m_x0[pp] )
	# 		l_dy = np.abs( cm_y0[jj] - m_y0[pp] )

	# 		if ( l_dx < 1) & ( l_dy < 1):

	# 			fill_s[pp] = 0

	# 		else:
	# 			fill_s[pp] = 1

	# 	id_nul = np.where( fill_s < 1 )[0]

	# 	cm_x0[id_nul] = np.nan
	# 	cm_y0[id_nul] = np.nan
	# 	cm_A0[id_nul] = np.nan
	# 	cm_B0[id_nul] = np.nan
	# 	cm_chi0[id_nul] = np.nan

	return cm_x0, cm_y0, cm_A0, cm_B0, cm_chi0, sub_ra0, sub_dec0

def tractor_peak_pos( img_file, gal_cat ):

	data = fits.open( img_file )
	img = data[0].data

	source = asc.read(gal_cat)
	Numb = np.array(source['NUMBER'][-1])
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])
	theta = np.array(source['THETA_IMAGE'])
	cx = np.array(source['X_IMAGE'])
	cy = np.array(source['Y_IMAGE'])
	p_type = np.array(source['CLASS_STAR'])

	Kron = 7
	a = Kron * A
	b = Kron * B

	major = a / 2
	minor = b / 2
	senior = np.sqrt(major**2 - minor**2)

	x_peak, y_peak = [], []

	for k in range( Numb ):

		xc = cx[k]
		yc = cy[k]

		lr = major[k]
		sr = minor[k]
		cr = senior[k]
		chi = theta[k] * np.pi / 180

		set_r = np.int(np.ceil(1.2 * lr))
		la0 = np.max( [np.int(xc - set_r), 0])
		la1 = np.min( [np.int(xc + set_r + 1), img.shape[1] ] )
		lb0 = np.max( [np.int(yc - set_r), 0] ) 
		lb1 = np.min( [np.int(yc + set_r + 1), img.shape[0] ] )

		cut_img = img[ lb0 : lb1, la0 : la1 ]
		x_p, y_p = np.where( cut_img == np.nanmax( cut_img ) )

		x_peak.append( x_p[0] + la0 )
		y_peak.append( y_p[0] + lb0 )

	x_peak, y_peak = np.array( x_peak ), np.array( y_peak )

	return cx, cy, x_peak, y_peak

### === ###
load = '/home/xkchen/fig_tmp/'
home = '/home/xkchen/data/SDSS/'
out_path = '/home/xkchen/data/SDSS/photo_files/pos_offset_correct_imgs/'

# cat_lis = [ 'low-age', 'hi-age' ]
# fig_name = [ 'younger', 'older' ]

"""
for mm in range( 2 ):

	band_str = band[ rank ]

	dat = pds.read_csv( load + 'bcg_M_simi_cat/%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[mm], band_str),)
	# dat = pds.read_csv( load + 'img_cat_2_28/low_BCG_star-Mass_%s-band_BCG-pos_cat.csv' % band_str ) # test sample
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

	Ns = len( z )
	# Ns = 50

	### peak position record
	for jj in range( Ns ):

		ra_g, dec_g, z_g = ra[jj], dec[jj], z[jj]

		Da_g = Test_model.angular_diameter_distance(z_g).value
		L_pix = Da_g * 10**3 * pixel / rad2asec

		##...SExTractor sources
		galx_file = home + 'photo_files/detect_source_cat/photo-z_img_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat' % (band_str, ra_g, dec_g, z_g)
		img_file = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band_str, ra_g, dec_g, z_g)

		# galx_file = home + 'source_detect_cat/cluster_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat' % (band_str, ra_g, dec_g, z_g) # test sample
		# img_file = home + 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band_str, ra_g, dec_g, z_g)

		mx, my, x_peak, y_peak = tractor_peak_pos( img_file, galx_file )

		##...BCG location
		img_data = fits.open( img_file )
		Head = img_data[0].header

		cx_g, cy_g = WCS_to_pixel_func( ra_g, dec_g, Head)

		keys = [ 'ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'L_cen_x', 'L_cen_y', 'x_peak', 'y_peak']
		values = [ ra_g, dec_g, z_g, cx_g, cy_g, mx, my, x_peak, y_peak ]
		fill = dict( zip( keys, values ) )
		data = pds.DataFrame( fill )
		data.to_csv( out_path + 'offset/%s-band_ra%.3f_dec%.3f_z%.3f_peak-pos.csv' % (band_str, ra_g, dec_g, z_g),)

	print( '%s band, done !' % band_str )

	commd.Barrier()

	### position compare
	for jj in range( Ns ):

		ra_g, dec_g, z_g = ra[jj], dec[jj], z[jj]

		img_file = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band_str, ra_g, dec_g, z_g)
		# img_file = home + 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band_str, ra_g, dec_g, z_g)# test sample
		img_data = fits.open( img_file )

		img = img_data[0].data
		Head_info = img_data[0].header

		## tractor source position
		g_cat = pds.read_csv( out_path + 'offset/%s-band_ra%.3f_dec%.3f_z%.3f_peak-pos.csv' % (band_str, ra_g, dec_g, z_g),)
		cen_x, cen_y = np.array( g_cat['L_cen_x'] ), np.array( g_cat['L_cen_y'] ) 
		peak_x, peak_y = np.array( g_cat['x_peak'] ), np.array( g_cat['y_peak'] )

		## star catalog
		star_file = home + 'photo_files/star_cats/source_SQL_Z%.3f_ra%.3f_dec%.3f.csv' % (z_g, ra_g, dec_g)
		# star_file = home + 'new_sql_star_cat/dr12/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g)# test sample
		cm_x0, cm_y0, cm_A0, cm_B0, cm_chi0 = star_pos_func( star_file, Head_info )[:5]

		Ns0 = len( cm_x0 )

		targ_order = []

		for ii in range( Ns0 ):

			ds = np.sqrt( ( cen_x - cm_x0[ii] )**2 + ( cen_y - cm_y0[ii] )**2 )
			id_x = np.where( ds == np.nanmin( ds ) )[0][0]
			targ_order.append( id_x )

		targ_order = np.array( targ_order )

		## stars in frame region
		id_limx = ( cm_x0 >= 0 ) & ( cm_x0 <= 2048 )
		id_limy = ( cm_y0 >= 0 ) & ( cm_y0 <= 1489 )
		id_lim = id_limx & id_limy

		lim_s_x0, lim_s_y0 = cm_x0[ id_lim ], cm_y0[ id_lim ]
		lim_order = targ_order[ id_lim ]

		## offset
		off_cen = np.sqrt( ( cen_x[ lim_order ] - lim_s_x0 )**2 + ( cen_y[ lim_order ] - lim_s_y0 )**2 )
		off_peak = np.sqrt( ( peak_x[ lim_order ] - lim_s_x0 )**2 + ( peak_y[ lim_order ] - lim_s_y0 )**2 )

		devi_cenx = cen_x[ lim_order ] - lim_s_x0
		devi_ceny = cen_y[ lim_order ] - lim_s_y0

		devi_pkx = peak_x[ lim_order ] - lim_s_x0
		devi_pky = peak_y[ lim_order ] - lim_s_y0

		##.. save the off set array
		keys = [ 'offset2cen', 'offset2peak', 'devi_cenx', 'devi_ceny', 'devi_pk_x', 'devi_pk_y' ]
		values = [ off_cen, off_peak, devi_cenx, devi_ceny, devi_pkx, devi_pky ]
		fill = dict( zip( keys, values ) )
		data = pds.DataFrame( fill )
		data.to_csv( out_path + 'offset/%s-band_ra%.3f_dec%.3f_z%.3f_star-pos-offset.csv' % (band_str, ra_g, dec_g, z_g),)

		##.. adjust in astropy.WCS
		# cm_x1, cm_y1, cm_A1, cm_B1, cm_chi1 = cc_star_pos_func( star_file, Head_info )[:5]
		# lim_x1, lim_y1 = [], []
		# Ns1 = len( cm_x1 )

		# targ_order_1 = []

		# for ii in range( Ns1 ):

		# 	ds = np.sqrt( ( cen_x - cm_x1[ii] )**2 + ( cen_y - cm_y1[ii] )**2 )
		# 	id_x = np.where( ds == np.nanmin( ds ) )[0][0]
		# 	targ_order_1.append( id_x )

		# targ_order_1 = np.array( targ_order_1 )

		# id_limx = ( cm_x1 >= 0 ) & ( cm_x1 <= 2048 )
		# id_limy = ( cm_y1 >= 0 ) & ( cm_y1 <= 1489 )
		# id_lim = id_limx & id_limy

		# lim_s_x1, lim_s_y1 = cm_x1[ id_lim ], cm_y1[ id_lim ]
		# lim_order = targ_order_1[ id_lim ]

		# off_cen_1 = np.sqrt( ( cen_x[ lim_order ] - lim_s_x1 )**2 + ( cen_y[ lim_order ] - lim_s_y1 )**2 )
		# off_peak_1 = np.sqrt( ( peak_x[ lim_order ] - lim_s_x1 )**2 + ( peak_y[ lim_order ] - lim_s_y1 )**2 )


		# off_bins = np.linspace( 0, 5, 35)
		# devi_bins = np.linspace( -5, 5, 70)

		# fig = plt.figure( figsize = (19.2, 4.8) )
		# ax0 = fig.add_axes([0.05, 0.09, 0.27, 0.80])
		# ax1 = fig.add_axes([0.35, 0.09, 0.27, 0.80])
		# ax2 = fig.add_axes([0.65, 0.09, 0.27, 0.80])

		# ax0.set_title('%s band' % band_str )
		# #
		# ax0.hist( off_cen, bins = off_bins, density = True, color = 'r', histtype = 'step', ls = '-', alpha = 0.5, 
		# 	label = 'sdss $position_{ref}$, offset2cen')
		# ax0.axvline( x = np.median(off_cen), ls = '-', color = 'r', alpha = 0.5,)

		# ax0.hist( off_peak, bins = off_bins, density = True, color = 'b', histtype = 'step', ls = '-', alpha = 0.5, 
		# 	label = 'sdss $position_{ref}$, offset2peak')
		# ax0.axvline( x = np.median(off_peak), ls = '-', color = 'b', alpha = 0.5,)
		# ax0.set_xlim( 0, 5 )

		# ax0.legend( loc = 1, frameon = False,)

		# ax0.set_xlabel('star position offset [# of pixels]')
		# ax0.set_ylabel('pdf')

		# ax1.hist( devi_cenx, bins = devi_bins, density = True, color = 'r', histtype = 'step', ls = '-', alpha = 0.5, label = 'old, x2cen offset')
		# ax1.axvline( x = np.median( devi_cenx), ls = '-', color = 'r', alpha = 0.5,)
		# ax1.hist( devi_pkx, bins = devi_bins, density = True, color = 'b', histtype = 'step', ls = '-', alpha = 0.5, label = 'old, x2peak offset')
		# ax1.axvline( x = np.median( devi_pkx), ls = '-', color = 'b', alpha = 0.5,)

		# ax1.set_xlim( -5, 5 )
		# ax1.set_xlabel( 'offset along row direction [# of pixels]')
		# ax1.legend( loc = 1, frameon = False,)

		# ax2.hist( devi_ceny, bins = devi_bins, density = True, color = 'r', histtype = 'step', ls = '-', alpha = 0.5, label = 'old, y2cen offset')
		# ax2.axvline( x = np.median( devi_ceny), ls = '-', color = 'r', alpha = 0.5,)
		# ax2.hist( devi_pky, bins = devi_bins, density = True, color = 'b', histtype = 'step', ls = '-', alpha = 0.5, label = 'old, y2peak offset')
		# ax2.axvline( x = np.median( devi_pky), ls = '-', color = 'b', alpha = 0.5,)

		# ax2.set_xlim( -5, 5 )
		# ax2.set_xlabel( 'offset along column direction [# of pixels]')
		# ax2.legend( loc = 1, frameon = False,)

		# plt.savefig('/home/xkchen/%s-band_ra%.3f_dec%.3f_z%.3f_offset-hist.png' % (band_str, ra_g, dec_g, z_g), dpi = 300)
		# plt.close()

	print( '%d, %s band done !' % (mm, band_str), )
"""

"""
### BCG position with offset adjust
for mm in range( 2 ):

	band_str = band[ rank ]

	dat = pds.read_csv( load + 'bcg_M_simi_cat/%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[mm], band_str),)
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

	Ns = len( z )

	off_pk_x = np.zeros( Ns )
	off_pk_y = np.zeros( Ns )

	for jj in range( Ns ):

		ra_g, dec_g, z_g = ra[jj], dec[jj], z[jj]

		##...
		pk_dat = pds.read_csv( out_path + 'offset/%s-band_ra%.3f_dec%.3f_z%.3f_peak-pos.csv' % (band_str, ra_g, dec_g, z_g),)
		cen_x, cen_y = np.array( pk_dat['bcg_x'] )[0], np.array( pk_dat['bcg_y'] )[0]

		off_dat = pds.read_csv( out_path + 'offset/%s-band_ra%.3f_dec%.3f_z%.3f_star-pos-offset.csv' % ( band_str, ra_g, dec_g, z_g),)
		x2cen_off_arr = np.array( off_dat[ 'devi_cenx' ] )
		y2cen_off_arr = np.array( off_dat[ 'devi_ceny' ] )

		x2pk_off_arr = np.array( off_dat[ 'devi_pk_x' ] )
		y2pk_off_arr = np.array( off_dat[ 'devi_pk_y' ] )

		medi_x2pk_off = np.median( x2pk_off_arr )
		medi_y2pk_off = np.median( y2pk_off_arr )

		p_pk_x, p_pk_y = cen_x + medi_x2pk_off, cen_y + medi_y2pk_off

		off_pk_x[jj] = p_pk_x
		off_pk_y[jj] = p_pk_y

	keys = [ 'ra', 'dec', 'z', 'bcg_x', 'bcg_y' ]
	values = [ ra, dec, z, off_pk_x, off_pk_y ]
	fill = dict( zip(keys, values) )
	data = pds.DataFrame( fill )
	data.to_csv( load + 'pkoffset_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_cat.csv' % (cat_lis[mm], band_str),)

	## z_ref case..
	cat_file = load + 'pkoffset_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_cat.csv' % (cat_lis[mm], band_str)
	out_file = load + 'pkoffset_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_cat_z-ref.csv' % (cat_lis[mm], band_str)
	zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)

	print( '%d, %s band done !' % (mm, band_str), )
"""

### 
band_str = band[ rank ]

for ii in range( 3 ):

	if ii == 0:
		cat_lis = [ 'low-rich', 'hi-rich' ]
		dat_file = load + 'bcg_M_simi_cat/%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv'

	if ii == 1:
		cat_lis = [ 'younger', 'older' ]
		dat_file = load + 'z_formed_cat/%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv'

	if ii == 2:
		cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']
		dat_file = load + 'photo_z_cat/%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv'

	for mm in range( 2 ):	

		dat = pds.read_csv( dat_file % (cat_lis[mm], band_str),)
		ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

		Ns = len( ra )

		off_pk_x = np.zeros( Ns )
		off_pk_y = np.zeros( Ns )

		for jj in range( Ns ):

			ra_g, dec_g, z_g = ra[jj], dec[jj], z[jj]

			Da_g = Test_model.angular_diameter_distance(z_g).value
			L_pix = Da_g * 10**3 * pixel / rad2asec

			#...bcg location
			img_file = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band_str, ra_g, dec_g, z_g)
			img_data = fits.open( img_file )

			Head = img_data[0].header
			cen_x, cen_y = WCS_to_pixel_func( ra_g, dec_g, Head)

			off_dat = pds.read_csv( out_path + 'offset/' + 
									'%s-band_ra%.3f_dec%.3f_z%.3f_star-pos-offset.csv' % ( band_str, ra_g, dec_g, z_g),)
			x2pk_off_arr = np.array( off_dat[ 'devi_pk_x' ] )
			y2pk_off_arr = np.array( off_dat[ 'devi_pk_y' ] )

			medi_x2pk_off = np.median( x2pk_off_arr )
			medi_y2pk_off = np.median( y2pk_off_arr )

			p_pk_x, p_pk_y = cen_x + medi_x2pk_off, cen_y + medi_y2pk_off

			off_pk_x[jj] = p_pk_x
			off_pk_y[jj] = p_pk_y

		keys = [ 'ra', 'dec', 'z', 'bcg_x', 'bcg_y' ]
		values = [ ra, dec, z, off_pk_x, off_pk_y ]
		fill = dict( zip(keys, values) )
		data = pds.DataFrame( fill )
		data.to_csv( load + 'pkoffset_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_cat.csv' % (cat_lis[mm], band_str),)

		## z_ref case..
		cat_file = load + 'pkoffset_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_cat.csv' % (cat_lis[mm], band_str)
		out_file = load + 'pkoffset_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_cat_z-ref.csv' % (cat_lis[mm], band_str)
		zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)

		print( '%d, %s band done !' % (mm, band_str), )

raise

### === ### random sample
#..(no need for BCG position correction, just calculating the offset)
for kk in range( 3 ):

	band_str = band[ kk ]

	dat = pds.read_csv( load + '/random_cat/2_28/random_%s-band_tot_remain_cat_set_200-grid_6.0-sigma.csv' % band_str )
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

	zN = len( z )
	m, n = divmod(zN, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	set_z, set_ra, set_dec = z[N_sub0 : N_sub1], ra[N_sub0 : N_sub1], dec[N_sub0 : N_sub1]

	Ns = len( set_z )

	for jj in range( Ns ):

		ra_g, dec_g, z_g = set_ra[jj], set_dec[jj], set_z[jj]

		Da_g = Test_model.angular_diameter_distance(z_g).value
		L_pix = Da_g * 10**3 * pixel / rad2asec

		##...SExTractor sources
		galx_file = home + 'source_detect_cat/random_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat' % (band_str, ra_g, dec_g, z_g)
		img_file = home + 'redMap_random/rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band_str, ra_g, dec_g, z_g)

		cen_x, cen_y, peak_x, peak_y = tractor_peak_pos( img_file, galx_file )

		img_data = fits.open( img_file )
		Head_info = img_data[0].header

		bcg_x, bcg_y = WCS_to_pixel_func( ra_g, dec_g, Head_info)

		## star catalog
		star_file = home + 'new_sql_star_cat/random/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g)
		cm_x0, cm_y0, cm_A0, cm_B0, cm_chi0 = star_pos_func( star_file, Head_info )[:5]

		Ns0 = len( cm_x0 )

		targ_order = []

		for ii in range( Ns0 ):

			ds = np.sqrt( ( cen_x - cm_x0[ii] )**2 + ( cen_y - cm_y0[ii] )**2 )
			id_x = np.where( ds == np.nanmin( ds ) )[0][0]
			targ_order.append( id_x )

		targ_order = np.array( targ_order )

		## stars in frame region
		id_limx = ( cm_x0 >= 0 ) & ( cm_x0 <= 2048 )
		id_limy = ( cm_y0 >= 0 ) & ( cm_y0 <= 1489 )
		id_lim = id_limx & id_limy

		lim_s_x0, lim_s_y0 = cm_x0[ id_lim ], cm_y0[ id_lim ]
		lim_order = targ_order[ id_lim ]

		## offset
		off_cen = np.sqrt( ( cen_x[ lim_order ] - lim_s_x0 )**2 + ( cen_y[ lim_order ] - lim_s_y0 )**2 )
		off_peak = np.sqrt( ( peak_x[ lim_order ] - lim_s_x0 )**2 + ( peak_y[ lim_order ] - lim_s_y0 )**2 )

		devi_cenx = cen_x[ lim_order ] - lim_s_x0
		devi_ceny = cen_y[ lim_order ] - lim_s_y0

		devi_pkx = peak_x[ lim_order ] - lim_s_x0
		devi_pky = peak_y[ lim_order ] - lim_s_y0

		##.. save the off set array
		keys = [ 'offset2cen', 'offset2peak', 'devi_cenx', 'devi_ceny', 'devi_pk_x', 'devi_pk_y' ]
		values = [ off_cen, off_peak, devi_cenx, devi_ceny, devi_pkx, devi_pky ]
		fill = dict( zip( keys, values ) )
		data = pds.DataFrame( fill )
		data.to_csv( out_path + 'offset/random_%s-band_ra%.3f_dec%.3f_z%.3f_star-pos-offset.csv' % (band_str, ra_g, dec_g, z_g),)

	print( '%s band done !' % band_str )

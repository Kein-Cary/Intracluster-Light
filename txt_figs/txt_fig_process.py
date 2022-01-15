"""
this file use for figure adjust
"""
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


### fig test
import glob
from img_pre_selection import cat_match_func

def img_cat_lis_func(img_file, ref_cat, out_put, sf_len, id_choice = True,):
	"""
	img_file : imgs need to capture information, only including data formats(.png, .pdf, .jpg, ...)
				and the path (in which imgs are saved.), /XX/XX_raX_decX_zX.xx
	ref_cat : catalog in which match the imgs information, .csv files

	out_put : informations of match those imgs, including [ra, dec, z, bcg_x, bcg_y]
				(bcg_x, bcg_y) is the location of BCG in image frame. .csv files	
	"""
	lis = glob.glob( img_file )
	name_lis = [ ll.split('/')[-1] for ll in lis ]

	tt_lis = name_lis[0].split('_')

	dete0 = ['ra' in ll for ll in tt_lis]
	index0 = dete0.index( True )

	sub_str = tt_lis[ index0 ].split('-')
	dete0_1 = [ 'ra' in ll for ll in sub_str ]
	index0_1 = dete0_1.index( True )

	dete1 = ['dec' in ll for ll in tt_lis]
	index1 = dete1.index( True )

	sub_str = tt_lis[ index1 ].split('-')
	dete1_1 = [ 'dec' in ll for ll in sub_str ]
	index1_1 = dete1_1.index( True )

	dete2 = ['-z0.' in ll for ll in tt_lis]
	index2 = dete2.index( True )

	sub_str = tt_lis[ index2 ].split('-')
	dete2_1 = [ 'z0.' in ll for ll in sub_str ]
	index2_1 = dete2_1.index( True )

	out_ra = [ ll.split('_')[ index0 ].split('-')[ index0_1 ][2:] for ll in name_lis ]
	out_dec = [ ll.split('_')[ index1 ].split('-')[ index1_1 ][3:] for ll in name_lis ]
	out_z = [ ll.split('_')[ index2 ].split('-')[ index2_1 ][1:] for ll in name_lis ]

	ref_dat = pds.read_csv( ref_cat )
	cat_ra, cat_dec, cat_z = np.array(ref_dat.ra), np.array(ref_dat.dec), np.array(ref_dat.z)
	clus_x, clus_y = np.array(ref_dat.bcg_x), np.array(ref_dat.bcg_y)

	lis_ra, lis_dec, lis_z, lis_x, lis_y, cat_order = cat_match_func( out_ra, out_dec, out_z, cat_ra, cat_dec, cat_z, clus_x, clus_y, sf_len, id_choice,)

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
	values = [lis_ra, lis_dec, lis_z, lis_x, lis_y]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_put )

	return

def star_pos_func( star_file, wcs_lis):

	## stars
	p_cat = pds.read_csv( star_file, skiprows = 1)
	set_ra = np.array( p_cat['ra'])
	set_dec = np.array( p_cat['dec'])
	set_mag = np.array( p_cat['r'])
	OBJ = np.array( p_cat['type'])
	xt = p_cat['Column1']
	flags = [str(qq) for qq in xt]

	x, y = wcs_lis.all_world2pix(set_ra * U.deg, set_dec * U.deg, 0)

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

	sub_A0 = lr_iso[ic] * 30 # 15
	sub_B0 = sr_iso[ic] * 30 # 15
	sub_chi0 = set_chi[ic]

	sub_ra0, sub_dec0 = set_ra[ic], set_dec[ic]

	# saturated source(may not stars)
	xa = ['SATURATED' in qq for qq in flags]
	xv = np.array(xa)
	idx = xv == True
	ipx = (idx)

	sub_x2 = x[ipx]
	sub_y2 = y[ipx]

	sub_A2 = lr_iso[ipx] * 75 # 5
	sub_B2 = sr_iso[ipx] * 75 # 5
	sub_chi2 = set_chi[ipx]

	## stars
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

	cm_x0, cm_y0 = m_x0 + 0, m_y0 + 0
	cm_A0, cm_B0, cm_chi0 = m_A0 + 0, m_B0 + 0, m_chi0 + 0

	for jj in range( len( d_x0 ) - 1 ):

		fill_s = np.ones( len(m_x0), )

		for pp in range( jj + 1, len( d_x0 ) ):

			l_dx = np.abs( cm_x0[jj] - m_x0[pp] )
			l_dy = np.abs( cm_y0[jj] - m_y0[pp] )

			if ( l_dx < 1) & ( l_dy < 1):

				fill_s[pp] = 0

			else:
				fill_s[pp] = 1

		id_nul = np.where( fill_s < 1 )[0]

		cm_x0[id_nul] = np.nan
		cm_y0[id_nul] = np.nan
		cm_A0[id_nul] = np.nan
		cm_B0[id_nul] = np.nan
		cm_chi0[id_nul] = np.nan

	## saturated pixle
	ddx = np.around( sub_x2 )
	d_x2 = np.array( list( set( ddx ) ) )

	m_A2, m_B2, m_chi2 = [], [], []
	m_x2, m_y2 = [], []

	for jj in range( len( d_x2 ) ):

		dex_2 = list( ddx ).index( d_x2[jj] )

		m_x2.append( sub_x2[dex_2] )
		m_y2.append( sub_y2[dex_2] )
		m_A2.append( sub_A2[dex_2] )
		m_B2.append( sub_B2[dex_2] )
		m_chi2.append( sub_chi2[dex_2] )

	m_A2, m_B2, m_chi2 = np.array( m_A2 ), np.array( m_B2 ), np.array( m_chi2 )
	m_x2, m_y2 = np.array( m_x2 ), np.array( m_y2 )

	cm_x2, cm_y2 = m_x2 + 0, m_y2 + 0
	cm_A2, cm_B2, cm_chi2 = m_A2 + 0, m_B2 + 0, m_chi2 + 0

	for jj in range( len( d_x2 ) - 1 ):

		fill_s = np.ones( len(m_x2), )

		for pp in range( jj + 1, len( d_x2 ) ):

			l_dx = np.abs( cm_x2[jj] - m_x2[pp] )
			l_dy = np.abs( cm_y2[jj] - m_y2[pp] )

			if ( l_dx < 1) & ( l_dy < 1):

				fill_s[pp] = 0

			else:
				fill_s[pp] = 1

		id_nul = np.where( fill_s < 1 )[0]

		cm_x2[id_nul] = np.nan
		cm_y2[id_nul] = np.nan
		cm_A2[id_nul] = np.nan
		cm_B2[id_nul] = np.nan
		cm_chi2[id_nul] = np.nan

	return cm_x0, cm_y0, cm_A0, cm_B0, cm_chi0, cm_x2, cm_y2, cm_A2, cm_B2, cm_chi2

def tmp_mask_func( img_file, gal_arr ):

	data = fits.open( img_file )
	img = data[0].data

	cx, cy, a, b, theta = gal_arr[:]

	major = a / 2
	minor = b / 2
	senior = np.sqrt(major**2 - minor**2)

	Numb = len( major )

	mask_path = np.ones((img.shape[0], img.shape[1]), dtype = np.float32)

	ox = np.linspace(0, img.shape[1] - 1, img.shape[1])
	oy = np.linspace(0, img.shape[0] - 1, img.shape[0])

	basic_coord = np.array(np.meshgrid(ox, oy))

	# masking 'galaxies'
	for k in range( Numb ):

		xc = cx[k]
		yc = cy[k]

		lr = major[k]
		sr = minor[k]
		cr = senior[k]
		chi = theta[k] * np.pi / 180

		if np.isnan( lr ):
			continue

		else:
			set_r = np.int( np.ceil(1.2 * lr) )
			la0 = np.max( [np.int(xc - set_r), 0])
			la1 = np.min( [np.int(xc + set_r + 1), img.shape[1] ] )
			lb0 = np.max( [np.int(yc - set_r), 0] ) 
			lb1 = np.min( [np.int(yc + set_r + 1), img.shape[0] ] )

			df1 = (basic_coord[0,:][lb0: lb1, la0: la1] - xc) * np.cos(chi) + (basic_coord[1,:][lb0: lb1, la0: la1] - yc) * np.sin(chi)
			df2 = (basic_coord[1,:][lb0: lb1, la0: la1] - yc) * np.cos(chi) - (basic_coord[0,:][lb0: lb1, la0: la1] - xc) * np.sin(chi)
			fr = df1**2 / lr**2 + df2**2 / sr**2
			jx = fr <= 1

			iu = np.where(jx == True)
			iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float32)
			iv[iu] = np.nan
			mask_path[lb0: lb1, la0: la1] = mask_path[lb0: lb1, la0: la1] * iv

	mask_img = mask_path * img

	return mask_img, mask_path


def ccp_mask_func( img_file, gal_arr ):

	data = fits.open( img_file )
	img = data[0].data

	cx, cy, a, b, theta = gal_arr[:]

	major = a / 2
	minor = b / 2
	senior = np.sqrt(major**2 - minor**2)

	Numb = len( major )

	N_wide = np.int( 100 * 2 )
	Ny, Nx = img.shape[0] + N_wide, img.shape[1] + N_wide
	mask_path = np.ones( (Ny, Nx), dtype = np.float32 )

	#. offset the position of objects
	cx, cy = cx + N_wide / 2, cy + N_wide / 2

	#. extend img with extra-pixel number
	extnd_img = np.ones( (Ny, Nx), dtype = np.float32 ) * -100

	extnd_img[ 100: -100, 100: -100] = img + 0.
	id_vx = extnd_img == -100
	extnd_img[ id_vx ] = np.nan

	ox = np.linspace(0, Nx - 1, Nx )
	oy = np.linspace(0, Ny - 1, Ny )

	basic_coord = np.array( np.meshgrid(ox, oy) )

	# masking 'galaxies'
	for k in range( Numb ):

		xc = cx[k]
		yc = cy[k]

		lr = major[k]
		sr = minor[k]
		cr = senior[k]
		chi = theta[k] * np.pi / 180

		if np.isnan( lr ):
			continue

		else:
			set_r = np.int( np.ceil(1.2 * lr) )
			la0 = np.max( [np.int(xc - set_r), 0])
			la1 = np.min( [np.int(xc + set_r + 1), Nx ] )
			lb0 = np.max( [np.int(yc - set_r), 0] ) 
			lb1 = np.min( [np.int(yc + set_r + 1), Ny ] )

			df1 = (basic_coord[0,:][lb0: lb1, la0: la1] - xc) * np.cos(chi) + (basic_coord[1,:][lb0: lb1, la0: la1] - yc) * np.sin(chi)
			df2 = (basic_coord[1,:][lb0: lb1, la0: la1] - yc) * np.cos(chi) - (basic_coord[0,:][lb0: lb1, la0: la1] - xc) * np.sin(chi)
			fr = df1**2 / lr**2 + df2**2 / sr**2
			jx = fr <= 1

			iu = np.where(jx == True)
			iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float32)
			iv[iu] = np.nan
			mask_path[lb0: lb1, la0: la1] = mask_path[lb0: lb1, la0: la1] * iv

	mask_img = mask_path * extnd_img

	return mask_img, mask_path


##.. info. match (masked images with BCG left)
ref_cat = '/home/xkchen/mywork/ICL/data/cat_select/match_2_28/cluster_tot-r-band_norm-img_cat.csv'
img_file = '/home/xkchen/tmp_run/data_files/figs/targ_1/spec-z_r-band_ra*-dec*-z*_compare.png'
out_cat = '/home/xkchen/tmp_run/data_files/figs/targ_info/tmp_img_cat_1.csv'

sf_len = 3
id_choice = True

# img_cat_lis_func(img_file, ref_cat, out_cat, sf_len, id_choice = id_choice,)



#... imgs
img_path = '/home/xkchen/tmp_run/data_files/figs/targ_info/z_photo_cat/'
galx_cat = '/home/xkchen/tmp_run/data_files/figs/targ_info/z_photo_cat/photo-z_img_r-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
star_cat = '/home/xkchen/tmp_run/data_files/figs/targ_info/z_photo_cat/source_SQL_Z%.3f_ra%.3f_dec%.3f.csv'

dat = pds.read_csv('/home/xkchen/tmp_run/data_files/figs/targ_info/z_spec_cat/tmp_img_cat_1.csv')
ra, dec, z_spec = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
imgx, imgy = np.array( dat['bcg_x'] ), np.array( dat['bcg_y'] )


#... redMaPPer information (use photo-z matched information)
inf_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/photo_cat/redMapper_z-photo_cat.csv')
inf_ra, inf_dec, inf_z = np.array( inf_dat['ra']), np.array( inf_dat['dec']), np.array( inf_dat['z'])
inf_ID, inf_rich = np.array( inf_dat['objID']), np.array( inf_dat['rich'])

ra_lis = ['%.5f' % ll for ll in inf_ra]
dec_lis = ['%.5f' % ll for ll in inf_dec]
z_lis = ['%.5f' % ll for ll in inf_z]


order_dex = extra_match_func( ra_lis, dec_lis, z_lis, ra, dec, z_spec, imgx, imgy,)[-1]

sub_objID = inf_ID[ order_dex ]
sub_rich = inf_rich[ order_dex ]
sub_ra, sub_dec, sub_z = inf_ra[ order_dex ], inf_dec[ order_dex ], inf_z[ order_dex ]


# for ii in range( 22, 23 ):
# for ii in ( 4, 13, 18, 25 ):
for ii in range( 4, 5 ):

	ra_g, dec_g, z_g = sub_ra[ii], sub_dec[ii], sub_z[ii]
	cen_x, cen_y = imgx[ii], imgy[ii]

	#. origin image
	data = fits.open( img_path + 'frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (ra_g, dec_g, z_g),)
	img = data[0].data

	head = data[0].header
	wcs_lis = awc.WCS( head )

	#. masked image
	data_m = fits.open( '/home/xkchen/tmp_run/data_files/figs/targ_info/z_photo_cat/' + 
						'photo-z_mask_r_ra%.3f_dec%.3f_z%.3f.fits' % (ra_g, dec_g, z_g),)
	img_m = data_m[0].data

	#. gri-combine image
	ref_img = mpimg.imread( '/home/xkchen/tmp_run/data_files/figs/frame-irg-002189-4-0142_cc.jpg' )


	## obj_cat
	source = asc.read( galx_cat % (ra_g, dec_g, z_g),)
	Numb = np.array( source['NUMBER'][-1] )
	A = np.array( source['A_IMAGE'] )
	B = np.array( source['B_IMAGE'] )
	theta = np.array( source['THETA_IMAGE'] )
	cx = np.array( source['X_IMAGE'] )
	cy = np.array( source['Y_IMAGE'] )
	p_type = np.array( source['CLASS_STAR'] )

	Kron = 16
	a = Kron * A
	b = Kron * B

	## stars
	p_cat = pds.read_csv( star_cat % (z_g, ra_g, dec_g), skiprows = 1)
	set_ra = np.array( p_cat['ra'])
	set_dec = np.array( p_cat['dec'])
	set_mag = np.array( p_cat['r'])
	OBJ = np.array( p_cat['type'])
	xt = p_cat['Column1']
	flags = [str(qq) for qq in xt]

	x, y = wcs_lis.all_world2pix(set_ra * U.deg, set_dec * U.deg, 0)

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

	sub_A0 = lr_iso[ic] * 30
	sub_B0 = sr_iso[ic] * 30
	sub_chi0 = set_chi[ic]

	sub_ra0, sub_dec0 = set_ra[ic], set_dec[ic]

	## galaxy, star seperate
	g_ra, g_dec = wcs_lis.all_pix2world( cx, cy, 0)

	coord_0 = SkyCoord( g_ra * U.deg, g_dec * U.deg )
	coord_1 = SkyCoord( sub_ra0 * U.deg, sub_dec0 * U.deg )

	# match stars to detected sources
	dex, d2d, d3d = coord_0.match_to_catalog_sky( coord_1 )

	id_lim = d2d.value <= 1e-3 # 1e-4
	print( np.sum( id_lim ) )

	dd_cx, dd_cy = cx[ id_lim == False ], cy[ id_lim == False ]
	dd_chi, dd_a, dd_b = theta[ id_lim == False ], a[ id_lim == False ], b[ id_lim == False ]

	star_file = star_cat % (z_g, ra_g, dec_g)
	pos_arr = star_pos_func( star_file, wcs_lis)
	cm_x0, cm_y0, cm_A0, cm_B0, cm_chi0, cm_x2, cm_y2, cm_A2, cm_B2, cm_chi2 = pos_arr[:]

	#. capture the merger mask of saturated pixels
	tmp_gal_arr = [ cm_x2, cm_y2, cm_A2, cm_B2, cm_chi2 ]
	tmp_files = img_path + 'frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (ra_g, dec_g, z_g)

	# tmp_mark_img, tmp_mark_arr = tmp_mask_func( tmp_files, tmp_gal_arr )

 	# extend the mask region of saturated pixels
	tmp_mark_img, tmp_mark_arr = ccp_mask_func( tmp_files, tmp_gal_arr )

	id_Nul = np.isnan( tmp_mark_arr )
	tmp_mark_arr[ id_Nul ] = -100.


	Da_z = Test_model.angular_diameter_distance( z_g ).value
	L_pix = Da_z * 10**3 * pixel / rad2asec

	R1Mpc = 1e3 / L_pix
	R200kpc = 2e2 / L_pix
	R100kpc = 1e2 / L_pix

	c_map = ['Greys', 'cividis', 'viridis'][0]
	v_min, v_max = -5e-3, 3e-2

	fig = plt.figure( figsize = (12.8, 9.4) )
	ax0 = fig.add_axes( [0.08, 0.53, 0.45, 0.46] )
	ax1 = fig.add_axes( [0.53, 0.53, 0.45, 0.46] )
	ax2 = fig.add_axes( [0.08, 0.07, 0.45, 0.46] )
	ax3 = fig.add_axes( [0.53, 0.07, 0.45, 0.46] )

	ax0.imshow( ref_img[::-1, :],)

	clust = Circle( xy = (cen_x, cen_y), radius = R1Mpc, fill = False, ec = 'w', ls = '--', linewidth = 1, alpha = 0.75,)
	ax0.add_patch(clust)

	clust = Circle( xy = (cen_x, cen_y), radius = R100kpc, fill = False, ec = 'w', ls = '-', linewidth = 1, alpha = 0.75,)
	ax0.add_patch(clust)

	ax0.set_xlim( -100, img.shape[1] + 100 )
	ax0.set_ylim( -100, img.shape[0] + 100 )

	ax0.set_xticklabels( [] )
	ax0.set_ylabel( 'Y [ Pixel # ]', fontsize = 14,)

	y_tick = np.arange(0, 1800, 300)
	ax0.set_yticks( y_tick )
	ax0.set_yticklabels( labels = ['%d' % pp for pp in y_tick],)
	
	x_tick = np.arange(0, 2400, 300)
	ax0.set_xticks( x_tick )
	ax0.set_xticklabels( labels = [],)
	ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15, top = True, right = True,)

	ax0.text( 100, 1400, s = '$ \\mathrm{SDSS} \; \; \\mathrm{J132109.53{+}643300.5}$', fontsize = 16, color = 'w',)
	ax0.text( 100, 1250, s = '$\\lambda$ : %.1f' % sub_rich[ii] + '   z : %.3f' % z_spec[ii], fontsize = 16, color = 'w',)

	# ax0.text( cen_x + 100 / L_pix, cen_y - 30, s = 'BCG', fontsize = 12, color = 'w',)
	ax0.text( cen_x - R1Mpc, cen_y, s = '$1\\mathrm{Mpc}$', fontsize = 12, color = 'w',)
	ax0.text( -50, 1500, s = '(a)', fontsize = 12, color = 'k',)


	ax1.imshow( img, origin = 'lower', cmap = c_map, vmin = -1e-3, vmax = 1e0, 
				norm = mpl.colors.SymLogNorm( linthresh = 0.001, linscale = 0.1, base = 10),)

	clust = Circle( xy = (cen_x, cen_y), radius = R1Mpc, fill = False, ec = 'r', ls = '--', linewidth = 1, alpha = 0.75, label = '1.0 Mpc')
	ax1.add_patch(clust)

	clust = Circle( xy = (cen_x, cen_y), radius = R100kpc, fill = False, ec = 'r', ls = '-', linewidth = 1, alpha = 0.75, label = '0.1 Mpc')
	ax1.add_patch(clust)

	ax1.set_xlim( -100, img.shape[1] + 100 )
	ax1.set_ylim( -100, img.shape[0] + 100 )

	ax1.set_xticklabels( [] )
	ax1.set_yticklabels( [] )

	ax1.set_yticks( y_tick )
	ax1.set_yticklabels( labels = [],)
	ax1.set_xticks( x_tick )
	ax1.set_xticklabels( labels = [],)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15, top = True, right = True,)
	ax1.text( -50, 1500, s = '(b)', fontsize = 12, color = 'k',)


	ax2.imshow( img, origin = 'lower', cmap = c_map, vmin = -1e-3, vmax = 1e0, 
			norm = mpl.colors.SymLogNorm( linthresh = 0.001, linscale = 0.1, base = 10),)

	Ng = len( dd_cx )
	for ll in range( Ng ):
		ellips = Ellipse( xy = (dd_cx[ll], dd_cy[ll]), width = dd_a[ll], height = dd_b[ll], angle = dd_chi[ll], fill = False, 
			ec = 'm', ls = '-', linewidth = 0.75, )
		ax2.add_patch( ellips )


	Ns0 = len( cm_A0 )
	for ll in range( Ns0 ):
		ellips = Ellipse( xy = (cm_x0[ll], cm_y0[ll]), width = cm_A0[ll], height = cm_B0[ll], angle = cm_chi0[ll], fill = False, 
			ec = 'c', ls = '-', linewidth = 0.75,)
		ax2.add_patch( ellips )


	# Ns2 = len( cm_A2 )
	# for ll in range( Ns2 ):
	# 	ellips = Ellipse( xy = (cm_x2[ll], cm_y2[ll]), width = cm_A2[ll], height = cm_B2[ll], angle = cm_chi2[ll], fill = False, 
	# 		ec = 'k', ls = '-', linewidth = 0.75,)
	# 	ax2.add_patch( ellips )

	# ax2.contour( tmp_mark_arr, origin = 'lower', levels = [1, 100], colors = ['k', 'w'], linestyles = '--', alpha = 0.75,)
	ax2.contour( tmp_mark_arr, origin = 'lower', levels = [1, 100], colors = ['k', 'w'], linestyles = '--', alpha = 0.75, 
				extent = (-100, img.shape[1] + 100, -100, img.shape[0] + 100), )

	ax2.set_xlim( -100, img.shape[1] + 100 )
	ax2.set_ylim( -100, img.shape[0] + 100 )

	ax2.set_xlabel( 'X [ Pixel # ]', fontsize = 14,)
	ax2.set_ylabel( 'Y [ Pixel # ]', fontsize = 14,)

	ax2.set_yticks( y_tick )
	ax2.set_yticklabels( labels = ['%d' % pp for pp in y_tick],)
	ax2.set_xticks( x_tick )
	ax2.set_xticklabels( labels = ['%d' % pp for pp in x_tick],)
	ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15, top = True, right = True,)
	ax2.text( -50, 1500, s = '(c)', fontsize = 12, color = 'k',)


	ax3.imshow( img_m, origin = 'lower', cmap = c_map, vmin = -1e-3, vmax = 1e0, 
			norm = mpl.colors.SymLogNorm( linthresh = 0.001, linscale = 0.1, base = 10),)
	# ax3.imshow( img_m, origin = 'lower', cmap = c_map, vmin = -1.2e-3, vmax = 1e0, 
	# 		norm = mpl.colors.SymLogNorm( linthresh = 0.001, linscale = 0.1, base = 10),)

	clust = Circle( xy = (cen_x, cen_y), radius = R1Mpc, fill = False, ec = 'r', ls = '--', linewidth = 1, alpha = 0.75,)
	ax3.add_patch(clust)

	clust = Circle( xy = (cen_x, cen_y), radius = R100kpc, fill = False, ec = 'r', ls = '-', linewidth = 1, alpha = 0.75,)
	ax3.add_patch(clust)

	ax3.set_xlim( -100, img.shape[1] + 100 )
	ax3.set_ylim( -100, img.shape[0] + 100 )

	ax3.set_yticklabels( [] )
	ax3.set_xlabel( 'X [ Pixel # ]', fontsize = 14,)

	ax3.set_yticks( y_tick )
	ax3.set_yticklabels( labels = [],)
	ax3.set_xticks( x_tick )
	ax3.set_xticklabels( labels = ['%d' % pp for pp in x_tick],)
	ax3.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15, top = True, right = True,)
	ax3.text( -50, 1500, s = '(d)', fontsize = 12, color = 'k',)

	# plt.savefig('/home/xkchen/img_process.png', dpi = 200)
	plt.savefig('/home/xkchen/img_process.pdf', dpi = 100)
	plt.close()

raise
